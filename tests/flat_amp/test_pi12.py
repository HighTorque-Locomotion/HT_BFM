"""Pi12 data/observation contracts, executable without starting Isaac Sim."""

import ast
import copy
import importlib.util
import math
import re
import xml.etree.ElementTree as ET
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from rsl_rl.utils.motion_loader_forpi_12dof import LEG_JOINT_NAMES, AMPLoaderPi12, selected_motion_files

from humanoidverse.flat_amp import VENDOR_ROOT
from humanoidverse.flat_amp.__main__ import build_invocation, build_parser
from humanoidverse.flat_amp.pi12_paths import motion_yaml, robot_urdf


def _quat_conjugate(q):
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def _quat_apply(q, v):
    t = 2.0 * torch.cross(q[..., 1:], v, dim=-1)
    return v + q[..., :1] * t + torch.cross(q[..., 1:], t, dim=-1)


def test_all_pi12_reward_selectors_match_urdf(urdf):
    """Evaluate the standalone Pi12 rewards without Isaac's renderer."""

    def configclass(cls):
        def initialize(self):
            for base in reversed(cls.__mro__):
                for name, value in vars(base).items():
                    if not name.startswith("_") and not callable(value):
                        setattr(self, name, copy.deepcopy(value))

        cls.__init__ = initialize
        return cls

    class RewardFunctions:
        def __getattr__(self, name):
            return name

    def entity(name, **kwargs):
        return SimpleNamespace(name=name, **kwargs)

    namespace = {
        "math": math,
        "configclass": configclass,
        "RewTerm": SimpleNamespace,
        "SceneEntityCfg": entity,
        "mdp": RewardFunctions(),
    }
    path = VENDOR_ROOT / "legged_lab/envs/pi/pi_12dof_cfg.py"
    tree = ast.parse(path.read_text())
    constants = {"LEG_JOINT_NAMES", "PI_LIN_COMMAND_THRESHOLD", "PI_YAW_COMMAND_THRESHOLD"}
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets)
    ]
    nodes.append(next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PiDof12RewardCfg"))
    for node in nodes:
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)

    reward = namespace["PiDof12RewardCfg"]()
    root = ET.parse(urdf)
    available = {
        "joint_names": [j.attrib["name"] for j in root.findall("joint") if j.attrib["type"] != "fixed"],
        "body_names": [link.attrib["name"] for link in root.findall("link")],
    }
    for term_name, term in vars(reward).items():
        if term is None:
            continue
        for parameter in getattr(term, "params", {}).values():
            for field, names in available.items():
                patterns = getattr(parameter, field, None)
                if patterns is None:
                    continue
                patterns = [patterns] if isinstance(patterns, str) else patterns
                for pattern in patterns:
                    assert any(re.fullmatch(pattern, name) for name in names), f"{term_name}:{field}: {pattern}"
                assert all(sum(bool(re.fullmatch(pattern, name)) for pattern in patterns) <= 1 for name in names)
    stand_patterns = reward.stand_still.params["asset_cfg"].joint_names
    assert {n for n in available["joint_names"] if any(re.fullmatch(p, n) for p in stand_patterns)} == set(LEG_JOINT_NAMES)
    assert reward.joint_deviation_arms is None
    assert reward.joint_deviation_waist_walk is None


@pytest.fixture
def urdf():
    path = robot_urdf()
    if not path.is_file():
        pytest.skip("Requires the supplied Pi12 URDF")
    return path


@pytest.fixture
def synthetic_motion(tmp_path):
    count = 20
    dt = 0.02
    names = [*reversed(LEG_JOINT_NAMES), "head_pitch_joint"]
    # Constant yaw of +90 degrees, moving along world +Y (= body +X).
    root_pos = np.zeros((count, 3))
    root_pos[:, 1] = np.arange(count) * dt
    quat = np.tile([np.sqrt(0.5), 0, 0, np.sqrt(0.5)], (count, 1))
    joints = np.zeros((count, len(names)))
    for index, name in enumerate(names):
        if "hip_pitch" in name:
            joints[:, index] = -0.25
        elif "calf" in name:
            joints[:, index] = 0.65
        elif "ankle_pitch" in name:
            joints[:, index] = -0.4
    joints[:, -1] = 100.0  # Upper-body values must have no effect.
    content = dict(joint_names=names, joint_pos=joints, base_pos_w=root_pos, base_quat_w=quat, framerate=1 / dt)
    path = tmp_path / "motion.npz"
    np.savez(path, **content)
    return path, content


def _loader(path, urdf):
    return AMPLoaderPi12("cpu", 0.02, motion_files=[str(path)], urdf_path=urdf)


def test_all_selected_experts_load_and_sample(urdf):
    if not motion_yaml().is_file():
        pytest.skip("Requires the Pi12 motion-selection YAML and NPZ files")
    files = selected_motion_files(motion_yaml())
    loader = AMPLoaderPi12(
        "cpu",
        0.02,
        motion_yaml=motion_yaml(),
        urdf_path=urdf,
        preload_transitions=True,
        num_preload_transitions=128,
    )
    assert loader.num_motions == len(files)
    assert loader.observation_dim == 36
    assert torch.isfinite(loader.all_trajectories_full).all()
    current, following = next(loader.feed_forward_generator(1, 32))
    assert current.shape == following.shape == (32, 36)


def test_joint_name_order_root_frame_and_quaternion_sign(synthetic_motion, urdf, tmp_path):
    path, content = synthetic_motion
    loader = _loader(path, urdf)
    features = loader.trajectories[0]
    torch.testing.assert_close(features[:, :3], torch.tensor([[1.0, 0, 0]]).expand(20, -1))
    torch.testing.assert_close(features[:, 3:6], torch.zeros(20, 3))
    expected_pose = torch.tensor([[-0.25, 0, 0, 0.65, -0.4, 0] * 2]).expand(20, -1)
    torch.testing.assert_close(features[:, 6:18], expected_pose)
    torch.testing.assert_close(features[:, 18:30], torch.zeros(20, 12))
    # Mirrored default pose gives matching foot X/Z, opposite Y.
    torch.testing.assert_close(features[:, 30:33] * torch.tensor([1, -1, 1]), features[:, 33:36], atol=1e-4, rtol=1e-4)
    perm = np.random.default_rng(42).permutation(13)
    content["joint_names"] = np.asarray(content["joint_names"])[perm]
    content["joint_pos"] = content["joint_pos"][:, perm]
    content["base_quat_w"][::2] *= -1
    reordered = tmp_path / "reordered.npz"
    np.savez(reordered, **content)
    torch.testing.assert_close(_loader(reordered, urdf).trajectories[0], features)


def test_missing_leg_joint_fails_instead_of_guessing_columns(synthetic_motion, urdf, tmp_path):
    _, content = synthetic_motion
    content["joint_names"][0] = "unknown_joint"
    broken = tmp_path / "broken.npz"
    np.savez(broken, **content)
    with pytest.raises(ValueError, match="missing joint"):
        _loader(broken, urdf)


def _environment_class():
    # Execute the actual tensor observation methods without importing Isaac's
    # renderer; state tensors below stand in for ArticulationData.
    path = VENDOR_ROOT / "legged_lab/envs/pi/pi_12dof_env.py"
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "PiDof12Env")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    namespace = {
        "VecEnv": object,
        "torch": torch,
        "quat_apply": _quat_apply,
        "quat_conjugate": _quat_conjugate,
        "get_euler_xyz_tensor": lambda quat: torch.zeros(quat.shape[0], 3),
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["PiDof12Env"]


def test_online_amp_and_actor_observations_match_expert_contract(synthetic_motion, urdf):
    path, content = synthetic_motion
    loader = _loader(path, urdf)
    features = loader.trajectories[0][:1]
    # Simulate a different PhysX joint/body order and a translated/rotated root.
    order = np.random.default_rng(12).permutation(12).tolist()
    names = [LEG_JOINT_NAMES[i] for i in order]
    q = features[:, 6:18][:, order]
    root_pos = torch.tensor([[3.0, 5.0, 0.351]])
    quat = torch.tensor(content["base_quat_w"][:1], dtype=torch.float32)
    body_state = torch.zeros(1, 4, 13)
    body_state[:, :, 3] = 1
    body_state[:, 0, :3] = root_pos
    body_state[:, 0, 3:7] = quat
    for body_id, offset in ((3, features[:, 30:33]), (1, features[:, 33:36])):
        body_state[:, body_id, :3] = root_pos + _quat_apply(quat, offset)
    data = SimpleNamespace(
        root_state_w=body_state[:, 0],
        body_state_w=body_state,
        root_lin_vel_b=features[:, :3],
        root_ang_vel_b=features[:, 3:6],
        joint_pos=q,
        joint_vel=torch.zeros_like(q),
        default_joint_pos=torch.zeros_like(q),
        default_joint_vel=torch.zeros_like(q),
        projected_gravity_b=torch.tensor([[0.0, 0, -1.0]]),
    )
    cls = _environment_class()
    env = cls.__new__(cls)
    env.amp_upper_body = False
    env.robot = SimpleNamespace(data=data, joint_names=names)
    env.left_leg_ids = [names.index(n) for n in LEG_JOINT_NAMES[:6]]
    env.right_leg_ids = [names.index(n) for n in LEG_JOINT_NAMES[6:]]
    env.feet_body_ids = [3, 1]
    env.torso_body_ids = [2]
    env.amp_obs_dim = 36
    torch.testing.assert_close(env.get_amp_obs_for_expert_trans(), features, atol=1e-6, rtol=1e-5)
    env.contact_sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w_history=torch.zeros(1, 3, 4, 3)))
    env.feet_cfg = SimpleNamespace(body_ids=[3, 1])
    env.command_generator = SimpleNamespace(command=torch.tensor([[0.4, 0, 0]]))
    env.obs_scales = SimpleNamespace(
        **dict.fromkeys(("ang_vel", "projected_gravity", "commands", "joint_pos", "joint_vel", "actions", "lin_vel"), 1.0)
    )
    env.action_buffer = SimpleNamespace(_circular_buffer=SimpleNamespace(buffer=torch.zeros(1, 1, 12)))
    env.phase = torch.zeros(1)
    actor, critic = env.compute_current_observations()
    assert actor.shape == (1, 47)
    assert critic.shape == (1, 52)
    torch.testing.assert_close(actor[:, 9:21], q)
    torch.testing.assert_close(critic[:, 47:50], features[:, :3])


def test_pi12_mirror_is_an_involution():
    path = VENDOR_ROOT / "legged_lab/envs/pi/symmetry.py"
    spec = importlib.util.spec_from_file_location("pi12_symmetry_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    env = SimpleNamespace(robot=SimpleNamespace(joint_names=list(reversed(LEG_JOINT_NAMES))))
    obs = torch.randn(4, 470)
    actions = torch.randn(4, 12)
    mirrored, mirrored_actions = module.compute_symmetric_states(obs, actions, env, "policy_history")
    twice, twice_actions = module.compute_symmetric_states(mirrored[4:], mirrored_actions[4:], env, "policy_history")
    torch.testing.assert_close(twice[4:], obs)
    torch.testing.assert_close(twice_actions[4:], actions)


def test_pi12_command_dispatch_and_profile():
    parser = build_parser()
    args = parser.parse_args(["train", "--task", "walk_pi_waq_12", "--controller", "amp"])
    module, options = build_invocation(args)
    assert module == "legged_lab.scripts.train"
    assert options[options.index("--task") + 1] == "walk_pi_waq_12"
    args.actuator_profile = "legacy"
    with pytest.raises(ValueError, match="5047"):
        build_invocation(args)


def test_invalid_yaml_reports_missing_motion(tmp_path):
    selection = tmp_path / "selection.yaml"
    selection.write_text("selected_files:\n- absent.npz\n")
    with pytest.raises(FileNotFoundError, match="absent.npz"):
        selected_motion_files(selection)


@pytest.mark.parametrize("reset_ids", [[1], [0, 1], []])
def test_pi12_preserves_terminal_features_before_reset(reset_ids):
    # Exercise the actual expanded step/reset methods, with only the simulator
    # and managers mocked. Reset deliberately overwrites the observed state.
    cls = _environment_class()
    env = cls.__new__(cls)
    env.num_envs, env.device, env.headless = 2, "cpu", True
    env.cfg = SimpleNamespace(sim=SimpleNamespace(decimation=1), robot=SimpleNamespace(critic_obs_history_length=1))
    env.step_dt = env.physics_dt = 0.02
    env.sim_step_counter = 0
    env.clip_actions, env.action_scale = 100.0, 0.1
    env.joint_encoder_bias = torch.zeros(2, 12)
    env.feet_cfg = SimpleNamespace(body_ids=[0, 1])
    env.feet_body_ids, env.torso_body_ids = [0, 1], [0]
    env.robot = SimpleNamespace(
        data=SimpleNamespace(
            default_joint_pos=torch.zeros(2, 12),
            body_lin_vel_w=torch.zeros(2, 2, 3),
            body_quat_w=torch.tensor([1.0, 0, 0, 0]).expand(2, 2, 4),
        ),
        set_joint_position_target=Mock(),
    )
    env.contact_sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w=torch.zeros(2, 2, 3)))
    env.episode_length_buf = torch.zeros(2, dtype=torch.long)
    env.phase_length_buf = torch.zeros(2, dtype=torch.long)
    env.gait_start = torch.zeros(2)
    env.reset_reason_buf = {}
    env.extras = {"terminal_amp_states": torch.full((1, 36), 99.0)}
    amp = torch.ones(2, 36)
    actor = torch.full((2, 47), 2.0)
    critic = torch.zeros(2, 52)
    env.get_amp_obs_for_expert_trans = lambda: amp
    env.compute_current_observations = lambda: (actor, critic)
    env.compute_observations = lambda: (actor, critic, actor, actor, critic)
    env._calculate_gait_para = Mock()
    dones = torch.zeros(2, dtype=torch.bool)
    dones[reset_ids] = True
    env.check_reset = lambda: (dones, torch.zeros(2, dtype=torch.bool))
    env.sim = Mock()
    env.scene = Mock()

    def reset_state(env_ids):
        amp[env_ids] = -1
        actor[env_ids] = -2

    env.scene.reset.side_effect = reset_state
    env.event_manager = SimpleNamespace(available_modes=[])
    env.reward_manager = Mock()
    env.reward_manager.compute.return_value = torch.zeros(2)
    env.reward_manager.reset.return_value = {}
    env.command_generator = Mock()
    env.actor_obs_buffer, env.critic_obs_buffer = Mock(), Mock()
    env.action_buffer = Mock()
    env.action_buffer.compute.side_effect = lambda actions: actions
    env._play_window = Mock()

    _, _, returned_dones, extras = env.step(torch.zeros(2, 12))
    torch.testing.assert_close(returned_dones, dones)
    torch.testing.assert_close(extras["terminal_amp_states"], torch.ones(len(reset_ids), 36))
    torch.testing.assert_close(extras["terminal_actor_obs"], torch.full((len(reset_ids), 47), 2.0))
    torch.testing.assert_close(amp[reset_ids], -torch.ones(len(reset_ids), 36))
    assert env._capture_amp_transition is False
    env._play_window.update.assert_called_once()
    # An explicit reset outside step must not overwrite the last transition.
    saved_terminal = extras["terminal_amp_states"].clone()
    env.reset(torch.tensor([0]))
    torch.testing.assert_close(extras["terminal_amp_states"], saved_terminal)


def test_pi12_clears_transition_capture_after_step_error():
    cls = _environment_class()
    env = cls.__new__(cls)
    env.action_buffer = Mock()
    env.action_buffer.compute.side_effect = RuntimeError("simulated action failure")
    with pytest.raises(RuntimeError, match="simulated action failure"):
        env.step(torch.zeros(2, 12))
    assert env._capture_amp_transition is False


def test_pi12_motor_uses_delayed_pd_path():
    class DelayedPD:
        def compute(self, action, joint_pos, joint_vel):
            self.delay_path_called = True
            return action

    path = VENDOR_ROOT / "legged_lab/assets/pi/pi_12dof.py"
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef))
    namespace = {"DelayedPDActuator": DelayedPD, "torch": torch, "Pi12HTMotorCfg": object}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    motor = namespace["Pi12HTMotor"].__new__(namespace["Pi12HTMotor"])
    motor._joint_vel = torch.zeros(2, 12)
    velocity = torch.randn(2, 12)
    action = object()
    assert motor.compute(action, torch.zeros(2, 12), velocity) is action
    assert motor.delay_path_called
    torch.testing.assert_close(motor._joint_vel, velocity)
