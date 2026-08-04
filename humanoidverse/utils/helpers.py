import os
import copy
import torch
from torch import nn
import numpy as np
import random

from collections.abc import Sequence
from typing import Any, List, Dict
from termcolor import colored
from loguru import logger
from omegaconf import DictConfig, OmegaConf

def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def pre_process_config(config) -> None:
    
    # compute observation_dim
    # config.robot.policy_obs_dim = -1
    # config.robot.critic_obs_dim = -1
    
    obs_dim_dict = dict()
    _obs_key_list = config.env.config.obs.obs_dict
    _aux_obs_key_list = config.env.config.obs.obs_auxiliary
    
    assert set(config.env.config.obs.noise_scales.keys()) == set(config.env.config.obs.obs_scales.keys())

    # convert obs_dims to list of dicts
    # import ipdb; ipdb.set_trace()
    import omegaconf
    from omegaconf import ListConfig
    if isinstance(config.env.config.obs.obs_dims, ListConfig):
        each_dict_obs_dims = {k: v for d in config.env.config.obs.obs_dims for k, v in d.items()}
        config.env.config.obs.obs_dims = each_dict_obs_dims
    logger.info(f"obs_dims: {config.env.config.obs.obs_dims}")
    auxiliary_obs_dims = {}
    for aux_obs_key, aux_config in _aux_obs_key_list.items():
        auxiliary_obs_dims[aux_obs_key] = 0
        for _key, _num in aux_config.items():
            try:
                assert _key in config.env.config.obs.obs_dims.keys()
                auxiliary_obs_dims[aux_obs_key] += config.env.config.obs.obs_dims[_key] * _num
            except:
                # import ipdb; ipdb.set_trace()
                logger.warning(f"aux_obs_key: {aux_obs_key} not found in obs_dims")
    logger.info(f"auxiliary_obs_dims: {auxiliary_obs_dims}")
    for obs_key, obs_config in _obs_key_list.items():
        obs_dim_dict[obs_key] = 0
        for key in obs_config:
            if key.endswith("_raw"): key = key[:-4]
            if key in config.env.config.obs.obs_dims.keys(): 
                obs_dim_dict[obs_key] += config.env.config.obs.obs_dims[key]
                logger.info(f"{obs_key}: {key} has dim: {config.env.config.obs.obs_dims[key]}")
            else:
                obs_dim_dict[obs_key] += auxiliary_obs_dims[key]
                logger.info(f"{obs_key}: {key} has dim: {auxiliary_obs_dims[key]}")
    config.robot.algo_obs_dim_dict = obs_dim_dict
    
    OmegaConf.set_struct(config.env.config.obs.obs_dims, False)
    config.env.config.obs.obs_dims.update(auxiliary_obs_dims) # ZL: adding auxiliary obs dims to obs_dims
    
    logger.info(f"algo_obs_dim_dict: {config.robot.algo_obs_dim_dict}")

    # compute action_dim for ppo
    # for agent in config.algo.config.network_dict.keys():
    #     for network in config.algo.config.network_dict[agent].keys():
    #         output_dim = config.algo.config.network_dict[agent][network].output_dim
    #         if output_dim == "action_dim":
    #             config.algo.config.network_dict[agent][network].output_dim = config.env.config.robot.actions_dim
                
    # print the config
    # logger.debug(f"PPO CONFIG")
    # logger.debug(f"{config.algo.config.module_dict}")
    # logger.debug(f"{config.algo.config.network_dict}")

def parse_observation(cls: Any, 
                      key_list: List, 
                      buf_dict: Dict, 
                      obs_scales: Dict, 
                      noise_scales: Dict,
                      current_noise_curriculum_value: Any,
                      use_noise: bool = True) -> None:
    """ Parse observations for the legged_robot_base class
    """
    # TOOD: Parse observations for manipulation tasks

    for obs_key in key_list:
        if use_noise:
            obs_noise = noise_scales[obs_key] * current_noise_curriculum_value
        else:
            obs_noise = 0.
        
        # print(f"obs_key: {obs_key}, obs_noise: {obs_noise}")
        
        actor_obs = getattr(cls, f"_get_obs_{obs_key}")().clone()
        obs_scale = obs_scales[obs_key]
        
       
        buf_dict[obs_key] = (actor_obs + (torch.rand_like(actor_obs)* 2. - 1.) * obs_noise) * obs_scale


def export_policy_as_jit(actor_critic, path):
    if hasattr(actor_critic, 'memory_a'):
        # assumes LSTM: TODO add GRU
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    else: 
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_1.pt')
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)

class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.
 
    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_lstm_1.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


def get_backward_observation(env, motion_id, use_root_height_obs: bool = False, velocity_multiplier: float = 1.0) -> torch.Tensor:
    from humanoidverse.utils.torch_utils import quat_apply, quat_mul, quat_rotate_inverse
    from humanoidverse.envs.legged_robot_motions.legged_robot_motions import compute_humanoid_observations_max, compute_humanoid_observations_max_with_contact
    
    motion_times = torch.arange(int(np.ceil((env._motion_lib._motion_lengths[motion_id]/env.dt).cpu()))).to(env.device) * env.dt
    
    # get blend motion state
    motion_state = env._motion_lib.get_motion_state(motion_id, motion_times)

    ref_body_pos = motion_state["rg_pos_t"]
    ref_body_rots = motion_state["rg_rot_t"]
    ref_body_vels = motion_state["body_vel_t"] * velocity_multiplier
    ref_body_angular_vels = motion_state["body_ang_vel_t"] * velocity_multiplier
    ref_dof_pos = motion_state["dof_pos"] - env.default_dof_pos[0]
    ref_dof_vel = motion_state["dof_vel"] * velocity_multiplier

    # The deployed PiPlus BFM encoder was generated from qpos with MuJoCo FK
    # using the 260424 robot model.  MotionLib uses the 260611 skeleton, whose
    # link frames are not identical.  Reconstruct the encoder bodies with the
    # deployed FK model so the ONNX encoder sees its training-time geometry.
    encoder_fk_xml = env.config.robot.get("bfm_encoder_fk_xml_file", None)
    fk_encoder_body_names = None
    if encoder_fk_xml:
        import mujoco

        print(f"BFM z encoder FK XML: {encoder_fk_xml}")
        fk_model = mujoco.MjModel.from_xml_path(str(encoder_fk_xml))
        fk_data = mujoco.MjData(fk_model)
        fk_encoder_body_names = list(env.config.robot.isaacsim_body_names)
        dof_names = list(env.config.robot.dof_names)
        body_ids = []
        qpos_addresses = []
        for body_name in fk_encoder_body_names:
            body_id = mujoco.mj_name2id(fk_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                raise ValueError(f"BFM FK XML is missing body {body_name!r}")
            body_ids.append(body_id)
        for dof_name in dof_names:
            joint_id = mujoco.mj_name2id(fk_model, mujoco.mjtObj.mjOBJ_JOINT, dof_name)
            if joint_id < 0:
                raise ValueError(f"BFM FK XML is missing joint {dof_name!r}")
            qpos_addresses.append(int(fk_model.jnt_qposadr[joint_id]))

        motion_dof = motion_state["dof_pos"].detach().cpu().numpy()
        root_pos = motion_state["rg_pos_t"][:, 0].detach().cpu().numpy()
        root_rot = motion_state["rg_rot_t"][:, 0].detach().cpu().numpy()  # xyzw
        frame_count = motion_dof.shape[0]
        body_pos_np = np.empty((frame_count, len(body_ids), 3), dtype=np.float32)
        body_rot_np = np.empty((frame_count, len(body_ids), 4), dtype=np.float32)
        for frame_index in range(frame_count):
            fk_data.qpos[:] = 0.0
            fk_data.qvel[:] = 0.0
            fk_data.qpos[:3] = root_pos[frame_index]
            fk_data.qpos[3:7] = root_rot[frame_index, [3, 0, 1, 2]]  # xyzw -> wxyz
            fk_data.qpos[qpos_addresses] = motion_dof[frame_index]
            mujoco.mj_kinematics(fk_model, fk_data)
            body_pos_np[frame_index] = fk_data.xpos[body_ids]
            body_rot_np[frame_index] = fk_data.xquat[body_ids][:, [1, 2, 3, 0]]

        # Match the onboard converter: frame zero is stationary and following
        # velocities are backward finite differences at the control rate.
        dt = float(env.dt)
        dof_vel_np = np.zeros_like(motion_dof, dtype=np.float32)
        body_vel_np = np.zeros_like(body_pos_np, dtype=np.float32)
        body_ang_vel_np = np.zeros_like(body_pos_np, dtype=np.float32)
        if frame_count > 1:
            dof_vel_np[1:] = (motion_dof[1:] - motion_dof[:-1]) / dt
            body_vel_np[1:] = (body_pos_np[1:] - body_pos_np[:-1]) / dt
            previous_conjugate = body_rot_np[:-1].copy()
            previous_conjugate[..., :3] *= -1.0
            delta_rot = quat_mul(
                torch.from_numpy(body_rot_np[1:]).reshape(-1, 4),
                torch.from_numpy(previous_conjugate).reshape(-1, 4),
                w_last=True,
            ).view(frame_count - 1, len(body_ids), 4).numpy()
            delta_xyz = delta_rot[..., :3]
            delta_w = np.clip(delta_rot[..., 3], -1.0, 1.0)
            angle = 2.0 * np.arctan2(np.linalg.norm(delta_xyz, axis=-1), delta_w)
            axis = delta_xyz / np.maximum(np.linalg.norm(delta_xyz, axis=-1, keepdims=True), 1e-8)
            body_ang_vel_np[1:] = axis * angle[..., None] / dt

        ref_body_pos = torch.from_numpy(body_pos_np).to(env.device)
        ref_body_rots = torch.from_numpy(body_rot_np).to(env.device)
        ref_body_vels = torch.from_numpy(body_vel_np * velocity_multiplier).to(env.device)
        ref_body_angular_vels = torch.from_numpy(body_ang_vel_np * velocity_multiplier).to(env.device)
        ref_dof_vel = torch.from_numpy(dof_vel_np * velocity_multiplier).to(env.device)

    if getattr(env, "motion_body_ids", None) is not None:
        # The deployed PiPlus z encoder consumes the Isaac body set plus three
        # virtual hand/head points.  They are encoder features, not MotionLib
        # joints, so derive them from their physical parent body states.
        encoder_body_names = list(env.config.robot.isaacsim_body_names)
        if fk_encoder_body_names is None:
            motion_body_names = list(env._motion_lib.mesh_parsers.body_names)
            missing_bodies = [name for name in encoder_body_names if name not in motion_body_names]
            if missing_bodies:
                raise ValueError(
                    "Motion data is missing bodies required by the BFM z encoder: "
                    f"{missing_bodies}"
                )
            encoder_body_ids = torch.tensor(
                [motion_body_names.index(name) for name in encoder_body_names],
                device=env.device,
                dtype=torch.long,
            )
            ref_body_pos = ref_body_pos[:, encoder_body_ids]
            ref_body_rots = ref_body_rots[:, encoder_body_ids]
            ref_body_vels = ref_body_vels[:, encoder_body_ids]
            ref_body_angular_vels = ref_body_angular_vels[:, encoder_body_ids]

        encoder_extensions = env.config.robot.get("bfm_encoder_extend_config", [])
        if encoder_extensions:
            parent_ids = []
            local_positions = []
            local_rotations = []
            for extension in encoder_extensions:
                parent_name = str(extension["parent_name"])
                if parent_name not in encoder_body_names:
                    raise ValueError(
                        f"BFM z encoder extension parent is not an Isaac body: {parent_name}"
                    )
                parent_ids.append(encoder_body_names.index(parent_name))
                local_positions.append(extension["pos"])
                # Config rotations are wxyz; HumanoidVerse tensors are xyzw.
                local_rotations.append(extension["rot"][1:] + extension["rot"][:1])

            parent_ids = torch.tensor(parent_ids, device=env.device, dtype=torch.long)
            local_positions = torch.tensor(
                local_positions, device=env.device, dtype=ref_body_pos.dtype
            ).unsqueeze(0).expand(ref_body_pos.shape[0], -1, -1)
            local_rotations = torch.tensor(
                local_rotations, device=env.device, dtype=ref_body_rots.dtype
            ).unsqueeze(0).expand(ref_body_rots.shape[0], -1, -1)
            parent_pos = ref_body_pos[:, parent_ids]
            parent_rot = ref_body_rots[:, parent_ids]
            parent_vel = ref_body_vels[:, parent_ids]
            parent_ang_vel = ref_body_angular_vels[:, parent_ids]
            world_offsets = quat_apply(
                parent_rot.reshape(-1, 4), local_positions.reshape(-1, 3), w_last=True
            ).view_as(local_positions)
            extension_pos = parent_pos + world_offsets
            extension_rot = quat_mul(
                parent_rot.reshape(-1, 4), local_rotations.reshape(-1, 4), w_last=True
            ).view_as(local_rotations)
            extension_vel = parent_vel + torch.cross(parent_ang_vel, world_offsets, dim=-1)
            ref_body_pos = torch.cat([ref_body_pos, extension_pos], dim=1)
            ref_body_rots = torch.cat([ref_body_rots, extension_rot], dim=1)
            ref_body_vels = torch.cat([ref_body_vels, extension_vel], dim=1)
            ref_body_angular_vels = torch.cat([ref_body_angular_vels, parent_ang_vel], dim=1)
            encoder_body_names.extend(str(extension["joint_name"]) for extension in encoder_extensions)

        print(
            "BFM z encoder bodies: "
            f"physical={len(env.config.robot.isaacsim_body_names)}, "
            f"extended={len(encoder_body_names) - len(env.config.robot.isaacsim_body_names)}, "
            f"total={len(encoder_body_names)}"
        )

    # construct observation
    if env.use_contact_in_obs_max:
        contact_binary = env.foot_contact_detect(ref_body_pos, ref_body_vels)
        obs_dict = compute_humanoid_observations_max_with_contact(
            ref_body_pos,
            ref_body_rots,
            ref_body_vels,
            ref_body_angular_vels,
            local_root_obs=True,
            root_height_obs=use_root_height_obs,
            contact_binary=contact_binary
        )
    else:
        obs_dict = compute_humanoid_observations_max(
            ref_body_pos,
            ref_body_rots,
            ref_body_vels,
            ref_body_angular_vels,
            local_root_obs=True,
            root_height_obs=use_root_height_obs,
        )
    max_local_self_obs = torch.cat([v for v in obs_dict.values()], dim=-1)
    print(f"BFM z encoder observation: state=52, privileged={max_local_self_obs.shape[-1]}")

    if env.config.obs.use_obs_filter:
        imu_body_idx = 0
        imu_body_name = env.config.robot.get("imu_body_name", None)
        if imu_body_name is not None:
            motion_body_names = encoder_body_names if getattr(env, "motion_body_ids", None) is not None else getattr(env, "motion_body_names", None)
            if motion_body_names is None:
                motion_body_names = env._motion_lib.mesh_parsers.body_names
            imu_body_idx = motion_body_names.index(imu_body_name)

        imu_quat = ref_body_rots[:, imu_body_idx]
        ref_ang_vel = ref_body_angular_vels[:, imu_body_idx]
        projected_gravity = quat_rotate_inverse(
            imu_quat,
            env.gravity_vec[0:1].repeat(max_local_self_obs.shape[0], 1),
            w_last=True
        )
        bogus_actions = ref_dof_pos

        bogus_history_actor = torch.cat([bogus_actions, ref_ang_vel, ref_dof_pos, ref_dof_vel, projected_gravity], dim=-1).repeat(1, 4)
        ref_dict = {
            "actions": bogus_actions,
            "ref_ang_vel": ref_ang_vel,
            "ref_dof_pos": ref_dof_pos,
            "ref_dof_vel": ref_dof_vel,
            "dof_pos": motion_state["dof_pos"],
            "fake_history": bogus_history_actor,
            "max_local_self_obs": max_local_self_obs,
            "projected_gravity": projected_gravity,
            "ref_body_pos": ref_body_pos,
            "ref_body_rots": ref_body_rots,
            "ref_body_vels": ref_body_vels,
            "ref_body_angular_vels": ref_body_angular_vels
        }
        state = torch.cat([ref_dof_pos,
                    ref_dof_vel,
                    projected_gravity,
                    ref_ang_vel], dim=-1)
        last_action = bogus_actions
        # TODO get obs from raw obs instead of the obs
        bfmzero_obs = {
            "state": state,
            "last_action": last_action,
            "privileged_state": max_local_self_obs
        }
        return bfmzero_obs, ref_dict
    else:
        ref_dict = {
            "max_local_self_obs": max_local_self_obs,
        }
        return max_local_self_obs, ref_dict


def _get_space_dim(space) -> int:
    if not hasattr(space, "shape") or len(space.shape) != 1:
        raise ValueError(f"Expected a 1D observation space, got {space}")
    return int(space.shape[0])


def _infer_actor_obs_layout(inference_model, history: bool) -> list[tuple[str, int]]:
    obs_space = getattr(inference_model, "obs_space", None)
    input_filter = getattr(inference_model.cfg.archi.actor, "input_filter", None)
    input_keys = getattr(input_filter, "key", None)

    if input_keys is None or not hasattr(obs_space, "spaces"):
        raise ValueError(
            "Cannot infer ONNX actor observation layout from model. "
            "Expected model.obs_space from checkpoint/model/init_kwargs.json and actor.input_filter.key, "
            "or pass obs_layout explicitly."
        )

    if isinstance(input_keys, str):
        input_keys = [input_keys]
    elif isinstance(input_keys, Sequence):
        input_keys = list(input_keys)
    else:
        raise TypeError(f"Unsupported actor input_filter.key type: {type(input_keys)}")

    layout = []
    for key in input_keys:
        if key == "history_actor" and not history:
            continue
        if key not in obs_space.spaces:
            raise KeyError(f"Actor input key {key!r} not found in obs_space keys {list(obs_space.spaces.keys())}")
        layout.append((key, _get_space_dim(obs_space.spaces[key])))
    return layout


def export_meta_policy_as_onnx(
    inference_model,
    path,
    exported_policy_name,
    example_obs_dict,
    z_dim,
    history: bool = False,
    obs_layout: list[tuple[str, int]] | None = None,
):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, exported_policy_name)
    inference_model = inference_model.eval()
    actor = copy.deepcopy(inference_model).to("cpu")
    actor_obs_layout = obs_layout or _infer_actor_obs_layout(inference_model, history=history)

    class PPOWrapper(nn.Module):
        def __init__(self, actor, actor_obs_layout):
            """
            model: The original PyTorch model.
            input_keys: List of input names as keys for the input dictionary.
            """
            super(PPOWrapper, self).__init__()
            self.actor = actor
            self.actor_obs_layout = actor_obs_layout

        def forward(self, actor_obs):
            """
            Dynamically creates a dictionary from the input keys and args.
            """
            actor_obs, ctx = actor_obs[:, :-z_dim], actor_obs[:, -z_dim:]
            actor_dict = {}
            start = 0
            for key, dim in self.actor_obs_layout:
                if dim is None:
                    actor_dict[key] = actor_obs[:, start:]
                    start = actor_obs.shape[-1]
                else:
                    end = start + dim
                    actor_dict[key] = actor_obs[:, start:end]
                    start = end

            return self.actor.act(actor_dict, ctx)

    wrapper = PPOWrapper(actor, actor_obs_layout=actor_obs_layout)
    example_input_list = example_obs_dict["actor_obs"]
    torch.onnx.export(
        wrapper,
        example_input_list,  # Pass x1 and x2 as separate inputs
        path,
        verbose=True,
        input_names=["actor_obs"],  # Specify the input names
        output_names=["action"],  # Name the output
        opset_version=13,  # Specify the opset version, if needed
    )
