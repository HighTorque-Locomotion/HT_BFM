"""Manual GPU/GUI smoke test: python tests/flat_amp/smoke_play_ui.py.

Run from the repository root with its Isaac Sim 5.1 Python environment.
"""

import copy
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from isaaclab.app import AppLauncher

launcher = AppLauncher(headless=False, device="cuda:0")
app = launcher.app
window = None
keyboard = None
env = None
try:
    import torch

    from humanoidverse.flat_amp import activate
    from humanoidverse.flat_amp.keyboard_control import KeyboardVelocity, refresh_play_observations

    activate()
    import legged_lab.envs  # noqa: F401
    from legged_lab.utils import task_registry

    from humanoidverse.flat_amp.play_ui import FlatAmpPlayWindow

    cfg, _ = map(copy.deepcopy, task_registry.get_cfgs("walk_pi_waq_12"))
    cfg.device = "cuda:0"
    cfg.scene.seed = 42
    cfg.scene.num_envs = 2
    cfg.scene.play_ui = True
    cfg.noise.add_noise = False
    cfg.domain_rand.events.push_robot = None
    env = task_registry.get_task_class("walk_pi_waq_12")(cfg, False)
    keyboard = KeyboardVelocity(cfg.commands.ranges)
    keyboard.connect()
    keyboard.apply(env)
    history, extras = env.get_observations()
    window = FlatAmpPlayWindow(env)
    visualizers = [*window.env.manager_visualizers.values(), window.contact_visualizer]
    for visualizer in visualizers:
        visualizer.set_debug_vis(True)
        for frame in visualizer._term_frames:
            frame.collapsed = False
    for _ in range(5):
        env.sim.render()
    forces_seen = False
    with torch.inference_mode():
        for step in range(30):
            if step == 5:
                window.ui_window_elements["viewer_env_index"].set_value(2)
                window._set_viewer_origin_type_fn("Robot")
            if step == 10:
                window.ui_window_elements["viewer_env_index"].set_value(1)
                assert all(v._env_idx == 0 for v in visualizers)
                keyboard.handle_key("W")
                keyboard.apply(env)
                refresh_play_observations(env, history, extras["observations"])
            if step == 15:
                for visualizer in visualizers:
                    visualizer.set_debug_vis(False)
                    visualizer.set_debug_vis(True)
                    for frame in visualizer._term_frames:
                        frame.collapsed = False
            history, _, _, extras = env.step(torch.zeros_like(env.action))
            forces_seen |= any(abs(values[2]) > 1 for _, values, _ in window.signals.terms("contact_forces", 0))
        assert forces_seen, "Ground-filtered foot sensors did not register contact"
        for visualizer in visualizers:
            for plot in visualizer._term_visualizers:
                assert len(plot.y_data[0]) > 1, visualizer.category
                assert len(plot.legends) == len(plot.y_data), visualizer.category
        plot = visualizers[0]._term_visualizers[0]
        count = len(plot.y_data[0])
        env.sim.render()
        env.sim.render()
        assert len(plot.y_data[0]) == count, "Render callbacks must not duplicate control-step samples"
        env.max_episode_length = 1
        env.step(torch.zeros_like(env.action))
        timeouts = window.env.manager_visualizers["termination_manager"]._term_visualizers[-1]
        assert timeouts.y_data[0][-1] == 1, "Timeout must be sampled before reset"
    window.close()
    window.close()
    assert env._play_window is None
    print("PLAY_UI_SMOKE_OK", flush=True)
except BaseException:
    traceback.print_exc()
    raise
finally:
    if window is not None:
        window.close()
    if keyboard is not None:
        keyboard.close()
    if env is not None:
        env.sim.clear_all_callbacks()
        env.sim.clear_instance()
    app.close(wait_for_replicator=False)
