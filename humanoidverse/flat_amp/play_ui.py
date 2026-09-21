"""HT_lab's native IsaacLab environment window adapted to the flat AMP loop.

Import only after AppLauncher has started Isaac Sim. The window, camera, scene
switches and LiveLinePlot controls come from the installed Isaac Lab dependency.
"""

from types import SimpleNamespace

import omni.ui as ui
from isaaclab.envs import ViewerCfg
from isaaclab.envs.ui import ManagerBasedRLEnvWindow, ViewportCameraController
from isaaclab.ui.widgets import LiveLinePlot, ManagerLiveVisualizer
from isaaclab.ui.widgets.manager_live_visualizer import ManagerLiveVisualizerCfg

from .play_ui_data import PlaySignals


class PlaybackVisualizer(ManagerLiveVisualizer):
    """Native plot controls sampled at control steps, rather than render callbacks."""

    def __init__(self, signals, category):
        super().__init__(signals, ManagerLiveVisualizerCfg(manager_name=category))
        self.signals = signals
        self.category = category
        self._term_visualizers = []
        self._term_frames = []
        self._debug_vis = False
        self._debug_vis_handle = None
        self._env_idx = 0

    def _set_debug_vis_impl(self, enabled):
        self._debug_vis = enabled
        self.debug_vis = enabled
        self._term_visualizers.clear()
        self._term_frames.clear()
        self._vis_frame.clear()
        self._vis_frame.visible = enabled
        if not enabled:
            return
        terms = self.signals.terms(self.category, self._env_idx)
        with self._vis_frame:
            with ui.VStack(spacing=2, height=0):
                if not terms:
                    ui.Label("No active curriculum for this flat playback task.", word_wrap=True)
                for name, values, legends in terms:
                    frame = ui.CollapsableFrame(
                        name, collapsed=self.category != "contact_forces", style={"border_color": 0xFF8A8777, "padding": 4}
                    )
                    self._term_frames.append(frame)
                    with frame:
                        plot = LiveLinePlot(y_data=[[v] for v in values], legends=legends, plot_height=150)
                        self._term_visualizers.append(plot)

    def _set_env_selection_impl(self, env_idx):
        if not 0 <= env_idx < self.signals.env.num_envs:
            raise ValueError(f"Environment index out of range: {env_idx}")
        self._env_idx = env_idx
        # Include env 0, which the upstream visualizer's > 0 check misses.
        for plot in self._term_visualizers:
            plot.clear()

    def update(self):
        if self._debug_vis:
            for (_, values, _), plot in zip(self.signals.terms(self.category, self._env_idx), self._term_visualizers):
                plot.add_datapoint(values)

    def close(self):
        self.set_debug_vis(False)


class FlatAmpPlayWindow(ManagerBasedRLEnvWindow):
    """The same IsaacLab panel as HT_lab, with flat AMP data adapters."""

    def __init__(self, env):
        self.ui_window = None
        self._closed = False
        self.physical_env = env
        self.signals = PlaySignals(env)
        viewer = ViewerCfg(eye=(3.0, 3.0, 2.0), lookat=(0.0, 0.0, 0.7))
        # A UI-only facade avoids replacing real training managers or config.
        facade = SimpleNamespace(cfg=SimpleNamespace(viewer=viewer), scene=env.scene, sim=env.sim, num_envs=env.num_envs)
        facade.viewport_camera_controller = ViewportCameraController(env, viewer)
        facade.manager_visualizers = {}
        for category in ("actions", "observations", "commands", "rewards", "curriculum", "termination"):
            manager_name = {
                "actions": "action_manager", "observations": "observation_manager",
                "commands": "command_manager", "rewards": "reward_manager",
            }.get(category, category + "_manager")
            visualizer = PlaybackVisualizer(self.signals, category)
            setattr(facade, manager_name, self.signals)
            facade.manager_visualizers[manager_name] = visualizer
        self.contact_visualizer = PlaybackVisualizer(self.signals, "contact_forces")
        super().__init__(facade, window_name="IsaacLab")
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("Command Markers", env.command_generator)
                    self._create_debug_vis_ui_element("Foot Contact Forces", self.contact_visualizer)
        env._play_window = self

    def update(self, raw_actions, joint_targets):
        """Called after reward/termination evaluation and before any episode reset."""
        self.signals.raw_actions = raw_actions
        self.signals.joint_targets = joint_targets
        for visualizer in self.env.manager_visualizers.values():
            visualizer.update()
        self.contact_visualizer.update()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.physical_env._play_window = None
        for visualizer in self.env.manager_visualizers.values():
            visualizer.close()
        self.contact_visualizer.close()
        self.env.viewport_camera_controller.__del__()
        super().__del__()

    def __del__(self):
        # Playback owns explicit cleanup before Kit shutdown.
        if getattr(self, "ui_window", None) is not None:
            super().__del__()
