"""RMB operation adapter for the dual-FR5 environment.

This is the initial operation scaffold. The full RMB rollout
requires robot ArmConfig integration, which is not enabled yet.
"""

import gymnasium as gym


class OperationRealDualFR5:
    def __init__(self, **kwargs):
        super().__init__()

    def setup_env(self, render_mode=None):
        self.env = gym.make(
            "robo_manip_baselines/RealDualFR5Env-v0",
            render_mode=render_mode,
        )

    def get_pre_motion_phases(self):
        # Never automatically move the arms before inference.
        return []
