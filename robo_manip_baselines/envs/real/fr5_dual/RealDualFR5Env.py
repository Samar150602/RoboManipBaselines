"""Read-only dual-FR5 Gymnasium environment for RMB integration testing."""

import time

import gymnasium as gym
import numpy as np
import rclpy

from gymnasium.spaces import Box, Dict
from rclpy.executors import SingleThreadedExecutor

from .FR5ROSInterface import FR5ROSInterface


class RealDualFR5Env(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, render_mode=None):
        super().__init__()

        self.dt = 1.0 / 15.0
        self.init_time = time.monotonic()

        self.action_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(12,),
            dtype=np.float64,
        )

        self.observation_space = Dict({
            "joint_pos": Box(
                low=-np.inf,
                high=np.inf,
                shape=(12,),
                dtype=np.float64,
            ),
        })

        self._owns_rclpy = not rclpy.ok()

        if self._owns_rclpy:
            rclpy.init()

        self.ros = FR5ROSInterface()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.ros)

        self._closed = False

    def _get_snapshot(self):
        q, images = self.ros.get_observation()

        obs = {
            "joint_pos": q,
        }

        info = {
            "rgb_images": images,
        }

        return obs, info

    def _wait_for_snapshot(self, timeout=10.0):
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            self.executor.spin_once(timeout_sec=0.05)

            try:
                return self._get_snapshot()
            except RuntimeError:
                pass

        raise TimeoutError(
            "Timed out waiting for dual-FR5 observations"
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.init_time = time.monotonic()

        # No robot reset movement.
        return self._wait_for_snapshot()

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)

        if action.shape != (12,):
            raise ValueError(
                f"Expected action shape (12,), got {action.shape}"
            )

        if not np.all(np.isfinite(action)):
            raise ValueError("Action contains NaN or Inf")

        # SHADOW MODE:
        # Never send the action to robot hardware.
        print(
            "[SHADOW] Predicted FR5 #1:",
            np.round(action[:6], 4),
        )
        print(
            "[SHADOW] Predicted FR5 #2:",
            np.round(action[6:], 4),
        )

        start = time.monotonic()

        # Process ROS messages during the nominal control interval.
        while time.monotonic() - start < self.dt:
            self.executor.spin_once(timeout_sec=0.01)

        obs, info = self._get_snapshot()

        reward = 0.0
        terminated = False
        truncated = False

        return obs, reward, terminated, truncated, info

    def close(self):
        if self._closed:
            return

        self._closed = True
        self.executor.remove_node(self.ros)
        self.ros.destroy_node()
        self.executor.shutdown()

        if self._owns_rclpy and rclpy.ok():
            rclpy.shutdown()