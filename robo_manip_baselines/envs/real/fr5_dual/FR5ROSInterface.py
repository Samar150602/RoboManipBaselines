"""ROS 2 observation interface for two FAIRINO FR5 arms and two D405 cameras.

Read-only by design: this class never publishes robot commands.
"""

import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState


TOPICS = {
    "robot1": "/fairino_1/joint_states",
    "robot2": "/fairino_2/joint_states",
    "front": "/camera_d405_1/color/image_rect_raw",
    "hand": "/camera_d405_2/color/image_rect_raw",
}


class FR5ROSInterface(Node):
    def __init__(self):
        super().__init__("rmb_dual_fr5_observer")

        self._lock = threading.Lock()
        self._joint = {}
        self._images = {}
        self._received_at = {}

        self.create_subscription(
            JointState,
            TOPICS["robot1"],
            lambda msg: self._joint_callback("robot1", msg),
            10,
        )
        self.create_subscription(
            JointState,
            TOPICS["robot2"],
            lambda msg: self._joint_callback("robot2", msg),
            10,
        )
        self.create_subscription(
            Image,
            TOPICS["front"],
            lambda msg: self._image_callback("front", msg),
            10,
        )
        self.create_subscription(
            Image,
            TOPICS["hand"],
            lambda msg: self._image_callback("hand", msg),
            10,
        )

        self.get_logger().info(
            "Dual FR5 observer started (READ-ONLY)"
        )

    def _joint_callback(self, name, msg):
        # Explicitly order by joint name rather than trusting message order.
        expected = [f"joint_{i}" for i in range(1, 7)]
        positions = dict(zip(msg.name, msg.position))

        try:
            q = np.array(
                [positions[joint] for joint in expected],
                dtype=np.float64,
            )
        except KeyError:
            self.get_logger().warning(
                f"{name}: missing expected joint names {expected}"
            )
            return

        if not np.all(np.isfinite(q)):
            return

        with self._lock:
            self._joint[name] = q
            self._received_at[name] = time.monotonic()

    def _image_callback(self, name, msg):
        if msg.encoding not in ("rgb8", "bgr8"):
            self.get_logger().warning(
                f"{name}: unsupported encoding {msg.encoding}"
            )
            return

        if msg.width <= 0 or msg.height <= 0:
            return

        channels = 3
        row_bytes = msg.width * channels

        if msg.step < row_bytes:
            return

        raw = np.frombuffer(msg.data, dtype=np.uint8)

        if raw.size < msg.height * msg.step:
            return

        image = raw[: msg.height * msg.step].reshape(
            msg.height, msg.step
        )[:, :row_bytes].reshape(
            msg.height, msg.width, channels
        ).copy()

        if msg.encoding == "bgr8":
            image = image[:, :, ::-1].copy()

        with self._lock:
            self._images[name] = image
            self._received_at[name] = time.monotonic()

    def get_observation(self, max_age=0.5):
        """Return a snapshot or raise if data is missing/stale.

        This checks freshness, not exact cross-topic synchronization.
        """
        required = ("robot1", "robot2", "front", "hand")

        with self._lock:
            missing = [
                key for key in required
                if key not in self._received_at
            ]

            if missing:
                raise RuntimeError(
                    f"Waiting for ROS topics: {missing}"
                )

            now = time.monotonic()
            stale = [
                key for key in required
                if now - self._received_at[key] > max_age
            ]

            if stale:
                raise RuntimeError(
                    f"Stale observations: {stale}"
                )

            q = np.concatenate(
                [self._joint["robot1"], self._joint["robot2"]]
            ).copy()

            images = {
                "front": self._images["front"].copy(),
                "hand": self._images["hand"].copy(),
            }

        return q, images

    def wait_for_observation(self, timeout=10.0):
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            try:
                return self.get_observation()
            except RuntimeError:
                time.sleep(0.05)

        raise TimeoutError(
            "No complete fresh dual-FR5 observation received"
        )