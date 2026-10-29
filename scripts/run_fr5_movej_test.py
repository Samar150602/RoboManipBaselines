#!/usr/bin/env python3

import math
import time

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from fairino_ctrl_msgs.srv import MoveJ


class MoveJTest(Node):

    def __init__(self):
        super().__init__("fr5_movej_test")

        self.joints = None
        self.last_received = None

        self.subscription = self.create_subscription(
            JointState,
            "/fairino_1/joint_states",
            self.joint_callback,
            10,
        )

        self.client = self.create_client(
            MoveJ,
            "/fairino_1/move_j",
        )

    def joint_callback(self, msg):

        names = [f"joint_{i}" for i in range(1, 7)]

        if not all(name in msg.name for name in names):
            return

        positions = dict(zip(msg.name, msg.position))

        self.joints = [
            math.degrees(positions[name])
            for name in names
        ]

        self.last_received = time.monotonic()


def main():

    rclpy.init()

    node = MoveJTest()

    print("Waiting for FR5 #1 joint states...")

    deadline = time.monotonic() + 10.0

    while rclpy.ok() and node.joints is None:
        if time.monotonic() > deadline:
            raise RuntimeError("Joint state timeout")

        rclpy.spin_once(node, timeout_sec=0.1)

    current = list(node.joints)

    target = current.copy()

    # First hardware test: +0.5 degrees on J1.
    target[0] += 0.5

    print("\n========== FIRST FR5 MOVEJ TEST ==========")

    print("Current joints:", current)
    print("Target joints: ", target)
    print("Movement: J1 +0.5 degrees")
    print("Speed: 5%")

    if not node.client.wait_for_service(timeout_sec=5.0):
        raise RuntimeError("MoveJ service unavailable")

    # Refuse to send if joint feedback is already stale.
    if time.monotonic() - node.last_received > 0.5:
        raise RuntimeError("Joint feedback is stale")

    print("\nConfirm physical emergency stop is accessible.")
    print("Confirm the J1 path is clear of both arms and obstacles.")
    print("Confirm FR5 #2 is stationary.")
    print("Confirm tool/user coordinate IDs 0 are correct.")

    answer = input("\nType MOVE to execute: ")

    if answer != "MOVE":
        print("Cancelled.")
        return

    # Refresh feedback immediately before constructing the request.
    node.joints = None
    deadline = time.monotonic() + 3.0

    while rclpy.ok() and node.joints is None:
        if time.monotonic() > deadline:
            raise RuntimeError("Fresh joint feedback unavailable")
        rclpy.spin_once(node, timeout_sec=0.05)

    target = list(node.joints)
    target[0] += 0.5

    request = MoveJ.Request()
    request.joint_positions_deg = target
    request.velocity_percent = 5.0

    print("\nSending MoveJ request...")

    future = node.client.call_async(request)

    deadline = time.monotonic() + 60.0

    while rclpy.ok() and not future.done():
        if time.monotonic() > deadline:
            print(
                "Service response timed out. Robot motion state is UNKNOWN."
            )
            print(
                "Do not retry. Inspect the robot and controller."
            )
            return

        rclpy.spin_once(node, timeout_sec=0.1)

    response = future.result()

    print("\n========== RESULT ==========")

    print("Success:", response.success)
    print("Error code:", response.error_code)
    print("Message:", response.message)


if __name__ == "__main__":
    main()
