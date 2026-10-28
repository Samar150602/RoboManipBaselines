#!/usr/bin/env python3

import argparse
import math

import numpy as np
import rclpy
from sensor_msgs.msg import JointState


ROBOT_IPS = {
    1: "192.168.58.2",
    2: "192.168.58.3",
}


class FR5MoveJTest:

    def __init__(self, robot_id):
        self.robot_id = robot_id
        self.current_joints = None

        self.node = rclpy.create_node(
            f"fr5_{robot_id}_movej_test"
        )

        self.subscription = self.node.create_subscription(
            JointState,
            f"/fairino_{robot_id}/joint_states",
            self.joint_callback,
            10,
        )

    def joint_callback(self, msg):
        joint_map = dict(zip(msg.name, msg.position))

        expected = [
            f"joint_{i}" for i in range(1, 7)
        ]

        if not all(name in joint_map for name in expected):
            return

        self.current_joints = np.array(
            [joint_map[name] for name in expected],
            dtype=float,
        )

    def get_current_joints(self):
        print("Waiting for joint states...")

        while (
            rclpy.ok()
            and self.current_joints is None
        ):
            rclpy.spin_once(
                self.node,
                timeout_sec=0.1,
            )

        if self.current_joints is None:
            raise RuntimeError(
                "No joint states received"
            )

        return self.current_joints.copy()

    def close(self):
        self.node.destroy_node()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--robot",
        type=int,
        choices=[1, 2],
        default=1,
    )

    parser.add_argument(
        "--joint",
        type=int,
        choices=range(1, 7),
        default=1,
    )

    parser.add_argument(
        "--increment-deg",
        type=float,
        default=0.0,
    )

    args = parser.parse_args()

    rclpy.init()

    tester = FR5MoveJTest(args.robot)

    try:
        current_rad = tester.get_current_joints()

        target_rad = current_rad.copy()

        target_rad[args.joint - 1] += math.radians(
            args.increment_deg
        )

        current_deg = np.rad2deg(current_rad)
        target_deg = np.rad2deg(target_rad)

        print("\n========== FR5 MOVEJ TEST ==========")

        print(f"Robot: FR5 #{args.robot}")
        print(f"IP: {ROBOT_IPS[args.robot]}")

        print("\nCurrent joints (degrees):")
        print(np.round(current_deg, 4))

        print("\nTarget joints (degrees):")
        print(np.round(target_deg, 4))

        print("\nJoint differences (degrees):")
        print(np.round(target_deg - current_deg, 4))

        print("\nDRY RUN ONLY")
        print("No commands were sent to the robot.")

    finally:
        tester.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
