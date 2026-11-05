#!/usr/bin/env python3

import sys
import time
from pathlib import Path

import numpy as np

SDK_ROOT = (
    Path.home()
    / "ros2_ws/src/fairino_ctrl/fairino_ctrl"
)

sys.path.insert(0, str(SDK_ROOT))

from fairino_ctrl.fairino_python_sdk import Robot


ROBOT_IP = "192.168.58.2"

SERVO_DT = 0.008
TEST_DURATION = 0.5


def checked_call(robot, name):
    error, value = getattr(robot, name)()

    if error != 0:
        raise RuntimeError(f"{name} failed: {error}")

    print(f"{name}: {value}")

    return value


def main():

    print("========== FR5 SERVOJ TEST ==========")
    print("Robot:", ROBOT_IP)
    print("Mode: HOLD CURRENT JOINT POSITIONS")
    print("No ACT predictions will be executed")

    robot = Robot.RPC(ROBOT_IP)
    servo_started = False

    try:
        if not robot.is_conect:
            raise RuntimeError("Robot SDK not connected")

        communication = checked_call(
            robot, "GetSDKComState"
        )

        emergency = checked_call(
            robot, "GetRobotEmergencyStopState"
        )

        safety = checked_call(
            robot, "GetSafetyStopState"
        )

        motion_done = checked_call(
            robot, "GetRobotMotionDone"
        )

        robot_errors = checked_call(
            robot, "GetRobotErrorCode"
        )

        joints = np.asarray(
            checked_call(robot, "GetActualJointPosDegree"),
            dtype=float,
        )

        if (
            communication != 0
            or emergency != 0
            or safety != [0, 0]
            or motion_done != 1
            or robot_errors != [0, 0]
        ):
            raise RuntimeError(
                "Robot state is not cleared for ServoJ testing"
            )

        if joints.shape != (6,) or not np.all(np.isfinite(joints)):
            raise RuntimeError("Invalid joint positions")

        print("\nCurrent joints:", joints.tolist())

        answer = input(
            "\nType SERVO to start the 0.5-second hold test: "
        ).strip()

        if answer != "SERVO":
            print("Cancelled")
            return

        error = robot.ServoMoveStart()

        if error != 0:
            raise RuntimeError(
                f"ServoMoveStart failed: {error}"
            )

        servo_started = True

        print("\nServo mode started")

        count = round(TEST_DURATION / SERVO_DT)
        next_deadline = time.perf_counter()

        for i in range(count):

            # Abort on excessive scheduling delay.
            if time.perf_counter() - next_deadline > 0.04:
                raise RuntimeError(
                    "Servo command deadline missed"
                )

            error = robot.ServoJ(
                joint_pos=joints.tolist(),
                axisPos=[0.0, 0.0, 0.0, 0.0],
                cmdT=SERVO_DT,
            )

            if error != 0:
                raise RuntimeError(
                    f"ServoJ failed at command {i}: {error}"
                )

            next_deadline += SERVO_DT

            remaining = next_deadline - time.perf_counter()

            if remaining > 0:
                time.sleep(remaining)

        print(f"\nSent {count} ServoJ hold commands")

    finally:

        if servo_started:
            print("Ending servo session...")
            try:
                error = robot.ServoMoveEnd()
                print("ServoMoveEnd return:", error)
            except Exception as exc:
                print("ServoMoveEnd failed:", exc)
                print("Verify robot state before further commands")

        try:
            robot.CloseRPC()
        except Exception:
            pass


if __name__ == "__main__":
    main()
