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


ROBOT_IP = "192.168.58.3"

SERVO_DT = 0.008
TEST_DURATION = 2
MOVE_DEG = 1.0


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

        

        count = round(TEST_DURATION / SERVO_DT)
        next_deadline = time.perf_counter()

        # Smooth joint-space trajectory.
        # Move J1 by +1 degree over 2 seconds.
        # Keep J2-J6 at their initial positions.

        start_joints = joints.copy()
        target_joints = start_joints.copy()
        target_joints[0] += MOVE_DEG

        # Check target against configured joint soft limits.
        err, limits = robot.GetJointSoftLimitDeg()

        if err != 0 or limits is None or len(limits) != 12:
            raise RuntimeError("Could not read joint limits")

        for j in range(6):
            low = limits[2 * j] + 2.0
            high = limits[2 * j + 1] - 2.0

            if not low <= target_joints[j] <= high:
                raise RuntimeError(
                    f"Joint {j+1} target outside permitted limits"
                )

        print("Starting joints:", start_joints)
        print("Target joints:", target_joints)

        error = robot.ServoMoveStart()

        if error != 0:
            raise RuntimeError(
                f"ServoMoveStart failed: {error}"
            )

        servo_started = True

        print("\nServo mode started")
        
        count = round(TEST_DURATION / SERVO_DT)
        next_deadline = time.perf_counter()

        for i in range(count + 1):

            # Normalized trajectory progress.
            alpha = i / count

            # Smoothstep trajectory:
            # starts and ends with zero velocity.
            alpha_smooth = 3 * alpha**2 - 2 * alpha**3

            command = (
                start_joints
                + alpha_smooth * (target_joints - start_joints)
            )

            # Check command timing.
            if time.perf_counter() - next_deadline > 0.04:
                raise RuntimeError("Servo streaming deadline missed")

            error = robot.ServoJ(
                joint_pos=command.tolist(),
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

        print(f"\nSent {count + 1} ServoJ trajectory commands")

        # Check measured final position.
        err, final_joints = robot.GetActualJointPosDegree()

        if err != 0:
            raise RuntimeError("Could not read final joints")

        final_joints = np.asarray(final_joints, dtype=float)

        print("Measured final joints:", final_joints)
        print(
            "Position error (deg):",
            target_joints - final_joints
        )

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
