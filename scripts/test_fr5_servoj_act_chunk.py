#!/usr/bin/env python3
"""One supervised FR5#1 ServoJ playback of selected ACT waypoints.

Only one robot connection. Run only AFTER stopping FR5#1's normal ROS SDK
controller (without resetting robot faults). This is NOT a closed-loop policy,
a collision checker, or a real-time safety-rated servo controller.
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

SDK_ROOT = Path.home() / 'ros2_ws/src/fairino_ctrl/fairino_ctrl'
sys.path.insert(0, str(SDK_ROOT))
from fairino_ctrl.fairino_python_sdk import Robot

IP = '192.168.58.2'
DT = 0.008


def read_checked(robot, method):
    code, value = getattr(robot, method)()
    if code != 0:
        raise RuntimeError(f'{method} SDK return {code}')
    return value


def healthy(robot):
    if read_checked(robot, 'GetSDKComState') != 0:
        raise RuntimeError('SDK communication fault')
    if read_checked(robot, 'GetRobotEmergencyStopState') != 0:
        raise RuntimeError('Emergency stop active')
    if read_checked(robot, 'GetSafetyStopState') != [0, 0]:
        raise RuntimeError('Safety stop active')
    if read_checked(robot, 'GetRobotErrorCode') != [0, 0]:
        raise RuntimeError('Robot fault active')
    if robot.GetSafetyCode() != 0:
        raise RuntimeError('Nonzero FAIRINO SDK safety code')


def joint_read(robot):
    joints = np.asarray(read_checked(robot, 'GetActualJointPosDegree'), dtype=float)
    if joints.shape != (6,) or not np.isfinite(joints).all():
        raise RuntimeError('Joint feedback invalid')
    return joints


def build_plan(start, targets, speed, accel, min_duration):
    """Create joint-space smoothstep waypoints and an estimated duration."""
    segments = []
    prev = start.copy()
    for target in targets:
        max_delta = float(np.max(np.abs(target - prev)))
        duration = max(
            min_duration,
            1.5 * max_delta / speed,              # smoothstep peak joint speed
            math.sqrt(6.0 * max_delta / accel),  # smoothstep peak acceleration
        )
        intervals = max(1, math.ceil(duration / DT))
        segments.append((prev.copy(), target.copy(), intervals))
        prev = target.copy()
    return segments, sum(intervals * DT for _, _, intervals in segments)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--trajectory', default='/tmp/fr5_act_chunk.npz')
    p.add_argument('--actions', type=int, default=3, help='First N predicted waypoints (max 25)')
    p.add_argument('--max-jump-deg', type=float, default=2.0)
    p.add_argument('--max-speed-deg-s', type=float, default=2.0)
    p.add_argument('--max-acc-deg-s2', type=float, default=8.0)
    p.add_argument('--min-segment-sec', type=float, default=0.25)
    p.add_argument('--max-total-sec', type=float, default=12.0)
    p.add_argument('--max-start-drift-deg', type=float, default=0.2)
    p.add_argument('--max-plan-age-sec', type=float, default=600.0)
    p.add_argument('--max-following-error-deg', type=float, default=1.5)
    args = p.parse_args()

    if not (1 <= args.actions <= 25):
        p.error('--actions must be between 1 and 25')
    if any(x <= 0 for x in (
        args.max_jump_deg, args.max_speed_deg_s, args.max_acc_deg_s2,
        args.min_segment_sec, args.max_total_sec, args.max_start_drift_deg,
        args.max_plan_age_sec, args.max_following_error_deg,
    )):
        p.error('All safety thresholds must be positive')

    with np.load(Path(args.trajectory).expanduser(), allow_pickle=False) as d:
        targets_all = np.asarray(d['target_deg'], dtype=float)
        captured = np.asarray(d['start_deg'], dtype=float)
        created = float(d['created_unix'])
    if targets_all.ndim != 2 or targets_all.shape[1] != 6 or len(targets_all) < args.actions:
        raise RuntimeError(f'Invalid trajectory shape {targets_all.shape}')
    targets = targets_all[:args.actions].copy()
    if captured.shape != (6,) or not np.isfinite(targets).all() or not np.isfinite(captured).all():
        raise RuntimeError('Invalid trajectory data')
    age = time.time() - created
    if not (0 <= age <= args.max_plan_age_sec):
        raise RuntimeError(f'ACT plan age {age:.1f}s exceeds allowed age; recapture')

    print('=== FR5 #1 ACT ServoJ chunk ===')
    print('Robot:', IP, '  ACT actions:', len(targets))
    print('Plan age:', round(age, 1), 'seconds')
    print('Captured FR5#1:', np.round(captured, 3))
    for i, target in enumerate(targets):
        print(f'ACT action {i}: {np.round(target, 3)}')
    print('FR5#2: no commands will be sent')

    robot = Robot.RPC(IP)
    started = False
    try:
        if not robot.is_conect:
            raise RuntimeError('Robot SDK did not connect')
        healthy(robot)
        if read_checked(robot, 'GetRobotMotionDone') != 1:
            raise RuntimeError('Robot motion not confirmed complete before test')
        actual_start = joint_read(robot)
        drift = float(np.max(np.abs(actual_start - captured)))
        print('Fresh current joints:', np.round(actual_start, 3))
        print('Max mismatch with ACT observation:', round(drift, 4), 'deg')
        if drift > args.max_start_drift_deg:
            raise RuntimeError('Robot has moved since ACT observation; recapture chunk')

        limits = np.asarray(read_checked(robot, 'GetJointSoftLimitDeg'), dtype=float)
        if limits.shape != (12,) or not np.isfinite(limits).all():
            raise RuntimeError('Bad joint soft limits')
        low, high = limits[0::2] + 2.0, limits[1::2] - 2.0
        if np.any(low >= high):
            raise RuntimeError('Invalid soft-limit intervals')
        if np.any(targets < low) or np.any(targets > high):
            raise RuntimeError('ACT waypoint exceeds joint soft-limit margin')
        if np.any(actual_start < low) or np.any(actual_start > high):
            raise RuntimeError('Current joints outside soft-limit margin')

        prevs = np.vstack([actual_start, targets[:-1]])
        diffs = targets - prevs
        max_jump = float(np.max(np.abs(diffs)))
        print('Largest ACT waypoint jump:', round(max_jump, 3), 'deg')
        if max_jump > args.max_jump_deg:
            raise RuntimeError('ACT waypoint jump exceeds configured limit')

        segments, duration = build_plan(
            actual_start, targets, args.max_speed_deg_s,
            args.max_acc_deg_s2, args.min_segment_sec,
        )
        print(f'Interpolated ServoJ stream: ~{duration:.2f}s at nominal {1/DT:.0f}Hz')
        print('Per waypoint six-joint change (deg):')
        for i, diff in enumerate(diffs):
            print(f' {i}: {np.round(diff, 3)}')
        if duration > args.max_total_sec:
            raise RuntimeError('Interpolated trajectory duration exceeds budget')

        print('\nWARNING: This does not verify Cartesian/inter-arm collision clearance.')
        print('Confirm FR5#2 stationary, complete swept path clear, no other SDK clients,')
        print('physical emergency stop tested, and previous faults resolved.')
        confirm = input('Type EXECUTE ACT to stream this one chunk: ').strip()
        if confirm != 'EXECUTE ACT':
            print('Cancelled; no motion command sent')
            return

        # Recheck immediately before entering servo mode.
        healthy(robot)
        start_now = joint_read(robot)
        if float(np.max(np.abs(start_now - actual_start))) > 0.1:
            raise RuntimeError('Joint configuration changed during confirmation; abort')
        if read_checked(robot, 'GetRobotMotionDone') != 1:
            raise RuntimeError('Robot not stationary; abort')

        rc = robot.ServoMoveStart()
        if rc != 0:
            raise RuntimeError(f'ServoMoveStart returned {rc}')
        started = True
        count = 0
        deadline = time.perf_counter()
        for idx, (q0, q1, intervals) in enumerate(segments):
            for k in range(1, intervals + 1):
                a = k / intervals
                fraction = a * a * (3.0 - 2.0 * a)
                cmd = q0 + fraction * (q1 - q0)
                if time.perf_counter() - deadline > 0.04:
                    raise RuntimeError('ServoJ command deadline missed: stop session')
                rc = robot.ServoJ(
                    joint_pos=cmd.tolist(), axisPos=[0.0] * 4, cmdT=DT,
                )
                if rc != 0:
                    raise RuntimeError(f'ServoJ return {rc} at segment {idx}, sample {k}')
                count += 1
                deadline += DT
                remaining = deadline - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)

        print(f'Sent {count} ServoJ commands (no mid-stream pauses)')
        end_rc = robot.ServoMoveEnd()
        started = False
        if end_rc != 0:
            raise RuntimeError(f'ServoMoveEnd returned {end_rc}')
        print('ServoMoveEnd return:', end_rc)
        time.sleep(0.12)
        healthy(robot)
        measured = joint_read(robot)
        following_error = float(np.max(np.abs(measured - targets[-1])))
        print('Final measured joints:', np.round(measured, 3))
        print('Last ACT waypoint:', np.round(targets[-1], 3))
        print('Max final position error:', round(following_error, 4), 'deg')
        if following_error > args.max_following_error_deg:
            raise RuntimeError('Final following error too large; no further commands')
        print('PASS: one ACT-predicted chunk completed within final-position tolerance')
    finally:
        if started:
            print('Ending servo mode...')
            try:
                print('ServoMoveEnd return:', robot.ServoMoveEnd())
            except Exception as exc:
                print('ServoMoveEnd FAILED:', exc)
                print('Robot state UNKNOWN. Use verified physical stop if necessary.')
        try:
            robot.CloseRPC()
        except Exception:
            pass


if __name__ == '__main__':
    main()
