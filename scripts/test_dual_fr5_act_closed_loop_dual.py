#!/usr/bin/env python3
"""Experimental dual-arm ACT -> ROS -> FAIRINO ServoJ.

Both existing fairino_ctrl nodes own their SDK connections. This process
reads live ROS observations and publishes 12 predicted joint targets in radians.

Default is PREVIEW ONLY. --execute and explicit terminal confirmation are
required to request real motion. This does NOT certify collision safety.
"""
import argparse
import json
import sys
import time

import numpy as np
import rclpy
import torch
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, String

from robo_manip_baselines.common import denormalize_data
from test_dual_fr5_act import load_policy, prepare_state, prepare_images

PREFIXES = {'fr1': '/fairino_1/act_servoj', 'fr2': '/fairino_2/act_servoj'}
SOURCES = {
    'fr1': '/fairino_1/joint_states',
    'fr2': '/fairino_2/joint_states',
    'front': '/camera_d405_1/color/image_rect_raw',
    'hand': '/camera_d405_2/color/image_rect_raw',
}


def joint_rad(msg):
    names = [f'joint_{n}' for n in range(1, 7)]
    if len(msg.name) != len(set(msg.name)) or len(msg.position) != len(msg.name):
        raise ValueError('Duplicate or mismatched joint names/positions')
    mapping = dict(zip(msg.name, msg.position))
    if not all(name in mapping for name in names):
        raise ValueError(f'Joint names must include {names}')
    q = np.asarray([mapping[name] for name in names], dtype=np.float64)
    if q.shape != (6,) or not np.isfinite(q).all():
        raise ValueError('Invalid joint positions')
    return q


def image_rgb(msg):
    if msg.encoding not in ('rgb8', 'bgr8'):
        raise ValueError(f'Expected rgb8/bgr8, got {msg.encoding!r}')
    if msg.width <= 0 or msg.height <= 0 or msg.step < msg.width * 3:
        raise ValueError('Invalid image dimensions/stride')
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    expected = msg.height * msg.step
    if raw.size != expected:
        raise ValueError(f'Image byte count {raw.size} != {expected}')
    rgb = raw.reshape(msg.height, msg.step)[:, :msg.width * 3]
    rgb = rgb.reshape(msg.height, msg.width, 3)
    if msg.encoding == 'bgr8':
        return rgb[:, :, ::-1].copy()
    return rgb.copy()


class Observations(Node):
    def __init__(self):
        super().__init__('act_dual_fr5_live_client')
        self.data = {}
        self.arrivals = {}
        self.seq = {k: 0 for k in SOURCES}
        self.statuses = {'fr1': None, 'fr2': None}
        self.status_at = {'fr1': 0.0, 'fr2': 0.0}
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.subscribers = []
        for key in ('fr1', 'fr2'):
            self.subscribers.append(self.create_subscription(
                JointState, SOURCES[key], lambda msg, k=key: self._joint(k, msg), qos))
        for key in ('front', 'hand'):
            self.subscribers.append(self.create_subscription(
                Image, SOURCES[key], lambda msg, k=key: self._image(k, msg), qos))
        self.target_pubs = {}
        self.arm_pubs = {}
        self.stop_pubs = {}
        self.status_subs = []
        for arm, prefix in PREFIXES.items():
            self.target_pubs[arm] = self.create_publisher(JointState, prefix + '/target', 10)
            self.arm_pubs[arm] = self.create_publisher(Bool, prefix + '/arm', 10)
            self.stop_pubs[arm] = self.create_publisher(Bool, prefix + '/stop', 10)
            self.status_subs.append(self.create_subscription(
                String, prefix + '/status',
                lambda msg, arm=arm: self._status(arm, msg),
                QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                           durability=DurabilityPolicy.TRANSIENT_LOCAL)))
        self.gripper_cmd_pubs = {
            "fr1": self.create_publisher(
                Bool, "/fairino_1/gripper_command", 10
            ),
            "fr2": self.create_publisher(
                Bool, "/fairino_2/gripper_command", 10
            )
        }

        self.gripper_states = {
            "fr1": None,
            "fr2": None,
        }

        self.gripper_state_subs = []

        for arm, topic in (
            ("fr1", "/fairino_1/gripper_state"),
            ("fr2", "/fairino_2/gripper_state"),
        ):
            self.gripper_state_subs.append(
                self.create_subscription(
                    Bool,
                    topic,
                    lambda msg, a=arm: self._gripper_state(a, msg),
                    10,
                )
            )

    def _gripper_state(self, arm, msg):
        self.gripper_states[arm] = float(msg.data)

    def send_gripper_command(self, arm, open_flag):

        msg = Bool()
        msg.data = bool(open_flag)

        self.gripper_cmd_pubs[arm].publish(msg)

    def _joint(self, key, msg):
        try:
            self.data[key] = joint_rad(msg)
            self.arrivals[key] = time.monotonic()
            self.seq[key] += 1
        except Exception as exc:
            self.get_logger().error(f'{key} invalid joint feedback: {exc}')

    def _image(self, key, msg):
        try:
            self.data[key] = image_rgb(msg)
            self.arrivals[key] = time.monotonic()
            self.seq[key] += 1
        except Exception as exc:
            self.get_logger().error(f'{key} camera frame invalid: {exc}')

    def _status(self, arm, msg):
        try:
            self.statuses[arm] = json.loads(msg.data)
            self.status_at[arm] = time.monotonic()
            print(f'[SERVO STATUS {arm}]', msg.data, flush=True)
        except (ValueError, TypeError):
            self.get_logger().error(f'Malformed {arm} bridge status: {msg.data}')

    def snapshot(self, since=None, max_age=0.30, max_spread=0.12):
        if len(self.data) != len(SOURCES):
            return None
        if since is not None and any(self.seq[k] <= since[k] for k in SOURCES):
            return None
        now = time.monotonic()
        times = list(self.arrivals.values())
        if len(times) != len(SOURCES):
            return None
        if max(now - t for t in times) > max_age:
            return None
        if max(times) - min(times) > max_spread:
            return None
        if any(v is None for v in self.gripper_states.values()):
            return None
        q = np.concatenate((self.data['fr1'], self.data['fr2']))
        obs = {
            'joint_pos': q.copy(),
            'gripper_state': np.array([
                self.gripper_states['fr1'],
                self.gripper_states['fr2'],
            ], dtype=np.float64),
        }
        info = {'rgb_images': {
            'front': self.data['front'].copy(),
            'hand': self.data['hand'].copy(),
        }}
        return obs, info, dict(self.seq)

    def wait_observation(self, since=None, timeout=6.0):
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.015)
            snap = self.snapshot(since=since)
            if snap is not None:
                return snap
        now = time.monotonic()

        print("\n========== OBSERVATION DIAGNOSTICS ==========")

        print("Received sources:", list(self.data.keys()))

        print("Missing sources:", [
            k for k in SOURCES if k not in self.data
        ])

        print("Source ages:", {
            k: round(now - self.arrivals[k], 3)
            if k in self.arrivals else None
            for k in SOURCES
        })

        if len(self.arrivals) == len(SOURCES):
            times = [self.arrivals[k] for k in SOURCES]
            print("Timestamp spread:", round(max(times) - min(times), 3))

        print("Gripper states:", self.gripper_states)
        print("============================================")

        raise RuntimeError(
            "No new, suitably synchronized camera + joint observations"
        )

    def send_targets(self, q12_rad):
        q = np.asarray(q12_rad, dtype=np.float64)
        if q.shape != (12,) or not np.isfinite(q).all():
            raise ValueError('Expected 12 finite joint positions in radians')
        for arm, q6 in (('fr1', q[:6]), ('fr2', q[6:12])):
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = [f'joint_{i}' for i in range(1, 7)]
            msg.position = q6.tolist()  # ROS values are RADIANS
            self.target_pubs[arm].publish(msg)

    def arm_both(self):
        msg = Bool(); msg.data = True
        for pub in self.arm_pubs.values():
            pub.publish(msg)

    def stop_both(self):
        msg = Bool(); msg.data = True
        for _ in range(5):
            for pub in self.stop_pubs.values():
                pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.04)


def inference(policy, meta, device, obs, info):
    q_t, measured = prepare_state(obs, meta, device)
    rgb_t = prepare_images(info, meta, device)
    with torch.inference_mode():
        chunk_t = policy(q_t, rgb_t)
    chunk = chunk_t[0].detach().cpu().numpy()
    if chunk.ndim != 2 or chunk.shape[1] != 14:
        raise RuntimeError(f'Unexpected ACT output shape {chunk.shape}')
    result = np.asarray([denormalize_data(a, meta['action']) for a in chunk], dtype=np.float64)
    if result.shape != chunk.shape or not np.isfinite(result).all():
        raise RuntimeError('ACT prediction contained NaN/Inf')
    return result, measured


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    ap.add_argument('--execute', action='store_true', help='Enable one timed real-hardware session')
    ap.add_argument('--steps', type=int, default=90)
    ap.add_argument('--hz', type=float, default=15.0)
    ap.add_argument('--max-duration', type=float, default=9.0)
    args = ap.parse_args()
    if args.steps < 1 or not (1.0 <= args.hz <= 15.0) or not (0 < args.max_duration <= 15.0):
        ap.error('Require steps>=1, 1<=hz<=15, 0<max-duration<=15s')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable (use --device cpu)')
    device = torch.device(args.device)
    policy, meta = load_policy(args.checkpoint, device)
    skip = int(meta['data']['skip'])
    chunk_len = int(meta['data']['chunk_size'])
    if skip < 1 or skip > chunk_len or chunk_len < 1:
        raise RuntimeError(f'Unexpected ACT skip/chunk size: {skip}/{chunk_len}')
    print(f'ACT predicts {chunk_len} actions; this test uses all actions in each chunk')
    rclpy.init()
    node = Observations()
    armed = False
    try:
        print('\nWaiting for real cameras and BOTH FR5 joint-state topics...', flush=True)
        obs, info, seq = node.wait_observation(timeout=12.0)
        preview_chunk, measured = inference(policy, meta, device, obs, info)
        print('\n=== LIVE ACT PREVIEW: BOTH FR5 ARMS ===')
        for arm, seg in (('FR5 #1', slice(0, 6)), ('FR5 #2', slice(6, 12))):
            q_now = np.rad2deg(measured[seg])
            q_proposed = np.rad2deg(preview_chunk[0, seg])
            print(f'{arm} measured:', np.round(q_now, 3))
            print(f'{arm} ACT:     ', np.round(q_proposed, 3))
            print(f'{arm} delta:   ', np.round(q_proposed-q_now, 3))
        print('ACT chunk:', preview_chunk.shape)
        print('Bridge statuses:', node.statuses)
        print("\n=== ACT GRIPPER PREDICTIONS ===")

        for i, action in enumerate(preview_chunk):
            g1, g2 = action[12:14]

            print(
                f"Action {i + 1:02d}: "
                f"FR1={g1:.3f}, FR2={g2:.3f}"
            )
        if not args.execute:
            print('\nPREVIEW ONLY. No servo targets or arm commands sent.')
            return
        if not all(isinstance(node.statuses[a], dict) and node.statuses[a].get('state') == 'idle'
                   for a in ('fr1', 'fr2')):
            raise RuntimeError(f'Both bridges must be IDLE; statuses={node.statuses}')
        print('\nWARNING: This is experimental physical ACT motion. No collision checking,')
        print('no gripper command, and the SDK may block on socket retries.')
        print('Verify swept-path clearance, active independent physical E-stop,')
        print('no existing controller fault, and no competing robot motion source.')
        confirm = input('Type START ACT BOTH ARMS to arm this bounded session: ').strip()
        if confirm != 'START ACT BOTH ARMS':
            print('Cancelled without motion.')
            return
        obs, info, seq = node.wait_observation(since=seq, timeout=4.0)
        fresh_chunk, fresh_q = inference(policy, meta, device, obs, info)
        delta_preview = np.rad2deg(
            fresh_chunk[0, :12] - preview_chunk[0, :12]
        )
        if np.max(np.abs(delta_preview)) > 1.0:
            raise RuntimeError('ACT first target changed >1 deg since preview; rerun preview')
        print(
            'Fresh ACT joint targets (deg):',
            np.round(np.rad2deg(fresh_chunk[0, :12]), 3)
        )

        print(
            'Fresh ACT gripper predictions:',
            np.round(fresh_chunk[0, 12:14], 3)
        )
        # Ensure that if ARM publication partially succeeds we request STOP for both.
        armed = True
        node.arm_both()
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.02)
            states = {a: (node.statuses[a] or {}).get('state') for a in ('fr1', 'fr2')}
            if all(v in ('armed', 'running') for v in states.values()):
                break
            if any(v in ('fault', 'disabled', 'stopping', 'stopped') for v in states.values()):
                raise RuntimeError(f'A bridge refused to arm: {node.statuses}')
        else:
            raise RuntimeError(f'No ARM acknowledgment from both bridges: {node.statuses}')
        period = 1.0 / args.hz
        # Both arms must reach each ACT waypoint before advancing.
        REACH_TOL_DEG = 0.75
        WAYPOINT_TIMEOUT_S = 3.0
        started = time.monotonic()
        next_tick = started
        action_chunk = fresh_chunk

        if any(node.gripper_states[a] is None for a in ("fr1", "fr2")):
            raise RuntimeError("Initial gripper states unavailable")

        gripper_requested = {
            arm: bool(node.gripper_states[arm])
            for arm in ("fr1", "fr2")
        }

        gripper_sent = gripper_requested.copy()

        for step in range(args.steps):
            if time.monotonic() - started >= args.max_duration:
                print('ACT session duration limit reached')
                break
            # The first action was explicitly previewed; after this, reobserve.
            if step == 0:
                current_obs, current_info, current_seq = obs, info, seq
            else:
                current_obs, current_info, current_seq = node.wait_observation(since=seq, timeout=0.30)
                seq = current_seq
            if step > 0 and step % chunk_len == 0:
                t0 = time.perf_counter()
                action_chunk, _ = inference(policy, meta, device, current_obs, current_info)
                latency = (time.perf_counter() - t0) * 1000
                print(f'ACT inference step={step}: {latency:.1f} ms')
            index = step % chunk_len
            action = action_chunk[index]

            arm_action = action[:12]
            gripper_action = action[12:14]

            # Update requested gripper states from ACT predictions.

            for arm, prediction in zip(
                ("fr1", "fr2"), gripper_action
            ):

                if prediction >= 0.75:
                    gripper_requested[arm] = True  # OPEN

                elif prediction <= 0.25:
                    gripper_requested[arm] = False  # CLOSE

            q_current = np.rad2deg(np.asarray(current_obs['joint_pos']))
            q_target = np.rad2deg(arm_action)
            delta = q_target - q_current
            print(f'step={step+1:03d} FR1={np.round(delta[:6], 2)} FR2={np.round(delta[6:], 2)}', flush=True)
            max_error = float(np.max(np.abs(delta)))
            worst_joint = int(np.argmax(np.abs(delta)))
    
            robot = 1 if worst_joint < 6 else 2
            joint = (worst_joint % 6) + 1
            if max_error > 12.0:
                print("\n==========ACT diagnostic===========")
                print(f"ACT step: {step + 1}")
                print(f"Chunk action: {index + 1}/{chunk_len}")
    
                print(f"Robot: FR5#{robot}")
                print(f"Joint: J{joint}")
    
                print(f"Measured joint: {q_current[worst_joint]:.3f} deg")
                print(f"ACT target: : {q_target[worst_joint]:.3f} deg")
                print(f"Difference: {delta[worst_joint]:+.3f} deg")
    
                print(f"fr1 diff", np.round(delta[:6], 2))
                print(f"fr2 diff", np.round(delta[6:], 2))
                print("\n=========================")
    
                raise RuntimeError(
                    "ACT target exceeds validated displacement limit"
                )

            # =====================================================
            # SYNCHRONIZED DUAL-ARM ACT WAYPOINT EXECUTION
            # =====================================================

            node.send_targets(arm_action)

            waypoint_start = time.monotonic()
            last_publish = waypoint_start
            session_timed_out = False

            while rclpy.ok():

                # Process feedback from BOTH robots.
                rclpy.spin_once(node, timeout_sec=0.01)

                now = time.monotonic()

                # Preserve the session duration limit.
                if now - started >= args.max_duration:
                    print("ACT session duration reached")
                    session_timed_out = True
                    break

                # Stop if either ServoJ controller reports a problem.
                for arm in ("fr1", "fr2"):
                    status = node.statuses[arm] or {}

                    if status.get("state") in (
                        "fault", "disabled", "stopping", "stopped"
                    ):
                        raise RuntimeError(
                            f"{arm} ServoJ stopped: {status}"
                        )

                # Require recent feedback from BOTH arms.
                for arm in ("fr1", "fr2"):
                    age = now - node.arrivals.get(arm, 0.0)

                    if age > 0.30:
                        raise RuntimeError(
                            f"{arm} joint feedback stale: {age:.3f}s"
                        )

                # Current measured joint positions, in degrees.
                measured = np.rad2deg(
                    np.concatenate((
                        node.data["fr1"],
                        node.data["fr2"],
                    ))
                )

                error = q_target - measured

                fr1_error = np.max(np.abs(error[:6]))
                fr2_error = np.max(np.abs(error[6:]))

                # Only advance when BOTH robots reached the waypoint.
                arms_reached = (
                    fr1_error <= REACH_TOL_DEG
                    and fr2_error <= REACH_TOL_DEG
                )

                if arms_reached:

                    # Command grippers only after BOTH arms
                    # have reached the current ACT waypoint.

                    for arm in ("fr1", "fr2"):

                        desired = gripper_requested[arm]

                        if gripper_sent[arm] != desired:

                            print(
                                f"ACT waypoint {step + 1}: "
                                f"{arm} gripper -> "
                                f"{'OPEN' if desired else 'CLOSE'}"
                            )

                            node.send_gripper_command(
                                arm,
                                desired
                            )

                            gripper_sent[arm] = desired

                    # Wait for both gripper states to match.
                    grippers_reached = all(
                        node.gripper_states[arm] is not None
                        and bool(node.gripper_states[arm])
                            == gripper_requested[arm]
                        for arm in ("fr1", "fr2")
                    )

                    if grippers_reached:

                        print(
                            f"Waypoint {step + 1} reached: "
                            f"FR1={fr1_error:.3f}°, "
                            f"FR2={fr2_error:.3f}°, "
                            f"grippers={node.gripper_states}"
                        )

                        break

                # Keep sending the SAME target so the bridge's
                # existing command watchdog does not expire.
                if now - last_publish >= period:
                    node.send_targets(arm_action)
                    last_publish = now

                # Never skip a waypoint because of timeout.
                if now - waypoint_start >= WAYPOINT_TIMEOUT_S:
                    raise RuntimeError(
                        f"Waypoint {step + 1} timeout: "
                        f"FR1 error={fr1_error:.2f}°, "
                        f"FR2 error={fr2_error:.2f}°"
                    )

            if session_timed_out:
                break
        print('ACT bounded session finished; requesting servo end.')
    finally:
        if armed:
            print('Sending /stop and awaiting bridge status...')
            node.stop_both()
            end_deadline = time.monotonic() + 3.0
            while time.monotonic() < end_deadline:
                rclpy.spin_once(node, timeout_sec=0.08)
                if all((node.statuses[a] or {}).get('state') in ('stopped', 'fault') for a in ('fr1', 'fr2')):
                    break
            print('Final bridge statuses:', node.statuses)
            if not all((node.statuses[a] or {}).get('state') in ('stopped', 'fault') for a in ('fr1', 'fr2')):
                print('WARNING: BOTH servo sessions have not been confirmed stopped.')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted by operator. Confirm robot is stopped.')
        sys.exit(130)
