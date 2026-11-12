"""Experimental single-owner FAIRINO ServoJ bridge for a pre-existing rclpy controller.

No Robot.RPC() is created here. The existing FairinoCtrl node remains sole SDK owner.
Hardware servo commands are OFF by default; enable_act_servoj must be true at node startup.
This bridge does NOT provide independent safety-rated stopping or collision detection.
"""

import json
import math
import threading
import time

import numpy as np
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy


class ActServoJBridge:
    PREFIX = '/fairino/act_servoj'
    DT = 0.008                                # Nominal servo command period
    MAX_VELOCITY_DEG_S = 5.0                  # Conservative initial test limit, NOT a safety rating
    MAX_ACCELERATION_DEG_S2 = 25.0
    MAX_TARGET_GAP_DEG = 29.0                 # Reject jumps, do not teleport to new ACT action
    MAX_TRACKING_ERROR_DEG = 3.0
    TARGET_TIMEOUT_S = 0.35
    FEEDBACK_TIMEOUT_S = 0.40
    MAX_RUN_S = 17.0                          # Bounded experimental session
    MAX_JITTER_S = 0.040                      # Python deadline fault
    SOFT_LIMIT_MARGIN_DEG = 2.0
    STATE_CHECK_PERIOD_S = 0.25

    def __init__(self, node, robot, robot_ip):
        self.node = node
        self.robot = robot
        self.robot_ip = robot_ip
        self.node.declare_parameter('enable_act_servoj', False)
        self.hw_enabled = bool(self.node.get_parameter('enable_act_servoj').value)
        self.lock = threading.Lock()
        self._reserved = False
        self._stop = threading.Event()
        self._thread = None
        self._target_deg = None
        self._target_at = 0.0
        self._feedback_deg = None
        self._feedback_at = 0.0
        self._last_fault = ''
        self._num_commands = 0
        self._fault_latched = False

        # ============================================================
        # ServoJ session tracking for pause/resume
        # ============================================================

        # Increases whenever a new ServoJ session is requested.
        # Allows the ACT rollout to identify the correct session.
        self._session_id = 0

        # True only after ServoMoveEnd() returns SDK code 0.
        self._servo_end_ok = False

        # Serializes status messages published by ROS callbacks
        # and the independent ServoJ worker thread.
        self._status_lock = threading.Lock()

        # Stores the last published status.
        self._last_state = None

        q = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                       reliability=ReliabilityPolicy.RELIABLE)
        self.status_pub = node.create_publisher(
            String, self.PREFIX + '/status',
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.sub_target = node.create_subscription(
            JointState, self.PREFIX + '/target', self._on_target, q)
        self.sub_fb = node.create_subscription(
            JointState, '/fairino/joint_states', self._on_feedback, 10)
        self.sub_arm = node.create_subscription(
            Bool, self.PREFIX + '/arm', self._on_arm, 10)
        self.sub_stop = node.create_subscription(
            Bool, self.PREFIX + '/stop', self._on_stop, 10)
        self._status('disabled' if not self.hw_enabled else 'idle', 'startup')
        node.get_logger().warning(
            f'ACT ServoJ bridge initialized for {self.robot_ip}; '
            f'hardware_enabled={self.hw_enabled}; '
            'independent physical E-stop required')

    @property
    def reserved(self):
        with self.lock:
            return self._reserved or self._fault_latched

    def _status(self, state, reason):

        # Protect status publication across the ROS executor
        # and the ServoJ worker thread.
        with self._status_lock:

            # Do not allow repeated or delayed STOP requests
            # to overwrite a completed STOP or FAULT status.
            if (
                state == 'stopping'
                and self._last_state in (
                    'idle',
                    'disabled',
                    'stopped',
                    'fault'
                )
            ):
                return

            self._last_state = state

            payload = {
                'state': state,
                'reason': str(reason),
                'ip': str(self.robot_ip),
                'commands': self._num_commands,

                # Identify the current ServoJ session.
                'session_id': self._session_id,

                # Acknowledgement from the actual SDK call.
                'servo_end_ok': self._servo_end_ok,
            }

            msg = String()
            msg.data = json.dumps(payload)

            self.status_pub.publish(msg)

            self.node.get_logger().info(
                'ACT_SERVOJ ' + msg.data
            )

    @staticmethod
    def _ordered(msg):
        keys = [f'joint_{j}' for j in range(1, 7)]
        if len(msg.name) != len(set(msg.name)) or not all(k in msg.name for k in keys):
            return None
        if len(msg.position) != len(msg.name):
            return None
        lookup = dict(zip(msg.name, msg.position))
        arr = np.asarray([lookup[k] for k in keys], dtype=float)
        if arr.shape != (6,) or not np.all(np.isfinite(arr)):
            return None
        return np.rad2deg(arr)

    def _on_feedback(self, msg):
        joints = self._ordered(msg)
        if joints is None:
            return
        with self.lock:
            self._feedback_deg = joints
            self._feedback_at = time.monotonic()

    def _fault_target(self, reason):

        # Only active sessions can fail because of a target.
        with self.lock:

            if not self._reserved:
                return

            self._last_fault = str(reason)
            self._fault_latched = True
            self._stop.set()

        self._status('fault', reason)

    def _on_target(self, msg):
        joints = self._ordered(msg)
        if joints is None:
            self._fault_target('Malformed joint target')
            return
        # Reject delayed ROS messages, rather than replaying an old queued target.
        stamp_ns = int(msg.header.stamp.sec) * 1000000000 + int(msg.header.stamp.nanosec)
        now_ns = self.node.get_clock().now().nanoseconds
        if stamp_ns <= 0 or not (0 <= now_ns - stamp_ns <= 250000000):
            self._fault_target('Stale or future target timestamp')
            return
        with self.lock:
            if not self._reserved:
                return
            self._target_deg = joints
            self._target_at = time.monotonic()

    def _on_arm(self, msg):
        if not msg.data:
            self.stop('disarmed by operator')
            return
        if not self.hw_enabled:
            self._status('disabled', 'enable_act_servoj=False')
            return
        with self.lock:

            # Prevent a new session while the previous worker
            # is still running or completing its cleanup.
            previous_worker_alive = (
                self._thread is not None
                and self._thread.is_alive()
            )

            busy = (
                self._reserved
                or self._fault_latched
                or previous_worker_alive
            )

            if not busy:

                self._reserved = True

                # A new ServoJ session begins here.
                self._session_id += 1

                # The new session has not been ended yet.
                self._servo_end_ok = False

                # Clear old diagnostics from a successful session.
                self._last_fault = ''

                # Discard targets from the previous session.
                self._target_deg = None
                self._target_at = 0.0

                # Reset the worker stop request.
                self._stop.clear()

                # Reset the command counter.
                self._num_commands = 0

                # Create a new ServoJ worker.
                thread = threading.Thread(
                    target=self._worker,
                    name='fr5-servoj-worker',
                    daemon=True
                )

                self._thread = thread
        if busy:
            self._status('fault' if self._fault_latched else 'busy', 'ServoJ active or previous fault latched; restart only after investigating')
            return
        self._status('armed', 'Waiting for first fresh ACT target')
        thread.start()

    def _on_stop(self, msg):
        if msg.data:
            self.stop('ROS operator stop requested')

    def stop(self, reason='stop requested'):

        # Check whether a ServoJ session is reserved.
        with self.lock:

            active = self._reserved

            if active:
                self._stop.set()

        # Publish STOPPING only for an active session.
        # Do not overwrite an existing STOPPED status.
        if active:
            self._status('stopping', reason)

    def shutdown(self):
        self.stop('controller shutdown')
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
        if t is not None and t.is_alive():
            self.node.get_logger().error(
                'ServoJ worker is still executing SDK call; '
                'DO NOT close SDK concurrently. Use physical E-stop if needed.')
            return False
        return True

    def _snapshot(self):
        with self.lock:
            return (None if self._target_deg is None else self._target_deg.copy(),
                    self._target_at,
                    None if self._feedback_deg is None else self._feedback_deg.copy(),
                    self._feedback_at)

    def _sdk_value(self, name):
        result = getattr(self.robot, name)()
        if not isinstance(result, (tuple, list)) or len(result) < 2 or result[0] != 0:
            raise RuntimeError(f'{name}: SDK error or malformed result {result}')
        return result[1]

    def _check_sdk_state(self):
        if self._sdk_value('GetSDKComState') != 0:
            raise RuntimeError('SDK communication abnormal')
        if self._sdk_value('GetRobotEmergencyStopState') != 0:
            raise RuntimeError('Emergency stop active')
        if self._sdk_value('GetSafetyStopState') != [0, 0]:
            raise RuntimeError('Safety stop active')
        if self._sdk_value('GetRobotErrorCode') != [0, 0]:
            raise RuntimeError('Robot error active')
        if self.robot.GetSafetyCode() != 0:
            raise RuntimeError('SDK safety code nonzero')

    def _worker(self):
        servo_started = False
        failed = False
        try:
            self._check_sdk_state()
            if self._sdk_value('GetRobotMotionDone') != 1:
                raise RuntimeError('Previous robot motion not complete')
            start = np.asarray(self._sdk_value('GetActualJointPosDegree'), dtype=float)
            limits_raw = np.asarray(self._sdk_value('GetJointSoftLimitDeg'), dtype=float)
            if start.shape != (6,) or limits_raw.shape != (12,) or not np.all(np.isfinite(start)) or not np.all(np.isfinite(limits_raw)):
                raise RuntimeError('Invalid initial joint state or limits')
            lo = limits_raw[::2] + self.SOFT_LIMIT_MARGIN_DEG
            hi = limits_raw[1::2] - self.SOFT_LIMIT_MARGIN_DEG
            if np.any(lo >= hi) or np.any(start < lo) or np.any(start > hi):
                raise RuntimeError('Current position outside soft-limit margins')
            # Check the ROS joint-state mapping against the SDK before ANY servo session.
            wait_deadline = time.monotonic() + 2.0
            while not self._stop.is_set():
                target, target_at, measured, feedback_at = self._snapshot()
                now = time.monotonic()
                if now >= wait_deadline:
                    raise RuntimeError('No fresh initial target and feedback')
                if (target is not None and measured is not None and
                        now - target_at < self.TARGET_TIMEOUT_S and
                        now - feedback_at < self.FEEDBACK_TIMEOUT_S):
                    break
                time.sleep(0.01)
            if self._stop.is_set():
                return
            if np.max(np.abs(measured - start)) > 2.0:
                raise RuntimeError('FR5 #1 ROS joint state disagrees with FR5 #1 SDK (>2 deg); mapping may be swapped')
            if np.any(target < lo) or np.any(target > hi):
                raise RuntimeError('Initial ACT target violates joint soft-limit margin')
            if np.max(np.abs(target - start)) > self.MAX_TARGET_GAP_DEG:
                raise RuntimeError('Initial ACT target too far from current joints')

            # Do not begin a ServoJ session if STOP has already
            # been requested during initial validation.
            if self._stop.is_set():
                return

            ret = self.robot.ServoMoveStart()
            if ret != 0:
                raise RuntimeError(f'ServoMoveStart returned {ret}')
            servo_started = True
            self._status('running', 'ServoMoveStart succeeded')
            q = start.copy()
            qdot = np.zeros(6)
            clock_start = time.monotonic()
            deadline = clock_start
            last_state_check = clock_start
            n = 0
            while not self._stop.is_set():
                now = time.monotonic()
                if now - clock_start >= self.MAX_RUN_S:
                    self._status('stopping', 'Session time limit reached')
                    break
                if now - deadline > self.MAX_JITTER_S:
                    raise RuntimeError(f'Python ServoJ scheduling late by {now - deadline:.3f}s')
                target, target_at, measured, feedback_at = self._snapshot()
                if target is None or now - target_at > self.TARGET_TIMEOUT_S:
                    raise RuntimeError('ACT target watchdog expired')
                if measured is None or now - feedback_at > self.FEEDBACK_TIMEOUT_S:
                    raise RuntimeError('Joint feedback watchdog expired')
                if np.max(np.abs(measured - q)) > self.MAX_TRACKING_ERROR_DEG:
                    raise RuntimeError('Measured joint positions diverged from commanded stream')
                if np.any(target < lo) or np.any(target > hi):
                    raise RuntimeError('ACT target outside joint margins')
                if np.max(np.abs(target - q)) > self.MAX_TARGET_GAP_DEG:
                    raise RuntimeError('ACT command jumped too far from current stream')
                if now - last_state_check > self.STATE_CHECK_PERIOD_S:
                    self._check_sdk_state()
                    last_state_check = now
                # Velocity and acceleration constrained target follower.
                requested_vel = np.clip((target - q) / self.DT,
                                        -self.MAX_VELOCITY_DEG_S, self.MAX_VELOCITY_DEG_S)
                change_vel = np.clip(requested_vel - qdot,
                                     -self.MAX_ACCELERATION_DEG_S2 * self.DT,
                                      self.MAX_ACCELERATION_DEG_S2 * self.DT)
                v_next = np.clip(qdot + change_vel,
                                 -self.MAX_VELOCITY_DEG_S, self.MAX_VELOCITY_DEG_S)
                move = np.sign(target - q) * np.minimum(np.abs(target - q), np.abs(v_next * self.DT))
                # Never command opposite to target direction at a reversal.
                move = np.where(np.sign(v_next) == np.sign(target - q), move, 0.0)
                q = q + move
                qdot = move / self.DT
                code = self.robot.ServoJ(q.tolist(), [0.0, 0.0, 0.0, 0.0], cmdT=self.DT)
                if code != 0:
                    raise RuntimeError(f'ServoJ returned SDK error {code}')
                n += 1
                self._num_commands = n
                deadline += self.DT
                sleep_left = deadline - time.monotonic()
                if sleep_left > 0:
                    time.sleep(sleep_left)
            self._status('stopping', f'Stream ended after {n} targets')
        except Exception as exc:
            failed = True
            self._last_fault = str(exc)
            self._status('fault', str(exc))
        finally:

            # ====================================================
            # END THE HARDWARE SERVO SESSION
            # ====================================================

            if servo_started:

                try:

                    # Ask FAIRINO SDK to end the ServoJ session.
                    end_code = self.robot.ServoMoveEnd()

                    self.node.get_logger().warning(
                        f'ServoMoveEnd returned {end_code}'
                    )

                    if end_code == 0:

                        # The SDK acknowledged successful session exit.
                        self._servo_end_ok = True

                    else:

                        # SDK did not confirm successful session exit.
                        failed = True

                        extra = f'ServoMoveEnd returned {end_code}'

                        self._last_fault = (
                            f'{self._last_fault}; {extra}'
                            if self._last_fault
                            else extra
                        )

                except Exception as exc:

                    failed = True

                    extra = f'ServoMoveEnd exception: {exc}'

                    self._last_fault = (
                        f'{self._last_fault}; {extra}'
                        if self._last_fault
                        else extra
                    )

                    self.node.get_logger().error(
                        f'ServoMoveEnd FAILED: {exc}'
                    )

            # ====================================================
            # RELEASE THE SESSION RESERVATION
            # ====================================================

            with self.lock:

                # Include faults detected by ROS target callbacks.
                failed = failed or self._fault_latched

                if failed:
                    self._fault_latched = True

                self._reserved = False

                # Old ACT targets must not survive into the
                # next ServoJ session.
                self._target_deg = None
                self._target_at = 0.0

            # ====================================================
            # PUBLISH FINAL SESSION RESULT
            # ====================================================

            self._status(
                'fault' if failed else 'stopped',
                self._last_fault if failed else 'Servo session ended'
            )
