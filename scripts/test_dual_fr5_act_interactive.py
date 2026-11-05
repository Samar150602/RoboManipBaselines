#!/usr/bin/env python3

"""
Dual FR5 ACT shadow inference test.

Reads:
    - 12 joint positions from the existing Gymnasium environment
    - RGB images from front and hand cameras

Runs:
    - RMB ACT checkpoint inference
    - RMB state normalization
    - RMB action denormalization

Safety:
    - No predicted actions are sent to env.step().
    - No robot command publishers are created.
"""

import argparse
import subprocess
import json
import os
import pickle
import sys
import time

import gymnasium as gym
import numpy as np
import torch

import robo_manip_baselines.envs

from robo_manip_baselines.common import normalize_data, denormalize_data

# Match RMB's ACT import mechanism.
RMB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ACT_ROOT = os.path.join(RMB_ROOT, "third_party", "act")

if ACT_ROOT not in sys.path:
    sys.path.insert(0, ACT_ROOT)

from detr.models.detr_vae import DETRVAE
from policy import ACTPolicy


ENV_ID = "robo_manip_baselines/RealDualFR5Env-v0"


def load_policy(checkpoint, device):
    checkpoint = os.path.abspath(checkpoint)

    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)

    meta_path = os.path.join(
        os.path.dirname(checkpoint),
        "model_meta_info.pkl",
    )

    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"Missing RMB metadata: {meta_path}"
        )

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    state_dim = len(meta["state"]["example"])
    action_dim = len(meta["action"]["example"])

    state_keys = meta["state"]["keys"]
    action_keys = meta["action"]["keys"]
    camera_names = meta["image"]["camera_names"]

    print("\n========== MODEL METADATA ==========")
    print("Checkpoint:", checkpoint)
    print("State keys:", state_keys)
    print("Action keys:", action_keys)
    print("State dimension:", state_dim)
    print("Action dimension:", action_dim)
    print("Cameras:", camera_names)
    print("Chunk size:", meta["data"]["chunk_size"])
    print("Skip:", meta["data"]["skip"])
    print("Device:", device)

    # Validate the exact model/data configuration.
    if state_keys != ["measured_joint_pos"]:
        raise ValueError(
            f"Unexpected state keys: {state_keys}"
        )

    if action_keys != ["command_joint_pos"]:
        raise ValueError(
            f"Unexpected action keys: {action_keys}"
        )

    if state_dim != 12 or action_dim != 12:
        raise ValueError(
            f"Expected 12D state/action, got {state_dim}/{action_dim}"
        )

    if camera_names != ["front", "hand"]:
        raise ValueError(
            f"Unexpected camera mapping: {camera_names}"
        )

    DETRVAE.set_state_dim(state_dim)
    DETRVAE.set_action_dim(action_dim)

    policy = ACTPolicy(meta["policy"]["args"])

    state_dict = torch.load(
        checkpoint,
        map_location=device,
        weights_only=True,
    )

    policy.load_state_dict(state_dict)
    policy.to(device)
    policy.eval()

    print("\n[OK] ACT checkpoint loaded successfully")

    return policy, meta


def prepare_state(obs, meta, device):
    state = np.asarray(
        obs["joint_pos"],
        dtype=np.float64,
    ).reshape(-1)

    if state.shape != (12,):
        raise ValueError(
            f"Expected 12 joint positions, got {state.shape}"
        )

    if not np.all(np.isfinite(state)):
        raise ValueError("Joint state contains NaN/Inf")

    # Same normalization as RMB RolloutBase.get_state().
    normalized = normalize_data(
        state,
        meta["state"],
    )

    if not np.all(np.isfinite(normalized)):
        raise ValueError("Normalized state contains NaN/Inf")

    state_tensor = torch.as_tensor(
        normalized,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    return state_tensor, state


def prepare_images(info, meta, device):
    rgb_images = info["rgb_images"]
    camera_names = meta["image"]["camera_names"]

    images = []

    for name in camera_names:
        image = np.asarray(rgb_images[name])

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Camera {name}: expected HWC RGB, got {image.shape}"
            )

        if image.dtype != np.uint8:
            raise ValueError(
                f"Camera {name}: expected uint8, got {image.dtype}"
            )

        images.append(image)

    if images[0].shape != images[1].shape:
        raise ValueError(
            "RMB expects both camera images to have the same dimensions"
        )

    # RMB image layout:
    # [batch, cameras, channels, height, width]
    image_array = np.stack(images, axis=0)
    image_array = np.moveaxis(image_array, -1, -3)

    # Equivalent to RMB's uint8 -> float32 [0, 1] transform.
    image_tensor = (
        torch.from_numpy(
            np.ascontiguousarray(image_array)
        )
        .to(device=device, dtype=torch.float32)
        .div_(255.0)
        .unsqueeze(0)
    )

    return image_tensor

def evaluate_act_prediction(current_joints, predicted_joints):
    """
    Compare ACT output against real robot joint positions.

    Both inputs must contain 12 joint positions in radians.
    No robot movement is commanded.
    """

    current = np.asarray(current_joints, dtype=np.float64)
    predicted = np.asarray(predicted_joints, dtype=np.float64)

    if current.shape != (12,) or predicted.shape != (12,):
        raise ValueError("Expected 12 joint positions")

    if not np.all(np.isfinite(current)):
        raise ValueError("Invalid current joint positions")

    if not np.all(np.isfinite(predicted)):
        raise ValueError("Invalid ACT prediction")

    current_deg = np.rad2deg(current)
    predicted_deg = np.rad2deg(predicted)

    displacement_deg = predicted_deg - current_deg

    print("\n========== ACT PREDICTION ANALYSIS ==========")

    for i in range(12):
        robot = 1 if i < 6 else 2
        joint = (i % 6) + 1

        print(
            f"FR5-{robot} J{joint}: "
            f"Current={current_deg[i]:8.3f}° "
            f"ACT={predicted_deg[i]:8.3f}° "
            f"Delta={displacement_deg[i]:+8.3f}°"
        )

    max_displacement = np.max(np.abs(displacement_deg))

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained ACT .ckpt file",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
    )

    args = parser.parse_args()

    if args.steps < 1:
        parser.error("--steps must be >= 1")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable. Try --device cpu"
        )

    device = torch.device(args.device)

    policy, meta = load_policy(
        args.checkpoint,
        device,
    )

    env = gym.make(ENV_ID)
    hardware_used = False

    try:
        obs, info = env.reset()
        print("\nACT INTERACTIVE TEST: PREVIEW for each step; at most ONE MoveJ command")
        print("Only FR5 #1 J1 may be commanded; no ACT action is passed to env.step().")

        for step in range(args.steps):
            # Fresh inference from the current observation on every cycle.
            state_tensor, current_joints = prepare_state(obs, meta, device)
            image_tensor = prepare_images(info, meta, device)

            t0 = time.perf_counter()
            with torch.inference_mode():
                chunk = policy(state_tensor, image_tensor)[0].detach().cpu().numpy()
            inference_ms = (time.perf_counter() - t0) * 1000
            if chunk.ndim != 2 or chunk.shape[1] != 12 or len(chunk) < 1:
                raise ValueError(f"Unexpected policy output: {chunk.shape}")

            # Use only the first action of a newly inferred chunk.
            prediction = np.asarray(
                denormalize_data(chunk[0], meta["action"]), dtype=np.float64
            )
            if prediction.shape != (12,) or not np.all(np.isfinite(prediction)):
                raise ValueError("Invalid policy prediction")

            current_deg = np.rad2deg(current_joints)
            predicted_deg = np.rad2deg(prediction)
            delta_deg = predicted_deg - current_deg
            # delta_j1 = float(delta_deg[0])

            # print(f"\n=== ACT STEP {step+1}/{args.steps} | inference {inference_ms:.1f} ms ===")
            # print("FR5 #1 current (deg):", np.round(current_deg[:6], 3))
            # print("FR5 #1 ACT     (deg):", np.round(predicted_deg[:6], 3))
            # print("FR5 #1 delta   (deg):", np.round(delta_deg[:6], 3))
            # print("FR5 #2 max predicted delta (deg):", round(float(np.max(np.abs(delta_deg[6:]))), 3))

            # eligible = np.isfinite(delta_j1) and delta_j1 >= 0.05
            # step_deg = min(delta_j1, 0.5) if eligible else 0.0
            # target_deg = current_deg[:6].copy()
            # target_deg[0] += step_deg
            # print(f"Proposed: J1 +{step_deg:.3f} deg at 5% (other joints unchanged)")

            current_fr1 = current_deg[6:].copy()
            predicted_fr1 = predicted_deg[6:].copy()

            delta_fr1 = predicted_fr1 - current_fr1

            # safety limit during testing
            MAX_DELTA_DEG = 3.0

            bounded_delta = np.clip(
                delta_fr1,
                -MAX_DELTA_DEG,
                MAX_DELTA_DEG
            )

            target_deg = current_fr1 + bounded_delta

            print(f"\n=== ACT STEP {step+1}/{args.steps}======")
            print()
            print(
                f"{'Joint':<8}"
                f"{'Current':>12}"
                f"{'ACT target':>14}"
                f"{'ACT delta':>12}"
                f"{'Proposed':>12}"
            )

            for j in range(6):
                print(
                    f"J{j+1:<7}"
                    f"{current_fr1[j]:>12.3f}"
                    f"{predicted_fr1[j]:>14.3f}"
                    f"{delta_fr1[j]:>+12.3f}"
                    f"{bounded_delta[j]:>+12.3f}"
                )

            print("\nFull ACT target:", np.round(predicted_fr1, 3))
            print("Proposed target:", np.round(target_deg, 3))
            print("Velocity: 5%")

            eligible = (
                np.all(np.isfinite(target_deg))
                and np.any(np.abs(bounded_delta) >= 0.05)
            )

            if hardware_used:
                print("Hardware one-shot already used: preview only.")
            elif not eligible:
                print("Prediction not eligible for positive hardware test.")

            choice = input("PREVIEW again [ENTER], MOVE once [MOVE], or exit [QUIT]: ").strip().upper()
            if choice == "QUIT":
                break
            if choice == "MOVE":
                if hardware_used or not eligible:
                    print("Movement unavailable; no service request sent.")
                else:
                    print("STOP: verify fresh feedback, actual target, clearance, other arm stationary,")
                    print("tool/user IDs, working physical E-stop, and controller one-shot checks.")
                    answer = input("Type CONFIRM for this ONE bounded movement: ").strip()
                    if answer == "CONFIRM":
                        request = json.dumps({
                            "joint_positions_deg": target_deg.tolist(),
                            "velocity_percent": 5.0,
                        })
                        # One request maximum even if command outcome is unknown.
                        # hardware_used = True
                        try:
                            out = subprocess.run(
                                ["ros2", "service", "call", "/fairino_2/move_j",
                                 "fairino_ctrl_msgs/srv/MoveJ", request],
                                text=True, capture_output=True, timeout=90,
                            )
                            print("SERVICE OUTPUT:\n", out.stdout)
                            if out.stderr:
                                print("STDERR:\n", out.stderr)
                            if out.returncode != 0:
                                print("ROS service command failed; do NOT retry blindly")
                        except subprocess.TimeoutExpired:
                            print("TIMEOUT: robot state UNKNOWN. Do NOT retry or continue execution.")
                            break
                    else:
                        print("Not confirmed. No movement requested.")

            # The known working shadow environment advances via placeholder.
            # Verify your installed env.step() does not drive hardware.
            placeholder = np.asarray(obs["joint_pos"], dtype=np.float64).copy()
            obs, _, terminated, truncated, info = env.step(placeholder)
            if terminated or truncated:
                print("Environment ended")
                break

        print("\nInteractive test completed. Maximum one hardware command was issued.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
