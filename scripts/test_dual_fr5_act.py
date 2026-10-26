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

    try:
        obs, info = env.reset()

        print("\n========== SHADOW INFERENCE ==========")

        skip = int(meta["data"]["skip"])
        if skip < 1:
            raise ValueError(f"Invalid skip: {skip}")

        action_chunk = None

        for step in range(args.steps):

            if step % skip == 0 or action_chunk is None:
                state_tensor, current_joints = prepare_state(
                    obs,
                    meta,
                    device,
                )

                image_tensor = prepare_images(
                    info,
                    meta,
                    device,
                )

                start = time.perf_counter()

                with torch.inference_mode():
                    prediction = policy(
                        state_tensor,
                        image_tensor,
                    )

                # RMB ACT returns [batch, chunk_size, action_dim].
                action_chunk = (
                    prediction[0]
                    .detach()
                    .cpu()
                    .numpy()
                )

                inference_ms = (
                    time.perf_counter() - start
                ) * 1000.0

                if (
                    action_chunk.ndim != 2
                    or action_chunk.shape[1] != 12
                ):
                    raise ValueError(
                        f"Unexpected ACT output shape: {action_chunk.shape}"
                    )

                print(
                    f"\n[INFERENCE] step={step}, "
                    f"chunk={action_chunk.shape}, "
                    f"time={inference_ms:.1f} ms"
                )

            # Use successive predictions from the current chunk.
            chunk_index = step % skip

            if chunk_index >= len(action_chunk):
                raise ValueError(
                    "ACT chunk is shorter than the inference skip"
                )

            normalized_action = action_chunk[chunk_index]

            predicted_action = np.asarray(
                denormalize_data(
                    normalized_action,
                    meta["action"],
                ),
                dtype=np.float64,
            )

            if predicted_action.shape != (12,):
                raise ValueError(
                    f"Unexpected action shape: {predicted_action.shape}"
                )

            if not np.all(np.isfinite(predicted_action)):
                raise ValueError(
                    "ACT prediction contains NaN/Inf"
                )

            print(f"\n[STEP {step + 1}/{args.steps}]")
            print(
                "Current FR5 #1:",
                np.round(current_joints[:6], 4),
            )
            print(
                "Current FR5 #2:",
                np.round(current_joints[6:], 4),
            )
            print(
                "ACT FR5 #1:",
                np.round(predicted_action[:6], 4),
            )
            print(
                "ACT FR5 #2:",
                np.round(predicted_action[6:], 4),
            )

            # CRITICAL:
            # Do not pass predicted_action to env.step().
            # Advance observations using current joint positions only.
            safe_placeholder = np.asarray(
                obs["joint_pos"],
                dtype=np.float64,
            ).copy()

            obs, _, terminated, truncated, info = env.step(
                safe_placeholder
            )

            if terminated or truncated:
                print("[INFO] Environment ended")
                break

            current_joints = np.asarray(
                obs["joint_pos"],
                dtype=np.float64,
            ).copy()

        print("\n========== ACT SHADOW TEST PASSED ==========")
        print("No ACT predictions were sent to the robots.")

    finally:
        env.close()


if __name__ == "__main__":
    main()
