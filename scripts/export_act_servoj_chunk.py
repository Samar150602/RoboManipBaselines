#!/usr/bin/env python3
"""Export one ACT prediction chunk to a local file. NEVER sends robot commands.

Run from RoboManipBaselines/scripts while FR5#1's normal ROS controller and
RealSense cameras are publishing observations. Uses the existing trained
checkpoint/normalization code in test_dual_fr5_act.py.
"""
import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import robo_manip_baselines.envs  # register environment
from robo_manip_baselines.common import denormalize_data
from test_dual_fr5_act import ENV_ID, load_policy, prepare_images, prepare_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out', default='/tmp/fr5_act_chunk.npz')
    ap.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = ap.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu')

    policy, meta = load_policy(args.checkpoint, torch.device(args.device))
    env = gym.make(ENV_ID)
    try:
        obs, info = env.reset()
        state_t, measured_rad = prepare_state(obs, meta, torch.device(args.device))
        image_t = prepare_images(info, meta, torch.device(args.device))
        with torch.inference_mode():
            prediction = policy(state_t, image_t)
        chunk = prediction[0].detach().cpu().numpy()
        if chunk.ndim != 2 or chunk.shape[1] != 12 or len(chunk) < 1:
            raise RuntimeError(f'Unexpected ACT chunk shape {chunk.shape}')
        predicted_rad = np.asarray(
            [denormalize_data(a, meta['action']) for a in chunk], dtype=np.float64
        )
        if predicted_rad.shape != chunk.shape or not np.isfinite(predicted_rad).all():
            raise RuntimeError('Invalid ACT prediction')
        if not np.isfinite(measured_rad).all():
            raise RuntimeError('Invalid robot observation')

        predicted_deg = np.rad2deg(predicted_rad[:, :6])
        measured_deg = np.rad2deg(measured_rad)
        deltas = predicted_deg - measured_deg[:6]

        print('\n=== ACT FR5 #1 CHUNK PREVIEW (NO MOVEMENT) ===')
        print('FR5 #1 observation (deg):', np.round(measured_deg[:6], 3))
        print('FR5 #2 observation (deg):', np.round(measured_deg[6:], 3))
        print('ACT shape:', predicted_deg.shape, 'training skip:', meta['data']['skip'])
        for i in range(len(predicted_deg)):
            print(f'{i:02d} target={np.round(predicted_deg[i],3)}  '
                  f'delta_from_start={np.round(deltas[i],3)}')
        print('Max delta from observation per joint (deg):',
              np.round(np.max(np.abs(deltas),axis=0), 3))
        if len(predicted_deg) > 1:
            print('Max adjacent action delta per joint (deg):',
                  np.round(np.max(np.abs(np.diff(predicted_deg, axis=0)),axis=0), 3))
        out = Path(args.out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            target_deg=predicted_deg,
            start_deg=measured_deg[:6],
            other_arm_start_deg=measured_deg[6:],
            created_unix=np.array(time.time()),
            training_skip=np.array(int(meta['data']['skip'])),
        )
        print(f'\nSaved preview on THIS PC: {out}')
        print('NO ROBOT COMMANDS WERE SENT BY THIS SCRIPT.')
    finally:
        env.close()


if __name__ == '__main__':
    main()
