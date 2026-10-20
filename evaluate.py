"""Run a trained agent (no human input).  python evaluate.py --ckpt checkpoints/basic_best.pt --render --video"""
import argparse, os

import cv2
import numpy as np
import torch

from env import DoomEnv
from model import SpikingDQN


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--render", action="store_true", help="show the ViZDoom window")
    p.add_argument("--video", action="store_true", help="save mp4 to videos/")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device)
    c = ck["cfg"]
    env = DoomEnv(c["scenario"], c["frame_skip"], c["img_size"], c["stack"], render=args.render)
    net = SpikingDQN(ck["n_actions"], c["stack"], c["img_size"], c["T"], c["beta"], c["hidden"]).to(device)
    net.load_state_dict(ck["model"]); net.eval()

    writer = None
    if args.video:
        os.makedirs("videos", exist_ok=True)
        writer = cv2.VideoWriter(f"videos/{c['scenario']}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 15, (480, 360))

    all_rates, scores = [], []
    for ep in range(args.episodes):
        s, done, total = env.reset(), False, 0.0
        while not done:
            with torch.no_grad():
                x = torch.from_numpy(s).to(device).float().div(255).unsqueeze(0)
                q, rates = net(x, return_rates=True)
            all_rates.append(rates)
            s, r, done = env.step(q.argmax(1).item())
            total += r
            if writer:
                f = cv2.resize(env.raw_frame(), (480, 360), interpolation=cv2.INTER_NEAREST)
                writer.write(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))
        scores.append(total); print(f"episode {ep+1}: reward {total:.1f}")
    if writer: writer.release()
    print(f"mean reward: {np.mean(scores):.1f}")
    r = np.mean(all_rates, axis=0)
    print("avg spike rate per layer (spikes/neuron/timestep): " + ", ".join(f"L{i+1}={v:.3f}" for i, v in enumerate(r)))


if __name__ == "__main__":
    main()
