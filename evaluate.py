"""Run a trained agent (no human input).
   python evaluate.py --ckpt checkpoints/basic_best.pt --episodes 30 --eps 0.02"""
import argparse, os, random

import cv2
import numpy as np
import torch

from env import DoomEnv
from model import build_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--eps", type=float, default=0.0, help="random-action probability at eval (0 = pure greedy)")
    p.add_argument("--render", action="store_true", help="show the ViZDoom window")
    p.add_argument("--video", action="store_true", help="save mp4 to videos/")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device)
    c = ck["cfg"]
    env = DoomEnv(c["scenario"], c["frame_skip"], c["img_size"], c["stack"], render=args.render)
    net = build_model(c, ck["n_actions"]).to(device)
    net.load_state_dict(ck["model"], strict=False); net.eval()

    writer = None
    if args.video:
        os.makedirs("videos", exist_ok=True)
        name = os.path.splitext(os.path.basename(args.ckpt))[0]
        writer = cv2.VideoWriter(f"videos/{name}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 15, (480, 360))

    all_rates, scores = [], []
    action_counts = {}      # how often each action is chosen (detects an agent stuck on one action)
    for ep in range(args.episodes):
        s, done, total = env.reset(), False, 0.0
        while not done:
            with torch.no_grad():
                x = torch.from_numpy(s).to(device).float().div(255).unsqueeze(0)
                q, rates = net(x, return_rates=True)
            all_rates.append(rates)
            a = random.randrange(ck["n_actions"]) if random.random() < args.eps else q.argmax(1).item()
            action_counts[a] = action_counts.get(a, 0) + 1
            s, r, done = env.step(a)
            total += r
            if writer:
                f = cv2.resize(env.raw_frame(), (480, 360), interpolation=cv2.INTER_NEAREST)
                writer.write(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))
        scores.append(total); print(f"episode {ep+1}: reward {total:.1f}")
    if writer: writer.release()
    sc = np.array(scores)
    print(f"mean reward: {sc.mean():.1f}   success rate (reward>0): {(sc > 0).mean()*100:.0f}%")
    print("action counts:", dict(sorted(action_counts.items())))
    r = np.mean(all_rates, axis=0)
    print("avg spike rate per layer (spikes/neuron/timestep): " + ", ".join(f"L{i+1}={v:.3f}" for i, v in enumerate(r)))


if __name__ == "__main__":
    main()
