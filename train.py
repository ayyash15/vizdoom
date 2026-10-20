"""Train the spiking DQN agent on ViZDoom.   python train.py --scenario basic --run_name v2"""
import argparse, copy, csv, os, random, time
from dataclasses import asdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import Config
from env import DoomEnv
from model import SpikingDQN
from replay import Replay


def evaluate(policy, env, device, episodes, eps=0.0, n_actions=3):
    """Returns (mean_reward, success_rate). Success = episode reward > 0 (monster killed)."""
    policy.eval()
    scores = []
    for _ in range(episodes):
        s, done, total = env.reset(), False, 0.0
        while not done:
            if random.random() < eps:
                a = random.randrange(n_actions)
            else:
                with torch.no_grad():
                    x = torch.from_numpy(s).to(device).float().div(255).unsqueeze(0)
                    a = policy(x).argmax(1).item()
            s, r, done = env.step(a)
            total += r
        scores.append(total)
    policy.train()
    return float(np.mean(scores)), float(np.mean(np.array(scores) > 0))


def main():
    cfg = Config()
    p = argparse.ArgumentParser()
    for k, v in asdict(cfg).items():
        p.add_argument(f"--{k}", type=type(v), default=v)
    p.add_argument("--resume", type=str, default=None)
    args = p.parse_args()
    cfg = Config(**{k: getattr(args, k) for k in asdict(cfg)})

    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")

    env = DoomEnv(cfg.scenario, cfg.frame_skip, cfg.img_size, cfg.stack, seed=cfg.seed)
    eval_env = DoomEnv(cfg.scenario, cfg.frame_skip, cfg.img_size, cfg.stack, seed=cfg.seed + 1)
    n = env.n_actions

    policy = SpikingDQN(n, cfg.stack, cfg.img_size, cfg.T, cfg.beta, cfg.hidden).to(device)
    if args.resume:
        policy.load_state_dict(torch.load(args.resume, map_location=device)["model"], strict=False)
    target = copy.deepcopy(policy).eval()
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    buf = Replay(cfg.buffer_size, cfg.stack, cfg.img_size)

    os.makedirs("checkpoints", exist_ok=True); os.makedirs("logs", exist_ok=True)
    tag = cfg.run_name or cfg.scenario
    log_f = open(f"logs/{tag}_episodes.csv", "w", newline="")
    log = csv.writer(log_f); log.writerow(["step", "episode_reward", "epsilon"])
    ev_f = open(f"logs/{tag}_evals.csv", "w", newline="")
    ev = csv.writer(ev_f); ev.writerow(["step", "mean_reward", "success_rate"])

    def save(name):
        torch.save({"model": policy.state_dict(), "cfg": asdict(cfg), "n_actions": n},
                   f"checkpoints/{tag}_{name}.pt")

    s, ep_r = env.reset(), 0.0
    best_key = (-1.0, -1e9)                    # (success_rate, mean_reward)
    ep_hist, loss_v = [], 0.0
    decay = cfg.eps_decay_frac * cfg.total_steps
    t0 = time.time()
    bar = tqdm(range(1, cfg.total_steps + 1), desc="train")
    for step in bar:
        eps = max(cfg.eps_end, cfg.eps_start - (cfg.eps_start - cfg.eps_end) * step / decay)
        if random.random() < eps:
            a = random.randrange(n)
        else:
            with torch.no_grad():
                a = policy(torch.from_numpy(s).to(device).float().div(255).unsqueeze(0)).argmax(1).item()
        ns, r, done = env.step(a)
        buf.add(s, a, r * cfg.reward_scale, ns, done)
        s, ep_r = ns, ep_r + r

        if done:
            ep_hist.append(ep_r); log.writerow([step, ep_r, round(eps, 3)]); log_f.flush()
            s, ep_r = env.reset(), 0.0

        if buf.n >= cfg.warmup and step % cfg.train_every == 0:
            bs, ba, br, bns, bd = buf.sample(cfg.batch_size, device)
            q = policy(bs).gather(1, ba.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                if cfg.double_dqn:
                    na = policy(bns).argmax(1, keepdim=True)        # policy picks action
                    nq = target(bns).gather(1, na).squeeze(1)       # target evaluates it
                else:
                    nq = target(bns).max(1)[0]
                y = br + cfg.gamma * nq * (1 - bd)
            loss = F.smooth_l1_loss(q, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
            opt.step(); loss_v = loss.item()

        if step % cfg.target_update == 0:
            target.load_state_dict(policy.state_dict())

        if step % 200 == 0 and ep_hist:
            bar.set_postfix(avg10=f"{np.mean(ep_hist[-10:]):.1f}", eps=f"{eps:.2f}", loss=f"{loss_v:.3f}")

        if step % cfg.eval_every == 0:
            mean_r, succ = evaluate(policy, eval_env, device, cfg.eval_episodes, cfg.eval_eps, n)
            ev.writerow([step, round(mean_r, 1), round(succ, 3)]); ev_f.flush()
            print(f"\n[eval] step {step}: mean reward {mean_r:.1f}, success {succ*100:.0f}%  ({(time.time()-t0)/60:.1f} min)")
            save("last"); save(f"step{step}")
            if (succ, mean_r) > best_key:
                best_key = (succ, mean_r); save("best"); print("  -> new best saved")

    save("last"); log_f.close(); ev_f.close()
    plt.figure(figsize=(8, 4))
    if ep_hist:
        k = min(20, len(ep_hist))
        plt.plot(np.convolve(ep_hist, np.ones(k) / k, mode="valid"), label=f"train (avg{k})")
    plt.xlabel("episode"); plt.ylabel("reward"); plt.title(f"SNN-DQN on {cfg.scenario} [{tag}]")
    plt.legend(); plt.tight_layout(); plt.savefig(f"logs/{tag}_curve.png", dpi=120)
    print(f"Done. Best eval: success {best_key[0]*100:.0f}%, mean reward {best_key[1]:.1f}")


if __name__ == "__main__":
    main()
