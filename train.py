"""Train a DQN agent on ViZDoom (cnn or snn).   python train.py --model cnn --run_name cnn_base"""
import argparse, copy, csv, math, os, random, time
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
from model import build_model
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

    combo = bool(cfg.combo_actions)
    env = DoomEnv(cfg.scenario, cfg.frame_skip, cfg.img_size, cfg.stack, seed=cfg.seed, combo=combo)
    eval_env = DoomEnv(cfg.scenario, cfg.frame_skip, cfg.img_size, cfg.stack, seed=cfg.seed + 1, combo=combo)
    n = env.n_actions
    print(f"actions: {n}  frame_skip: {cfg.frame_skip}  combo: {combo}")

    policy = build_model(cfg, n).to(device)
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
    ep_hist, loss_v, evals = [], 0.0, []          # evals: (step, mean_reward, success)
    decay = cfg.eps_decay_frac * cfg.total_steps
    t0 = time.time()
    bar = tqdm(range(1, cfg.total_steps + 1), desc="train")
    for step in bar:
        eps = max(cfg.eps_end, cfg.eps_start - (cfg.eps_start - cfg.eps_end) * step / decay)
        if cfg.lr_decay:
            prog = step / cfg.total_steps
            lr = cfg.lr * (cfg.lr_final_frac + (1 - cfg.lr_final_frac) * 0.5 * (1 + math.cos(math.pi * prog)))
            for g in opt.param_groups:
                g["lr"] = lr
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
            if cfg.rate_reg > 0 and cfg.model == "snn":
                q_all, rate = policy(bs, return_reg=True)      # rate = mean spike rate (differentiable)
            else:
                q_all, rate = policy(bs), 0.0
            q = q_all.gather(1, ba.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                if cfg.double_dqn:
                    na = policy(bns).argmax(1, keepdim=True)        # policy picks action
                    nq = target(bns).gather(1, na).squeeze(1)       # target evaluates it
                else:
                    nq = target(bns).max(1)[0]
                y = br + cfg.gamma * nq * (1 - bd)
            loss = F.smooth_l1_loss(q, y) + cfg.rate_reg * rate
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
            opt.step(); loss_v = loss.item()

        if step % cfg.target_update == 0:
            target.load_state_dict(policy.state_dict())

        if step % 200 == 0 and ep_hist:
            bar.set_postfix(avg10=f"{np.mean(ep_hist[-10:]):.1f}", eps=f"{eps:.2f}", loss=f"{loss_v:.3f}")

        if step % cfg.eval_every == 0:
            mean_r, succ = evaluate(policy, eval_env, device, cfg.eval_episodes, cfg.eval_eps, n)
            evals.append((step, mean_r, succ))
            ev.writerow([step, round(mean_r, 1), round(succ, 3)]); ev_f.flush()
            print(f"\n[eval] step {step}: mean reward {mean_r:.1f}, success {succ*100:.0f}%  ({(time.time()-t0)/60:.1f} min)")
            save("last"); save(f"step{step}")

    save("last"); log_f.close(); ev_f.close()

    # ---- final: re-score the top-k checkpoints on identical spawns, keep the best ----
    ok = [e for e in evals if e[2] >= 0.95] or evals
    cands = sorted(ok, key=lambda e: (e[1], e[2]), reverse=True)[:cfg.final_top_k]
    final = []
    for step, m, sc in cands:
        net = build_model(cfg, n).to(device)
        net.load_state_dict(torch.load(f"checkpoints/{tag}_step{step}.pt", map_location=device)["model"], strict=False)
        random.seed(1234)                                   # same eval randomness for every candidate
        fenv = DoomEnv(cfg.scenario, cfg.frame_skip, cfg.img_size, cfg.stack, seed=cfg.seed + 2, combo=combo)
        fm, fs = evaluate(net, fenv, device, cfg.final_eval_episodes, 0.0, n)
        fenv.close()
        final.append((fm, fs, step))
        print(f"[final] step {step}: periodic {m:.1f} -> re-eval {fm:.1f} (success {fs*100:.0f}%, {cfg.final_eval_episodes} eps)")
    if final:
        fm, fs, bstep = sorted(final, key=lambda t: (t[1] >= 0.95, t[0]), reverse=True)[0]
        torch.save(torch.load(f"checkpoints/{tag}_step{bstep}.pt", map_location="cpu"),
                   f"checkpoints/{tag}_best.pt")
        with open(f"logs/{tag}_final.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["step", "mean_reward", "success_rate"])
            for a_, b_, c_ in final:
                w.writerow([c_, round(a_, 2), round(b_, 3)])
        print(f"Done. Best checkpoint: step {bstep}, mean reward {fm:.1f}, success {fs*100:.0f}% "
              f"({cfg.final_eval_episodes} episodes)")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    if ep_hist:
        k = min(20, len(ep_hist))
        ax[0].plot(np.convolve(ep_hist, np.ones(k) / k, mode="valid"), label=f"train (avg{k})")
        ax[0].legend()
    ax[0].set_xlabel("episode"); ax[0].set_ylabel("reward")
    if evals:
        ax[1].plot([e[0] for e in evals], [e[1] for e in evals], marker="o")
    ax[1].set_xlabel("step"); ax[1].set_ylabel("eval mean reward")
    fig.suptitle(f"{cfg.model.upper()}-DQN on {cfg.scenario} [{tag}]")
    fig.tight_layout(); fig.savefig(f"logs/{tag}_curve.png", dpi=120)


if __name__ == "__main__":
    main()
