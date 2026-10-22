"""Best-case (oracle) search for the aiming action space. NOT RUN YET. CPU only, no training, no GPU.
For every fixed seed it tries many open-loop programs [turn actions...] -> fire, and keeps the best per seed.
This is an UPPER BOUND with hindsight (per-seed best program), not a learnable policy.
    python oracle_aim.py --aim_mode delta --combo 1 --seeds 100 --aim_unit 1.0
Sweep --aim_unit (0.5, 1, 2) because delta units are UNVERIFIED. Compare with the ~84 strafe ceiling."""
import argparse, itertools, json
import numpy as np
from env import DoomEnv

p = argparse.ArgumentParser()
p.add_argument("--aim_mode", default="delta"); p.add_argument("--aim_deltas", default="1,3,8,20")
p.add_argument("--aim_unit", type=float, default=1.0); p.add_argument("--combo", type=int, default=1)
p.add_argument("--seeds", type=int, default=100); p.add_argument("--seed0", type=int, default=100000)
p.add_argument("--max_turns", type=int, default=2); p.add_argument("--fire_cap", type=int, default=40)
a = p.parse_args()

env = DoomEnv("basic", 1, 64, 4, seed=a.seed0, combo=bool(a.combo),
              aim_mode=a.aim_mode, aim_deltas=a.aim_deltas, aim_unit=a.aim_unit)
g, nd = env.game, len(a.aim_deltas.split(",")) * 2
turn_ids, atk = list(range(nd)), nd                      # delta layout: [turns..., ATTACK, (turn+ATTACK)...]
combo_ids = list(range(nd + 1, 2 * nd + 1)) if a.combo else []
progs = [()]
for k in range(1, a.max_turns + 1):
    progs += list(itertools.product(turn_ids, repeat=k))
progs += [t + (c,) for k in range(0, a.max_turns) for t in itertools.product(turn_ids, repeat=k) for c in combo_ids]

def run(seed, prog):
    g.set_seed(seed); g.new_episode(); tot, t = 0.0, 0
    for ai in prog:
        tot += g.make_action(env.actions[ai], 1); t += 1
        if g.is_episode_finished(): return tot, t
    for _ in range(a.fire_cap):
        tot += g.make_action(env.actions[atk], 1); t += 1
        if g.is_episode_finished(): break
    return tot, t

best = []
for i in range(a.seeds):
    s = a.seed0 + i
    r = max((run(s, pr) + (pr,) for pr in progs), key=lambda x: x[0])
    best.append(r)
sc = np.array([b[0] for b in best]); ln = np.array([b[1] for b in best])
print(json.dumps(dict(seeds=a.seeds, mean=sc.mean(), std=sc.std(ddof=1), median=float(np.median(sc)),
                      success=float((sc > 0).mean()), mean_tics=ln.mean(), aim_unit=a.aim_unit, combo=a.combo)))
