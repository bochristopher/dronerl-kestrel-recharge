"""Break a checkpoint's episode reward into deliveries, crashes and charging steps."""
import sys, collections
import numpy as np, torch
from common_setup import make_env, OFFICIAL_SEEDS, BASELINES
from fast_eval import load
from torch_impl.helpers.rl_helpers import set_seed

def run(path, seeds):
    agents = [load(path)] + [load(p) for p in BASELINES]
    rows = []
    for seed in seeds:
        env = make_env(); set_seed(env, seed); state = env.reset()
        c = collections.Counter(); acts = collections.Counter()
        with torch.no_grad():
            for _ in range(1000):
                a = {i: ag([state[i]])[0].argmax().item() for i, ag in enumerate(agents)}
                acts[a[0]] += 1
                state, r, d, _, _ = env.step(a)
                c[round(r[0], 2)] += 1
        rows.append((seed, c[1], c[-1], c[-0.1], dict(acts)))
        print(f"seed {seed}: deliveries={c[1]} crashes={c[-1]} charge_steps={c[-0.1]} score={c[1]-c[-1]-0.1*c[-0.1]:.1f} actions={dict(sorted(acts.items()))}", flush=True)
    return rows

if __name__ == "__main__":
    run(sys.argv[1], [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else OFFICIAL_SEEDS)
