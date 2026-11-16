"""Replica of the official DroneRacerEvaluator scoring loop without video rendering.

Same env params, same 5 baseline opponents, same agent ordering ("YOU" sorts first -> drone 0),
same seeding. With --seeds official it reproduces the official evaluator's score exactly.
"""
import argparse
import numpy as np
import torch
from common_setup import make_env, OFFICIAL_SEEDS, BASELINES
from torch_impl.agents.dqn import BaseDQNFactory
from torch_impl.helpers.rl_helpers import set_seed


def load(path):
    return BaseDQNFactory.from_checkpoint(path).create_qnetwork()[0]


def evaluate(model, seeds=OFFICIAL_SEEDS, steps=1000, opponents=None):
    opponents = opponents or [load(p) for p in BASELINES]
    agents = [model] + opponents  # sorted names: "YOU" < "baseline-1" < ... < "baseline-5"
    scores = []
    with torch.no_grad():
        for seed in seeds:
            env = make_env()
            set_seed(env, seed)
            state = env.reset()
            ep = np.zeros(len(agents))
            for _ in range(steps):
                actions = {i: a([state[i]])[0].argmax().item() for i, a in enumerate(agents)}
                state, rewards, _, _, _ = env.step(actions)
                ep += np.array(list(rewards.values()))
            scores.append(ep)
    scores = np.array(scores)
    return scores[:, 0].mean(), scores[:, 0].std(), scores


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--seeds", default="official", help="'official' or 'val' (100 held-out seeds) or comma list")
    a = ap.parse_args()
    if a.seeds == "official":
        seeds = OFFICIAL_SEEDS
    elif a.seeds == "val":
        seeds = list(range(10_000, 10_100))
    else:
        seeds = [int(s) for s in a.seeds.split(",")]
    opp = [load(p) for p in BASELINES]
    for m in a.models:
        mean, std, _ = evaluate(load(m), seeds, opponents=opp)
        print(f"{m}: mean={mean:.2f} std={std:.2f} over {len(seeds)} seeds")
