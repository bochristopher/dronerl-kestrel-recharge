"""Improved DQN trainer for the DroneRL 2026 evaluation setting.

Differences from the starter's train_torch.py (the network classes and checkpoint format are
the starter's own, so checkpoints load with BaseDQNFactory.from_checkpoint):
  * trains in the exact evaluation configuration: 11x11 grid, 6 drones, the 5 official baseline
    checkpoints as opponents, our drone in slot 0, 1000-step episodes
  * several environments stepped in lock-step with batched inference
  * Double DQN target, n-step returns, Huber loss, large batches, soft target updates,
    gradient clipping, linear epsilon schedule
  * crashes are treated as non-terminal by default (the drone respawns, the episode goes on)
  * periodic greedy evaluation on held-out seeds (never the official seeds); best checkpoint kept
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from common_setup import make_env, BASELINES
from fast_eval import evaluate, load
from torch_impl.agents.dqn import ConvQNetworkFactory, DenseQNetworkFactory, DenseQNetwork
from torch_impl.helpers.rl_helpers import set_seed


def save_checkpoint(net, factory, path):
    # identical metadata layout to DQNAgent.save() in the starter kit
    metadata = {
        "network_type": "dense" if isinstance(net, DenseQNetwork) else "conv",
        "dense_layers": str(factory.dense_layers),
        "obs_shape": str(factory.obs_shape),
        "action_shape": str(factory.action_shape),
    }
    if hasattr(factory, "conv_layers"):
        metadata["conv_layers"] = str(tuple({k: l[k] for k in ("out_channels", "kernel_size", "stride", "padding")}
                                            for l in factory.conv_layers))
    sd = {k: v.detach().cpu().contiguous() for k, v in net.state_dict().items()}
    save_file(sd, path, metadata=metadata)


class Replay:
    def __init__(self, cap, obs_shape):
        self.cap, self.n, self.i = cap, 0, 0
        self.s = np.zeros((cap,) + obs_shape, np.float16)
        self.s2 = np.zeros((cap,) + obs_shape, np.float16)
        self.a = np.zeros(cap, np.int64)
        self.r = np.zeros(cap, np.float32)
        self.g = np.zeros(cap, np.float32)  # discount to apply to bootstrap (0 if terminal)

    def add(self, s, a, r, s2, g):
        j = self.i
        self.s[j], self.a[j], self.r[j], self.s2[j], self.g[j] = s, a, r, s2, g
        self.i = (j + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, b):
        idx = np.random.randint(0, self.n, b)
        return (torch.from_numpy(self.s[idx].astype(np.float32)), torch.from_numpy(self.a[idx]),
                torch.from_numpy(self.r[idx]), torch.from_numpy(self.s2[idx].astype(np.float32)),
                torch.from_numpy(self.g[idx]))


def q_forward(net, x):
    """Batched forward on a float tensor [B,7,7,6] using the starter network's layers."""
    if isinstance(net, DenseQNetwork):
        return net.network(x.reshape(x.shape[0], -1))
    return net.network(x.permute(0, 3, 1, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps", type=int, default=2_000_000, help="total learner transitions")
    ap.add_argument("--n_envs", type=int, default=16)
    ap.add_argument("--net", choices=["conv", "dense"], default="conv")
    ap.add_argument("--conv", default='[{"out_channels":32,"kernel_size":3,"stride":1,"padding":1},{"out_channels":32,"kernel_size":3,"stride":1,"padding":1}]')
    ap.add_argument("--dense", default="128")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--gamma", type=float, default=0.97)
    ap.add_argument("--nstep", type=int, default=3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--updates_per_step", type=int, default=1, help="gradient updates per lock-step of all envs")
    ap.add_argument("--memory", type=int, default=500_000)
    ap.add_argument("--warmup", type=int, default=20_000)
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--eps_start", type=float, default=1.0)
    ap.add_argument("--eps_end", type=float, default=0.02)
    ap.add_argument("--eps_frac", type=float, default=0.3)
    ap.add_argument("--crash_terminal", action="store_true")
    ap.add_argument("--pickup_bonus", type=float, default=0.0, help="training-only shaping; eval uses true rewards")
    ap.add_argument("--opp_transitions", action="store_true", help="also store baseline opponents' transitions")
    ap.add_argument("--eval_every", type=int, default=100_000)
    ap.add_argument("--eval_seeds", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None, help="optional checkpoint to start from")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out = os.path.join("runs", a.name)
    os.makedirs(out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(out, "config.json"), "w"), indent=1)
    log = open(os.path.join(out, "log.jsonl"), "a")

    obs_shape, act_shape = (7, 7, 6), (5,)
    dense = tuple(int(x) for x in a.dense.split(",") if x)
    if a.init:
        from torch_impl.agents.dqn import BaseDQNFactory
        factory = BaseDQNFactory.from_checkpoint(a.init)
    elif a.net == "conv":
        factory = ConvQNetworkFactory(obs_shape, act_shape, conv_layers=tuple(json.loads(a.conv)), dense_layers=dense)
    else:
        factory = DenseQNetworkFactory(obs_shape, act_shape, hidden_layers=dense)
    qnet, _ = factory.create_qnetwork()
    tnet, _ = factory.create_qnetwork()
    tnet.load_state_dict(qnet.state_dict())
    opt = torch.optim.Adam(qnet.parameters(), lr=a.lr)
    opponents = [load(p) for p in BASELINES]
    for o in opponents:
        o.requires_grad_(False)

    # each env carries its own python RNG state so lock-stepping does not entangle them
    envs, states, rngs = [], [], []
    for e in range(a.n_envs):
        env = make_env()
        set_seed(env, 1_000_000 * (a.seed + 1) + e)
        st = env.reset()
        envs.append(env); states.append(st); rngs.append(random.getstate())
    ep_t = [0] * a.n_envs
    n_ag = 6 if a.opp_transitions else 1
    nbuf = [[[] for _ in range(n_ag)] for _ in range(a.n_envs)]
    replay = Replay(a.memory, obs_shape)
    disc = [a.gamma ** k for k in range(a.nstep + 1)]

    def push(buf, final=False):
        # emit n-step transitions from the head of buf
        while buf and (len(buf) >= a.nstep or final):
            R, g, s2 = 0.0, 0.0, None
            for k, (s, act, r, nxt, term) in enumerate(buf[:a.nstep]):
                R += disc[k] * r
                s2 = nxt
                if term:
                    g = 0.0
                    break
                g = disc[k + 1]
            replay.add(buf[0][0], buf[0][1], R, s2, g)
            buf.pop(0)

    total, t0, best, nxt_eval, upd = 0, time.time(), -1e9, a.eval_every, 0
    recent = []
    ep_ret = [0.0] * a.n_envs
    while total < a.steps:
        eps = max(a.eps_end, a.eps_start - (a.eps_start - a.eps_end) * total / (a.eps_frac * a.steps))
        obs = np.stack([np.stack([states[e][i] for i in range(6)]) for e in range(a.n_envs)]).astype(np.float32)
        x = torch.from_numpy(obs)
        with torch.no_grad():
            acts = np.zeros((a.n_envs, 6), np.int64)
            acts[:, 0] = q_forward(qnet, x[:, 0]).argmax(1).numpy()
            for i, o in enumerate(opponents, start=1):
                acts[:, i] = q_forward(o, x[:, i]).argmax(1).numpy()
        rand = np.random.rand(a.n_envs) < eps
        acts[rand, 0] = np.random.randint(0, 5, rand.sum())
        for e, env in enumerate(envs):
            random.setstate(rngs[e])
            nstate, rew, dones, _, _ = env.step({i: int(acts[e, i]) for i in range(6)})
            ep_t[e] += 1
            reset = ep_t[e] >= 1000
            for i in range(n_ag):
                r = rew[i]
                ep_ret[e] += r if i == 0 else 0.0
                if a.pickup_bonus and r == 0 and (not states[e][i][3, 3, 1]) and nstate[i][3, 3, 1] and not dones[i]:
                    r += a.pickup_bonus  # picked up a package this step (training-only shaping)
                term = bool(dones[i]) and a.crash_terminal
                nbuf[e][i].append((obs[e, i], int(acts[e, i]), r, nstate[i].astype(np.float32), term))
                if term:
                    push(nbuf[e][i], final=True)
                else:
                    push(nbuf[e][i])
            if reset:
                for i in range(n_ag):
                    push(nbuf[e][i], final=True)  # truncation: bootstrap from last next-state
                recent.append(ep_ret[e]); ep_ret[e] = 0.0
                nstate = env.reset(); ep_t[e] = 0
            states[e] = nstate
            rngs[e] = random.getstate()
        total += a.n_envs

        if replay.n >= a.warmup:
            for _ in range(a.updates_per_step):
                s, act, r, s2, g = replay.sample(a.batch)
                q = q_forward(qnet, s).gather(1, act[:, None]).squeeze(1)
                with torch.no_grad():
                    a2 = q_forward(qnet, s2).argmax(1, keepdim=True)  # double DQN
                    tgt = r + g * q_forward(tnet, s2).gather(1, a2).squeeze(1)
                loss = F.smooth_l1_loss(q, tgt)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(qnet.parameters(), 10.0)
                opt.step(); upd += 1
                with torch.no_grad():
                    for p, tp in zip(qnet.parameters(), tnet.parameters()):
                        tp.mul_(1 - a.tau).add_(a.tau * p)

        if total >= nxt_eval or total >= a.steps:
            nxt_eval += a.eval_every
            seeds = list(range(10_000, 10_000 + a.eval_seeds))  # held-out validation seeds
            state_py, state_np = random.getstate(), np.random.get_state()
            m, sd, _ = evaluate(qnet, seeds, opponents=opponents)
            random.setstate(state_py); np.random.set_state(state_np)
            save_checkpoint(qnet, factory, os.path.join(out, "last.safetensors"))
            if m > best:
                best = m
                save_checkpoint(qnet, factory, os.path.join(out, "best.safetensors"))
            rec = {"steps": total, "updates": upd, "eps": round(eps, 3), "val_mean": round(m, 2), "val_std": round(sd, 2),
                   "best": round(best, 2), "train_ep_return": round(float(np.mean(recent[-32:])), 2) if recent else None,
                   "sec": round(time.time() - t0)}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()


if __name__ == "__main__":
    main()
