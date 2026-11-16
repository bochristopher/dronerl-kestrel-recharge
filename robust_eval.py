"""Held-out robustness check: mean, std and share of 'bad' episodes (< 20) over seeds 20000-20099."""
import sys
from fast_eval import evaluate, load
seeds = list(range(20000, 20100))
for p in sys.argv[1:]:
    m, s, sc = evaluate(load(p), seeds)
    y = sc[:, 0]
    print(f"{p}: mean={m:.2f} std={s:.2f} min={y.min():.1f} bad(<20)={int((y < 20).sum())}/100", flush=True)
