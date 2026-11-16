"""Window-only scripted planner (diagnostic + optional demonstration source).

Uses ONLY the 7x7x6 observation that the network sees. Channels:
0 drones, 1 packets (or carried packet at own cell), 2 dropzones, 3 stations, 4 charge/100, 5 obstacles/out-of-bounds.
Actions: 0 LEFT, 1 DOWN, 2 RIGHT, 3 UP, 4 STAY.
"""
from collections import deque
import numpy as np

DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
C = 3


def plan(obs, rng=None):
    obs = np.asarray(obs)
    charge = int(round(obs[C, C, 4] * 100))
    carrying = obs[C, C, 1] > 0.5
    blocked = obs[:, :, 5] > 0.5
    others = [(y, x) for y in range(7) for x in range(7) if obs[y, x, 0] > 0.5 and (y, x) != (C, C)]
    danger = np.zeros((7, 7), bool)
    for (y, x) in others:
        for dy, dx in DIRS:
            yy, xx = y + dy, x + dx
            if 0 <= yy < 7 and 0 <= xx < 7:
                danger[yy, xx] = True

    def bfs(avoid_danger):
        dist = -np.ones((7, 7), int)
        first = {}
        dist[C, C] = 0
        q = deque()
        for a, (dy, dx) in enumerate(DIRS[:4]):
            y, x = C + dy, C + dx
            if 0 <= y < 7 and 0 <= x < 7 and not blocked[y, x] and not (avoid_danger and danger[y, x]):
                dist[y, x] = 1; first[(y, x)] = a; q.append((y, x))
        while q:
            y, x = q.popleft()
            for dy, dx in DIRS[:4]:
                yy, xx = y + dy, x + dx
                if 0 <= yy < 7 and 0 <= xx < 7 and dist[yy, xx] < 0 and not blocked[yy, xx]:
                    dist[yy, xx] = dist[y, x] + 1; first[(yy, xx)] = first[(y, x)]; q.append((yy, xx))
        return dist, first

    on_station = obs[C, C, 3] > 0.5
    moves_left = charge // 10 - 1  # non-station steps before the battery hits 0
    for avoid in (True, False):
        dist, first = bfs(avoid)
        st = [(dist[y, x], (y, x)) for y in range(7) for x in range(7) if obs[y, x, 3] > 0.5 and dist[y, x] > 0]
        tgt_ch = 2 if carrying else 1
        goals = [(dist[y, x], (y, x)) for y in range(7) for x in range(7)
                 if obs[y, x, tgt_ch] > 0.5 and dist[y, x] > 0 and (y, x) != (C, C)]
        # nearest station distance from each goal (approx via manhattan inside window)
        def station_after(g):
            if not st and not on_station:
                return 99
            cands = [abs(g[0] - s[1][0]) + abs(g[1] - s[1][1]) for s in st]
            return min(cands) if cands else 99
        if on_station and charge < 100 and (not goals or min(goals)[0] + 1 > moves_left + 2):
            if not (avoid and danger[C, C]):
                return 4  # keep charging
        feasible = [g for g in goals if g[0] <= moves_left and (g[0] + station_after(g[1]) <= moves_left + 1 or g[0] == moves_left or tgt_ch == 2)]
        if feasible and charge > 20:
            return first[min(feasible)[1]]
        if st:
            d, s = min(st)
            if d <= moves_left + 1:
                return first[s]
        if goals:
            return first[min(goals)[1]]
        # explore: step to a free cell, prefer away from walls
        opts = [a for a, (dy, dx) in enumerate(DIRS[:4]) if (C + dy, C + dx) in first]
        if opts:
            return opts[(rng.integers(len(opts)) if rng is not None else 0)]
    return 4


class HeuristicNet:
    """Callable with the same interface the evaluator uses: agent([obs]) -> q-values."""
    def __init__(self):
        self.rng = np.random.default_rng(0)

    def __call__(self, states):
        import torch
        a = plan(states[0], self.rng)
        q = torch.zeros(1, 5); q[0, a] = 1.0
        return q


if __name__ == "__main__":
    from fast_eval import evaluate
    m, s, _ = evaluate(HeuristicNet(), list(range(10_000, 10_020)))
    print(f"heuristic val20: {m:.2f} ± {s:.2f}")
