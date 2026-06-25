"""Efficiency frontier: per-step wall-clock time vs horizon T. The Reeve k-NN-UCB baselines
do an O(t) search over k each round, so their per-step cost grows with T; RLE-UCB reads k off
a running variance statistic (O(1) selection), so its per-step cost grows much more slowly.
One dataset (Magic, A=2), single timing run per T. Figure: us/step vs T (log-log)."""
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets10 import load
from zhang_data import StreamEnv
from bandits import ALGOS

OUT = "../figures"
X, y, A = load("Magic", 1120, None)
TS = [1000, 2000, 4000, 8000, 16000]
METH = [("LinUCB", "LinUCB", {"alpha": 1.0}, "#c44"),
        ("$k$-NN UCB (Reeve)", "kNN-UCB-Reeve", {"c": 1.0}, "#4a4"),
        ("Naive (Lin+$k$-NN)", "NaiveReeve", {"c": 1.0}, "#a86"),
        ("RLE-UCB", "RLE-UCB", {"c": 0.5}, "#27a")]


def per_step_us(key, kw, Tt):
    rng = np.random.default_rng(0); env = StreamEnv(X, y, A, Tt, 1000)
    algo = ALGOS[key](X.shape[1], A, **kw)
    t0 = time.perf_counter()
    for t in range(Tt):
        ctxs, rew, exp = env.step(t, rng); a = algo.select(ctxs); algo.update(a, ctxs, rew[a])
    return 1e6 * (time.perf_counter() - t0) / Tt


plt.figure(figsize=(5.4, 3.9))
data = {}
for disp, key, kw, col in METH:
    us = [per_step_us(key, kw, Tt) for Tt in TS]
    data[disp] = us
    lw = 2.4 if disp == "RLE-UCB" else 1.4
    plt.plot(TS, us, marker="o", color=col, lw=lw, label=disp)
    print(f"{disp:22s} " + "  ".join(f"T={t}:{u:6.0f}us" for t, u in zip(TS, us)), flush=True)
plt.xscale("log"); plt.yscale("log")
plt.xlabel("horizon $T$"); plt.ylabel("per-step time ($\\mu$s)")
plt.title("Efficiency frontier: per-step cost vs. horizon (Magic)")
plt.legend(fontsize=8); plt.grid(alpha=0.3, which="both"); plt.tight_layout()
plt.savefig(f"{OUT}/efficiency_frontier.png", dpi=150); plt.close()
print(f"[saved {OUT}/efficiency_frontier.png]", flush=True)
import json; json.dump({"T": TS, "us_per_step": data}, open("results/efficiency_frontier.json", "w"), indent=2)
