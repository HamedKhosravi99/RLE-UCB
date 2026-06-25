"""Cumulative-regret trajectories over time on representative real benchmarks (one per
category). Shows HOW regret accumulates, not just the endpoint. Best hyperparameter per
method (read from results/bench_*.json), averaged over 3 seeds. Honest selection: a
hybrid and a linear set RLE-UCB wins, and a local set where the k-NN specialist wins."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets10 import load
from zhang_data import StreamEnv
from bandits import ALGOS

OUT = "../figures"
DS = [("Satimage", 182, None, "hybrid"), ("Optdigits", 28, None, "linear"), ("Phoneme", 1489, None, "local")]
# (display, ALGOS key, kwarg name)
METH = [("LinUCB", "LinUCB", "alpha", "#c44"), ("$k$-NN UCB", "kNN-UCB-Reeve", "c", "#4a4"),
        ("Naive", "NaiveReeve", "c", "#a86"), ("RLE-UCB", "RLE-UCB", "c", "#27a")]
SEEDS = list(range(20)); T = 3000


def curve(key, kw, X, y, A, Tt, seed):
    rng = np.random.default_rng(seed); env = StreamEnv(X, y, A, Tt, 1000 + seed)
    algo = ALGOS[key](X.shape[1], A, **kw); cum = np.empty(Tt); c = 0.0
    for t in range(Tt):
        ctxs, rew, exp = env.step(t, rng); a = algo.select(ctxs); algo.update(a, ctxs, rew[a])
        c += float(np.max(exp) - exp[a]); cum[t] = c
    return cum


fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
for ax, (name, did, cap, cat) in zip(axes, DS):
    X, y, A = load(name, did, cap); Tt = min(T, len(X))
    cfg = json.load(open(f"results/bench_{name}.json"))["models"]
    for disp, key, kwn, col in METH:
        bc = cfg[ {"LinUCB":"LinUCB","kNN-UCB-Reeve":"kNN-UCB","NaiveReeve":"Naive","RLE-UCB":"RLE-UCB"}[key] ]["best_config"]
        kw = {kwn: bc} if key != "SquareCB-lin" else {kwn: bc, "seed": 0}
        cs = np.mean([curve(key, dict(kw), X, y, A, Tt, s) for s in SEEDS], 0)
        lw = 2.4 if disp == "RLE-UCB" else 1.3
        ax.plot(np.arange(Tt), cs, color=col, lw=lw, label=disp)
    ax.set_title(f"{name} ({cat})"); ax.set_xlabel("round $t$"); ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    print(f"{name} ({cat}) done, T={Tt}", flush=True)
axes[0].set_ylabel("cumulative regret")
plt.tight_layout(); plt.savefig(f"{OUT}/regret_curves.png", dpi=140); plt.close()
print(f"[saved {OUT}/regret_curves.png]", flush=True)
