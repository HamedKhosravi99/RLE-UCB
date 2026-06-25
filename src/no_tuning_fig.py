"""No-tuning evidence (always-update benchmarks, where per-arm counts grow so the 1/(N+1)
annealing engages). For RLE-UCB sweep its exploration constant c; for LinUCB sweep alpha; over
a wide grid (>1000x range). Plot cumulative regret vs the exploration parameter. RLE-UCB should
be ~flat (tuning-free); LinUCB should vary. Same format as the RLE-UCB exploration ablation."""
import sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets10 import load
from zhang_data import StreamEnv
from bandits import run_episode, ALGOS

OUT = "../figures"
DS = [("Magic", 1120, None, "hybrid"), ("Mushroom", 24, None, "local"), ("Letter", 6, None, "local")]
GRID = [0.02, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
T, SEEDS = 3000, list(range(20))
data = {}
fig, axes = plt.subplots(1, len(DS), figsize=(4.2 * len(DS), 3.4))


def sweep(key, pname, X, y, A, Tt):
    mean, std = [], []
    for v in GRID:
        vals = [run_episode(ALGOS[key], StreamEnv(X, y, A, Tt, 1000 + s), Tt, s, {pname: v}) for s in SEEDS]
        mean.append(float(np.mean(vals))); std.append(float(np.std(vals)))
    return np.array(mean), np.array(std)


for ax, (name, did, cap, cat) in zip(axes, DS):
    X, y, A = load(name, did, cap); Tt = min(T, len(X))
    rle, rle_sd = sweep("RLE-UCB", "c", X, y, A, Tt)
    lin, lin_sd = sweep("LinUCB", "alpha", X, y, A, Tt)
    data[name] = {"grid": GRID, "RLE-UCB": rle.tolist(), "RLE-UCB_std": rle_sd.tolist(),
                  "LinUCB": lin.tolist(), "LinUCB_std": lin_sd.tolist()}
    rs = (rle.max() - rle.min()) / rle.mean(); ls = (lin.max() - lin.min()) / lin.mean()
    ax.plot(GRID, lin, marker="s", color="#c44", linewidth=1.5, label=f"LinUCB ($\\alpha$), spread {ls:.0%}")
    ax.fill_between(GRID, lin - lin_sd, lin + lin_sd, color="#c44", alpha=0.15)
    ax.plot(GRID, rle, marker="o", color="#27a", linewidth=2.4, label=f"RLE-UCB ($c$), spread {rs:.0%}")
    ax.fill_between(GRID, rle - rle_sd, rle + rle_sd, color="#27a", alpha=0.15)
    ax.set_xscale("log"); ax.set_xlabel("exploration parameter"); ax.set_title(f"{name} ({cat})")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    print(f"{name:10s} [{cat}] T={Tt} ({len(SEEDS)} seeds): RLE-UCB regret {rle.min():.0f}-{rle.max():.0f} "
          f"(spread {rs:.1%}, std~{rle_sd.mean():.0f})  LinUCB {lin.min():.0f}-{lin.max():.0f} "
          f"(spread {ls:.1%}, std~{lin_sd.mean():.0f})", flush=True)
axes[0].set_ylabel("cumulative regret")
plt.tight_layout(); plt.savefig(f"{OUT}/no_tuning.png", dpi=140); plt.close()
print(f"[saved {OUT}/no_tuning.png]", flush=True)
json.dump(data, open("results/no_tuning.json", "w"), indent=2)
