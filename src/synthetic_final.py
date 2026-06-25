"""Regenerate the synthetic separation triangle with the FINAL method
(residualized k-NN + uncertainty-adaptive exploration) and produce the figures
the paper references. Prints the regret table and saves env1/2/3 PNGs."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bandits import ALGOS, ArmState
from run_synthetic import EnvHybrid, EnvLinear, EnvNonlinear

OUT = "../figures"
T, SEEDS = 15000, list(range(5))


def run_curve(AlgoCls, env, T, seed, kw=None):
    rng = np.random.default_rng(seed)
    algo = AlgoCls(env.d, env.A, **(kw or {}))
    cur = np.empty(T); tot = 0.0
    for t in range(T):
        ctxs, rew, exp = env.step(t, rng)
        a = algo.select(ctxs); algo.update(a, ctxs, rew[a])
        tot += float(np.max(exp) - exp[a]); cur[t] = tot
    return cur


def mean_curve(AlgoCls, EnvCls, kw=None):
    curves = [run_curve(AlgoCls, EnvCls(), T, s, kw) for s in SEEDS]
    return np.mean(curves, axis=0)


def best_final(EnvCls, c_grid=(0.05, 0.1, 0.3)):
    best = None
    for c in c_grid:
        m = mean_curve(ALGOS["RLE-UCB-final"], EnvCls, {"c": c})
        if best is None or m[-1] < best[1][-1]:
            best = (c, m)
    return best


ENVS = [("Env1 Hybrid (d=5,d'=2)", EnvHybrid, "env1_lnucb_dominates.png", "RLE-UCB"),
        ("Env2 Linear (d=10)",     EnvLinear, "env2_linucb_dominates.png", "LinUCB"),
        ("Env3 Nonlinear (d=3)",   EnvNonlinear, "env3_knn_dominates.png", "kNN-UCB")]

print(f"=== Synthetic (final method), T={T}, {len(SEEDS)} seeds ===", flush=True)
print(f"{'Env':28s}{'LinUCB':>10s}{'kNN-UCB':>10s}{'RLE-UCB':>12s}", flush=True)
results = {}
for name, EnvCls, fname, _ in ENVS:
    lin = mean_curve(ALGOS["LinUCB"], EnvCls, {"alpha": 1.0})
    knn = mean_curve(ALGOS["kNN-UCB"], EnvCls, {"alpha": 1.0, "k": 5})
    cbest, fin = best_final(EnvCls)
    results[name] = (lin, knn, fin, cbest)
    print(f"{name:28s}{lin[-1]:10.0f}{knn[-1]:10.0f}{fin[-1]:12.0f}   (c={cbest})", flush=True)
    plt.figure(figsize=(5, 3.4))
    plt.plot(lin, label="LinUCB"); plt.plot(knn, label="kNN-UCB")
    plt.plot(fin, label="RLE-UCB", linewidth=2)
    plt.xlabel("round $t$"); plt.ylabel("cumulative regret")
    plt.title(name); plt.legend(); plt.tight_layout()
    plt.savefig(f"{OUT}/{fname}", dpi=130); plt.close()
    print(f"    saved {fname}", flush=True)

print("\nLaTeX table rows:")
for name, EnvCls, _, _ in ENVS:
    lin, knn, fin, _ = results[name]
    print(f"  {name}: LinUCB={lin[-1]:.0f}  kNN={knn[-1]:.0f}  RLE-UCB={fin[-1]:.0f}", flush=True)
