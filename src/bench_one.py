"""Run ALL in-domain models on ONE real dataset (raw features: ordinal-encode categoricals
+ standardize + L2-normalize rows; NO PCA / dim-reduction). Disjoint per-arm protocol,
reward 1 if arm==label, regret=#mistakes. Fair tuning: every model gets a 3-value grid of
its single exploration hyperparameter; best config reported. Saves results/bench_<name>.json
with per-config, per-seed regret/reward/time for reproducibility. Usage: python bench_one.py <Name> [T] [nseed]"""
import sys, json, time, os
import numpy as np
from datasets10 import load
from zhang_data import StreamEnv
from bandits import run_episode, ALGOS

# 15 standard real classification-as-bandit datasets (name -> openml id, cap N)
SPECS15 = {
    "Adult": (1590, 30000), "Mushroom": (24, None), "Magic": (1120, None),
    "Shuttle": (40685, 30000), "Letter": (6, None), "Pendigits": (32, None),
    "Optdigits": (28, None), "Satimage": (182, None), "Segment": (36, None),
    "MNIST": (554, 20000), "Vehicle": (54, None), "Spambase": (44, None),
    "Waveform": (60, None), "Phoneme": (1489, None), "EEG": (1471, None),
}
MODELS = [  # (display, ALGOS key, kwargs builder, 3-value grid)  -- NOT neural/kernel
    ("LinUCB",      "LinUCB",        lambda v, s: {"alpha": v},            [0.25, 1.0, 4.0]),
    ("kNN-UCB",     "kNN-UCB-Reeve", lambda v, s: {"c": v},                [0.5, 1.0, 2.0]),
    ("Naive",       "NaiveReeve",    lambda v, s: {"c": v},                [0.5, 1.0, 2.0]),
    ("ModelSelect", "ModelSelect",   lambda v, s: {"C": v},                [0.5, 1.0, 2.0]),
    ("SquareCB",    "SquareCB-lin",  lambda v, s: {"gamma": v, "seed": s}, [1.0, 10.0, 50.0]),
    ("RLE-UCB",     "RLE-UCB",       lambda v, s: {"c": v},                [0.05, 0.5, 5.0]),
]


def main():
    name = sys.argv[1]
    T = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    nseed = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    seeds = list(range(nseed))
    did, cap = SPECS15[name]
    X, y, A = load(name, did, cap); Tt = min(T, len(X))
    res = {"dataset": name, "N": int(len(X)), "d": int(X.shape[1]), "A": int(A),
           "T": int(Tt), "seeds": nseed, "models": {}}
    print(f"=== {name}: N={len(X)} d={X.shape[1]} A={A} T={Tt} seeds={nseed} ===", flush=True)
    for disp, key, kwfn, grid in MODELS:
        configs = {}; best = None
        for v in grid:
            t0 = time.time()
            regs = [run_episode(ALGOS[key], StreamEnv(X, y, A, Tt, 1000 + s), Tt, s, kwfn(v, s)) for s in seeds]
            dt = (time.time() - t0) / len(seeds)
            m, sd = float(np.mean(regs)), float(np.std(regs))
            configs[str(v)] = {"regret_mean": m, "regret_std": sd, "reward": 1 - m / Tt,
                               "time_per_run_s": dt, "us_per_step": 1e6 * dt / Tt,
                               "per_seed_regret": [float(r) for r in regs]}
            if best is None or m < best["best_regret"]:
                best = {"best_config": v, "best_regret": m, "best_std": sd,
                        "best_reward": 1 - m / Tt, "time_per_run_s": dt, "us_per_step": 1e6 * dt / Tt}
        best["all_configs"] = configs
        res["models"][disp] = best
        print(f"  {disp:12s} best regret {best['best_regret']:8.1f}+-{best['best_std']:5.1f} "
              f"(p={best['best_config']})  reward {best['best_reward']:.3f}  {best['us_per_step']:7.0f} us/step",
              flush=True)
        os.makedirs("results", exist_ok=True)             # checkpoint after each model
        json.dump(res, open(f"results/bench_{name}.json", "w"), indent=2)
    print(f"[saved results/bench_{name}.json]", flush=True)


if __name__ == "__main__":
    main()
