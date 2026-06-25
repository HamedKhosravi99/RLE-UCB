"""Two revision experiments (run on the 14 real benchmarks, StreamEnv protocol, T=min(3000,N)).

(1) CONTROLLED RESIDUALIZATION ABLATION. Isolate residualization by holding the exploration
    bonus, the variance-guided k-rule, and the subspace step IDENTICAL: RLE-UCB (residualize=True,
    k-NN averages Y - mu^T x) vs the SAME algorithm with raw-reward k-NN (residualize=False = the
    naive Lin+kNN hybrid, matched exploration / k / subspace). Both grid-searched over the SAME c
    grid, so the only thing that differs is residualization.

(2) FIXED-DEFAULT-c. RLE-UCB at a single fixed c=0.5 across ALL datasets vs its own best-of-grid,
    to show the tuning-free claim does not depend on per-dataset tuning.

Saves results/ablation_resid.json.  Usage: python3 ablation_resid.py [nseed] [Name,Name,...]"""
import sys, json, time, os
import numpy as np
from datasets10 import load
from zhang_data import StreamEnv
from bandits import run_episode, ALGOS
from bench_one import SPECS15

NAMES = ["Adult", "Magic", "Shuttle", "Letter", "Pendigits", "Optdigits", "Satimage",
         "Segment", "Vehicle", "Spambase", "Waveform", "Phoneme", "EEG", "MNIST"]   # 14 (no Mushroom)
CGRID = [0.05, 0.5, 5.0]
CDEFAULT = 0.5


def best_of_grid(X, y, A, T, seeds, extra):
    best = None
    for c in CGRID:
        regs = [run_episode(ALGOS["RLE-UCB"], StreamEnv(X, y, A, T, 1000 + s), T, s, {"c": c, **extra})
                for s in seeds]
        m = float(np.mean(regs))
        if best is None or m < best[1]:
            best = (c, m, float(np.std(regs)))
    return best                                            # (best_c, mean, std)


def main():
    nseed = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    names = sys.argv[2].split(",") if len(sys.argv) > 2 else NAMES
    seeds = list(range(nseed))
    out = {}
    try:
        out = json.load(open("results/ablation_resid.json"))
    except Exception:
        pass
    hdr = f"{'Dataset':10s}{'RLE(resid)':>12s}{'Naive-match':>12s}{'resid%':>9s}{'RLE c=0.5':>11s}{'RLE best':>10s}{'fixed%':>8s}"
    print(f"nseed={nseed}\n{hdr}", flush=True)
    for name in names:
        t0 = time.time()
        did, cap = SPECS15[name]; X, y, A = load(name, did, cap); T = min(3000, len(X))
        rle = best_of_grid(X, y, A, T, seeds, {"residualize": True})
        nai = best_of_grid(X, y, A, T, seeds, {"residualize": False})
        regs05 = [run_episode(ALGOS["RLE-UCB"], StreamEnv(X, y, A, T, 1000 + s), T, s, {"c": CDEFAULT})
                  for s in seeds]
        fixed = float(np.mean(regs05))
        resid_pct = 100.0 * (nai[1] - rle[1]) / nai[1]          # residualization gain over naive-matched
        fixed_pct = 100.0 * (fixed - rle[1]) / rle[1]           # fixed-default penalty vs best-grid
        out[name] = {"rle_best": rle[1], "rle_best_c": rle[0], "rle_best_std": rle[2],
                     "naive_matched_best": nai[1], "naive_matched_c": nai[0],
                     "resid_gain_pct": resid_pct, "rle_fixed_c0.5": fixed,
                     "fixed_vs_best_pct": fixed_pct, "T": T, "nseed": nseed,
                     "secs": round(time.time() - t0, 1)}
        print(f"{name:10s}{rle[1]:12.1f}{nai[1]:12.1f}{resid_pct:8.1f}%{fixed:11.1f}{rle[1]:10.1f}{fixed_pct:+7.1f}%",
              flush=True)
        os.makedirs("results", exist_ok=True)
        json.dump(out, open("results/ablation_resid.json", "w"), indent=2)
    done = [n for n in names if n in out]
    gains = [out[n]['resid_gain_pct'] for n in done]
    fdiff = [out[n]['fixed_vs_best_pct'] for n in done]
    print(f"\n[residualization gain over naive-matched]  median {np.median(gains):+.1f}%  mean {np.mean(gains):+.1f}%  "
          f"(wins {sum(g > 0 for g in gains)}/{len(gains)})")
    print(f"[fixed c=0.5 vs best-grid]  median {np.median(fdiff):+.1f}%  mean {np.mean(fdiff):+.1f}%  worst {np.max(fdiff):+.1f}%")
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
