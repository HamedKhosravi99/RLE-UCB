"""Aggregate results/bench_*.json into (1) a categorized main results table (regret, reward,
time) across all 15 real datasets, (2) a dataset categorization (linear- / local- / hybrid-
dominated, by the LinUCB-vs-kNN gap), and (3) a no-tuning sensitivity table: relative spread
of best-vs-worst regret across each model's hyperparameter grid (small spread = no tuning
needed). Emits both a console summary and LaTeX rows."""
import json, glob, os
import numpy as np

ORDER = ["Phoneme", "EEG", "Magic", "Spambase", "Waveform", "Vehicle", "Segment",
         "Satimage", "Pendigits", "Optdigits", "Shuttle", "Letter", "Mushroom", "Adult", "MNIST"]
MODELS = ["LinUCB", "kNN-UCB", "Naive", "ModelSelect", "SquareCB", "RLE-UCB"]
res = {}
for f in glob.glob("results/bench_*.json"):
    if f.endswith("bench_summary.json"):
        continue
    d = json.load(open(f)); res[d["dataset"]] = d
names = [n for n in ORDER if n in res] + [n for n in res if n not in ORDER]


def categorize(r):
    L = r["models"]["LinUCB"]["best_regret"]; K = r["models"]["kNN-UCB"]["best_regret"]
    lo, hi = min(L, K), max(L, K)
    ratio = hi / max(lo, 1e-9)
    if ratio < 1.15:
        return "hybrid"          # neither specialist dominates
    return "linear" if L < K else "local"


def relspread(model_block):
    regs = [c["regret_mean"] for c in model_block["all_configs"].values()]
    return (max(regs) - min(regs)) / max(np.mean(regs), 1e-9)


print(f"{'Dataset':11s}{'cat':8s}{'d':>5}{'A':>4}{'T':>6}   " + "".join(f"{m:>11s}" for m in MODELS), flush=True)
print("-" * 130, flush=True)
cats = {}; rows = []
for n in names:
    r = res[n]; cat = categorize(r); cats[n] = cat
    regrets = {m: r["models"][m]["best_regret"] for m in MODELS}
    winner = min(regrets, key=regrets.get)
    cells = []
    for m in MODELS:
        v = regrets[m]; mark = "*" if m == winner else " "
        cells.append(f"{v:9.0f}{mark} ")
    print(f"{n:11s}{cat:8s}{r['d']:>5}{r['A']:>4}{r['T']:>6}   " + "".join(cells), flush=True)
    rows.append((n, cat, r))

# reward + time + win summary
print("\n=== reward (1-regret/T) and time (us/step), best config ===", flush=True)
print(f"{'Dataset':11s}   " + "".join(f"{m:>11s}" for m in MODELS), flush=True)
for n in names:
    r = res[n]
    rw = "".join(f"{r['models'][m]['best_reward']:11.3f}" for m in MODELS)
    print(f"{n:11s} rw" + rw, flush=True)
    tm = "".join(f"{r['models'][m]['us_per_step']:11.0f}" for m in MODELS)
    print(f"{'':11s} us" + tm, flush=True)

# win counts by category
print("\n=== wins by category (lowest regret) ===", flush=True)
for cat in ["linear", "local", "hybrid"]:
    ds = [n for n in names if cats[n] == cat]
    wins = {m: 0 for m in MODELS}
    for n in ds:
        w = min(MODELS, key=lambda m: res[n]["models"][m]["best_regret"]); wins[w] += 1
    print(f"  {cat:8s} ({len(ds):2d} sets): " + ", ".join(f"{m} {wins[m]}" for m in MODELS if wins[m]), flush=True)

# no-tuning sensitivity: relative regret spread across each model's grid
print("\n=== no-tuning: relative regret spread across hyperparameter grid (smaller=less tuning) ===", flush=True)
print(f"{'Dataset':11s}   {'LinUCB(a)':>12s}{'kNN-UCB(c)':>12s}{'RLE-UCB(c)':>12s}", flush=True)
spreads = {"LinUCB": [], "kNN-UCB": [], "RLE-UCB": []}
for n in names:
    r = res[n]["models"]
    sl, sk, sr = relspread(r["LinUCB"]), relspread(r["kNN-UCB"]), relspread(r["RLE-UCB"])
    for m, s in [("LinUCB", sl), ("kNN-UCB", sk), ("RLE-UCB", sr)]:
        spreads[m].append(s)
    print(f"{n:11s}   {sl:11.1%}{sk:11.1%}{sr:11.1%}", flush=True)
print(f"{'MEDIAN':11s}   " + "".join(f"{np.median(spreads[m]):11.1%}" for m in ["LinUCB", "kNN-UCB", "RLE-UCB"]), flush=True)
print(f"{'MEAN':11s}   " + "".join(f"{np.mean(spreads[m]):11.1%}" for m in ["LinUCB", "kNN-UCB", "RLE-UCB"]), flush=True)

# LaTeX main table rows (mean +- std)
print("\n=== LaTeX main table rows (dataset & cat & d & A & per-model best regret mean+-std) ===", flush=True)
for n in names:
    r = res[n]
    reg = {m: r["models"][m]["best_regret"] for m in MODELS}
    sd = {m: r["models"][m]["best_std"] for m in MODELS}
    w = min(reg, key=reg.get)
    cells = " & ".join((f"\\textbf{{{reg[m]:.0f}}}$\\pm${sd[m]:.0f}" if m == w
                        else f"{reg[m]:.0f}$\\pm${sd[m]:.0f}") for m in MODELS)
    print(f"{n} & {cats[n]} & {r['d']} & {r['A']} & {cells} \\\\", flush=True)
    print(f"  seeds={r['seeds']}", flush=True)

# LaTeX appendix: reward + time per dataset
print("\n=== LaTeX appendix reward rows (reward = 1-regret/T) ===", flush=True)
for n in names:
    r = res[n]["models"]
    rw = {m: r[m]["best_reward"] for m in MODELS}
    w = max(rw, key=rw.get)
    cells = " & ".join((f"\\textbf{{{rw[m]:.3f}}}" if m == w else f"{rw[m]:.3f}") for m in MODELS)
    print(f"{n} & {cells} \\\\", flush=True)
print("\n=== LaTeX appendix time rows (us/step, best config) ===", flush=True)
for n in names:
    r = res[n]["models"]
    tm = {m: r[m]["us_per_step"] for m in MODELS}
    cells = " & ".join(f"{tm[m]:.0f}" for m in MODELS)
    print(f"{n} & {cells} \\\\", flush=True)

json.dump({"categories": cats, "spreads_median": {m: float(np.median(spreads[m])) for m in spreads}},
          open("results/bench_summary.json", "w"), indent=2)
print("\n[saved results/bench_summary.json]", flush=True)
