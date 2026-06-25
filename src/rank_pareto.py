"""Two summary metrics over the 15 in-domain datasets (20 seeds), from results/bench_*.json:
(1) Critical-difference / average-rank diagram (Friedman test + Nemenyi post-hoc) -- the standard
    multi-dataset comparison: rank the 6 methods per dataset by regret, average, draw the CD bar.
(2) Pareto frontier: average normalized regret vs median per-step time -- the regret/cost trade-off."""
import json, glob
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "../figures"
MODELS = ["LinUCB", "kNN-UCB", "Naive", "ModelSelect", "SquareCB", "RLE-UCB"]
LABEL = {"kNN-UCB": "$k$-NN UCB", "Naive": "Naive", "ModelSelect": "ModSel",
         "SquareCB": "SqCB", "LinUCB": "LinUCB", "RLE-UCB": "RLE-UCB"}   # display labels (figures)
EXCLUDE = {"Mushroom"}            # near-deterministic 2-arm problem (degenerate bandit) -- dropped
res = {}
for f in glob.glob("results/bench_*.json"):
    if f.endswith("summary.json"):
        continue
    d = json.load(open(f))
    if d["dataset"] in EXCLUDE:
        continue
    res[d["dataset"]] = d
names = sorted(res)
N, k = len(names), len(MODELS)

# regret matrix (datasets x methods) and per-dataset ranks (1 = best/lowest regret)
R = np.array([[res[n]["models"][m]["best_regret"] for m in MODELS] for n in names])
ranks = np.array([stats.rankdata(row) for row in R])          # 1=lowest regret
avg_rank = ranks.mean(0)
# Friedman test across datasets
fr_stat, fr_p = stats.friedmanchisquare(*[R[:, j] for j in range(k)])
# Nemenyi critical difference (alpha=0.05): q for k=6 is 2.850
q_alpha = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949}[k]
CD = q_alpha * np.sqrt(k * (k + 1) / (6.0 * N))
print(f"Friedman chi2={fr_stat:.1f} p={fr_p:.2e}  (N={N} datasets, k={k} methods)")
print(f"Nemenyi CD (alpha=0.05) = {CD:.2f}")
order = np.argsort(avg_rank)
for j in order:
    print(f"  avg-rank {avg_rank[j]:.2f}  {MODELS[j]}")

# --- Figure 1: average-rank bar chart with Nemenyi significance vs. the best method ---
# A method is statistically tied with the top method (RLE-UCB) iff |rank gap| <= CD.
top = order[0]                                    # best-ranked method index
sig = {j: (avg_rank[j] - avg_rank[top] > CD) for j in range(k)}   # True => sig. worse than top
fig, ax = plt.subplots(figsize=(6.2, 3.2))
ys = np.arange(k)
for yi, j in enumerate(order):                    # best at top
    col = "#27a" if MODELS[j] == "RLE-UCB" else ("#bbb" if not sig[j] else "#d88")
    ax.barh(k - 1 - yi, avg_rank[j], color=col, edgecolor="k", linewidth=0.6, height=0.66)
    tag = "" if j == top else ("  (n.s.)" if not sig[j] else "  *")
    ax.text(avg_rank[j] + 0.07, k - 1 - yi, f"{avg_rank[j]:.2f}{tag}", va="center", fontsize=8)
ax.set_yticks(range(k)); ax.set_yticklabels([LABEL[MODELS[j]] for j in order[::-1]], fontsize=9)
for lab in ax.get_yticklabels():
    if lab.get_text() == "RLE-UCB":
        lab.set_fontweight("bold")
ax.set_xlabel(f"average rank over {N} datasets (1 = best regret)")
ax.set_xlim(0, k + 0.6)
# significance threshold: methods with rank beyond (best + CD) are sig. worse than RLE-UCB
thr = avg_rank[top] + CD
ax.axvline(avg_rank[top], color="#27a", ls=":", lw=1)
ax.axvline(thr, color="k", ls="--", lw=1)
ax.text(thr + 0.04, 0.15, f"RLE-UCB $+$ CD\n($={thr:.2f}$; beyond $=$ sig.)", fontsize=7.2, va="center")
ax.set_title(f"Friedman $p={fr_p:.1e}$ (Nemenyi CD $=$ {CD:.2f});  $*$ $=$ sig. worse than RLE-UCB", fontsize=8.5)
plt.tight_layout(); plt.savefig(f"{OUT}/rank_cd.png", dpi=150); plt.close()
print(f"[saved {OUT}/rank_cd.png]  sig-worse-than-top: " +
      ", ".join(MODELS[j] for j in range(k) if sig[j]))

# normalized-regret summary (kept for the prose / json; the regret-cost Pareto FIGURE is removed)
norm = R / R.min(1, keepdims=True)                # per dataset: best method = 1.0
avg_norm = norm.mean(0)
for j in np.argsort(avg_norm):
    print(f"  norm-regret {avg_norm[j]:.3f}  {MODELS[j]}")
json.dump({"avg_rank": dict(zip(MODELS, avg_rank.tolist())), "CD": float(CD),
           "friedman_p": float(fr_p), "avg_norm_regret": dict(zip(MODELS, avg_norm.tolist())),
           "n_datasets": N, "datasets": names},
          open("results/rank_pareto.json", "w"), indent=2)
