# RLE-UCB: Residualized Local Estimation with Upper Confidence Bounds

Code and figures for the contextual-bandit method **RLE-UCB** (Residualized Local
Estimation with Upper Confidence Bounds). RLE-UCB pairs a ridge estimate of the global
linear trend with a **residualized** `k`-nearest-neighbour estimate of the nonlinear
deviation `delta_a = m_a - mu_a^T x`, learned in the residual's lower intrinsic dimension
when it is detectable, and adds an uncertainty-adaptive exploration bonus. It targets
*misspecified* linear contextual bandits, where the reward is mostly linear with a
structured nonlinear correction that neither a global linear model nor a purely local
estimator handles well alone.

## Repository layout

```
RLE-UCB/
├── src/                  all experiment code (run scripts from this directory)
│   ├── bandits.py        RLE-UCB and the baseline algorithms + the run-episode/run-grid harness
│   ├── datasets10.py     OpenML classification-as-bandit loader (fetched on first use, then cached)
│   ├── zhang_data.py     the bandit stream (StreamEnv) and protocol loaders
│   ├── run_synthetic.py  the synthetic environments (hybrid / linear / nonlinear)
│   ├── synthetic_final.py  synthetic separation triangle (Env 1–4) + figures
│   ├── bench_one.py      in-domain benchmark on one real dataset
│   ├── bench_aggregate.py  aggregate the per-dataset results into the main table
│   ├── ablation_resid.py controlled residualization ablation + fixed-default check
│   ├── rank_pareto.py    average-rank / critical-difference diagram
│   ├── regret_curves.py  cumulative-regret trajectories
│   ├── efficiency_frontier.py  per-step time vs horizon
│   └── no_tuning_fig.py  exploration-sensitivity (tuning-free) study
├── figures/              the figures used in the paper (PNG)
├── requirements.txt
└── README.md
```

## Algorithms (`src/bandits.py`, `ALGOS` registry)

| Key             | Method |
|-----------------|--------|
| `RLE-UCB`       | **RLE-UCB** (this work): ridge linear trend + residualized `k`-NN on the recovered residual-active subspace + uncertainty-adaptive exploration |
| `LinUCB`        | LinUCB (Li et al. 2010; Chu et al. 2011) |
| `kNN-UCB-Reeve` | `k`-NN UCB (Reeve, Mellor & Brown 2018) |
| `NaiveReeve`    | naive Lin + `k`-NN hybrid (no residualization) |
| `ModelSelect`   | UCB meta-selector over `{LinUCB, k-NN}` (a basic bandit model-selection baseline) |
| `SquareCB-lin`  | SquareCB (Foster & Rakhlin 2020) with a linear/ridge oracle |

## Reproduce

```bash
pip install -r requirements.txt
cd src

# Synthetic separation triangle (Env 1–4) and its figures:
python3 synthetic_final.py

# In-domain real benchmarks: run one dataset at a time (writes src/results/bench_<Name>.json),
# then aggregate into the main results table:
python3 bench_one.py Vehicle 3000 20      # <Name> [T] [nseed]
#   ... repeat for each dataset (Adult, Magic, Shuttle, Letter, Pendigits, Optdigits,
#       Satimage, Segment, Vehicle, Spambase, Waveform, Phoneme, EEG, MNIST) ...
python3 bench_aggregate.py

# Controlled residualization ablation (+ fixed-default-c check):
python3 ablation_resid.py 20              # [nseed] [Name,Name,...]

# Figures (each writes to ../figures/):
python3 rank_pareto.py                    # -> rank_cd.png
python3 regret_curves.py                  # -> regret_curves.png
python3 efficiency_frontier.py            # -> efficiency_frontier.png
python3 no_tuning_fig.py                  # -> no_tuning.png
```

The real-benchmark datasets are pulled from **OpenML** by ID (see `bench_one.py`); the first
run downloads and caches them. Result files are written under `src/results/` at run time;
the table/figure scripts read them once generated. All runs use 20 random seeds and report
the best of a three-value exploration grid per method, matching the paper.

## Paper item → script

| Paper item                                   | Script |
|----------------------------------------------|--------|
| Synthetic table (Env 1–4) + figures          | `synthetic_final.py` (`env1_hybrid`, `env2_linucb_dominates`, `env3_knn_dominates`) |
| In-domain table (14 datasets)                | `bench_one.py` (per dataset) + `bench_aggregate.py` |
| Residualization ablation + fixed-`c` check   | `ablation_resid.py` |
| Average-rank / critical-difference diagram   | `rank_pareto.py` → `rank_cd.png` |
| Regret trajectories                          | `regret_curves.py` → `regret_curves.png` |
| Per-step efficiency frontier                 | `efficiency_frontier.py` → `efficiency_frontier.png` |
| Tuning-free exploration study                | `no_tuning_fig.py` → `no_tuning.png` |

## Requirements

`numpy`, `scipy`, `scikit-learn`, `pandas`, `matplotlib` (see `requirements.txt`).
