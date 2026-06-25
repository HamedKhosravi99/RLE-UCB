"""10 real classification-as-bandit datasets, one IDENTICAL pipeline for every model:
ordinal-encode categoricals -> standardize -> L2-normalize rows. Disjoint per-arm
ridge (== Zhang kd encoding), reward 1 iff arm==label, regret = #mistakes.
No per-dataset or per-model engineering: same preprocessing, same protocol everywhere."""
import os, numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

DATA = os.path.join(os.path.dirname(__file__), "data"); os.makedirs(DATA, exist_ok=True)

# (name, openml data_id, cap N)  -- standard bandit-classification datasets
SPECS = [
    ("Mushroom",  24,   None),
    ("Adult",     1590, 30000),
    ("Magic",     1120, None),
    ("Shuttle",   40685, 30000),
    ("Letter",    6,    None),
    ("Pendigits", 32,   None),
    ("Optdigits", 28,   None),
    ("Satimage",  182,  None),
    ("Segment",   36,   None),
    ("MNIST",     554,  20000),
]


def load(name, data_id, cap=None):
    cache = os.path.join(DATA, f"ds_{name}.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True); X, y, A = z["X"], z["y"], int(z["A"]); return X, y, A
    from sklearn.datasets import fetch_openml
    d = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    df = d.frame.copy()
    target = d.target.name if hasattr(d.target, "name") and d.target.name else df.columns[-1]
    y_raw = df[target].astype(str).values
    Xdf = df.drop(columns=[target])
    cols = []
    for c in Xdf.columns:
        col = Xdf[c]
        if col.dtype == object or str(col.dtype).startswith("category"):
            cols.append(LabelEncoder().fit_transform(col.astype(str)).astype(float))
        else:
            cols.append(col.astype(float).fillna(col.astype(float).median()).values)
    X = np.column_stack(cols)
    X = StandardScaler().fit_transform(X)
    nrm = np.linalg.norm(X, axis=1, keepdims=True); nrm[nrm == 0] = 1.0; X = X / nrm
    y = LabelEncoder().fit_transform(y_raw)
    if cap and len(X) > cap:
        rng = np.random.default_rng(0); idx = rng.choice(len(X), cap, replace=False); X, y = X[idx], y[idx]
    A = int(len(np.unique(y)))
    np.savez(cache, X=X, y=y, A=A)
    return X, y, A


if __name__ == "__main__":
    for name, did, cap in SPECS:
        try:
            X, y, A = load(name, did, cap)
            print(f"OK   {name:10s} N={len(X):6d}  d={X.shape[1]:3d}  classes={A}", flush=True)
        except Exception as e:
            print(f"FAIL {name:10s} {type(e).__name__}: {e}", flush=True)
