"""Zhang et al. (2021)-faithful dataset loaders + bandit stream.

Key fidelity choices (from Zhang NeurTS Sec 5.1 / A.1):
  - categoricals ORDINAL/integer-encoded (NOT one-hot)  -> matches their Input Dims
    (Adult 15, Magic 12, Mushroom 23, MNIST 784: ordinal feats + 1 bias term)
  - standardize, then L2-normalize each row to unit norm
  - reward 1 if pulled arm == label else 0; regret = total mistakes
  - T = 10000 (Mushroom 8124); single reshuffled pass per run (no replacement)
"""
import os, urllib.request, socket
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

socket.setdefaulttimeout(60)
DATA = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA, exist_ok=True)
UCI = "https://archive.ics.uci.edu/ml/machine-learning-databases"
ROUNDS = {"Adult": 10000, "Magic": 10000, "Mushroom": 8124, "MNIST": 10000}


def _get(url, fname):
    p = os.path.join(DATA, fname)
    if not os.path.exists(p):
        urllib.request.urlretrieve(url, p)
    return p


STANDARDIZE = False  # Zhang recipe: raw ordinal codes + bias, L2-normalize rows (validated on Mushroom)

def _finalize(Xnum, Xcat, y, add_bias=True):
    """Ordinal-encode categoricals, (optionally standardize), +bias, L2-normalize rows."""
    cols = []
    if Xnum is not None and Xnum.shape[1]:
        a = Xnum.astype(float)
        cols.append(StandardScaler().fit_transform(a) if STANDARDIZE else a)
    if Xcat is not None and Xcat.shape[1]:
        enc = np.column_stack([LabelEncoder().fit_transform(Xcat[:, j])
                               for j in range(Xcat.shape[1])]).astype(float)
        cols.append(StandardScaler().fit_transform(enc) if STANDARDIZE else enc)
    X = np.column_stack(cols)
    if add_bias:
        X = np.column_stack([X, np.ones(len(X))])
    nrm = np.linalg.norm(X, axis=1, keepdims=True); nrm[nrm == 0] = 1.0
    X = X / nrm
    classes = sorted(np.unique(y))
    y = np.array([list(classes).index(v) for v in y])
    return X, y, len(classes)


def load_adult():
    p = _get(f"{UCI}/adult/adult.data", "adult.data")
    cols = ["age","workclass","fnlwgt","education","education-num","marital","occupation",
            "relationship","race","sex","capgain","caploss","hours","country","label"]
    df = pd.read_csv(p, header=None, names=cols, na_values=" ?", skipinitialspace=True).dropna()
    y = df["label"].values
    num = ["age","fnlwgt","education-num","capgain","caploss","hours"]
    cat = ["workclass","education","marital","occupation","relationship","race","sex","country"]
    return _finalize(df[num].values, df[cat].values.astype(str), y)   # 6+8+bias = 15


def load_magic():
    p = _get(f"{UCI}/magic/magic04.data", "magic04.data")
    df = pd.read_csv(p, header=None)
    y = df.iloc[:, -1].values
    return _finalize(df.iloc[:, :-1].values, None, y)                 # 10 + bias = 11 (~12)


def load_mushroom():
    p = _get(f"{UCI}/mushroom/agaricus-lepiota.data", "mushroom.data")
    df = pd.read_csv(p, header=None)                                  # keep '?' as a category
    y = df.iloc[:, 0].values
    return _finalize(None, df.iloc[:, 1:].values.astype(str), y)      # N=8124, 22 + bias = 23


def load_mnist(n=None):
    from sklearn.datasets import fetch_openml
    cache = os.path.join(DATA, "mnist_raw.npz")
    if os.path.exists(cache):
        z = np.load(cache); X, y = z["X"], z["y"]
    else:
        mn = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
        X, y = mn.data.astype(float) / 255.0, mn.target.astype(int)
        np.savez(cache, X=X, y=y)
    nrm = np.linalg.norm(X, axis=1, keepdims=True); nrm[nrm == 0] = 1.0
    X = X / nrm
    return X, y, 10                                                   # 784, no bias


LOADERS = {"Adult": load_adult, "Magic": load_magic,
           "Mushroom": load_mushroom, "MNIST": load_mnist}


class StreamEnv:
    """Single reshuffled pass (no replacement); cycles if T>N."""
    def __init__(self, X, y, A, T, seed):
        self.X, self.y, self.A, self.d = X, y, A, X.shape[1]
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(X))
        if T > len(X):
            reps = int(np.ceil(T / len(X)))
            perm = np.concatenate([rng.permutation(len(X)) for _ in range(reps)])
        self.idx = perm[:T]; self.T = T
    def step(self, t, rng):
        i = self.idx[t]; x = self.X[i]; lab = self.y[i]
        rew = np.zeros(self.A); rew[lab] = 1.0
        return [x] * self.A, rew, rew
