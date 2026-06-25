"""Synthetic Env 1-3 (the separation 'triangle') for the residualized RLE-UCB."""
import sys, time
import numpy as np
from bandits import run_grid


class EnvHybrid:      # Env1: d=5, residual on (x0,x1), shared linear mu + smooth bilinear residual
    d, A = 5, 2
    mu = np.array([2.0, 1.6, 1.2, 0.8, 0.4])
    eps, sigma = 0.6, 0.1
    def step(self, t, rng):
        x = rng.uniform(-1, 1, self.d)
        res = x[0] * x[1]                          # smooth, Lipschitz, orthogonal to linear
        delta = np.array([+self.eps * res, -self.eps * res])
        exp = self.mu @ x + delta
        rew = exp + rng.normal(0, self.sigma, self.A)
        return [x, x], rew, exp


class EnvManifold:    # Env4: contexts on a 2-D linear subspace of R^5 (intrinsic dim d'=2)
    d, A = 5, 2
    mu = np.array([2.0, 1.6, 1.2, 0.8, 0.4])
    eps, sigma = 0.6, 0.1
    def __init__(self, seed=0):
        r = np.random.default_rng(7)
        M = r.standard_normal((5, 2)); M /= np.linalg.norm(M, axis=0, keepdims=True)
        self.M = M
    def step(self, t, rng):
        z = rng.uniform(-1, 1, 2)
        x = self.M @ z                              # context lives on a 2-D subspace
        res = z[0] * z[1]
        delta = np.array([+self.eps * res, -self.eps * res])
        exp = self.mu @ x + delta
        rew = exp + rng.normal(0, self.sigma, self.A)
        return [x, x], rew, exp


class EnvLinear:      # Env2: d=10, pure linear, well-separated arms
    d, A = 10, 2
    sigma = 0.1
    def __init__(self, seed=0):
        r = np.random.default_rng(1234)
        self.W = np.stack([r.uniform(-1, 1, self.d), r.uniform(-1, 1, self.d)])
        self.W[1] = -self.W[0]                     # well-separated
    def step(self, t, rng):
        x = rng.uniform(-1, 1, self.d)
        exp = self.W @ x
        rew = exp + rng.normal(0, self.sigma, self.A)
        return [x, x], rew, exp


class EnvNonlinear:   # Env3: d=3, pure nonlinear (mu=0), smooth bilinear residual
    d, A = 3, 2
    eps, sigma = 0.6, 0.1
    def step(self, t, rng):
        x = rng.uniform(-1, 1, self.d)
        res = x[0] * x[1]
        delta = np.array([+self.eps * res, -self.eps * res])
        exp = 1.0 + delta
        rew = exp + rng.normal(0, self.sigma, self.A)
        return [x, x], rew, exp


if __name__ == "__main__":
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 15000
    nseed = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    seeds = list(range(nseed))
    envs = {"Env1 Hybrid (d=5,d'=2)": EnvHybrid(),
            "Env2 Linear (d=10)": EnvLinear(),
            "Env3 Nonlinear (d=3)": EnvNonlinear()}
    algos = ["LinUCB", "kNN-UCB", "RawHybrid", "RLE-UCB-res"]
    t0 = time.time()
    res = run_grid(envs, algos, T, seeds)
    print(f"\n=== Synthetic cumulative regret (T={T}, {nseed} seeds), mean +/- std ===")
    for ename, row in res.items():
        print(f"\n{ename}")
        for aname in algos:
            m, s, _ = row[aname]
            print(f"   {aname:24s} {m:9.1f} +/- {s:6.1f}")
    print(f"\n[elapsed {time.time()-t0:.1f}s]")
