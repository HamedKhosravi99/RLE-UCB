"""
RLE-UCB and baseline contextual-bandit algorithms.

Algorithms (disjoint per-arm ridge, shared d-dim context):
  - LinUCB
  - kNN-UCB            (fixed k, raw-reward k-NN + UCB bonus)
  - RawHybrid          ((Lin+kNN)-UCB on RAW rewards = the raw, non-residualized hybrid)
  - RLE_UCB_res       (Path B: residualized k-NN + reward-modulated exploration)

All algorithms share the same simulation loop and regret accounting so comparisons
are apples-to-apples under one harness.
"""
import numpy as np


# --------------------------------------------------------------------------- #
#  Per-arm ridge state                                                        #
# --------------------------------------------------------------------------- #
class ArmState:
    def __init__(self, d, lam):
        self.d = d
        self.A = lam * np.eye(d)          # Sigma_a
        self.Ainv = np.eye(d) / lam
        self.b = np.zeros(d)
        self._N = 0
        self._cap = 64
        self.Xbuf = np.empty((self._cap, d), dtype=np.float64)   # preallocated history
        self.Ybuf = np.empty(self._cap, dtype=np.float64)
        self._ysum = 0.0; self._ysq = 0.0      # running stats for mean/var
        self._mu = np.zeros(d)
        self._dirty = False

    @property
    def N(self):
        return self._N

    @property
    def Xhist(self):
        return self.Xbuf[:self._N]
    @property
    def Yhist(self):
        return self.Ybuf[:self._N]

    def ymean(self):
        return self._ysum / self._N if self._N else 0.0
    def yvar(self):
        if self._N < 2:
            return 0.0
        m = self._ysum / self._N
        return max(self._ysq / self._N - m * m, 0.0)

    def mu(self):
        if self._dirty:
            self._mu = self.Ainv @ self.b
            self._dirty = False
        return self._mu

    def width(self, x):
        return np.sqrt(max(x @ self.Ainv @ x, 0.0))

    def update(self, x, y):
        self.A += np.outer(x, x)
        Ainv_x = self.Ainv @ x
        denom = 1.0 + x @ Ainv_x
        self.Ainv -= np.outer(Ainv_x, Ainv_x) / denom
        self.b += y * x
        if self._N == self._cap:                  # grow buffer (amortized O(1))
            self._cap *= 2
            self.Xbuf = np.resize(self.Xbuf, (self._cap, self.d))
            self.Ybuf = np.resize(self.Ybuf, self._cap)
        self.Xbuf[self._N] = x
        self.Ybuf[self._N] = y
        self._N += 1
        self._ysum += y; self._ysq += y * y
        self._dirty = True


def _knn_idx(Xhist_arr, x, k):
    d2 = np.einsum("ij,ij->i", Xhist_arr - x, Xhist_arr - x)
    if k >= len(d2):
        return np.arange(len(d2))
    return np.argpartition(d2, k)[:k]


# --------------------------------------------------------------------------- #
#  Algorithms                                                                 #
# --------------------------------------------------------------------------- #
class LinUCB:
    name = "LinUCB"
    def __init__(self, d, A, lam=1.0, alpha=1.0, **kw):
        self.arms = [ArmState(d, lam) for _ in range(A)]
        self.alpha = alpha
    def select(self, ctxs):
        s = []
        for a, st in enumerate(self.arms):
            x = ctxs[a]
            s.append(st.mu() @ x + self.alpha * st.width(x))
        return int(np.argmax(s))
    def update(self, a, ctxs, y):
        self.arms[a].update(ctxs[a], y)


class kNNUCB:
    name = "kNN-UCB"
    def __init__(self, d, A, lam=1.0, alpha=1.0, k=5, **kw):
        self.arms = [ArmState(d, lam) for _ in range(A)]
        self.alpha = alpha; self.k = k; self.t = 0
    def select(self, ctxs):
        self.t += 1
        s = []
        for a, st in enumerate(self.arms):
            x = ctxs[a]
            if st.N >= self.k:
                Xa = st.Xhist
                idx = _knn_idx(Xa, x, self.k)
                ya = st.Yhist[idx]
                est = ya.mean()
                bonus = self.alpha * np.sqrt(2.0 * np.log(self.t + 1) / self.k)
            else:
                est = 0.0
                bonus = self.alpha * 2.0           # optimistic until enough data
            s.append(est + bonus)
        return int(np.argmax(s))
    def update(self, a, ctxs, y):
        self.arms[a].update(ctxs[a], y)


class RawHybrid:
    """(Lin+kNN)-UCB on RAW rewards = the raw, non-residualized hybrid (fixed alpha)."""
    name = "RawHybrid (Lin+kNN, raw)"
    def __init__(self, d, A, lam=1.0, alpha=1.0, tmin=2, tmax=10, **kw):
        self.arms = [ArmState(d, lam) for _ in range(A)]
        self.alpha = alpha; self.tmin = tmin; self.tmax = tmax
    def _k(self, st):
        v = st.yvar()
        return int(np.ceil(self.tmin + (self.tmax - self.tmin) * min(v, 1.0)))
    def select(self, ctxs):
        s = []
        for a, st in enumerate(self.arms):
            x = ctxs[a]
            lin = st.mu() @ x
            k = self._k(st)
            if st.N >= k:
                Xa = st.Xhist
                idx = _knn_idx(Xa, x, k)
                f = st.Yhist[idx].mean()      # RAW rewards
            else:
                f = 0.0
            s.append(lin + f + self.alpha * st.width(x))
        return int(np.argmax(s))
    def update(self, a, ctxs, y):
        self.arms[a].update(ctxs[a], y)


class RLE_UCB_res:
    """Path B: residualized k-NN + reward-modulated exploration."""
    name = "RLE-UCB (residualized)"
    def __init__(self, d, A, lam=1.0, alpha0=1.0, kappa=0.5, tmin=2, tmax=10, **kw):
        self.arms = [ArmState(d, lam) for _ in range(A)]
        self.alpha0 = alpha0; self.kappa = kappa
        self.tmin = tmin; self.tmax = tmax
        self.g_sum = 0.0; self.g_n = 0
    def _k(self, st):
        v = st.yvar()
        return int(np.ceil(self.tmin + (self.tmax - self.tmin) * min(v, 1.0)))
    def select(self, ctxs):
        g = self.g_sum / self.g_n if self.g_n > 0 else 0.0
        s = []
        for a, st in enumerate(self.arms):
            x = ctxs[a]
            mu = st.mu()
            lin = mu @ x
            k = self._k(st)
            if st.N >= k:
                Xa = st.Xhist
                idx = _knn_idx(Xa, x, k)
                ya = st.Yhist[idx]
                xa = Xa[idx]
                resid = ya - xa @ mu                       # RESIDUALIZED target  <-- Path B
                f = resid.mean()
            else:
                f = 0.0
            n_a = st.ymean()
            alpha_a = self.alpha0 / (st.N + 1) * abs(self.kappa * g + (1 - self.kappa) * n_a)
            s.append(lin + f + alpha_a * st.width(x))
        return int(np.argmax(s))
    def update(self, a, ctxs, y):
        self.arms[a].update(ctxs[a], y)
        self.g_sum += y; self.g_n += 1


class OrigRLEUCB:
    """Raw, adaptive-attention variant (kept for ablation):
       UCB = (mu^T x + RAW adaptive-kNN) + alpha_att * width,
       k = round(theta_min + (theta_max-theta_min)*Var(z)),  RAW k-NN (no residualization),
       alpha = alpha0/(N+1) * (kappa*g + (1-kappa)*n_a)   [reward-modulated attention]."""
    name = "RLE-UCB (raw + attention)"
    def __init__(self, d, A, lam=1.0, alpha0=1.0, kappa=0.5, tmin=2, tmax=10, **kw):
        self.arms = [ArmState(d, lam) for _ in range(A)]
        self.alpha0 = alpha0; self.kappa = kappa
        self.tmin = tmin; self.tmax = tmax
        self.g_sum = 0.0; self.g_n = 0
    def _k(self, st):
        return int(round(self.tmin + (self.tmax - self.tmin) * st.yvar()))
    def select(self, ctxs):
        g = self.g_sum / self.g_n if self.g_n > 0 else 0.0
        s = []
        for a, st in enumerate(self.arms):
            x = ctxs[a]
            lin = st.mu() @ x
            k = max(1, self._k(st))
            if st.N >= 1:
                kk = min(k, st.N)
                idx = _knn_idx(st.Xhist, x, kk)
                f = st.Yhist[idx].mean()                 # RAW rewards (original)
            else:
                f = 0.0
            n_a = st.ymean()
            alpha_a = self.alpha0 / (st.N + 1) * (self.kappa * g + (1 - self.kappa) * n_a)
            s.append(lin + f + alpha_a * st.width(x))
        return int(np.argmax(s))
    def update(self, a, ctxs, y):
        self.arms[a].update(ctxs[a], y)
        self.g_sum += y; self.g_n += 1


class RLE_UCB_final:
    """FINAL method: residualized k-NN + uncertainty-adaptive exploration bonus
       bonus = c*( sqrt(log(t+2))*||x||_{Sigma^-1} + u_k + sigma_hat/sqrt(k) )."""
    name = "RLE-UCB (final)"
    def __init__(self, d, A, lam=1.0, c=0.1, tmin=2, tmax=10, **kw):
        self.arms = [ArmState(d, lam) for _ in range(A)]
        self.c = c; self.tmin = tmin; self.tmax = tmax; self.t = 0
    def _k(self, st):
        return int(np.ceil(self.tmin + (self.tmax - self.tmin) * min(st.yvar(), 1.0)))
    def select(self, ctxs):
        self.t += 1
        s = []
        for a, st in enumerate(self.arms):
            x = ctxs[a]; mu = st.mu(); lin = mu @ x
            k = self._k(st); u_k = 0.0; f = 0.0
            if st.N >= k:
                Xa = st.Xhist; idx = _knn_idx(Xa, x, k)
                d2 = np.einsum("ij,ij->i", Xa[idx] - x, Xa[idx] - x)
                f = (st.Yhist[idx] - Xa[idx] @ mu).mean()      # residualized
                u_k = float(np.sqrt(d2.max()))                 # empirical kNN radius
            sig = np.sqrt(max(st.yvar(), 1e-6))
            bonus = self.c * (np.sqrt(np.log(self.t + 1)) * st.width(x) + u_k + sig / np.sqrt(max(k, 1)))
            s.append(lin + f + bonus)
        return int(np.argmax(s))
    def update(self, a, ctxs, y):
        self.arms[a].update(ctxs[a], y)


class kNNUCB_Reeve:
    """Faithful k-NN UCB (Reeve, Mellor & Brown, ALT 2018): the index searches over
    ALL k each round -- index_a = inf_k ( mu_k + c*sqrt(log t / k) + r_k ), where r_k is
    the distance to the k-th neighbor. Select argmax_a index_a. O(N log N)/arm/step."""
    name = "kNN-UCB (Reeve)"
    def __init__(self, d, A, lam=1.0, c=1.0, **kw):
        self.arms = [ArmState(d, lam) for _ in range(A)]; self.c = c; self.t = 0
    def _index(self, st, x):
        n = st.N
        if n == 0:
            return np.inf                                   # explore unplayed arms first
        d2 = np.einsum("ij,ij->i", st.Xhist - x, st.Xhist - x)
        order = np.argsort(d2)
        r = np.sqrt(d2[order])                              # sorted distances; r[k-1]=r_k
        ymean = np.cumsum(st.Yhist[order]) / np.arange(1, n + 1)   # mu_k for k=1..n
        width = self.c * np.sqrt(np.log(self.t + 1) / np.arange(1, n + 1))
        return float((ymean + width + r).min())             # inf over k
    def select(self, ctxs):
        self.t += 1
        return int(np.argmax([self._index(self.arms[a], ctxs[a]) for a in range(len(self.arms))]))
    def update(self, a, ctxs, y):
        self.arms[a].update(ctxs[a], y)


class NaiveReeve:
    """Naive Lin + k-NN-UCB hybrid: adds the ridge linear prediction to the faithful
    Reeve k-NN-UCB index (raw rewards). Double-counts the linear part on purpose -- this
    is the naive baseline the residualized design is meant to improve on."""
    name = "Naive (Lin+kNN-UCB)"
    def __init__(self, d, A, lam=1.0, c=1.0, **kw):
        self.arms = [ArmState(d, lam) for _ in range(A)]; self.c = c; self.t = 0
    def select(self, ctxs):
        self.t += 1; s = []
        for a, st in enumerate(self.arms):
            x = ctxs[a]; lin = st.mu() @ x; n = st.N
            if n == 0:
                s.append(np.inf); continue
            d2 = np.einsum("ij,ij->i", st.Xhist - x, st.Xhist - x); order = np.argsort(d2)
            r = np.sqrt(d2[order]); ymean = np.cumsum(st.Yhist[order]) / np.arange(1, n + 1)
            width = self.c * np.sqrt(np.log(self.t + 1) / np.arange(1, n + 1))
            s.append(lin + float((ymean + width + r).min()))
        return int(np.argmax(s))
    def update(self, a, ctxs, y):
        self.arms[a].update(ctxs[a], y)


class RLE_UCB:
    """RLE-UCB: ridge linear trend + residualized k-NN on a residual-relevant SUBSPACE
    metric + uncertainty-adaptive exploration with a per-arm 1/(N+1) decay.
      bonus = c/(N+1) * ( sqrt(log t)*||x||_Sig^-1 + u_k + sigma/sqrt(k) ).
    The residualized k-NN restricts its neighbour metric to the coordinates the residual
    actually depends on, recovered ONLINE from the residualized rewards Ytilde = Y - mu^T x
    via a relevance score that detects interactions (corr(Ytilde^2, x_j^2) fires for
    delta=x0*x1 even though corr(Ytilde,x_j)=0). When the residual is sparse it then faces
    the d'-dimensional packing of the residual-active subspace; when it is dense/rotated
    the score keeps all coordinates, recovering the plain full-metric k-NN (does no harm).
    The 1/(N+1) factor makes c essentially tuning-free on always-update protocols.
    Set subspace=False for the full-metric ablation."""
    name = "RLE-UCB"
    def __init__(self, d, A, lam=1.0, c=0.1, tmin=2, tmax=10, subspace=True,
                 recompute=50, min_N_sel=400, fixed_explore=False, no_knn=False,
                 eps_explore=0.0, seed=0, W=np.inf, shared_ridge=False,
                 residualize=True, **kw):
        self.arms = [ArmState(d, lam) for _ in range(A)]
        # shared_ridge=True -> ONE pooled ridge for the COMMON trend mu* (updated every round on the
        # selected arm's reward), justified when arms share the linear trend (Assumption A2 of the
        # decoupling proof); the linear term then cancels in the arm comparison -> exact decoupling.
        self.shared_ridge = shared_ridge
        self.shared = ArmState(d, lam) if shared_ridge else None
        self.c = c; self.tmin = tmin; self.tmax = tmax; self.t = 0
        # W = norm bound for the Algorithm-1 ridge projection (clip ||mu_hat||<=W). Default inf
        # (= no projection, as in all reported runs: max||mu_hat||<=3.5 across datasets, so any
        # finite W>=4 leaves every number unchanged; the projection is a theoretical safeguard).
        self.W = W
        # residualize=True (default): k-NN averages the RESIDUAL Y - mu^T x (RLE-UCB).
        # residualize=False: k-NN averages the RAW reward Y (the naive Lin+kNN hybrid, but with
        # RLE-UCB's IDENTICAL bonus, k-rule, and subspace) -- the controlled residualization ablation.
        self.residualize = residualize
        self.subspace = subspace; self.recompute = recompute; self.min_N_sel = min_N_sel
        self.fixed_explore = fixed_explore   # True -> plain c*||x||_Sig (LinUCB-style fixed bonus)
        self.no_knn = no_knn                 # True -> linear only (ablate the k-NN component)
        # eps_explore>0 -> forced exploration: pull a uniform-random arm w.p. eps_t = eps_explore*(t)^{-1/3},
        # supplying each arm i.i.d. contexts decoupled from the UCB rule (Assumption: i.i.d. decoupling).
        self.eps_explore = eps_explore; self.rng = np.random.default_rng(seed)
        self._sel = [None] * A; self._last = [0] * A
    def _k(self, st):
        return int(np.ceil(self.tmin + (self.tmax - self.tmin) * min(st.yvar(), 1.0)))
    def _relevance(self, st, mu):
        X = st.Xhist; rt = st.Yhist - X @ mu           # residualized rewards
        if X.shape[0] > 2000:                          # cap cost: relevance is a population quantity
            X = X[-2000:]; rt = rt[-2000:]
        def corr2(y, Z):                               # squared corr of y with each column of Z
            yc = y - y.mean(); Zc = Z - Z.mean(0)
            num = yc @ Zc
            den = np.sqrt((yc @ yc) * np.einsum("ij,ij->j", Zc, Zc) + 1e-18)
            return (num / den) ** 2
        return corr2(rt, X) + corr2(rt * rt, X * X)    # mean-dependence + interaction/variance-dependence
    def _mask(self, a, st, mu):
        if (not self.subspace) or st.N < self.min_N_sel:
            return None                                # off, or too little data -> full metric
        if self._sel[a] is None or st.N - self._last[a] >= self.recompute:
            sc = self._relevance(st, mu)
            # Restrict ONLY when the residual is CLEARLY sparse: keep coordinates above an
            # absolute noise floor (~1/N under the null), and restrict only if that set is
            # small (<= half) and captures >=90% of total relevance; else keep all (no harm).
            floor = max(0.02, 10.0 / max(st.N, 1))
            relevant = sc >= floor; tot = sc.sum()
            if relevant.sum() and relevant.sum() <= 0.5 * st.d and sc[relevant].sum() >= 0.9 * tot:
                mask = relevant
            else:
                mask = np.ones(st.d, dtype=bool)
            self._sel[a] = mask; self._last[a] = st.N
        return self._sel[a]
    def select(self, ctxs):
        self.t += 1
        if self.eps_explore > 0.0:                          # forced exploration (decoupling device)
            eps_t = min(1.0, self.eps_explore * self.t ** (-1.0 / 3.0))
            if self.rng.random() < eps_t:
                return int(self.rng.integers(len(self.arms)))
        s = []
        for a, st in enumerate(self.arms):
            x = ctxs[a]
            rid = self.shared if self.shared_ridge else st   # shared (pooled) or per-arm ridge
            mu = rid.mu()
            if np.isfinite(self.W):                       # Algorithm-1 projection onto ||.||<=W ball
                nmu = np.linalg.norm(mu)
                if nmu > self.W:
                    mu = mu * (self.W / nmu)
            lin = mu @ x
            k = max(1, self._k(st)); u_k = 0.0; f = 0.0
            if (not self.no_knn) and st.N >= k:
                Xa = st.Xhist; mask = self._mask(a, st, mu)
                if mask is None or mask.all():            # nothing pruned -> full-metric k-NN
                    idx = _knn_idx(Xa, x, k)
                    d2 = np.einsum("ij,ij->i", Xa[idx] - x, Xa[idx] - x)
                else:                                     # restrict the metric to the relevant subspace
                    diff = (Xa - x)[:, mask]
                    d2f = np.einsum("ij,ij->i", diff, diff)
                    kk = min(k, len(d2f))
                    idx = np.argpartition(d2f, kk - 1)[:kk] if kk < len(d2f) else np.arange(len(d2f))
                    d2 = d2f[idx]
                f = (st.Yhist[idx] - Xa[idx] @ mu).mean() if self.residualize \
                    else st.Yhist[idx].mean()        # residualized vs raw k-NN (ablation knob)
                u_k = float(np.sqrt(d2.max()))
            if self.fixed_explore:
                bonus = self.c * rid.width(x)                      # fixed LinUCB-style exploration
            else:
                sig = np.sqrt(max(st.yvar(), 1e-6))
                base = np.sqrt(np.log(self.t + 1)) * rid.width(x) + u_k + sig / np.sqrt(max(k, 1))
                bonus = self.c / (st.N + 1) * base                 # per-arm 1/(N_t^a+1) decay
            s.append(lin + f + bonus)
        return int(np.argmax(s))
    def update(self, a, ctxs, y):
        self.arms[a].update(ctxs[a], y)
        if self.shared_ridge:                 # pooled ridge sees every round's (context, selected reward)
            self.shared.update(ctxs[a], y)


RLE_UCB_Sub = RLE_UCB   # backward-compatible alias: the subspace metric is now part of RLE-UCB


class RLE_UCB_grow(RLE_UCB):
    """RLE-UCB with a GROWING neighborhood cap theta_max(t)=min(kcap, max(10, ceil(sqrt(N)))),
    so k can grow with the arm's history (drives the local variance to zero -> the
    rate-optimal variant), still chosen in O(1) from the variance (no search over k)."""
    name = "RLE-UCB (grow-k)"
    def __init__(self, d, A, lam=1.0, c=0.1, tmin=2, kcap=200, **kw):
        super().__init__(d, A, lam=lam, c=c, tmin=tmin, tmax=10)
        self.kcap = kcap
    def _k(self, st):
        tmax = min(self.kcap, max(10, int(np.ceil(np.sqrt(max(st.N, 1))))))
        return int(np.ceil(self.tmin + (tmax - self.tmin) * min(st.yvar(), 1.0)))


class ModelSelectUCB:
    """In-domain baseline: model selection over base learners {LinUCB, kNN-UCB}.
    A UCB meta-selector picks which base learner acts each round (scale-robust);
    only the acting learner updates (true bandit model-selection feedback).
    Answers the 'why not just let an algorithm pick linear-vs-local?' question."""
    name = "ModelSelect{Lin,kNN}"
    def __init__(self, d, A, lam=1.0, C=1.0, **kw):
        self.bases = [LinUCB(d, A, lam=lam, alpha=1.0), kNNUCB_Reeve(d, A, lam=lam, c=1.0)]
        self.sum = [0.0, 0.0]; self.cnt = [0, 0]; self.t = 0; self.C = C; self._b = 0
    def select(self, ctxs):
        self.t += 1
        if min(self.cnt) == 0:
            self._b = int(np.argmin(self.cnt))
        else:
            ucb = [self.sum[i] / self.cnt[i] + self.C * np.sqrt(np.log(self.t + 1) / self.cnt[i])
                   for i in range(2)]
            self._b = int(np.argmax(ucb))
        return self.bases[self._b].select(ctxs)
    def update(self, a, ctxs, y):
        b = self._b
        self.bases[b].update(a, ctxs, y)
        self.sum[b] += y; self.cnt[b] += 1


class SquareCBLinear:
    """In-domain baseline: SquareCB (Foster & Rakhlin 2020) with a LINEAR (ridge)
    regression oracle. Inverse-gap sampling on the oracle's predictions; inherits
    the linear misspecification, so it cannot escape the eps-mis gap on structured
    nonlinear rewards (that is the point of comparing to it)."""
    name = "SquareCB (linear oracle)"
    def __init__(self, d, A, lam=1.0, gamma=10.0, seed=0, **kw):
        self.arms = [ArmState(d, lam) for _ in range(A)]; self.A = A; self.gamma = gamma
        self.rng = np.random.default_rng(seed)
    def select(self, ctxs):
        yhat = np.array([self.arms[a].mu() @ ctxs[a] for a in range(self.A)])
        b = int(np.argmax(yhat))
        p = np.zeros(self.A)
        for a in range(self.A):
            if a != b:
                p[a] = 1.0 / (self.A + self.gamma * max(yhat[b] - yhat[a], 0.0))
        p[b] = max(1.0 - p.sum(), 0.0)
        p = np.clip(p, 1e-12, None); p = p / p.sum()
        return int(self.rng.choice(self.A, p=p))           # inverse-gap sampling
    def update(self, a, ctxs, y):
        self.arms[a].update(ctxs[a], y)


ALGOS = {
    "LinUCB": LinUCB,
    "kNN-UCB": kNNUCB,
    "RawHybrid": RawHybrid,
    "RLE-UCB-res": RLE_UCB_res,
    "RLE-UCB-orig": OrigRLEUCB,
    "RLE-UCB-final": RLE_UCB_final,
    "ModelSelect": ModelSelectUCB,
    "SquareCB-lin": SquareCBLinear,
    "kNN-UCB-Reeve": kNNUCB_Reeve,
    "NaiveReeve": NaiveReeve,
    "RLE-UCB": RLE_UCB,
    "RLE-UCB-Sub": RLE_UCB_Sub,
    "RLE-UCB-grow": RLE_UCB_grow,
}


# --------------------------------------------------------------------------- #
#  Simulation loop                                                            #
# --------------------------------------------------------------------------- #
def run_episode(AlgoCls, env, T, seed, algo_kwargs=None):
    """
    env.step(t, rng) -> (ctxs[A][d], reward_vector[A], expected_reward_vector[A])
      ctxs:    per-arm context (shared base context allowed)
      reward:  stochastic reward the learner *receives* if it pulls that arm
      exp_rew: expected reward used for regret accounting
    Returns cumulative regret over T steps.
    """
    rng = np.random.default_rng(seed)
    d, A = env.d, env.A
    algo = AlgoCls(d, A, **(algo_kwargs or {}))
    cum_regret = 0.0
    for t in range(T):
        ctxs, rew, exp_rew = env.step(t, rng)
        a = algo.select(ctxs)
        algo.update(a, ctxs, rew[a])
        cum_regret += float(np.max(exp_rew) - exp_rew[a])
    return cum_regret


def run_grid(envs, algos, T, seeds, algo_kwargs=None):
    out = {}
    for ename, env in envs.items():
        out[ename] = {}
        for aname in algos:
            vals = [run_episode(ALGOS[aname], env, T, s, algo_kwargs) for s in seeds]
            out[ename][aname] = (float(np.mean(vals)), float(np.std(vals)), vals)
    return out
