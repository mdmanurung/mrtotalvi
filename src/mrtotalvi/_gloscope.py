"""GloScope-style divergences between per-sample cell distributions.

A Python port of the GloScope estimators (Wang, Torous, Gong and Purdom, Genome Biol 2024,
25:259, doi:10.1186/s13059-024-03398-1). Parity with the R package was checked against GloScope
1.4.0 and 2.0.1. Each sample is the distribution of its cells in a latent space. Samples are
compared by a divergence between those distributions, and classical MDS of the divergence
matrix gives a sample-level embedding.

The k-NN estimators (``dens="KNN"``) are the estimators with exact R parity: KL is
``FNN::KL.dist`` (``KL(p||q) + KL(q||p)``), and JS is GloScope's plug-in form, whose
within-sample neighbour query counts the point itself. Parity was verified on Mule to 1e-13.

GMM values (``dens="GMM"``) do not match R GloScope. GloScope fits mclust model ``"VVE"``.
``gmm_model="full"`` (default) fits scikit-learn ``GaussianMixture(covariance_type="full")``, which
is mclust ``"VVV"`` with k-means initialisation, and picks the component count by BIC as mclust does.
``gmm_model="vve"`` fits a Gaussian mixture with varying volume and shape and one shared
orientation (mclust ``"VVE"``) by the clean-room implementation described below.

As in GloScope, the Monte Carlo KL redraws ``r`` points from the fitted mixture for every
sample pair and every direction: ``KL(p||q)`` is the mean of ``log p(x) - log q(x)`` over
``x ~ p``, and the two directions are summed. Component counts of each draw follow the share
of the sample's cells that the fitted model assigns to each component (hard classification),
as in R, not the mixing weights. Each (pair, direction) draw has its own random stream,
derived from the seed and the sorted-label rank of the two samples, so the result does not
depend on the order of ``samples`` or on how pairs are scheduled.

Clean-room VVE (``gmm_model="vve"``)
------------------------------------
The VVE code is written from the literature and the mclust help pages only. No mclust R or
Fortran source was read, and no GPL code is included; the package licence stays BSD-3. It uses
numpy and scipy only. Sources:

- Browne and McNicholas (2014), Adv Data Anal Classif 8:217-226 (arXiv 1302.2102): the MM
  algorithm for the common-orientation M-step.
- Fraley and Raftery (2002), J Am Stat Assoc 97:611-631: model-based clustering, EM and BIC.
- Scrucca, Fop, Murphy and Raftery (2016), R Journal 8(1):289-317: mclust 5 model names, the
  hc subset option, the VVE estimator, and the BIC convention (larger is better).
- Scrucca and Raftery (2015), Adv Data Anal Classif 9:447-460: the scaled SVD transform.
- Banfield and Raftery (1993), Biometrics 49:803-821: model-based agglomerative clustering,
  which merges the pair of clusters that gives the largest classification likelihood.
- Fraley (1998), SIAM J Sci Comput 20:270-281, is cited by the mclust help pages for the hc
  algorithms. It was not available, so none of it is used (see below).

As in R GloScope, Monte Carlo draws split the ``r`` points over components by the
hard-classification proportions of the sample's cells, while log-densities use the fitted
mixing weights (G143). This holds for both ``gmm_model`` values.

EM starts from model-based agglomerative clustering with unconstrained covariances (hcModelName
"VVV", hcUse "SVD", subset 2000 cells; the mclust.options defaults) on the scaled-SVD data
(Banfield and Raftery 1993). Each step merges the pair of clusters with the smallest increase of
``sum_k n_k log|Sigma_k|``, the classification-likelihood criterion. The published text of the
treatment of clusters with fewer points than dimensions (Fraley 1998) was not available, so this
port makes its own choice: ``Sigma_k = (W_k + s0 I) / (n_k + 1)``, with ``W_k`` the scatter of
the cluster and ``s0`` the mean variance of the transformed data (one pseudo-observation at the
overall mean variance; mclust documents ``alpha = beta = 1`` as defaults, but not their role).
EM has no iteration cap, as in mclust 6.1.4 (``emControl()$itmax`` is 2147483647), with
relative tolerance 1e-5.

This is a clean-room implementation. It does NOT reproduce R mclust. The merge tree differs
from R's: on Mule sample d233 the partitions agree with R's ``hclass`` at an adjusted Rand
index of 0.45 to 0.68 for 5 to 20 groups. On Mule (20 samples, 5 seeds) the port picks G=15 in
81 of 100 fits and G=10 in 19, against G=5/10/15/20 in 34/46/16/4 for R, and reaches a higher
log-likelihood in 96 of 100 fits. Its divergences correlate with R's at Pearson 0.76 (median of 5
seeds; R against R: 0.93) and Spearman 0.87 (R against R: 0.95); the chosen G agrees with R's
in 4 of 20 samples (R against R: 11.5). It fails the strict reproducibility gate (bench report
R098, figure ``vve_port_vs_mclust_v2.png``). The free-parameter count and BIC formula match mclust.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.special import logsumexp
from scvi import settings

MIN_CELLS = 500  # GloScope warns below this many cells per sample


# --- Clean-room VVE mixture (see the module docstring) ----------------------------------------------------------

_VVE_SUBSET = 2000  # hc subset size for the initial partition when a sample has more cells (mclust default)
_VVE_TOL = 1e-5  # relative log-likelihood tolerance of EM (mclust emControl default)
_VVE_EPS = np.finfo(float).eps  # a variance below eps * the largest one counts as singular


class _VVEError(Exception):
    """EM hit a singular or empty component; this component count is skipped."""


def _vve_n_params(g: int, d: int) -> int:
    """Free parameters of VVE as in mclust ``nVarParams``: means, weights and G + d(d-1)/2 + G(d-1) covariance terms."""
    return g * d + (g - 1) + g + d * (d - 1) // 2 + g * (d - 1)


def _vve_bic(loglik: float, n: int, g: int, d: int) -> float:
    """BIC with the mclust sign convention (larger is better)."""
    return 2 * loglik - _vve_n_params(g, d) * np.log(n)


def _svd_transform(x: np.ndarray) -> np.ndarray:
    """Scaled SVD (Scrucca and Raftery 2015, sec. 4.5): centre, scale to unit variance, ``Z = U D^(1/2)``."""
    xs = (x - x.mean(0)) / x.std(0, ddof=0)
    u, s, _ = np.linalg.svd(xs, full_matrices=False)
    return u * np.sqrt(s)


def _vve_mstep(x, z, rot, inner_tol=1.49e-8, inner_max=500):
    """VVE M-step: weights, means, shared orientation and per-component variances.

    The orientation comes from pairwise sweeps (Browne and McNicholas 2014): per column pair, with the
    variances fixed, the rotation in their plane minimises a 2x2 quadratic form.
    """
    n, d = x.shape
    nk = z.sum(0)
    mean = (z.T @ x) / nk[:, None]
    w = np.empty((len(nk), d, d))
    for k in range(len(nk)):
        y = x - mean[k]
        w[k] = (y * z[:, [k]]).T @ y
    if rot is None:
        rot = np.linalg.eigh(w.sum(0))[1][:, ::-1]
    rot = rot.copy()
    wd = np.einsum("kij,jl->kil", w, rot)
    omega = np.einsum("ji,kji->ki", rot, wd) / nk[:, None]
    f_old = (nk * np.log(omega).sum(1)).sum()
    for _ in range(inner_max):
        inv = 1.0 / omega
        for i in range(d - 1):
            for j in range(i + 1, d):
                q = rot[:, [i, j]]
                wq = wd[:, :, [i, j]]
                t = np.einsum("ia,kib->kab", q, wq)
                m = np.einsum("k,kab->ab", inv[:, i] - inv[:, j], t)
                m = 0.5 * (m + m.T)
                vec = np.linalg.eigh(m)[1]
                rot[:, [i, j]] = q @ vec
                wd[:, :, [i, j]] = wq @ vec
        omega = np.einsum("ji,kji->ki", rot, wd) / nk[:, None]
        f = (nk * np.log(omega).sum(1)).sum()
        done = abs(f - f_old) <= inner_tol * (1 + abs(f))
        f_old = f
        if done:
            break
    return nk / n, mean, rot, omega


def _vve_component_logpdf(x, pro, mean, rot, omega):
    """``(n, G)`` array of ``log pro_k + log N(x | mean_k, rot diag(omega_k) rot')``."""
    d = x.shape[1]
    cols = np.empty((x.shape[0], len(pro)))
    for k in range(len(pro)):
        y = (x - mean[k]) @ rot
        cols[:, k] = np.log(pro[k]) - 0.5 * (d * np.log(2 * np.pi) + np.log(omega[k]).sum() + (y**2 / omega[k]).sum(1))
    return cols


def _vve_estep(x, pro, mean, rot, omega):
    cols = _vve_component_logpdf(x, pro, mean, rot, omega)
    ll = logsumexp(cols, axis=1)
    return np.exp(cols - ll[:, None]), ll.sum()


def _vve_fit(x, z0, tol=_VVE_TOL, max_iter=None):
    """EM for VVE from initial responsibilities ``z0`` of shape ``(n, G)``; ``max_iter=None`` means no cap."""
    if z0.sum(0).min() < 1:
        raise _VVEError("empty component at start")
    pro, mean, rot, omega = _vve_mstep(x, z0, None)
    ll_old, trace = -np.inf, []
    it = 0
    while max_iter is None or it < max_iter:
        it += 1
        z, ll = _vve_estep(x, pro, mean, rot, omega)
        if not np.isfinite(ll) or z.sum(0).min() < 1e-3 or omega.min() <= _VVE_EPS * omega.max():
            raise _VVEError("singular or empty component")
        trace.append(ll)
        if abs(ll - ll_old) <= tol * (1 + abs(ll)):
            break
        ll_old = ll
        pro, mean, rot, omega = _vve_mstep(x, z, rot)
    else:
        # Stopped at the cap right after an M-step: refresh loglik and z for the returned parameters.
        z, ll = _vve_estep(x, pro, mean, rot, omega)
    return {"pro": pro, "mean": mean, "rot": rot, "omega": omega, "loglik": ll, "z": z, "loglik_trace": trace}


def _vvv_hc(t: np.ndarray, gs: list[int]) -> dict[int, np.ndarray]:
    """Model-based agglomerative clustering with unconstrained covariances; partitions for each count in ``gs``.

    Starts from singletons. Each step merges the pair of clusters with the smallest increase of
    ``sum_k n_k log|Sigma_k|`` (Banfield and Raftery 1993), where ``Sigma_k = (W_k + s0 I) / (n_k + 1)`` keeps the
    covariance of small clusters non-singular (this port's own choice; see the module docstring). Returns
    ``{g: labels}`` with labels ``0..g-1``.
    """
    n, d = t.shape
    s0 = np.trace(np.cov(t.T, bias=True)) / d
    count = np.ones(n)
    mean = t.copy()
    scatter = np.zeros((n, d, d))
    logdet = np.full(n, d * np.log(s0 / 2))  # singleton: Sigma = s0 I / 2

    def merge_cost(i, js):
        nij = count[i] + count[js]
        diff = mean[i] - mean[js]
        wij = (
            scatter[i] + scatter[js] + (count[i] * count[js] / nij)[:, None, None] * diff[:, :, None] * diff[:, None, :]
        )
        sigma = (wij + s0 * np.eye(d)) / (nij + 1)[:, None, None]
        return nij * np.linalg.slogdet(sigma)[1] - count[i] * logdet[i] - count[js] * logdet[js]

    cost = np.full((n, n), np.inf)  # upper triangle: cost[i, j] for i < j
    for i in range(n - 1):
        js = np.arange(i + 1, n)
        cost[i, js] = merge_cost(i, js)
    label = np.arange(n)
    alive = np.ones(n, dtype=bool)
    out = {}
    for k in range(n, min(gs), -1):
        i, j = np.unravel_index(np.argmin(cost), cost.shape)
        nij = count[i] + count[j]
        diff = mean[i] - mean[j]
        scatter[i] += scatter[j] + (count[i] * count[j] / nij) * np.outer(diff, diff)
        mean[i] = (count[i] * mean[i] + count[j] * mean[j]) / nij
        count[i] = nij
        logdet[i] = np.linalg.slogdet((scatter[i] + s0 * np.eye(d)) / (nij + 1))[1]
        alive[j] = False
        label[label == j] = i
        cost[j, :] = np.inf
        cost[:, j] = np.inf
        others = np.flatnonzero(alive)
        others = others[others != i]
        c = merge_cost(i, others)
        above = others > i
        cost[i, others[above]] = c[above]
        cost[others[~above], i] = c[~above]
        if k - 1 in gs:
            out[k - 1] = np.unique(label, return_inverse=True)[1]
    if n in gs:
        out[n] = np.arange(n)
    return out


def _vve_init(x: np.ndarray, gs: list[int], rng: np.random.Generator) -> dict[int, np.ndarray]:
    """Initial responsibilities per component count.

    VVV model-based agglomerative clustering of the scaled-SVD data of a seeded subset (at most 2000 cells), one
    M-step on the subset, one E-step on all cells (the mclust way of starting EM from a subset).
    """
    n = x.shape[0]
    idx = np.sort(rng.choice(n, _VVE_SUBSET, replace=False)) if n > _VVE_SUBSET else np.arange(n)
    partitions = _vvv_hc(_svd_transform(x[idx]), gs)
    out = {}
    for g in gs:
        zs = (partitions[g][:, None] == np.arange(g)[None, :]).astype(float)
        if zs.sum(0).min() < 2:
            continue
        try:
            pro, mean, rot, omega = _vve_mstep(x[idx], zs, None)
            out[g] = _vve_estep(x, pro, mean, rot, omega)[0]
        except (np.linalg.LinAlgError, FloatingPointError):
            continue
    return out


class _VVEMixture:
    """Fitted VVE mixture with the parts of the scikit-learn API that this module uses."""

    def __init__(self, fit: dict, bic: float):
        self.weights_ = fit["pro"]
        self.means_ = fit["mean"]
        self.rot_ = fit["rot"]
        self.omega_ = fit["omega"]
        self.covariances_ = np.einsum("ij,kj,lj->kil", fit["rot"], fit["omega"], fit["rot"])
        self.n_components = len(fit["pro"])
        self.loglik_ = fit["loglik"]
        self.bic_ = bic
        self.hard_weights_ = np.bincount(fit["z"].argmax(1), minlength=self.n_components) / fit["z"].shape[0]

    def score_samples(self, x: np.ndarray) -> np.ndarray:
        return logsumexp(_vve_component_logpdf(x, self.weights_, self.means_, self.rot_, self.omega_), axis=1)


def _gmm_vve(x: np.ndarray, num_components: tuple[int, ...], random_state: int | None) -> _VVEMixture:
    """Fit VVE for each candidate component count; keep the largest mclust-convention BIC."""
    n, d = x.shape
    candidates = sorted({min(g, n - 1) for g in num_components})
    inits = _vve_init(x, candidates, np.random.default_rng(random_state))
    fits = []
    for g in candidates:
        if g not in inits:
            continue
        try:
            fit = _vve_fit(x, inits[g])
        except (_VVEError, np.linalg.LinAlgError):
            continue
        fits.append((_vve_bic(fit["loglik"], n, g, d), fit))
    if not fits:
        raise RuntimeError(f"No VVE fit succeeded for any component count in {candidates}; try gmm_model='full'.")
    bic, fit = max(fits, key=lambda t: t[0])
    return _VVEMixture(fit, bic)


def _gmm(x: np.ndarray, num_components: tuple[int, ...], random_state: int | None, gmm_model: str = "full"):
    """Fit one GMM per candidate component count; keep the best BIC (``gmm_model``: ``"full"`` or ``"vve"``)."""
    if gmm_model == "vve":
        return _gmm_vve(x, num_components, random_state)
    from sklearn.mixture import GaussianMixture

    candidates = sorted({min(g, x.shape[0] - 1) for g in num_components})
    fits = [GaussianMixture(g, covariance_type="full", random_state=random_state).fit(x) for g in candidates]
    best = min(fits, key=lambda m: m.bic(x))
    best.hard_weights_ = np.bincount(best.predict(x), minlength=best.n_components) / x.shape[0]
    return best


def _sample_gmm(model, r: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``r`` points from a fitted full-covariance mixture; component counts follow its hard-classification shares."""
    counts = rng.multinomial(r, model.hard_weights_)
    if hasattr(model, "rot_"):  # VVE: scale standard normals by sqrt(omega) and rotate (no Cholesky)
        d = model.means_.shape[1]
        return np.vstack(
            [
                model.means_[c] + (rng.standard_normal((n, d)) * np.sqrt(model.omega_[c])) @ model.rot_.T
                for c, n in enumerate(counts)
                if n
            ]
        )
    parts = [rng.multivariate_normal(model.means_[c], model.covariances_[c], size=n) for c, n in enumerate(counts) if n]
    return np.vstack(parts)


def _kth_distance(query: np.ndarray, reference: np.ndarray, k: int, self_match: bool) -> np.ndarray:
    """Distance from each query point to its k-th neighbour in ``reference``."""
    from sklearn.neighbors import NearestNeighbors

    n_neighbors = k + 1 if self_match else k
    dist, _ = NearestNeighbors(n_neighbors=n_neighbors).fit(reference).kneighbors(query)
    return dist[:, -1]


def _knn_kl(x: np.ndarray, y: np.ndarray, k: int) -> float:
    """Directed KL(x || y) by the k-NN estimator of FNN::KL.divergence, ``log(m / n)`` term included."""
    n, d = x.shape
    m = y.shape[0]
    rho = _kth_distance(x, x, k, self_match=True)
    nu = _kth_distance(x, y, k, self_match=False)
    return float(d * np.mean(np.log(nu) - np.log(rho)) + np.log(m / n))


def _knn_js(x: np.ndarray, y: np.ndarray, k: int) -> float:
    """Plug-in k-NN Jensen-Shannon divergence, as GloScope's ``.calc_JS``.

    As there (``RANN::nn2`` with the sample as its own query), the within-sample k-th
    neighbour counts the point itself, so it is the (k-1)-th other point.
    """
    n, d = x.shape
    m = y.shape[0]
    x_in_x = _kth_distance(x, x, k, self_match=False) ** d
    x_in_y = _kth_distance(x, y, k, self_match=False) ** d
    y_in_y = _kth_distance(y, y, k, self_match=False) ** d
    y_in_x = _kth_distance(y, x, k, self_match=False) ** d
    term_x = np.log(2 * m * x_in_y / (m * x_in_y + (n - 1) * x_in_x))
    term_y = np.log(2 * n * y_in_x / (n * y_in_x + (m - 1) * y_in_y))
    return float(term_x.mean() / 2 + term_y.mean() / 2)


def sample_divergence(
    embedding: npt.ArrayLike,
    sample_ids: npt.ArrayLike,
    samples: list | None = None,
    dens: Literal["GMM", "KNN"] = "GMM",
    dist_metric: Literal["KL", "JS"] = "KL",
    r: int = 10000,
    num_components: tuple[int, ...] = (5, 10, 15, 20),
    k: int = 50,
    random_state: int | None = None,
    gmm_model: Literal["full", "vve"] = "full",
) -> tuple[np.ndarray, list]:
    """Symmetric divergence between the cell distributions of every pair of samples.

    Parameters
    ----------
    embedding
        ``(n_cell, n_dim)`` cell coordinates.
    sample_ids
        ``(n_cell,)`` sample label of each cell.
    samples
        Sample order of the output. Defaults to the sorted unique labels.
    dens
        ``"GMM"`` fits a Gaussian mixture per sample; ``"KNN"`` uses k-nearest-neighbour
        estimators with no density fit.
    dist_metric
        ``"KL"``: ``KL(p||q) + KL(q||p)``. ``"JS"``: Jensen-Shannon divergence.
    r
        Monte Carlo draws per sample pair and direction (GMM only).
    num_components
        Candidate component counts for the GMM, chosen by BIC; capped at ``n_cell - 1``.
    k
        Neighbour rank for ``dens="KNN"``. Samples with ``k`` or fewer cells get NaN.
    gmm_model
        Covariance model of the GMM. ``"full"``: scikit-learn full covariances (mclust ``VVV``).
        ``"vve"``: varying volume and shape with a shared orientation (mclust ``VVE``,
        GloScope's model), by a clean-room implementation that does not reproduce R mclust.
        Both keep per-pair draws and hard-classification proportions.
    random_state
        Seed of the GMM fits and the Monte Carlo draws. ``None`` uses
        :attr:`scvi.settings.seed`. Each (pair, direction) draw has its own stream derived
        from the seed and the sorted-label rank of the samples, so results do not depend on
        the order of ``samples`` or on pair scheduling.

    Returns
    -------
    ``(n_sample, n_sample)`` divergence matrix with a zero diagonal, and the sample order.
    """
    if dens not in ("GMM", "KNN"):
        raise ValueError(f"dens must be 'GMM' or 'KNN', not {dens!r}.")
    if dist_metric not in ("KL", "JS"):
        raise ValueError(f"dist_metric must be 'KL' or 'JS', not {dist_metric!r}.")
    if gmm_model not in ("full", "vve"):
        raise ValueError(f"gmm_model must be 'full' or 'vve', not {gmm_model!r}.")
    x = np.asarray(embedding, dtype=np.float64)
    ids = np.asarray(sample_ids)
    if x.ndim != 2 or ids.shape != (x.shape[0],):
        raise ValueError("embedding must be (n_cell, n_dim) and sample_ids (n_cell,).")
    samples = list(samples) if samples is not None else sorted(set(ids.tolist()))
    groups = {s: x[ids == s] for s in samples}
    small = [s for s, g in groups.items() if g.shape[0] < MIN_CELLS]
    if small:
        warnings.warn(
            f"{len(small)} samples have fewer than {MIN_CELLS} cells; their divergences may be unreliable.",
            UserWarning,
            stacklevel=2,
        )
    seed = random_state if random_state is not None else settings.seed

    usable = {s: g.shape[0] > (k if dens == "KNN" else 1) for s, g in groups.items()}
    dropped = [s for s, ok in usable.items() if not ok]
    if dropped:
        warnings.warn(
            f"Samples with too few cells for dens={dens!r} get NaN divergences: {dropped}.",
            UserWarning,
            stacklevel=2,
        )

    if dens == "GMM":
        # Rank in the sorted labels, not the position in ``samples``, keys every random stream.
        rank = {s: i for i, s in enumerate(sorted(set(ids.tolist())))}
        models = {}
        for s in samples:
            if usable[s]:
                models[s] = _gmm(groups[s], num_components, None if seed is None else seed + rank[s], gmm_model)
        root = np.random.SeedSequence(seed)

    n = len(samples)
    out = np.full((n, n), np.nan)
    for i, s in enumerate(samples):
        if not usable[s]:
            continue
        out[i, i] = 0.0
        for j in range(i):
            t = samples[j]
            if not usable[t]:
                continue
            if dens == "GMM":
                lo, hi = sorted((rank[s], rank[t]))
                # Fresh draws from each sample of the pair, one stream per direction.
                rng_s, rng_t = (
                    np.random.default_rng(
                        np.random.SeedSequence(root.entropy, spawn_key=(lo, hi, int(rank[u] > rank[v])))
                    )
                    for u, v in ((s, t), (t, s))
                )
                x_s, x_t = _sample_gmm(models[s], r, rng_s), _sample_gmm(models[t], r, rng_t)
                s_in_s, s_in_t = models[s].score_samples(x_s), models[t].score_samples(x_s)
                t_in_t, t_in_s = models[t].score_samples(x_t), models[s].score_samples(x_t)
            if dens == "GMM" and dist_metric == "KL":
                value = (s_in_s - s_in_t).mean() + (t_in_t - t_in_s).mean()
            elif dens == "GMM":
                mix_s = np.logaddexp(s_in_s, s_in_t) - np.log(2)
                mix_t = np.logaddexp(t_in_t, t_in_s) - np.log(2)
                value = (s_in_s - mix_s).mean() / 2 + (t_in_t - mix_t).mean() / 2
            elif dist_metric == "KL":
                value = _knn_kl(groups[s], groups[t], k) + _knn_kl(groups[t], groups[s], k)
            else:
                value = _knn_js(groups[s], groups[t], k)
            out[i, j] = out[j, i] = value
    return out, samples


def classical_mds(distances: npt.ArrayLike, n_components: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Classical (Torgerson) MDS, as R's ``cmdscale`` used by GloScope's ``plotMDS``.

    Returns the ``(n, n_components)`` coordinates and the eigenvalues of every axis kept.
    Axes with non-positive eigenvalues are dropped, with a warning.
    """
    d = np.asarray(distances, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distances must be a square matrix.")
    if not np.isfinite(d).all():
        raise ValueError("distances contain NaN or inf; drop those samples first.")
    n = d.shape[0]
    centre = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * centre @ (d**2) @ centre
    eigval, eigvec = np.linalg.eigh((b + b.T) / 2)
    order = np.argsort(eigval)[::-1][:n_components]
    eigval, eigvec = eigval[order], eigvec[:, order]
    keep = eigval > 0
    if not keep.all():
        warnings.warn(
            f"Only {int(keep.sum())} of {n_components} MDS axes have positive eigenvalues; the rest are dropped.",
            UserWarning,
            stacklevel=2,
        )
    return eigvec[:, keep] * np.sqrt(eigval[keep]), eigval[keep]
