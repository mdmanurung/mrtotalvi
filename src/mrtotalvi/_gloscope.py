"""GloScope-style divergences between per-sample cell distributions.

A Python port of the GloScope estimators (Wang, Torous, Gong and Purdom, Genome Biol 2024,
25:259, doi:10.1186/s13059-024-03398-1). Parity with the R package was checked against GloScope
1.4.0 and 2.0.1. Each sample is the distribution of its cells in a latent space. Samples are
compared by a divergence between those distributions, and classical MDS of the divergence
matrix gives a sample-level embedding.

The k-NN estimators (``dens="KNN"``) are the estimators with exact R parity: KL is
``FNN::KL.dist`` (``KL(p||q) + KL(q||p)``), and JS is GloScope's plug-in form, whose
within-sample neighbour query counts the point itself. Parity was verified on Mule to 1e-13.

GMM values (``dens="GMM"``) do not match R GloScope yet. GloScope fits mclust model ``"VVE"``.
This port fits scikit-learn ``GaussianMixture(covariance_type="full")``, which is mclust
``"VVV"`` with k-means initialisation, and picks the component count by BIC as mclust does.

As in GloScope, the Monte Carlo KL redraws ``r`` points from the fitted mixture for every
sample pair and every direction: ``KL(p||q)`` is the mean of ``log p(x) - log q(x)`` over
``x ~ p``, and the two directions are summed. Component counts of each draw follow the share
of the sample's cells that the fitted model assigns to each component (hard classification),
as in R, not the mixing weights. Each (pair, direction) draw has its own random stream,
derived from the seed and the sorted-label rank of the two samples, so the result does not
depend on the order of ``samples`` or on how pairs are scheduled.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import numpy.typing as npt
from scvi import settings

MIN_CELLS = 500  # GloScope warns below this many cells per sample


def _gmm(x: np.ndarray, num_components: tuple[int, ...], random_state: int | None):
    """Fit one full-covariance GMM per candidate component count; keep the lowest BIC."""
    from sklearn.mixture import GaussianMixture

    candidates = sorted({min(g, x.shape[0] - 1) for g in num_components})
    fits = [GaussianMixture(g, covariance_type="full", random_state=random_state).fit(x) for g in candidates]
    best = min(fits, key=lambda m: m.bic(x))
    best.hard_weights_ = np.bincount(best.predict(x), minlength=best.n_components) / x.shape[0]
    return best


def _sample_gmm(model, r: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``r`` points from a fitted full-covariance mixture; component counts follow its hard-classification shares."""
    counts = rng.multinomial(r, model.hard_weights_)
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
                models[s] = _gmm(groups[s], num_components, None if seed is None else seed + rank[s])
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
