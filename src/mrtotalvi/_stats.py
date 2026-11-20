"""Shared u-space statistical APIs for Mr multimodal models."""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pandas as pd
import torch
import xarray as xr
from scipy.special import logsumexp
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal
from tqdm import tqdm

from scvi import settings


def _validate_sample_level_covariates(obs, sample_key: str, cov_keys: list[str]) -> None:
    """Raise ValueError if any cov in cov_keys is not constant within sample_key."""
    for cov in cov_keys:
        if cov not in obs.columns:
            continue
        n_per_sample = obs.groupby(sample_key, observed=True)[cov].nunique()
        bad = n_per_sample[n_per_sample > 1]
        if len(bad):
            raise ValueError(
                f"sample_cov_key '{cov}' is not constant within sample_key '{sample_key}' "
                f"for the following samples: {bad.index.tolist()}. "
                f"Each sample (sample_key level) must map to exactly one covariate value; "
                f"otherwise drop_duplicates() picks an arbitrary value and the design matrix "
                f"is incorrect. To test a within-sample condition (e.g. timepoint), retrain "
                f"with a more granular sample_key (e.g. 'donor_timepoint')."
            )


# Absolute cosine between a fixed covariate and another design column above which the
# per-covariate Wald statistic is flagged as mixed (a warning only; no output changes).
_CORRELATED_DESIGN_TOL = 0.1


def _rowwise_max_excluding_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {tuple(matrix.shape)}.")
    n_cols = matrix.shape[1]
    if n_cols == 1:
        return matrix[:, 0]
    mask = ~torch.eye(n_cols, dtype=torch.bool, device=matrix.device)
    return torch.where(mask, matrix, torch.tensor(-float("inf"), device=matrix.device)).max(
        dim=1
    ).values


def _as_indices(adata, indices):
    if indices is None:
        return np.arange(adata.n_obs)
    return np.asarray(indices)


def _loader_order(adata, indices):
    """Indices to hand the data loader, and the permutation back to request order.

    scvi-tools sorts indices inside each minibatch for backed AnnData, so rows would come
    back in sorted order, and h5py rejects repeated indices. For backed data each cell is
    loaded once in ascending order, and the returned index array maps those rows back to
    the request order; otherwise it is ``None``.
    """
    if indices is None or not getattr(adata, "isbacked", False):
        return indices, None
    unique_indices, inverse = np.unique(np.asarray(indices), return_inverse=True)
    return unique_indices, inverse


def collect_u_posterior(model, adata=None, indices=None, batch_size: int = 256):
    """Collect posterior Normal parameters for ``u`` from a model dataloader."""
    adata = model._validate_anndata(adata)
    indices = _as_indices(adata, indices)
    loader_indices, inverse = _loader_order(adata, indices)
    dataloader = model._make_data_loader(
        adata=adata, indices=loader_indices, batch_size=batch_size
    )

    locs = []
    scales = []
    was_training = model.module.training
    model.module.eval()
    try:
        with torch.inference_mode():
            for tensors in dataloader:
                inference_inputs = model.module._get_inference_input(tensors)
                outputs = model.module.inference(**inference_inputs)
                qu = outputs.get("qu")
                if qu is None:
                    qu = outputs.get("qz")
                if qu is None:
                    locs.append(outputs["qz_m"])
                    scales.append(torch.sqrt(outputs["qz_v"]))
                else:
                    locs.append(qu.loc)
                    scales.append(qu.scale)
    finally:
        model.module.train(was_training)

    loc, scale = torch.cat(locs, dim=0), torch.cat(scales, dim=0)
    if inverse is not None:
        loc, scale = loc[inverse], scale[inverse]
    return loc, scale


def get_aggregated_posterior(
    model,
    *,
    adata=None,
    sample_key: str,
    sample: str | int | None = None,
    indices=None,
    batch_size: int = 256,
) -> MixtureSameFamily:
    """Compute the aggregated posterior mixture over ``u``."""
    model._check_if_trained(warn=False)
    adata = model._validate_anndata(adata)
    indices = _as_indices(adata, indices)
    if sample is not None:
        indices = np.intersect1d(indices, np.where(adata.obs[sample_key] == sample)[0])
    if len(indices) == 0:
        raise ValueError("Cannot build an aggregated posterior from zero cells.")

    loc, scale = collect_u_posterior(model, adata=adata, indices=indices, batch_size=batch_size)
    weights = torch.ones(loc.shape[0], device=loc.device, dtype=loc.dtype) / loc.shape[0]
    return MixtureSameFamily(Categorical(probs=weights), Independent(Normal(loc, scale), 1))


def _knn_sample_counts(
    u: np.ndarray,
    sample_codes: np.ndarray,
    n_samples: int,
    k: int,
    omit_original_sample: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Count reference cells per sample among each query cell's ``k`` nearest neighbours.

    ``sample_codes`` holds the column index of each cell's sample in the analysed sample
    list, or ``-1`` for cells of samples outside the analysis (queried but never reference).
    The reference of query cell ``i`` is every analysed cell except ``i`` itself, and, when
    ``omit_original_sample`` is true, except every cell of ``i``'s own sample.

    Returns ``counts`` and ``n_ref``, both ``(n_cells, n_samples)``. ``counts[i, s]`` is
    the number of the ``k`` neighbours from sample ``s``; ``n_ref[i, s]`` is the number of
    reference cells of sample ``s`` available to cell ``i`` (0 for its own sample when
    omitted).
    """
    from sklearn.neighbors import NearestNeighbors

    n_cells = u.shape[0]
    in_ref = sample_codes >= 0
    ref_idx = np.flatnonzero(in_ref)
    sizes = np.bincount(sample_codes[in_ref], minlength=n_samples)
    counts = np.zeros((n_cells, n_samples), dtype=np.int64)
    n_ref = np.tile(sizes, (n_cells, 1))
    own = np.flatnonzero(in_ref)
    if omit_original_sample:
        n_ref[own, sample_codes[own]] = 0
    else:
        n_ref[own, sample_codes[own]] -= 1  # the cell itself is not its own neighbour

    def _count(query: np.ndarray, ref: np.ndarray, drop_self: bool) -> None:
        if ref.size < k + int(drop_self):
            raise ValueError(
                f"density_estimator='knn' needs at least knn_k={k} reference cells for "
                f"every query cell; found {ref.size}. Lower knn_k."
            )
        nn = NearestNeighbors(n_neighbors=k + int(drop_self)).fit(u[ref])
        nbr = ref[nn.kneighbors(u[query], return_distance=False)]
        if drop_self:
            is_self = nbr == query[:, None]
            # a query cell outside the reference, or a tie that pushed it out, loses its
            # farthest neighbour instead
            is_self[~is_self.any(axis=1), -1] = True
            keep = ~is_self
            nbr = nbr[keep].reshape(len(query), k)
        codes = sample_codes[nbr]
        for sample in range(n_samples):
            counts[query, sample] = (codes == sample).sum(axis=1)

    if omit_original_sample:
        for sample in range(n_samples):
            query = np.flatnonzero(sample_codes == sample)
            if query.size:
                _count(query, ref_idx[sample_codes[ref_idx] != sample], drop_self=False)
        query = np.flatnonzero(~in_ref)
        if query.size:
            _count(query, ref_idx, drop_self=False)
    else:
        _count(np.flatnonzero(in_ref), ref_idx, drop_self=True)
        query = np.flatnonzero(~in_ref)
        if query.size:
            _count(query, ref_idx, drop_self=False)
    return counts, n_ref


def _sqrtm_batch(xtmx: torch.Tensor) -> torch.Tensor:
    """Matrix square root of each ``(n_cov, n_cov)`` matrix in a batch.

    A singular ``X^T M X`` (a cell with too few admissible samples) makes
    ``scipy.linalg.sqrtm`` return inf or NaN, which would turn the Wald statistic into NaN
    and make the Chi2 CDF raise. Non-finite entries are set to 0, as in the MrVI reference,
    so such a cell gets a zero statistic instead of aborting the run.
    """
    import scipy.linalg

    root = torch.stack(
        [
            torch.from_numpy(
                scipy.linalg.sqrtm(m.cpu().numpy()).real.astype(np.float32)
            )
            for m in xtmx
        ]
    ).to(xtmx.device)
    return torch.nan_to_num(root, nan=0.0, posinf=0.0, neginf=0.0)


def _knn_log_density(counts: np.ndarray, n_ref: np.ndarray) -> np.ndarray:
    """``log(count + 0.5) - log(n_ref)``; NaN where the group has no reference cells."""
    with np.errstate(divide="ignore"):
        out = np.log(counts + 0.5) - np.log(n_ref)
    return np.where(n_ref > 0, out, np.nan)


def _leave_one_cell_out(ap, u_tensor, lp: np.ndarray, own: np.ndarray) -> np.ndarray:
    """Remove each cell's own Normal component from its own sample's mixture density.

    ``lp`` is ``ap.log_prob(u_tensor)`` with uniform weights ``1/n``. For a cell whose
    component is ``own[i] >= 0`` the result is ``log((n * p - c_i) / (n - 1))``, computed in
    log space as ``logsumexp(component log-probs without c_i) - log(n - 1)``. Cells with
    ``own[i] < 0`` keep ``lp``. A one-component mixture has nothing left: NaN.
    """
    hit = np.nonzero(own >= 0)[0]
    if hit.size == 0:
        return lp
    lp = lp.copy()
    n_comp = ap.component_distribution.base_dist.loc.shape[0]
    if n_comp == 1:
        lp[hit] = np.nan
        return lp
    comp = ap.component_distribution.log_prob(u_tensor[hit].unsqueeze(-2))
    comp[torch.arange(hit.size, device=comp.device), torch.as_tensor(own[hit], device=comp.device)] = (
        -float("inf")
    )
    lp[hit] = (torch.logsumexp(comp, dim=-1) - np.log(n_comp - 1)).detach().cpu().numpy()
    return lp


def differential_abundance(
    model,
    *,
    adata=None,
    sample_key: str,
    sample_cov_keys: list[str] | None = None,
    sample_subset: list[str] | None = None,
    compute_log_enrichment: bool = False,
    omit_original_sample: bool = True,
    donor_key: str | None = None,
    batch_size: int = 128,
    n_mc_samples: int = 1,
    validated_sample_order: list | None = None,
    density_estimator: Literal["posterior", "knn", "scale_floor"] = "posterior",
    knn_k: int = 50,
    scale_floor: float | None = None,
) -> xr.Dataset:
    """Compute MrVI-style differential abundance log probabilities over ``u``.

    With ``omit_original_sample=False`` (also forced by ``donor_key``) every cell is scored
    leave-one-cell-out: its own Normal component is removed from its own sample's aggregated
    posterior (in log space, renormalised by ``n_s - 1``), as ``local_sample_enrichment`` and
    ``get_outlier_cell_sample_pairs`` already do. **This departs from upstream MrVI**, which
    keeps the component; that bias made every cell look enriched for its own group (on a
    null stub, +67.8 nats for ``"posterior"``, +9.1 with ``scale_floor=0.4`` and +32.6 with
    ``donor_key``). The default ``omit_original_sample=True`` is unchanged. A sample with a
    single cell has no leave-one-out density: its per-sample ``log_probs`` for that cell are
    NaN and it carries no weight in group aggregates.

    Group results are NaN, never ``-inf`` or ``inf``, when a cell's group has no reference
    sample: a group made only of the cell's own sample under ``omit_original_sample=True``,
    or only of a one-cell own sample under ``False``. Enrichments involving such a group are
    NaN. A ``UserWarning`` gives the number of affected cells.

    Parameters
    ----------
    density_estimator
        How per-sample densities over ``u`` are estimated.

        ``"posterior"`` (default) is the MrVI aggregated posterior: one Normal kernel per
        cell, using that cell's posterior scale. It reproduces earlier outputs exactly.
        Per-cell posterior scales are often far smaller than the spacing between
        neighbouring cells, which makes the estimate noise-dominated: pairs of samples with
        no real abundance change can score tens of nats apart.

        The same noise has a directional effect. ``logmeanexp`` over noise-dominated
        per-sample log densities behaves like a maximum, so a group of more samples scores
        higher, and ``omit_original_sample=True`` removes one sample from a cell's own
        group, which pushes cells away from their own group. With unequal group sizes the
        sign of the default log-enrichment can track samples per group (null stub, A with 2
        samples against B with 8: mean about -18.7 nats for A against B, ``scale_floor=0.4``
        gave -1.6 and ``"knn"`` gave 0.0). This matches the MrVI formula. Use ``"knn"`` or
        ``"scale_floor"`` for unbalanced designs.

        ``"scale_floor"`` is the same aggregated posterior with every kernel scale raised to
        ``max(scale, scale_floor)``. It needs ``scale_floor`` (a positive float; about 0.4
        calibrated a planted-effect test). ``log_probs`` stay true log densities, now of a
        smoothed mixture, and every other output keeps its meaning. ``n_mc_samples`` and
        ``donor_key`` work as for ``"posterior"``.

        ``"knn"`` replaces kernels with neighbour counts in the point-estimate ``u``
        (``n_mc_samples`` must be 1). For each query cell, take its ``knn_k`` nearest
        reference cells, where the reference is every cell of an analysed sample except the
        cell itself, and also except the cell's own sample when
        ``omit_original_sample=True``. With ``k_g`` of those neighbours in group ``g`` and
        ``n_g`` reference cells in ``g``, the group score is
        ``log(k_g + 0.5) - log(n_g)``. It is a log density of ``g`` at the cell up to an
        additive term, ``-log(volume of the neighbourhood)``, that is the same for every
        group. Consequences for the outputs:

        - ``{key}_log_probs`` and the log-enrichments are exactly this score and its
          differences. Enrichment of ``g`` against the rest is
          ``log(k_g + 0.5) - log(k_rest + 0.5) - log(n_g / n_rest)``, the neighbour-count
          ratio with pseudocount and sample-size correction.
        - Per-sample ``log_probs`` use the same score with ``g`` a single sample. They are
          **not** comparable with ``"posterior"`` values or across cells, because the
          neighbourhood volume differs per cell. Use differences (the enrichments) across
          cells. Within one cell they are comparable across samples of similar size only:
          the pseudocount 0.5 is fixed while the expected neighbour count of sample ``g``
          is ``knn_k * n_g / N`` (``N`` the reference size). Where that is below about 1, the
          score tends to ``log(0.5 / n_g)``, which favours the smaller sample (null stub with
          samples of 2000, 2000, 2000 and 30 cells and ``knn_k=50``: the 30-cell sample
          ranked first for every cell). Group scores pool counts but share the bias when a
          pooled group's expected count ``knn_k * n_g / N`` is below about 1 (null stub with
          groups of 4000 and 30 cells, ``knn_k=50``: mean log-enrichment +0.65 towards the
          30-cell group); a warning names such groups.
        - A group with no reference cells for a cell (its own sample under
          ``omit_original_sample=True``) scores NaN, where ``"posterior"`` gives ``-inf``.
        - Group scores are sample-size corrected by cell counts (pooled cells), not by the
          equal-sample ``logmeanexp`` that ``"posterior"`` uses. Samples with more cells
          weigh more in a group.
        - ``donor_key`` is not supported (the centering acts on densities, not counts).
        - Reference cells are those of ``sample_subset`` samples when given.

    knn_k
        Neighbours per query cell for ``"knn"``; ignored otherwise. Must be at least 1 and
        not exceed the reference size.
    scale_floor
        Minimum per-dimension kernel scale for ``"scale_floor"``. Must be ``None`` for the
        other estimators.
    n_mc_samples
        Number of Monte-Carlo draws from ``q(u|x)`` used to estimate the
        per-cell log-probability under each sample's aggregated posterior.

        When ``n_mc_samples=1`` (default) the posterior mean is used as a
        plug-in estimate (``give_mean=True``).  This is fast and
        **deterministic** given fixed model weights.

        When ``n_mc_samples > 1``, ``K`` draws ``u_k ~ q(u|x)`` are taken and
        the estimate is ``log(1/K * Σ_k p(u_k | sample))``, i.e.
        ``logsumexp(log_probs_k) - log(K)``.  This scores the cell's posterior
        ``q(u|x)`` against each sample mixture: the log of the expected density under
        ``q(u|x)``, a smoothed and stochastic alternative to the plug-in value
        ``log p(E[u])``. It is not a debiased version of the plug-in (the Jensen
        inequality is ``E[log p(u)] ≤ log E[p(u)]``, which does not relate the two), so
        outputs from ``n_mc_samples=1`` and ``> 1`` sit on different scales and should not
        be mixed. It costs ``K`` forward passes.

        **Important:** ``n_mc_samples > 1`` does *not* reduce cross-training-
        seed variance in DA results.  The seed variance originates from
        training (the u-encoder converges to a different attractor per seed),
        not from the evaluation estimator.  Once model weights are fixed, DA
        is deterministic for ``n_mc_samples=1``; for ``n_mc_samples > 1`` set
        :attr:`scvi.settings.seed` before calling to ensure reproducibility.
    """
    adata = model._validate_anndata(adata)

    if n_mc_samples < 1:
        raise ValueError("n_mc_samples must be >= 1.")
    if density_estimator not in ("posterior", "knn", "scale_floor"):
        raise ValueError(
            "density_estimator must be 'posterior', 'knn' or 'scale_floor'; "
            f"got {density_estimator!r}."
        )
    if density_estimator == "scale_floor":
        if scale_floor is None or not scale_floor > 0:
            raise ValueError("density_estimator='scale_floor' needs scale_floor > 0.")
    elif scale_floor is not None:
        raise ValueError("scale_floor is only used with density_estimator='scale_floor'.")
    if int(knn_k) != knn_k or knn_k < 1:
        raise ValueError("knn_k must be an integer >= 1.")
    if density_estimator == "knn":
        if n_mc_samples != 1:
            raise ValueError("density_estimator='knn' uses the posterior mean; set n_mc_samples=1.")
        if donor_key is not None:
            raise ValueError("donor_key is not supported with density_estimator='knn'.")

    if sample_cov_keys:
        _validate_sample_level_covariates(adata.obs, sample_key, list(sample_cov_keys))

    if validated_sample_order is not None:
        sample_values = list(validated_sample_order)
    else:
        sample_values = list(model.sample_order) if hasattr(model, "sample_order") else list(
            adata.obs[sample_key].unique()
        )
        if sample_subset is not None:
            sample_values = [s for s in sample_values if s in set(sample_subset)]

    donor_column_groups: list[list[int]] = []
    if donor_key is not None:
        if omit_original_sample:
            raise ValueError(
                "donor_key cannot be combined with omit_original_sample=True: the "
                "cell's own component inflates its own sample's density, and that "
                "inflated value enters the donor mean that is subtracted from the "
                "donor's other samples. Pass omit_original_sample=False."
            )
        if not sample_cov_keys:
            warnings.warn(
                "donor_key only changes the covariate aggregates; without "
                "sample_cov_keys it has no effect on the output.",
                UserWarning,
                stacklevel=2,
            )
        sample_info_da = (
            adata.obs[[sample_key, donor_key]]
            .drop_duplicates(subset=sample_key)
            .set_index(sample_key)
        )
        single_sample_donors = []
        for donor_id in sample_info_da[donor_key].unique():
            donor_samples = set(
                sample_info_da.index[sample_info_da[donor_key] == donor_id].tolist()
            )
            col_idx = [i for i, name in enumerate(sample_values) if name in donor_samples]
            if len(col_idx) == 1:
                single_sample_donors.append(donor_id)
            elif len(col_idx) > 1:
                donor_column_groups.append(col_idx)
        if single_sample_donors:
            raise ValueError(
                "donor_key centering needs at least two analysed samples per donor; "
                f"donors with one sample: {single_sample_donors}. A single-sample "
                "donor would keep uncentered log-densities and be pooled with "
                "centered ones. Drop them with sample_subset or omit donor_key."
            )

    # Per-cell modality-weight models cannot accept held-out adata in get_latent_representation.
    # DA is always run on the training data, so adata=None (→ model.adata) is equivalent.
    _repr_adata = (
        None
        if getattr(getattr(model, "module", None), "modality_weights", None) == "cell"
        else adata
    )
    if n_mc_samples == 1:
        us_list = [
            model.get_latent_representation(
                _repr_adata, give_mean=True, give_z=False, batch_size=batch_size
            )
        ]
    else:
        us_list = [
            model.get_latent_representation(
                _repr_adata, give_mean=False, give_z=False, batch_size=batch_size
            )
            for _ in range(n_mc_samples)
        ]

    log_probs = []
    knn_counts = knn_n_ref = None
    n_splits = max(int(np.ceil(adata.n_obs / batch_size)), 1)
    if density_estimator == "knn":
        sample_codes = pd.Index(sample_values).get_indexer(adata.obs[sample_key].to_numpy())
        knn_counts, knn_n_ref = _knn_sample_counts(
            us_list[0], sample_codes, len(sample_values), int(knn_k), omit_original_sample
        )
        sample_sequence = []
        log_probs = [_knn_log_density(knn_counts, knn_n_ref)]
    else:
        sample_sequence = sample_values
    for sample_name in tqdm(sample_sequence, desc="Aggregated posterior log probabilities"):
        ap = model.get_aggregated_posterior(
            adata=adata,
            sample=sample_name,
            batch_size=batch_size,
        )
        if density_estimator == "scale_floor":
            base = ap.component_distribution.base_dist
            ap = MixtureSameFamily(
                ap.mixture_distribution,
                Independent(Normal(base.loc, base.scale.clamp(min=scale_floor)), 1),
            )
        _dtype = ap.component_distribution.base_dist.loc.dtype
        _device = ap.component_distribution.base_dist.loc.device
        # Component j of ``ap`` is the j-th cell (in adata order) of this sample. With
        # omit_original_sample=False a cell of this sample must not score against its own
        # component, so ``own_component`` maps each cell to that component (-1 = not in sample).
        own_component = None
        if not omit_original_sample:
            own_component = np.full(adata.n_obs, -1, dtype=np.int64)
            in_sample = np.where(adata.obs[sample_key].to_numpy() == sample_name)[0]
            own_component[in_sample] = np.arange(in_sample.shape[0])
        draw_log_probs = []
        for us in us_list:
            chunk_lps = []
            for rows, u_rep in zip(
                np.array_split(np.arange(adata.n_obs), n_splits),
                np.array_split(us, n_splits),
                strict=True,
            ):
                u_tensor = torch.as_tensor(u_rep, dtype=_dtype, device=_device)
                lp = ap.log_prob(u_tensor).detach().cpu().numpy()
                if own_component is not None:
                    lp = _leave_one_cell_out(ap, u_tensor, lp, own_component[rows])
                chunk_lps.append(lp)
            draw_log_probs.append(np.concatenate(chunk_lps, axis=0))  # (n_cells,)
        if n_mc_samples == 1:
            cell_lp = draw_log_probs[0]
        else:
            cell_lp = logsumexp(np.stack(draw_log_probs, axis=0), axis=0) - np.log(n_mc_samples)
        log_probs.append(cell_lp[:, None])

    log_probs = log_probs[0] if density_estimator == "knn" else np.concatenate(log_probs, axis=1)

    # Scores used for covariate aggregation. ``log_probs`` stays uncentered in the output.
    agg_log_probs = log_probs
    if donor_column_groups:
        # Within-donor centering: subtract the per-donor mean across that donor's
        # samples so that donor blocks are absorbed before covariate aggregation.
        # A NaN sample (a one-cell own sample) is left out of the donor mean.
        agg_log_probs = log_probs.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for col_idx in donor_column_groups:
                agg_log_probs[:, col_idx] -= np.nanmean(
                    agg_log_probs[:, col_idx], axis=1, keepdims=True
                )

    coords = {"cell_name": adata.obs_names.to_numpy(), "sample": np.asarray(sample_values)}
    data_vars = {"log_probs": (["cell_name", "sample"], log_probs)}
    log_probs_ds = xr.Dataset(data_vars, coords=coords)

    if not sample_cov_keys:
        return log_probs_ds

    for key in sample_cov_keys:
        n_cov_values = len(adata.obs[key].unique())
        if n_cov_values > len(sample_values) / 2:
            warnings.warn(
                f"The covariate '{key}' has {n_cov_values} unique values across "
                f"{len(sample_values)} samples; DA covariates should be discrete.",
                UserWarning,
                stacklevel=2,
            )

    sample_info = adata.obs[[sample_key, *sample_cov_keys]].drop_duplicates(subset=sample_key)
    agg_ds = xr.DataArray(
        agg_log_probs,
        dims=["cell_name", "sample"],
        coords={"cell_name": adata.obs_names.to_numpy(), "sample": np.asarray(sample_values)},
    )

    empty_group = np.zeros(adata.n_obs, dtype=bool)  # cells with a group that has no reference

    def aggregate(samples: np.ndarray) -> np.ndarray:
        if density_estimator == "knn":
            cols = pd.Index(sample_values).get_indexer(samples)
            return _knn_log_density(
                knn_counts[:, cols].sum(axis=1), knn_n_ref[:, cols].sum(axis=1)
            )
        sample_log_probs = agg_ds.loc[{"sample": samples}].values
        if omit_original_sample:
            sample_one_hot = np.zeros((adata.n_obs, len(samples)))
            for i, sample_name in enumerate(samples):
                sample_one_hot[adata.obs[sample_key].to_numpy() == sample_name, i] = 1
            n_ref = (1 - sample_one_hot).sum(axis=1)
            sample_log_probs = np.where(sample_one_hot, -np.inf, sample_log_probs)
            result = logsumexp(sample_log_probs, axis=1) - np.log(np.maximum(n_ref, 1))
            result[n_ref == 0] = np.nan  # group is only the cell's own sample
        else:
            # NaN marks a sample whose leave-one-cell-out density is empty (a one-cell own
            # sample); it carries no weight, and a group with no usable sample is NaN.
            usable = ~np.isnan(sample_log_probs)
            n_ref = usable.sum(axis=1)
            result = logsumexp(np.where(usable, sample_log_probs, -np.inf), axis=1) - np.log(
                np.maximum(n_ref, 1)
            )
            result[n_ref == 0] = np.nan
        empty_group[n_ref == 0] = True
        return result

    cov_log_probs = {}
    cov_log_enrichs = {}
    if density_estimator == "knn" and sample_cov_keys:
        # The fixed 0.5 pseudocount favours a group whose expected neighbour count is
        # below about 1.
        cells_per_sample = adata.obs[sample_key].value_counts()
        n_reference = cells_per_sample.reindex(sample_values).fillna(0).sum()
        small_groups = []
        for key in sample_cov_keys:
            for value in sample_info[key].unique():
                samples = sample_info.loc[sample_info[key] == value, sample_key]
                n_group = cells_per_sample.reindex(samples).fillna(0).sum()
                if 0 < knn_k * n_group / n_reference < 1:
                    small_groups.append(f"{key}={value}")
        if small_groups:
            warnings.warn(
                "knn group scores are biased towards groups whose expected neighbour "
                f"count knn_k * n_g / N is below 1: {small_groups}. Increase knn_k or "
                "use density_estimator='posterior'.",
                UserWarning,
                stacklevel=2,
            )
    for key in sample_cov_keys:
        per_value = {}
        per_enrich = {}
        for value in sample_info[key].unique():
            samples = sample_info.loc[sample_info[key] == value, sample_key].to_numpy()
            if sample_subset is not None:
                samples = np.intersect1d(samples, np.asarray(sample_subset))
            if len(samples) == 0:
                continue
            val_log_probs = aggregate(samples)
            per_value[value] = val_log_probs
            if compute_log_enrichment:
                rest = np.setdiff1d(np.asarray(sample_values), samples)
                if len(rest) == 0:
                    continue
                per_enrich[value] = val_log_probs - aggregate(rest)
        values = list(per_value)
        cov_log_probs[key] = (values, np.vstack([per_value[v] for v in values]).T)
        if per_enrich:
            enrich_values = list(per_enrich)
            cov_log_enrichs[key] = (
                enrich_values,
                np.vstack([per_enrich[v] for v in enrich_values]).T,
            )

    if empty_group.any():
        warnings.warn(
            f"{int(empty_group.sum())} cells have a group with no reference sample once "
            "their own cell or sample is left out; their log probabilities and "
            "enrichments for that group are NaN.",
            UserWarning,
            stacklevel=2,
        )

    coords = {"cell_name": adata.obs_names.to_numpy(), "sample": np.asarray(sample_values)}
    for key, (values, _) in cov_log_probs.items():
        coords[key] = np.asarray(values)

    data_vars = {"log_probs": (["cell_name", "sample"], log_probs)}
    for key, (_, values) in cov_log_probs.items():
        data_vars[f"{key}_log_probs"] = (["cell_name", key], values)
    if compute_log_enrichment:
        for key, (values, arr) in cov_log_enrichs.items():
            enrich_dim = f"{key}_enrichment"
            coords[enrich_dim] = np.asarray(values)
            data_vars[f"{key}_log_enrichs"] = (["cell_name", enrich_dim], arr)

    return xr.Dataset(data_vars, coords=coords)


def _construct_design_matrix(
    sample_info,
    sample_cov_keys: list[str],
    normalize_design_matrix: bool = True,
    donor_key: str | None = None,
) -> tuple:
    """Build a float design matrix from sample-level metadata.

    Parameters
    ----------
    sample_info
        DataFrame indexed by sample name; must contain all ``sample_cov_keys``
        (and ``donor_key`` if provided).
    sample_cov_keys
        Column names for fixed-effect covariates.
    normalize_design_matrix
        Min-max normalize each fixed-effect column to [0, 1].
    donor_key
        Optional column to add donor dummy columns after the fixed effects as
        nuisance covariates (not normalized, not reported in output betas).

    Returns
    -------
    Xmat : ``torch.FloatTensor`` of shape ``(n_samples, n_covariates)``
    covariate_names : ``np.ndarray`` of covariate names
    n_fixed : int — number of reported fixed-effect columns
    """
    import pandas as pd

    parts: list[np.ndarray] = []
    names: list[str] = []

    for key in sample_cov_keys:
        cov = sample_info[key]
        # pandas >= 3 gives string columns a "str" dtype, not object; treat every
        # non-numeric column (object, string, categorical) as categorical.
        if hasattr(cov, "cat") or not pd.api.types.is_numeric_dtype(cov):
            if hasattr(cov, "cat"):
                cov = cov.cat.remove_unused_categories()
            dummies = pd.get_dummies(cov, drop_first=True)
            parts.append(dummies.values.astype(np.float32))
            names.extend([f"{key}_{c}" for c in dummies.columns])
        else:
            parts.append(cov.values[:, None].astype(np.float32))
            names.append(key)

    if parts:
        xmat_fixed = np.concatenate(parts, axis=1)
    else:
        xmat_fixed = np.zeros((len(sample_info), 0), np.float32)
    names_fixed = np.array(names, dtype=str)

    if normalize_design_matrix and xmat_fixed.shape[1] > 0:
        xmin = xmat_fixed.min(axis=0)
        xmax = xmat_fixed.max(axis=0)
        scale = np.where(xmax - xmin > 1e-6, xmax - xmin, 1.0)
        xmat_fixed = (xmat_fixed - xmin) / scale

    n_fixed = xmat_fixed.shape[1]

    if donor_key is not None:
        # donor_key may equal the DataFrame's index name (when donor_key == model.sample_key).
        if donor_key == sample_info.index.name:
            cov = sample_info.index.to_series()
        else:
            cov = sample_info[donor_key]
        if hasattr(cov, "cat"):
            cov = cov.cat.remove_unused_categories()
        donor_dummies = pd.get_dummies(cov, drop_first=True)
        donor_mat = donor_dummies.values.astype(np.float32)
        donor_names = np.array([f"{donor_key}_{c}" for c in donor_dummies.columns], dtype=str)
        if n_fixed > 0:
            xmat = np.concatenate([xmat_fixed, donor_mat], axis=1)
            names_all = np.concatenate([names_fixed, donor_names])
        else:
            xmat = donor_mat
            names_all = donor_names
    else:
        xmat = xmat_fixed
        names_all = names_fixed

    return torch.tensor(xmat, dtype=torch.float32), names_all, n_fixed


def _wald_pvalue(statistic: np.ndarray, df: np.ndarray) -> np.ndarray:
    """Chi-square survival function in float64: a large statistic gives a small p, not 0."""
    from scipy.stats import chi2

    return chi2.sf(np.asarray(statistic, dtype=np.float64), np.asarray(df, dtype=np.float64))


def _differential_expression(
    model,
    *,
    adata=None,
    sample_cov_keys: list[str],
    sample_subset: list[str] | None = None,
    indices=None,
    batch_size: int = 128,
    normalize_design_matrix: bool = True,
    mc_samples: int = 50,
    filter_inadmissible_samples: bool = False,
    store_lfc: bool = False,
    donor_key: str | None = None,
    delta: float | None = 0.3,
    lambd: float = 0.0,
    store_baseline: bool = False,
    eps_lfc: float = 1e-6,
    use_vmap: bool = False,
    validated_sample_order: list | None = None,
    **filter_samples_kwargs,
) -> xr.Dataset:
    """MrVI-style latent-space differential expression for Mr multimodal models.

    Fits a cell-specific weighted least-squares linear model on the sample-
    specific residual ``eps_d = z_d - z_base`` (the donor latent shift),
    following the approach of MrVI (Boyeau et al., 2023).

    Parameters
    ----------
    model
        A trained ``MrTotalVI`` or ``MrMultiVI`` instance.
    adata
        AnnData/MuData to compute DE on.  Defaults to the training data.
    sample_cov_keys
        Sample-level covariate column names in ``adata.obs``.
    sample_subset
        Restrict DE to these sample names only.
    indices
        Cell indices (default: all cells).
    batch_size
        Dataloader batch size.
    normalize_design_matrix
        Min-max normalize each covariate column to [0, 1].
    mc_samples
        Monte-Carlo draws from ``q(u)``.  ``1`` uses the posterior mean.
    filter_inadmissible_samples
        Weight out outlier cell-sample pairs via aggregated-posterior scores.
    store_lfc
        If ``True``, compute and store gene/protein log2-fold-changes (LFC) in
        the returned dataset.  Requires ``model.module`` to implement
        ``compute_h_from_x_eps``.  LFC is batch-marginalized by weighting over
        unique batch values observed in each mini-batch. ``lfc_std`` is the square
        root of the weighted mean of per-batch MC variances, as in upstream MrVI. It
        ignores the between-batch term and the covariance between batches that share
        MC draws, so it is not the SD of the batch-marginal LFC; it is never smaller.
    donor_key
        Column in ``adata.obs`` identifying the donor.  Adds donor dummy
        columns as nuisance covariates; only fixed-effect betas are returned.
    delta
        Effect-size threshold for the posterior-DE probability (PDE):
        ``pde = P(|lfc| >= delta)``.  Only used when ``store_lfc=True``.
        Set to ``None`` to skip PDE computation.
    lambd
        L2 regularisation added to ``X^T M X`` before inversion.
    store_baseline
        When ``store_lfc=True``, also store the batch-marginalized baseline
        expression ``h(x, eps_mean)`` (the null decode) as
        ``baseline_expression`` in the returned dataset.
    eps_lfc
        Small constant added to expression values before taking log2 to avoid
        log(0).  Default ``1e-6``.
    use_vmap
        Currently ignored (reserved for future opt-in vmap acceleration).
        The LFC path always uses explicit Python loops, which are safe with
        BatchNorm-based decoders.
    **filter_samples_kwargs
        Forwarded to :func:`get_outlier_cell_sample_pairs`.

    Notes
    -----
    ``pvalue`` is the chi-square survival function of the Wald statistic, in float64.
    ``padj`` is Benjamini-Hochberg over one family: every analysed cell times every fixed
    covariate. Per-cell tests share one model fit, so they are strongly dependent, and the
    adjusted values change with how many cells and covariates are analysed.
    """
    from scipy.stats import false_discovery_control

    from scvi import REGISTRY_KEYS as RK

    if store_lfc and not hasattr(model.module, "compute_h_from_x_eps"):
        raise NotImplementedError(
            "store_lfc requires model.module.compute_h_from_x_eps, "
            "which is not implemented for this module."
        )
    if not sample_cov_keys:
        raise ValueError("sample_cov_keys must contain at least one sample-level covariate.")
    if donor_key is not None:
        warnings.warn(
            "donor_key adds donor dummies as nuisance covariates.  This is only valid "
            "when each donor spans multiple levels of the sample_cov_keys covariates "
            "(e.g. multiple time points).  If each donor maps to exactly one condition "
            "the design is collinear and fixed-effect betas will be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    model._check_if_trained(warn=False)
    adata = model._validate_anndata(adata)

    if mc_samples < 1:
        raise ValueError("mc_samples must be >= 1.")
    _cov_to_check = list(sample_cov_keys)
    if donor_key and donor_key != model.sample_key:
        _cov_to_check.append(donor_key)
    _validate_sample_level_covariates(adata.obs, model.sample_key, _cov_to_check)

    n_sample = model.summary_stats.n_sample
    if validated_sample_order is not None:
        sample_order_kept = np.asarray(validated_sample_order)
        position = {sample: i for i, sample in enumerate(model.sample_order)}
        sample_indices_kept = [position[sample] for sample in sample_order_kept]
    else:
        sample_mask = (
            np.isin(model.sample_order, list(sample_subset))
            if sample_subset is not None
            else np.ones(n_sample, dtype=bool)
        )
        sample_order_kept = model.sample_order[sample_mask]
        sample_indices_kept = np.where(sample_mask)[0].tolist()
    n_sample_kept = len(sample_indices_kept)

    # Admissibility mask: (n_cells_total, n_sample_kept)
    if filter_inadmissible_samples:
        sample_key = model.sample_key
        outliers = get_outlier_cell_sample_pairs(
            model,
            adata=adata,
            sample_key=sample_key,
            **filter_samples_kwargs,
        )
        admissible_samples = (
            outliers["is_admissible"]
            .sel(sample=sample_order_kept)
            .values.astype(np.float32)
        )
    else:
        admissible_samples = np.ones((adata.n_obs, n_sample_kept), dtype=np.float32)

    # Design matrix from sample-level metadata.
    # Exclude donor_key from the column list when it equals model.sample_key — that
    # column will become the index after set_index(), so selecting it twice would
    # produce duplicate columns and a tuple MultiIndex.
    cov_cols = list(sample_cov_keys) + ([donor_key] if donor_key else [])
    cov_cols_for_select = [c for c in cov_cols if c != model.sample_key]
    sample_info = (
        adata.obs[[model.sample_key] + cov_cols_for_select]
        .drop_duplicates(subset=model.sample_key)
        .set_index(model.sample_key)
        .loc[sample_order_kept]
    )
    Xmat, Xmat_names, n_fixed = _construct_design_matrix(
        sample_info,
        list(sample_cov_keys),
        normalize_design_matrix=normalize_design_matrix,
        donor_key=donor_key,
    )
    n_covariates = Xmat.shape[1]
    device = model.device
    Xmat = Xmat.to(device)

    # The per-covariate Wald statistic multiplies betas by sqrt(X^T M X), as upstream MrVI
    # does, so a fixed covariate that is not orthogonal to another design column picks up
    # that column's effect. Only the sum over covariates is the joint Wald statistic.
    design_norm = torch.linalg.vector_norm(Xmat, dim=0).clamp(min=1e-12)
    design_cosine = (Xmat.T @ Xmat) / torch.outer(design_norm, design_norm)
    design_cosine.fill_diagonal_(0.0)
    if bool((design_cosine[:n_fixed].abs() > _CORRELATED_DESIGN_TOL).any()):
        warnings.warn(
            "Design columns are correlated: per-covariate effect_size, pvalue and padj "
            "mix effects between them (upstream MrVI behaviour); beta is unaffected and "
            "only the sum of effect_size over covariates is the joint Wald statistic.",
            UserWarning,
            stacklevel=2,
        )
    if filter_inadmissible_samples and not np.all(admissible_samples > 0):
        warnings.warn(
            "Rejected (inadmissible) samples still enter the per-cell centring and "
            "scaling of eps before the weighted fit (upstream MrVI behaviour), so they "
            "can still move beta and the Wald statistic.",
            UserWarning,
            stacklevel=2,
        )

    scdl = model._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

    beta_list: list[np.ndarray] = []
    effect_size_list: list[np.ndarray] = []
    pvalue_list: list[np.ndarray] = []
    all_cell_names: list[str] = []

    lfc_list: list[np.ndarray] = []
    lfc_std_list: list[np.ndarray] = []
    pde_list: list[np.ndarray] | None = [] if (store_lfc and delta is not None) else None
    baseline_list: list[np.ndarray] | None = [] if (store_lfc and store_baseline) else None

    was_training = model.module.training
    model.module.eval()
    try:
        with torch.inference_mode():
            for tensors in tqdm(scdl, desc="MrVI-style DE"):
                cell_idx = tensors[RK.INDICES_KEY].long().flatten().cpu().numpy()
                n_cells = len(cell_idx)
                all_cell_names.extend(adata.obs_names[cell_idx].tolist())

                inf_inputs = model.module._get_inference_input(tensors)
                base_out = model.module.inference(**inf_inputs)
                qu = base_out.get("qu")
                if qu is None:
                    raise RuntimeError(
                        "model.module.inference() did not return 'qu'; "
                        "cannot compute MC eps samples."
                    )

                def _eps_from_u(
                    u: torch.Tensor,
                    batch_n_cells: int = n_cells,
                ) -> torch.Tensor:
                    # For each kept sample d, compute eps_d = qz(u, d)[1].
                    # u is held fixed across donors: counterfactual substitution.
                    eps_list = []
                    for d_idx in sample_indices_kept:
                        cf = torch.full(
                            (batch_n_cells, 1),
                            d_idx,
                            dtype=torch.long,
                            device=u.device,
                        )
                        _, eps_d, _ = model.module.qz(u, cf, deterministic=True)
                        eps_list.append(eps_d)
                    # (n_cells, n_sample_kept, n_latent)
                    return torch.stack(eps_list, dim=1)

                if mc_samples == 1:
                    u_samples = [qu.mean]
                    eps_batch = _eps_from_u(qu.mean).unsqueeze(0)
                else:
                    u_samples = [qu.rsample() for _ in range(mc_samples)]
                    eps_batch = torch.stack(
                        [_eps_from_u(u_i) for u_i in u_samples], dim=0
                    )
                # eps_batch: (mc, n_cells, n_sample_kept, n_latent)

                # Admissibility for this batch
                admiss = torch.from_numpy(admissible_samples[cell_idx]).to(device)
                # (n_cells, n_sample_kept)
                n_per_cell = admiss.sum(1)  # (n_cells,)

                # X^T M X per cell: einsum over sample dimension s
                # Xmat: (n_sample_kept, n_cov)  →  s,k
                # admiss: (n_cells, n_sample_kept)  →  n,s
                xtmx = (
                    torch.einsum("sk,ns,sl->nkl", Xmat, admiss, Xmat)
                    + lambd * torch.eye(n_covariates, device=device).unsqueeze(0)
                )  # (n_cells, n_cov, n_cov)
                prefactor = _sqrtm_batch(xtmx)  # (n_cells, n_cov, n_cov)
                inv_ = torch.vmap(torch.linalg.pinv)(xtmx)  # (n_cells, n_cov, n_cov)
                # Amat = (X^T M X)^{-1} X^T M  →  (n_cells, n_cov, n_sample_kept)
                Amat = torch.einsum("nkj,sj,ns->nks", inv_, Xmat, admiss)

                # Standardise eps across sample dimension (dim=2)
                eps_mean = eps_batch.mean(dim=2, keepdim=True)
                eps_std = eps_batch.std(dim=2, keepdim=True)
                eps_norm = (eps_batch - eps_mean) / (eps_std + 1e-6)
                # (mc, n_cells, n_sample_kept, n_latent)

                # OLS per cell per MC draw
                # Amat: (n_cells, n_cov, n_sample)  →  n,k,s
                # eps_norm: (mc, n_cells, n_sample, n_latent)  →  a,n,s,d
                # betas: (mc, n_cells, n_cov, n_latent)
                betas = torch.einsum("nks,ansd->ankd", Amat, eps_norm)

                # Wald: prefactor: (n_cells, n_cov, n_cov)  →  n,l,k
                # betas_norm: (mc, n_cells, n_cov, n_latent)  →  a,n,l,d
                betas_norm = torch.einsum("nlk,ankd->anld", prefactor, betas)
                # ts: mean over mc, sum over latent → (n_cells, n_cov)
                ts = (betas_norm**2).sum(-1).mean(0)

                # Chi2 p-values with df = n_admissible_per_cell
                df = n_per_cell.clamp(min=1).unsqueeze(-1).expand_as(ts)
                pvals = _wald_pvalue(ts.cpu().numpy(), df.cpu().numpy())  # (n_cells, n_cov)

                # Un-standardise betas and average over MC draws
                betas_rescaled = (betas * eps_std).mean(0)  # (n_cells, n_cov, n_latent)

                beta_list.append(betas_rescaled[:, :n_fixed, :].cpu().numpy())
                effect_size_list.append(ts[:, :n_fixed].cpu().numpy())
                pvalue_list.append(pvals[:, :n_fixed])

                # ---- gene/protein LFC block ----
                if store_lfc:
                    # betas_eps: per-draw betas in original eps space
                    # shape (mc, n_cells, n_fixed, n_latent)
                    betas_eps = (betas * eps_std)[:, :, :n_fixed, :]

                    # null eps position: mean over MC draws and sample dim
                    # eps_mean: (mc, n_cells, 1, n_latent) → average to (n_cells, n_latent)
                    eps_mean_cell = eps_mean.mean(0).squeeze(1)  # (n_cells, n_latent)

                    # Build kwargs for compute_h_from_x_eps from inference inputs;
                    # batch_index is overridden per counterfactual batch below.
                    h_kwargs = dict(inf_inputs)
                    # _get_inference_input does not return extra_eps or cf_sample,
                    # so h_kwargs can be passed directly to compute_h_from_x_eps.

                    # Batch-marginalised LFC: decode under every registered batch with
                    # uniform weights, as the MrVI reference does (mrvi_torch/_model.py
                    # default branch). Weighting by the batches present in the current
                    # mini-batch made LFC depend on batch_size and cell order.
                    batch_index_obs = inf_inputs["batch_index"]  # (n_cells, 1)
                    n_batch_total = int(model.summary_stats.n_batch)
                    unique_batches = torch.arange(
                        n_batch_total,
                        device=batch_index_obs.device,
                        dtype=batch_index_obs.dtype,
                    )
                    batch_weights = torch.full(
                        (n_batch_total,),
                        1.0 / n_batch_total,
                        device=batch_index_obs.device,
                    )

                    # Accumulators (initialized lazily on first batch value)
                    lfc_wsum: torch.Tensor | None = None   # (n_fixed, n_cells, n_features)
                    lfc_var_wsum: torch.Tensor | None = None
                    pde_wsum: torch.Tensor | None = None
                    baseline_wsum: torch.Tensor | None = None

                    for b_idx, b_val in enumerate(unique_batches):
                        w_b = batch_weights[b_idx]
                        batch_cf = torch.full_like(batch_index_obs, b_val.item())
                        h_kwargs_b = dict(h_kwargs)
                        h_kwargs_b["batch_index"] = batch_cf

                        # Cache inference aux once per batch value.  library_gene (and
                        # qz_m for MrMultiVI) depend only on x/batch_index, not on
                        # u_anchor, so they are identical for all mc_samples draws.
                        # Without this cache, compute_h_from_x_eps would re-run the
                        # full encoder mc_samples*(1+n_fixed) times per batch value.
                        _lfc_aux = None
                        if hasattr(model.module, "_infer_lfc_aux"):
                            _lfc_aux = model.module._infer_lfc_aux(**h_kwargs_b)

                        # CRN: null decode per MC draw, sharing u_samples[mc_idx]
                        # with the contrast decode below.  x_0 varies with u so
                        # lfc_std now captures u-posterior uncertainty (not just
                        # regression uncertainty in beta).
                        x0_mc = [] if store_baseline else None
                        log_x0_mc = []
                        for mc_idx in range(mc_samples):
                            x_0 = model.module.compute_h_from_x_eps(
                                extra_eps=eps_mean_cell,
                                u_anchor=u_samples[mc_idx],
                                _lfc_aux=_lfc_aux,
                                **h_kwargs_b,
                            )  # (n_cells, n_features)
                            if x0_mc is not None:
                                x0_mc.append(x_0)
                            log_x0_mc.append(torch.log2(x_0 + eps_lfc))

                        if lfc_wsum is None:
                            n_features_lfc = log_x0_mc[0].shape[-1]
                            lfc_wsum = torch.zeros(
                                n_fixed, n_cells, n_features_lfc, device=device
                            )
                            lfc_var_wsum = torch.zeros_like(lfc_wsum)
                            if delta is not None:
                                pde_wsum = torch.zeros_like(lfc_wsum)
                            if store_baseline:
                                baseline_wsum = torch.zeros(
                                    n_cells, n_features_lfc, device=device
                                )

                        if store_baseline:
                            baseline_wsum += w_b * torch.stack(x0_mc, 0).mean(0)

                        # Contrast decode per covariate × MC draw; share u_anchor (CRN)
                        lfc_mc_cov_list = []
                        for k in range(n_fixed):
                            lfc_mc_k = []
                            for mc_idx in range(mc_samples):
                                extra = (
                                    betas_eps[mc_idx, :, k, :] + eps_mean_cell
                                )  # (n_cells, n_latent)
                                x_1 = model.module.compute_h_from_x_eps(
                                    extra_eps=extra,
                                    u_anchor=u_samples[mc_idx],
                                    _lfc_aux=_lfc_aux,
                                    **h_kwargs_b,
                                )  # (n_cells, n_features)
                                lfc_mc_k.append(
                                    torch.log2(x_1 + eps_lfc) - log_x0_mc[mc_idx]
                                )
                            lfc_mc_cov_list.append(
                                torch.stack(lfc_mc_k, 0)
                            )  # (mc, n_cells, n_features)
                        lfc_mc_cov = torch.stack(
                            lfc_mc_cov_list, 0
                        )  # (n_fixed, mc, n_cells, n_features)

                        lfc_wsum += w_b * lfc_mc_cov.mean(1)
                        # correction=0 (population variance) because mc_samples=1 is valid;
                        # correction=1 yields NaN via 0/0 when mc_samples=1.
                        # Note: this accumulates within-batch MC variance only; the
                        # between-batch term sum(w*(mu_b-mu)^2) is omitted — matching the
                        # MRVI reference (mrvi_torch/_model.py:1444-1448), so lfc_std is
                        # slightly conservative when batches have different mean LFC.
                        lfc_var_wsum += w_b * lfc_mc_cov.var(1, correction=0)
                        if delta is not None:
                            pde_wsum += w_b * (lfc_mc_cov.abs() >= delta).float().mean(1)

                    # Permute from (n_fixed, n_cells, n_features) to
                    # (n_cells, n_fixed, n_features).
                    lfc_list.append(lfc_wsum.permute(1, 0, 2).cpu().numpy())
                    lfc_std_list.append(
                        torch.sqrt(lfc_var_wsum).permute(1, 0, 2).cpu().numpy()
                    )
                    if delta is not None:
                        pde_list.append(pde_wsum.permute(1, 0, 2).cpu().numpy())
                    if store_baseline:
                        baseline_list.append(baseline_wsum.cpu().numpy())
    finally:
        model.module.train(was_training)

    beta = np.concatenate(beta_list, 0)
    effect_size = np.concatenate(effect_size_list, 0)
    pvalue = np.concatenate(pvalue_list, 0)
    padj = false_discovery_control(pvalue.flatten(), method="bh").reshape(pvalue.shape)

    data_vars: dict = {
        "beta": (["cell_name", "covariate", "latent_dim"], beta),
        "effect_size": (["cell_name", "covariate"], effect_size),
        "pvalue": (["cell_name", "covariate"], pvalue),
        "padj": (["cell_name", "covariate"], padj),
    }
    coords: dict = {
        "cell_name": np.asarray(all_cell_names),
        "covariate": Xmat_names[:n_fixed],
        "latent_dim": np.arange(beta.shape[-1]),
    }

    if store_lfc:
        lfc = np.concatenate(lfc_list, 0)          # (n_cells_total, n_fixed, n_features)
        lfc_std = np.concatenate(lfc_std_list, 0)
        data_vars["lfc"] = (["cell_name", "covariate", "feature"], lfc)
        data_vars["lfc_std"] = (["cell_name", "covariate", "feature"], lfc_std)
        coords["feature"] = np.arange(lfc.shape[-1])
        if pde_list is not None:
            pde = np.concatenate(pde_list, 0)
            data_vars["pde"] = (["cell_name", "covariate", "feature"], pde)
        if baseline_list is not None:
            baseline = np.concatenate(baseline_list, 0)
            data_vars["baseline_expression"] = (["cell_name", "feature"], baseline)

    return xr.Dataset(data_vars, coords=coords)


def get_outlier_cell_sample_pairs(
    model,
    *,
    adata=None,
    sample_key: str,
    subsample_size: int | None = 5_000,
    quantile_threshold: float = 0.05,
    admissibility_threshold: float = 0.0,
    batch_size: int = 256,
    random_state: int | None = None,
) -> xr.Dataset:
    """Compute admissible cell-sample pairs using aggregated posteriors over ``u``.

    One global threshold is used: the ``quantile_threshold`` quantile of leave-one-out
    scores pooled over all samples. A cell scored against its own sample excludes its own
    component, so about ``quantile_threshold`` of own-sample pairs fail by design (0.95
    admissible at the default). The private DE filter therefore drops about that share of
    cells from their own sample. ``get_counterfactual_latent`` uses a different rule (one
    threshold per target sample), so its ``admissible`` flag is a different quantity.

    A registered sample with no cells in ``adata`` gets a NaN, inadmissible column and adds
    no threshold scores. A sample whose mixture has one component (a one-cell sample, or
    ``subsample_size=1``) gives that cell a NaN, inadmissible score for its own sample and
    adds no threshold score. ``random_state`` seeds the per-sample subsample; ``None`` uses
    :attr:`scvi.settings.seed` when set, otherwise the subsample is not reproducible.
    """
    rng = np.random.default_rng(random_state if random_state is not None else settings.seed)
    adata = model._validate_anndata(adata)
    _repr_adata = (
        None
        if getattr(getattr(model, "module", None), "modality_weights", None) == "cell"
        else adata
    )
    us = model.get_latent_representation(
        _repr_adata,
        give_mean=True,
        give_z=False,
        batch_size=batch_size,
    )

    sample_values = list(model.sample_order) if hasattr(model, "sample_order") else list(
        adata.obs[sample_key].unique()
    )
    log_probs = []
    thresholds = []
    n_splits = max(int(np.ceil(adata.n_obs / batch_size)), 1)
    for sample_name in tqdm(sample_values, desc="Outlier cell-sample scores"):
        sample_idxs = np.where(adata.obs[sample_key].to_numpy() == sample_name)[0]
        if sample_idxs.size == 0:
            # A registered sample absent from this adata has no mixture: every pair is
            # NaN and inadmissible, and it adds no threshold scores.
            log_probs.append(np.full((us.shape[0], 1), np.nan, dtype=np.float32))
            continue
        if subsample_size is not None and sample_idxs.shape[0] > subsample_size:
            # Sorted, so component j is the j-th index whatever order the loader keeps.
            sample_idxs = np.sort(rng.choice(sample_idxs, size=subsample_size, replace=False))

        ap = model.get_aggregated_posterior(
            adata=adata,
            indices=sample_idxs,
            batch_size=batch_size,
        )
        in_sample = torch.as_tensor(
            us[sample_idxs],
            dtype=ap.component_distribution.base_dist.loc.dtype,
            device=ap.component_distribution.base_dist.loc.device,
        )
        # A one-component mixture has no leave-one-out score, so it adds nothing to the
        # pooled threshold (its 1x1 entry would be the cell's own self-density).
        if sample_idxs.shape[0] > 1:
            in_comp_log_probs = ap.component_distribution.log_prob(in_sample.unsqueeze(-2))
            thresholds.append(
                _rowwise_max_excluding_diagonal(in_comp_log_probs).detach().cpu()
            )

        # A cell inside this sample's subsample must not match its own component,
        # mirroring the leave-one-out thresholds above.
        own_component = np.full(us.shape[0], -1, dtype=np.int64)
        own_component[sample_idxs] = np.arange(sample_idxs.shape[0])

        sample_log_probs = []
        for rows in np.array_split(np.arange(us.shape[0]), n_splits):
            u_tensor = torch.as_tensor(
                us[rows],
                dtype=ap.component_distribution.base_dist.loc.dtype,
                device=ap.component_distribution.base_dist.loc.device,
            )
            comp_log_probs = ap.component_distribution.log_prob(u_tensor.unsqueeze(-2))
            chunk_own = own_component[rows]
            hit = np.nonzero(chunk_own >= 0)[0]
            if hit.size:
                comp_log_probs[
                    torch.as_tensor(hit, device=comp_log_probs.device),
                    torch.as_tensor(chunk_own[hit], device=comp_log_probs.device),
                ] = -float("inf")
            sample_log_probs.append(comp_log_probs.max(dim=1, keepdim=True).values.cpu().numpy())
        sample_scores = np.concatenate(sample_log_probs, axis=0)
        if sample_idxs.shape[0] == 1:
            # The only component is the cell itself: no independent reference, so NaN
            # (inadmissible), as in differential_abundance.
            sample_scores[sample_idxs] = np.nan
        log_probs.append(sample_scores)

    if not thresholds:
        raise ValueError(
            f"No registered sample of '{sample_key}' has at least 2 cells in adata; "
            "leave-one-out thresholds are undefined."
        )
    threshold_values = torch.cat(thresholds).numpy()
    global_threshold = np.quantile(threshold_values, q=quantile_threshold)
    log_probs = np.concatenate(log_probs, axis=1)
    log_ratios = log_probs - global_threshold

    coords = {"cell_name": adata.obs_names.to_numpy(), "sample": np.asarray(sample_values)}
    return xr.Dataset(
        {
            "log_probs": (["cell_name", "sample"], log_probs),
            "log_ratios": (["cell_name", "sample"], log_ratios),
            "is_admissible": (
                ["cell_name", "sample"],
                log_ratios > admissibility_threshold,
            ),
        },
        coords=coords,
    )
