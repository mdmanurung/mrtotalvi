"""Ambient ADT calibration from cell-free droplets (ADR-0011, PR 1).

The profile follows scAR's ``setup_anndata`` (Novartis/scar, ``scar/main/_setup.py``):
droplets with at least ``min_raw_counts`` counts that are not called cells are scored
under a multinomial with the current ambient profile; droplets whose joint
log-probability is at least ``log(prob) * n_features`` are cell-free; the profile is
re-estimated from them, ``iterations`` times. scAR pools all droplets into one profile
and applies no shrinkage, and so does this module (ADR-0011, C4 and C6).

The per-protein log statistics are DSB's step-I ambient statistics (niaid/dsb,
``R/dsb.r``), kept on two scales: ``log1p`` seeds the totalVI background prior
(C1, C2) and ``log(x + 10)`` feeds the optional z-scored encoder input (C12).
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

MIN_CELL_FREE_DROPLETS = 50
DSB_PSEUDOCOUNT = 10.0


def feature_digest(names) -> str:
    """Return a SHA-256 digest of an ordered feature-name list."""
    joined = "\x1f".join(str(n) for n in names)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


@dataclass
class AmbientProfile:
    """Pooled ambient ADT calibration with provenance.

    Attributes
    ----------
    feature_names
        Ordered protein names the statistics refer to.
    profile
        Ambient composition, shape ``(n_features,)``, sums to one.
    log1p_mean, log1p_sd
        Per-protein mean and SD of ``log1p`` counts over cell-free droplets.
    dsb_log_mean, dsb_log_sd
        Per-protein mean and SD of ``log(x + 10)`` over cell-free droplets.
    n_cell_free
        Number of droplets called cell-free in the last iteration.
    settings
        Algorithm settings, for provenance.
    """

    feature_names: list
    profile: np.ndarray
    log1p_mean: np.ndarray
    log1p_sd: np.ndarray
    dsb_log_mean: np.ndarray
    dsb_log_sd: np.ndarray
    n_cell_free: int
    settings: dict = field(default_factory=dict)

    @property
    def digest(self) -> str:
        return feature_digest(self.feature_names)

    def aligned(self, names) -> AmbientProfile:
        """Return the profile reordered to ``names``; fails on any mismatch of the name set."""
        names = [str(n) for n in names]
        own = [str(n) for n in self.feature_names]
        if sorted(names) != sorted(own) or len(set(names)) != len(names):
            raise ValueError(
                "Ambient profile features do not match the registered protein axis: "
                f"missing {sorted(set(names) - set(own))}, extra {sorted(set(own) - set(names))}."
            )
        order = np.array([own.index(n) for n in names])
        return AmbientProfile(
            feature_names=names,
            profile=self.profile[order],
            log1p_mean=self.log1p_mean[order],
            log1p_sd=self.log1p_sd[order],
            dsb_log_mean=self.dsb_log_mean[order],
            dsb_log_sd=self.dsb_log_sd[order],
            n_cell_free=self.n_cell_free,
            settings=dict(self.settings),
        )


def _log_stats(counts: np.ndarray, pseudocount: float) -> tuple[np.ndarray, np.ndarray]:
    logged = np.log(counts + pseudocount)
    return logged.mean(axis=0), logged.std(axis=0, ddof=1)


def estimate_adt_ambient(
    raw_counts,
    raw_barcodes,
    cell_barcodes,
    feature_names,
    adt_features=None,
    prob: float = 0.995,
    min_raw_counts: int = 2,
    iterations: int = 3,
) -> AmbientProfile:
    """Estimate a pooled ambient ADT profile from unfiltered droplets.

    Droplets are selected on every feature of ``raw_counts`` (RNA and ADT), as scAR
    does: its cutoff ``log(prob) * n_features`` is calibrated for a whole
    transcriptome and no droplet passes it on an ADT-only matrix. The profile and
    log statistics are then computed on ``adt_features`` only.

    Parameters
    ----------
    raw_counts
        Unfiltered droplets x features count matrix (dense or scipy sparse).
    raw_barcodes
        Barcodes of the rows of ``raw_counts``.
    cell_barcodes
        Barcodes called as cells; never used as cell-free droplets.
    feature_names
        Names of the columns of ``raw_counts``.
    adt_features
        The ADT features to profile; ``None`` uses every column.
    prob, min_raw_counts, iterations
        scAR ``setup_anndata`` defaults.
    """
    raw_barcodes = np.asarray(raw_barcodes).astype(str)
    if len(set(raw_barcodes)) != raw_barcodes.size:
        raise ValueError("raw_barcodes must be unique.")
    feature_names = [str(n) for n in feature_names]
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("feature_names must be unique.")
    adt_features = feature_names if adt_features is None else [str(n) for n in adt_features]
    missing = sorted(set(adt_features) - set(feature_names))
    if missing:
        raise ValueError(f"adt_features not in feature_names: {missing}.")
    counts = sp.csr_matrix(raw_counts, dtype=np.float64)
    if counts.shape != (raw_barcodes.size, len(feature_names)):
        raise ValueError(f"raw_counts has shape {counts.shape}; expected ({raw_barcodes.size}, {len(feature_names)}).")
    data = counts.data
    if not np.all(np.isfinite(data)) or (data.size and data.min() < 0) or not np.allclose(data, np.round(data)):
        raise ValueError("raw_counts must be finite non-negative integer counts.")
    if not 0.0 < prob < 1.0:
        raise ValueError("prob must lie in (0, 1).")
    if int(iterations) != iterations or iterations < 1:
        raise ValueError("iterations must be a positive integer.")

    is_cell = np.isin(raw_barcodes, np.asarray(cell_barcodes).astype(str))
    totals_all = np.asarray(counts.sum(axis=1)).ravel()
    keep = totals_all >= min_raw_counts
    candidates = counts[keep & ~is_cell]
    if candidates.shape[0] < MIN_CELL_FREE_DROPLETS:
        raise ValueError(
            f"Only {candidates.shape[0]} non-cell droplets pass min_raw_counts={min_raw_counts}; "
            f"at least {MIN_CELL_FREE_DROPLETS} are needed."
        )
    from scipy.special import gammaln

    pooled = np.asarray(counts[keep].sum(axis=0)).ravel()
    profile_all = pooled / pooled.sum()
    threshold = np.log(prob) * counts.shape[1]
    totals = np.asarray(candidates.sum(axis=1)).ravel()
    log_coef = (
        gammaln(totals + 1)
        - np.asarray(candidates.copy().tocsr()._with_data(gammaln(candidates.data + 1)).sum(axis=1)).ravel()
    )
    for _ in range(iterations):
        # Full multinomial log-pmf of each droplet under the current profile (scAR).
        log_prob = log_coef + candidates @ np.log(np.clip(profile_all, 1e-300, None))
        selected = log_prob >= threshold
        if selected.sum() < MIN_CELL_FREE_DROPLETS:
            raise ValueError(
                f"Only {int(selected.sum())} cell-free droplets at prob={prob} over "
                f"{counts.shape[1]} features; at least {MIN_CELL_FREE_DROPLETS} are needed. "
                "scAR's cutoff assumes droplets scored over the whole transcriptome."
            )
        summed = np.asarray(candidates[selected].sum(axis=0)).ravel()
        profile_all = summed / summed.sum()

    cols = [feature_names.index(n) for n in adt_features]
    cell_free = candidates[selected][:, cols].toarray()
    adt_sum = cell_free.sum(axis=0)
    if adt_sum.sum() <= 0:
        raise ValueError("Cell-free droplets carry no ADT counts.")
    profile = adt_sum / adt_sum.sum()
    absent = [n for n, total in zip(adt_features, adt_sum, strict=True) if total == 0]
    if absent:
        # Their ambient share is 0 and their log SD is 0, so a z-scored encoder input
        # for them is undefined (the model refuses it).
        warnings.warn(
            f"No ADT counts in any cell-free droplet for {absent}; their ambient statistics are not estimable.",
            UserWarning,
            stacklevel=2,
        )
    l1_mean, l1_sd = _log_stats(cell_free, 1.0)
    dsb_mean, dsb_sd = _log_stats(cell_free, DSB_PSEUDOCOUNT)
    stats = (profile, l1_mean, l1_sd, dsb_mean, dsb_sd)
    if not all(np.all(np.isfinite(x)) for x in stats):
        raise ValueError("Ambient statistics are not finite.")
    return AmbientProfile(
        feature_names=list(adt_features),
        profile=profile,
        log1p_mean=l1_mean,
        log1p_sd=l1_sd,
        dsb_log_mean=dsb_mean,
        dsb_log_sd=dsb_sd,
        n_cell_free=int(selected.sum()),
        settings={
            "method": "scar_iterative_pooled",
            "prob": prob,
            "min_raw_counts": min_raw_counts,
            "iterations": iterations,
            "n_selection_features": int(counts.shape[1]),
            "n_candidates": int(candidates.shape[0]),
        },
    )
