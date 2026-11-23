"""Planted CITE-seq simulator with known ambient, nonspecific and native ADT counts.

ADR-0011 benchmark. Every ADT count is NB(A + N + S, theta) with
A = c_i * a (free ambient, pooled composition a), N = softplus(eta + lam * h_i)
(cell-associated nonspecific binding, larger in an FcR-high cell type) and S the
native target signal (zero for isotypes and for absent markers). Cell-free droplets
draw Poisson totals spread by the same composition a.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

N_DONORS = 8  # donors 0-3 condition "ctrl", 4-7 condition "stim"
N_CELL_TYPES = 4  # type 3 is FcR-high (monocyte-like)
N_GENES = 200
N_TARGETS = 20
ISOTYPES = ["IgG1_iso", "IgG2a_iso"]
ABSENT = [17, 18, 19]  # targets with no native signal anywhere
INDUCED = [15, 16]  # low-abundance targets induced in stim donors, cell type 1
THETA = 20.0
N_TRANSCRIPTOME = 20000


def _nb(rng, mu, theta):
    return rng.poisson(rng.gamma(theta, mu / theta))


def simulate_citeseq(
    seed: int = 0,
    cells_per_donor: int = 400,
    n_empty: int = 6000,
    stim_ambient_fold: float = 1.0,
    batch_specific_ambient: bool = False,
):
    """Return ``(adata, truth, empties)``.

    ``adata`` has RNA counts in ``X`` and ADT counts in ``obsm["protein"]`` (DataFrame).
    ``truth`` holds the planted per-cell ``native``, ``ambient`` and ``nonspecific``
    means (cells x proteins DataFrames). ``empties`` is a dict with the unfiltered
    droplet matrix, barcodes and feature names for ambient calibration.
    """
    rng = np.random.default_rng(seed)
    proteins = [f"P{t:02d}" for t in range(N_TARGETS)] + ISOTYPES
    n_prot = len(proteins)
    target = np.array([p not in ISOTYPES for p in proteins])

    n_cells = N_DONORS * cells_per_donor
    donor = np.repeat(np.arange(N_DONORS), cells_per_donor)
    batch = donor % 2  # two capture wells, crossed with condition
    stim = donor >= 4
    cell_type = rng.integers(0, N_CELL_TYPES, n_cells)

    # RNA: cell-type programs plus a stim program in cell type 1.
    gene_base = rng.lognormal(-1.0, 1.0, N_GENES)
    type_fc = rng.lognormal(0.0, 0.8, (N_CELL_TYPES, N_GENES))
    stim_fc = np.ones(N_GENES)
    stim_fc[:10] = 4.0
    rna_mu = gene_base * type_fc[cell_type]
    rna_mu[(cell_type == 1) & stim] *= stim_fc
    lib = rng.lognormal(np.log(1500), 0.3, n_cells)
    rna_mu = rna_mu / rna_mu.sum(axis=1, keepdims=True) * lib[:, None]
    rna = _nb(rng, rna_mu, 5.0)

    # Native protein: each present target is positive in 1-2 cell types.
    native_type = np.zeros((N_CELL_TYPES, n_prot))
    for t in range(N_TARGETS):
        if t in ABSENT or t in INDUCED:
            continue
        pos = rng.choice(N_CELL_TYPES, size=rng.integers(1, 3), replace=False)
        native_type[pos, t] = rng.lognormal(np.log(150), 0.6, pos.size)
    native = native_type[cell_type] * rng.lognormal(0.0, 0.3, (n_cells, n_prot))
    for t in INDUCED:
        native[(cell_type == 1) & stim, t] = rng.lognormal(np.log(15), 0.3, ((cell_type == 1) & stim).sum())
    native[:, ~target] = 0.0

    # Free ambient: pooled composition, per-cell burden around 25 counts.
    composition = rng.dirichlet(np.full(n_prot, 2.0))
    composition[~target] *= 0.5
    composition /= composition.sum()
    # Optional second staining pool: well 1 gets a shifted ambient composition.
    compositions = np.stack([composition, composition])
    if batch_specific_ambient:
        other = rng.dirichlet(np.full(n_prot, 2.0))
        other[~target] *= 0.5
        compositions[1] = 0.3 * composition + 0.7 * other / other.sum()
    burden = rng.lognormal(np.log(25), 0.5, n_cells)
    # Confounding stress test: stim donors carry more free ambient.
    burden = burden * np.where(stim, stim_ambient_fold, 1.0)
    ambient = burden[:, None] * compositions[batch]

    # Nonspecific binding: higher h in the FcR-high cell type.
    h = rng.normal(0.0, 1.0, n_cells) + 1.5 * (cell_type == 3)
    eta = rng.normal(1.0, 0.4, n_prot)
    lam = rng.uniform(0.4, 0.9, n_prot)
    nonspecific = np.log1p(np.exp(eta[None, :] + lam[None, :] * h[:, None]))

    adt = _nb(rng, ambient + nonspecific + native, THETA)

    obs = pd.DataFrame(
        {
            "donor": pd.Categorical([f"d{d}" for d in donor]),
            "batch": pd.Categorical([f"well{b}" for b in batch]),
            "condition": pd.Categorical(np.where(stim, "stim", "ctrl"), categories=["ctrl", "stim"]),
            "cell_type": pd.Categorical([f"ct{c}" for c in cell_type]),
        },
        index=[f"cell{i}" for i in range(n_cells)],
    )
    adata = ad.AnnData(X=rna.astype(np.float32), obs=obs)
    adata.var_names = [f"G{g:03d}" for g in range(N_GENES)]
    adata.obsm["protein"] = pd.DataFrame(adt.astype(np.float32), index=obs.index, columns=proteins)

    truth = {
        key: pd.DataFrame(value, index=obs.index, columns=proteins)
        for key, value in {"native": native, "ambient": ambient, "nonspecific": nonspecific}.items()
    }
    truth["h"] = pd.Series(h, index=obs.index)
    truth["burden"] = pd.Series(burden, index=obs.index)
    truth["composition"] = pd.Series(composition, index=proteins)

    # Unfiltered droplets over a whole transcriptome (N_TRANSCRIPTOME genes) plus ADT,
    # because scAR's cell-free cutoff scales with the number of features.
    empty_batch = np.arange(n_empty) % 2
    empty_tot = rng.poisson(rng.lognormal(np.log(8), 0.5, n_empty))
    empty_adt = np.stack([rng.multinomial(n, compositions[b]) for n, b in zip(empty_tot, empty_batch, strict=True)])
    rna_comp = rng.dirichlet(np.full(N_TRANSCRIPTOME, 0.05))
    empty_rna_tot = rng.poisson(rng.lognormal(np.log(3), 0.6, n_empty))
    rows = np.repeat(np.arange(n_empty), empty_rna_tot)
    cols = rng.choice(N_TRANSCRIPTOME, size=rows.size, p=rna_comp)
    empty_rna = sp.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(n_empty, N_TRANSCRIPTOME))
    cell_rna = sp.hstack([sp.csr_matrix(rna), sp.csr_matrix((n_cells, N_TRANSCRIPTOME - N_GENES))])
    raw = sp.vstack(
        [sp.hstack([cell_rna, sp.csr_matrix(adt)]), sp.hstack([empty_rna, sp.csr_matrix(empty_adt)])]
    ).tocsr()
    raw_barcodes = list(obs.index) + [f"empty{i}" for i in range(n_empty)]
    features = [f"TX{g:05d}" for g in range(N_TRANSCRIPTOME)] + proteins
    empties = {
        "raw_counts": raw,
        "raw_barcodes": raw_barcodes,
        "feature_names": features,
        "adt_features": proteins,
        "droplet_batch": np.concatenate([[f"well{b}" for b in batch], [f"well{b}" for b in empty_batch]]),
    }
    return adata, truth, empties
