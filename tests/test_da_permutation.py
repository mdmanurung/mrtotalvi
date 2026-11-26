"""Donor-level permutation test of differential abundance (roadmap decisions G29, G45)."""

from math import comb

import numpy as np
import pandas as pd
import pytest
import scvi

import mrtotalvi._stats as stats
from mrtotalvi import MrTotalVI


def _obs(n_per_level=8, cells_per_sample=3):
    samples = [f"s{i:02d}" for i in range(2 * n_per_level)]
    rows = [
        {"sample": s, "cond": "A" if i < n_per_level else "B", "ct": "T" if c < 2 else "M"}
        for i, s in enumerate(samples)
        for c in range(cells_per_sample)
    ]
    obs = pd.DataFrame(rows)
    obs.index = [f"c{i}" for i in range(len(obs))]
    return obs, samples


def _stub(monkeypatch, per_sample_value):
    """Replace the density pass with an aggregator whose group score is the mean sample value."""
    values = pd.Series(per_sample_value)

    def fake(*_args, **_kwargs):
        def aggregate(group_samples):
            return np.full(_stub.n_cells, values.loc[list(group_samples)].mean())

        return None, aggregate

    monkeypatch.setattr(stats, "differential_abundance", fake)


def _run(obs, samples, **kwargs):
    adata = type("A", (), {"obs": obs})()
    _stub.n_cells = len(obs)
    return stats.differential_abundance_test(
        None,
        adata=adata,
        sample_key="sample",
        sample_cov_key="cond",
        groupby="ct",
        validated_sample_order=samples,
        density_estimator="knn",
        **kwargs,
    )


def test_label_permutations_exact_and_random():
    rng = np.random.default_rng(0)
    splits, exact = stats._label_permutations(4, 8, 10000, 99, rng)
    assert exact and len(splits) == comb(8, 4) == 70
    splits, exact = stats._label_permutations(8, 16, 100, 99, rng)
    assert not exact and len(splits) == 99


def test_perfect_separation_gives_the_smallest_exact_pvalue(monkeypatch):
    obs, samples = _obs()
    _stub(monkeypatch, {s: (1.0 if i < 8 else 0.0) for i, s in enumerate(samples)})
    # 8 v 8 has C(16, 8) = 12870 splits, above the default max_exact of 10000 (G45).
    assert not _run(obs, samples, n_permutations=9)["exact"].any()
    result = _run(obs, samples, max_exact=20000)
    # Two-sided: the observed split and its complement both reach |statistic| = 1.
    assert result["exact"].all()
    assert result["n_permutations"].iloc[0] == comb(16, 8)
    np.testing.assert_allclose(result["statistic"].to_numpy(), 1.0)
    np.testing.assert_allclose(result["pvalue"].to_numpy(), 2 / comb(16, 8))


def test_no_difference_gives_pvalue_one(monkeypatch):
    obs, samples = _obs()
    _stub(monkeypatch, dict.fromkeys(samples, 0.5))
    result = _run(obs, samples, max_exact=20000)
    np.testing.assert_allclose(result["statistic"].to_numpy(), 0.0)
    np.testing.assert_allclose(result["pvalue"].to_numpy(), 1.0)
    np.testing.assert_allclose(result["padj"].to_numpy(), 1.0)


def test_random_permutations_use_plus_one_pvalue(monkeypatch):
    obs, samples = _obs()
    _stub(monkeypatch, {s: (1.0 if i < 8 else 0.0) for i, s in enumerate(samples)})
    result = _run(obs, samples, max_exact=10, n_permutations=199, random_state=1)
    assert not result["exact"].any()
    # With 199 random splits the observed extreme is almost never matched: p = (1 + b) / 200.
    assert (result["pvalue"] >= 1 / 200).all()
    assert (result["pvalue"] <= 5 / 200).all()


def test_design_checks(monkeypatch):
    obs, samples = _obs(n_per_level=4)
    _stub(monkeypatch, dict.fromkeys(samples, 0.0))
    with pytest.raises(ValueError, match="at least 8 samples"):
        _run(obs, samples)
    with pytest.raises(ValueError, match="'posterior' is not calibrated"):
        stats.differential_abundance_test(
            None,
            adata=type("A", (), {"obs": obs})(),
            sample_key="sample",
            sample_cov_key="cond",
            groupby="ct",
            validated_sample_order=samples,
            density_estimator="posterior",
        )


def test_model_method_runs_end_to_end():
    scvi.settings.seed = 0
    adata = scvi.data.synthetic_iid(batch_size=40, n_genes=8, n_proteins=3, n_batches=2, n_labels=2)
    adata.obs["sample"] = np.asarray([f"s{i % 4}" for i in range(adata.n_obs)])
    adata.obs["cond"] = np.where(adata.obs["sample"].isin(["s0", "s1"]), "A", "B")
    MrTotalVI.setup_anndata(adata, protein_expression_obsm_key="protein_expression", sample_key="sample")
    model = MrTotalVI(adata, sample_key="sample", n_latent=3, n_latent_u=2)
    model.train(max_epochs=1, accelerator="cpu", batch_size=40)
    result = model.differential_abundance_test(
        "cond", "labels", density_estimator="knn", knn_k=5, min_samples_per_group=2
    )
    assert set(result.index) == set(adata.obs["labels"].astype(str))
    assert result["exact"].all() and (result["n_permutations"] == comb(4, 2)).all()
    assert result["pvalue"].between(0, 1).all()
