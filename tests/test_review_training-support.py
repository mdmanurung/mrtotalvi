"""Training-support flag: per-donor training counts, warning, train_support output."""

import warnings

import numpy as np
import pytest
import scvi
import torch

from mrtotalvi import MrTotalVI


def _adata(n_samples):
    adata = scvi.data.synthetic_iid(batch_size=4, n_genes=5, n_proteins=3, n_regions=2, n_batches=2, n_labels=2)
    adata.obs["sample"] = np.asarray([f"sample_{i % n_samples}" for i in range(adata.n_obs)])
    MrTotalVI.setup_anndata(
        adata, protein_expression_obsm_key="protein_expression", sample_key="sample", batch_key="batch"
    )
    return adata


@pytest.fixture(scope="module")
def fitted():
    scvi.settings.seed = 0
    adata = _adata(3)
    model = MrTotalVI(
        adata,
        sample_key="sample",
        n_latent=3,
        n_latent_u=2,
        hierarchy_mode="centered_v2",
        u_encoder_mode="sample_blind",
    )
    with pytest.warns(UserWarning, match="no training cells"):
        model.train(max_epochs=1, accelerator="cpu", train_size=0.5, validation_size=0.25, batch_size=8)
    return model, adata


def test_counts_match_split_and_donor_is_flagged(fitted):
    model, adata = fitted
    codes = adata.obs["_scvi_sample"].to_numpy()
    expected = np.bincount(codes[model.train_indices_], minlength=3)
    assert model.n_train_cells_per_sample_.tolist() == expected.tolist()
    assert 0 in expected
    result = model.get_counterfactual_latent(
        adata=adata, indices=np.asarray(model.train_indices_[:2]), target_samples=list(model.sample_order), n_draws=1
    )
    flag = result["train_support"].to_numpy()
    assert flag.dtype == np.int8
    assert flag[0].tolist() == (expected > 0).astype(int).tolist()
    # target_support is a separate quantity and still counts validation/test cells.
    empty = int(np.flatnonzero(expected == 0)[0])
    assert result["target_support"].to_numpy()[:, empty].all()


def test_checkpoint_round_trip_and_old_checkpoint_unknown(fitted, tmp_path):
    model, adata = fitted
    model.save(tmp_path / "new", save_anndata=False)
    loaded = MrTotalVI.load(tmp_path / "new", adata=adata)
    assert loaded.n_train_cells_per_sample_.tolist() == model.n_train_cells_per_sample_.tolist()

    path = tmp_path / "new" / "model.pt"
    state = torch.load(path, weights_only=False)
    state["attr_dict"].pop("n_train_cells_per_sample_")
    torch.save(state, path)
    old = MrTotalVI.load(tmp_path / "new", adata=adata)
    assert old.n_train_cells_per_sample_ is None
    result = old.get_counterfactual_latent(
        adata=adata, indices=np.arange(2), target_samples=list(old.sample_order), n_draws=1
    )
    assert (result["train_support"].to_numpy() == -1).all()


def test_no_warning_when_every_donor_has_training_cells():
    scvi.settings.seed = 0
    adata = _adata(2)
    model = MrTotalVI(adata, sample_key="sample", n_latent=3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.train(max_epochs=1, accelerator="cpu", batch_size=64, early_stopping=False)
    assert not [w for w in caught if "no training cells" in str(w.message)]
    assert (model.n_train_cells_per_sample_ > 0).all()
