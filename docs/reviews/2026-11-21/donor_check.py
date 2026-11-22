"""Run from the repository root with PYTHONPATH=. and the test environment."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scvi
import torch

from mrtotalvi import MrTotalVI
from tests.test_mrtotalvi import DONOR_LEVELS, donor_recovery_adata


def compare(adata, design, seed, train, test):
    scvi.settings.seed = seed
    MrTotalVI.setup_anndata(
        adata,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
    )
    model = MrTotalVI(adata, sample_key="sample")
    model.train(
        max_epochs=100,
        n_epochs_kl_warmup=20,
        accelerator="cpu",
        devices=1,
        batch_size=128,
        early_stopping=False,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        external_indexing=[train, np.array([], dtype=int), test],
    )
    donors = [f"d{i}" for i in range(8)]
    latents = model.get_local_sample_representation(indices=test).sel(sample=donors)
    z = torch.as_tensor(latents.values.copy()).reshape(-1, model.module.n_latent)
    with torch.inference_mode():
        # All cells belong to batch zero. A common library cancels in the ratio.
        px, _, _ = model.module.decoder(
            z,
            torch.ones(len(z), 1),
            torch.zeros(len(z), 1, dtype=torch.long),
        )
        rates = px["rate"].reshape(len(test), len(donors), -1)
        ratios = (rates[:, :, :12].sum(-1) / rates[:, :, 12:24].sum(-1)).log()
    by_donor = ratios.mean(0).numpy()
    truth = 0.4 * DONOR_LEVELS if design == "Planted" else np.zeros(len(donors))
    return pd.DataFrame(
        {
            "design": design,
            "seed": seed,
            "donor": donors,
            "truth": truth,
            "mean_log_ratio": by_donor,
            "centered_log_ratio": by_donor - by_donor.mean(),
        }
    )


def plot(results, directory):
    donors = [f"d{i}" for i in np.argsort(DONOR_LEVELS)]
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(8.2, 3.6), layout="constrained")
    for ax, design in zip(axes, ("Planted", "Balanced null"), strict=True):
        subset = results.loc[results.design == design].set_index("donor")
        for seed, color in enumerate(("#1b6b75", "#99542d", "#7765a2")):
            frame = subset.loc[subset.seed == seed].loc[donors]
            ax.plot(donors, frame.centered_log_ratio, "o-", color=color, ms=3, lw=1, label=f"Seed {seed}")
        ax.plot(donors, frame.truth, color="black", ls="--", label="Truth")
        ax.set(title=design, xlabel="Target donor")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Centered RNA block log ratio (nats)")
    axes[1].legend(frameon=False, ncol=2, loc="upper left", fontsize=8)
    fig.savefig(directory / "donor_check.png", dpi=180)


if __name__ == "__main__":
    torch.set_num_threads(1)
    directory = Path(__file__).resolve().parent
    base = donor_recovery_adata.__wrapped__()
    groups = base.obs.groupby(["sample", "state"], observed=True).indices
    rng = np.random.default_rng(20261121)
    null = base.copy()
    null.obs["sample"] = null.obs["sample"].astype(object)
    train_rows, test_rows = [], []
    for rows in groups.values():
        train_rows.extend(rows[:24])
        test_rows.extend(rows[24:])
        for indices in (rows[:24], rows[24:]):
            balanced = np.tile(np.arange(8), len(indices) // 8)
            names = [f"d{i}" for i in rng.permutation(balanced)]
            null.obs.loc[null.obs_names[indices], "sample"] = names
    train, test = np.sort(train_rows), np.sort(test_rows)
    # Direct comparator for the planted contrast in this balanced design.
    blocks = (
        pd.DataFrame(
            {
                "sample": base.obs["sample"].to_numpy()[train],
                "first": base.X[train, :12].sum(axis=1),
                "second": base.X[train, 12:24].sum(axis=1),
            }
        )
        .groupby("sample")
        .sum()
        .reindex([f"d{i}" for i in range(8)])
    )
    pooled = np.log(blocks["first"] / blocks["second"])
    pooled_error = pooled - pooled.mean() - 0.4 * DONOR_LEVELS
    print("Pooled training-count span and RMSE:", np.ptp(pooled), np.mean(pooled_error**2) ** 0.5)
    results = pd.concat(
        [
            compare(data.copy(), design, seed, train, test)
            for design, data in (("Planted", base), ("Balanced null", null))
            for seed in (0, 1, 2)
        ],
        ignore_index=True,
    )
    results.to_csv(directory / "donor_check.tsv", sep="\t", index=False)
    plot(results, directory)
