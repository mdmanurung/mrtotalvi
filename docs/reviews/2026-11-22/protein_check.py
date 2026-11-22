"""Estimate a known donor protein contrast in two crossed batches. Run with PYTHONPATH=."""

from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scvi
import torch
from scvi.model import TOTALVI
from tests.test_mrtotalvi import DONOR_LEVELS

from mrtotalvi import MrTotalVI

DONORS = [f"d{i}" for i in range(len(DONOR_LEVELS))]
DATA_SEEDS = (731, 997)
FIT_SEEDS = (0, 1, 2)
STRATA = ["sample", "batch", "state"]
METHODS = {
    "Training counts": "#31688e",
    "totalVI factual": "#b55109",
    "MrTotalVI factual": "#13806e",
    "MrTotalVI target swap": "#80478e",
}


def simulate(data_seed, planted, *, holdout):
    # Separate count streams keep RNA and nuisance antibodies identical across designs.
    streams = np.random.SeedSequence([data_seed, int(holdout)]).spawn(3)
    rna_rng, marker_rng, other_rng = map(np.random.default_rng, streams)
    obs, rna, protein, population_means = [], [], [], []
    for donor, level in zip(DONORS, DONOR_LEVELS, strict=True):
        for batch in (0, 1):
            for state in (0, 1):
                common_batch = batch == int(level > 0)
                n = 12 if holdout else (24 if common_batch else 8)
                state_sign, batch_sign = 2 * state - 1, 2 * batch - 1
                log_ratio = 0.1 * level * planted + 0.3 * state_sign + 0.8 * batch_sign
                means = 16 * np.exp(np.array([log_ratio, -log_ratio]) / 2)
                population_means.extend([means] * n)
                markers = marker_rng.poisson(means, size=(n, 2))
                other = other_rng.poisson(np.where(np.arange(6) % 2 == state, 16, 4), (n, 6))
                protein.append(np.column_stack((markers, other)))
                logits = np.repeat([state_sign, -state_sign, 0.3 * batch_sign, -0.3 * batch_sign], 12)
                weights = np.exp(logits)
                rna.append(rna_rng.multinomial(1200, weights / weights.sum(), size=n))
                obs.extend([(donor, f"b{batch}", f"s{state}")] * n)
    obs = pd.DataFrame(obs, columns=STRATA)
    split = "test" if holdout else "train"
    obs.index = pd.Index([f"{split}-{i}" for i in range(len(obs))])
    data = ad.AnnData(np.vstack(rna).astype("float32"), obs=obs)
    data.var_names = [f"g{i}" for i in range(data.n_vars)]
    data.obsm["protein"] = pd.DataFrame(
        np.vstack(protein).astype("float32"), index=obs.index, columns=[f"P{i}" for i in range(8)]
    )
    np.testing.assert_allclose(donor_contrast(obs, population_means), 0.1 * DONOR_LEVELS * planted, atol=1e-12)
    return data


def donor_contrast(obs, markers):
    """Average the four within-stratum log ratios of means, then center across donors."""
    frame = obs[STRATA].copy()
    frame[["P0", "P1"]] = np.asarray(markers, dtype="float64")
    means = frame.groupby(STRATA, observed=True)[["P0", "P1"]].mean()
    if len(means) != 4 * len(DONORS) or not (np.isfinite(means) & (means > 0)).all().all():
        raise ValueError("Each donor needs two positive, finite marker means in all four strata.")
    ratios = np.log(means.P0 / means.P1).groupby(level="sample").mean().loc[DONORS]
    return ratios.to_numpy() - ratios.mean()


@torch.inference_mode()
def decode_markers(model, z, batch):
    """Exact total protein means conditional on the supplied latent and observed batch."""
    batch = torch.as_tensor(batch, dtype=torch.long).reshape(-1, 1)
    params = model.module.generative(torch.as_tensor(z), torch.ones(len(z), 1), batch, label=torch.zeros_like(batch))
    py = params["py_"]
    back = torch.exp(py["back_alpha"] + 0.5 * py["back_beta"].square())
    p_back = torch.sigmoid(py["mixing"])
    mean = params["per_batch_efficiency"] * back * (p_back + (1 - p_back) * py["fore_scale"])
    return mean[:, :2].numpy()


def fit(data, model_class, seed, epochs=100):
    train = data.copy()
    scvi.settings.seed = seed
    sample_kw = {"sample_key": "sample"} if model_class is MrTotalVI else {}
    model_class.setup_anndata(train, protein_expression_obsm_key="protein", batch_key="batch", **sample_kw)
    model = model_class(train, n_latent=30, empirical_protein_background_prior=False, **sample_kw)
    model.train(
        train_size=1.0,
        max_epochs=epochs,
        n_epochs_kl_warmup=20,
        batch_size=128,
        accelerator="cpu",
        devices=1,
        early_stopping=False,
        reduce_lr_on_plateau=False,
        lr=0.002,
        plan_kwargs={"eps": 1e-8, "weight_decay": 1e-8},
        enable_progress_bar=False,
        logger=False,
        enable_checkpointing=False,
    )
    np.testing.assert_array_equal(np.sort(model.train_indices), np.arange(train.n_obs))
    np.testing.assert_array_equal(model.protein_state_registry.column_names, train.obsm["protein"].columns)
    model.module.eval()
    batch = train.obs["_scvi_batch"].to_numpy()
    z = model.get_latent_representation(give_mean=True)
    markers = decode_markers(model, z, batch)
    factual = donor_contrast(train.obs, markers)
    if model_class is TOTALVI:
        return {"totalVI factual": factual}

    _, rows = np.unique(batch, return_index=True)
    reference = model.module._deterministic_decoder_parameters(
        torch.as_tensor(z[rows]),
        torch.ones(len(rows), 1),
        torch.as_tensor(batch[rows, None], dtype=torch.long),
    )
    np.testing.assert_allclose(markers[rows], reference["protein_total_mean"][:, :2], rtol=2e-6)

    # Exactly eight anchors per source donor, state and batch; identical cells for every target.
    anchors = np.flatnonzero(train.obs.groupby(STRATA, observed=True).cumcount() < 8)
    source = train.obs.iloc[anchors]
    local = (
        model.get_local_sample_representation(indices=anchors)
        .sel(cell_name=source.index.to_numpy(), sample=DONORS)
        .transpose("cell_name", "sample", "latent_dim")
        .values
    )
    own = pd.Index(DONORS).get_indexer(source["sample"])
    np.testing.assert_allclose(local[np.arange(len(anchors)), own], z[anchors], atol=2e-5, rtol=1e-5)
    batch_codes = np.repeat(batch[anchors], len(DONORS))
    decoded = decode_markers(model, local.reshape(-1, local.shape[-1]), batch_codes)
    target = pd.DataFrame(
        {
            "sample": np.tile(DONORS, len(anchors)),
            "state": np.repeat(source.state.to_numpy(), len(DONORS)),
            "batch": np.repeat(source.batch.to_numpy(), len(DONORS)),
        }
    )
    return {"MrTotalVI factual": factual, "MrTotalVI target swap": donor_contrast(target, decoded)}


def run(fit_seeds=FIT_SEEDS, epochs=100):
    frames = []
    for design, planted in (("Planted", True), ("Null", False)):
        truth = 0.1 * DONOR_LEVELS * planted
        for data_seed in DATA_SEEDS:
            train = simulate(data_seed, planted, holdout=False)
            test = simulate(data_seed, planted, holdout=True)
            pd.testing.assert_index_equal(train.var_names, test.var_names)
            pd.testing.assert_index_equal(train.obsm["protein"].columns, test.obsm["protein"].columns)
            if (train.n_obs, test.n_obs) != (512, 384) or not train.obs_names.intersection(test.obs_names).empty:
                raise ValueError("Unexpected train/test split.")
            estimates = {
                ("Training counts", None): donor_contrast(train.obs, train.obsm["protein"][["P0", "P1"]]),
                ("Held-out counts", None): donor_contrast(test.obs, test.obsm["protein"][["P0", "P1"]]),
            }
            for seed in fit_seeds:
                for model_class in (MrTotalVI, TOTALVI):
                    print(
                        f"{design} data={data_seed} seed={seed} model={model_class.__name__} epochs={epochs}",
                        flush=True,
                    )
                    for name, estimate in fit(train, model_class, seed, epochs).items():
                        estimates[name, seed] = estimate
            for (name, seed), estimate in estimates.items():
                frames.append(
                    pd.DataFrame(
                        {
                            "design": design,
                            "data_seed": data_seed,
                            "fit_seed": seed,
                            "method": name,
                            "donor": DONORS,
                            "truth": truth,
                            "estimate": estimate,
                        }
                    )
                )
    results = pd.concat(frames, ignore_index=True)
    results["fit_seed"] = results["fit_seed"].astype("Int64")
    return results


def plot(results, directory):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10, 6.6),
        layout="constrained",
        gridspec_kw={"height_ratios": [1.35, 1]},
    )
    axes[0, 1].sharey(axes[0, 0])
    axes[0, 1].tick_params(labelleft=False)
    donors = np.asarray(DONORS)[np.argsort(DONOR_LEVELS)]
    for col, design in enumerate(("Planted", "Null")):
        top, bottom = axes[:, col]
        data = results.loc[results.design == design].copy()
        top.plot(donors, data.groupby("donor").truth.first().loc[donors], "k--", lw=1.6, label="Truth")
        held = (
            data.loc[data.method == "Held-out counts"]
            .pivot(index="donor", columns="data_seed", values="estimate")
            .loc[donors]
        )
        top.scatter(
            np.repeat(np.arange(len(donors)), len(DATA_SEEDS)),
            held.to_numpy().ravel(),
            color="0.45",
            marker="x",
            s=16,
            label="Held-out counts",
        )
        for pos, (method, color) in enumerate(METHODS.items()):
            subset = data.loc[data.method == method].copy()
            profiles = subset.groupby("donor").estimate
            profile = profiles.mean().loc[donors]
            if method == "MrTotalVI factual":
                top.fill_between(
                    donors,
                    profiles.min().loc[donors],
                    profiles.max().loc[donors],
                    color=color,
                    alpha=0.15,
                    lw=0,
                )
                top.plot(donors, profile, "o-", color=color, mfc="white", lw=2, ms=6, label=method)
            elif method == "MrTotalVI target swap":
                top.plot(donors, profile, "s--", color=color, lw=1, ms=2.5, label=method)
            else:
                top.plot(donors, profile, "o-", color=color, lw=1.2, ms=3.5, label=method)
            subset["sq_error"] = (subset.estimate - subset.truth) ** 2
            errors = np.sqrt(subset.groupby(["data_seed", "fit_seed"], dropna=False).sq_error.mean())
            for i, data_seed in enumerate(DATA_SEEDS):
                values = errors.loc[data_seed].to_numpy()
                offset = np.linspace(-0.06, 0.06, len(values)) if len(values) > 1 else 0
                bottom.scatter(
                    pos + (i - 0.5) * 0.26 + offset,
                    values,
                    s=25,
                    marker=("o", "^")[i],
                    color=color,
                    edgecolor="white",
                    lw=0.4,
                )
        top.set_title(("A   Planted protein effect", "B   Zero donor effect")[col], loc="left")
        bottom.set_title(("C   Error against planted truth", "D   Error against zero")[col], loc="left")
        bottom.set_xticks(
            range(4),
            [
                "Training\ncounts",
                "totalVI\nfactual",
                "MrTotalVI\nfactual",
                "MrTotalVI\ntarget swap",
            ],
            fontsize=9,
        )
        bottom.set_xlim(-0.5, 3.5)
        for ax in (top, bottom):
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="0.91")
            ax.set_axisbelow(True)
    axes[0, 0].set_ylabel("Centered protein log ratio (nats)")
    axes[1, 0].set_ylabel("RMSE across eight donors (nats)")
    axes[1, 0].set_ylim(bottom=0)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_ylabel("RMSE (nats, log scale)")
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    for i, seed in enumerate(DATA_SEEDS):
        axes[1, 1].scatter([], [], color="0.35", marker=("o", "^")[i], label=f"Data seed {seed}")
    axes[1, 1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.savefig(directory / "protein_check.png", dpi=180)
    plt.close(fig)


def plot_epochs(primary, longer, directory):
    data = pd.concat([primary.assign(epochs=100), longer.assign(epochs=300)])
    data["sq_error"] = (data.estimate - data.truth) ** 2
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9), layout="constrained")
    for ax, design in zip(axes, ("Planted", "Null"), strict=True):
        group = data.loc[data.design == design]
        counts = group.loc[group.method == "Training counts"]
        baseline = np.sqrt(counts.groupby("data_seed").sq_error.mean())
        ax.axhspan(baseline.min(), baseline.max(), color="0.8", label="Training counts range")
        for method in ("MrTotalVI factual", "totalVI factual"):
            paired = group.loc[(group.method == method) & (group.fit_seed == 0)]
            rmse = np.sqrt(paired.groupby(["data_seed", "epochs"]).sq_error.mean()).unstack()
            for i, seed in enumerate(DATA_SEEDS):
                ax.plot(
                    [100, 300],
                    rmse.loc[seed, [100, 300]],
                    ("o-", "^--")[i],
                    color=METHODS[method],
                    lw=1.5,
                    label=method if i == 0 else None,
                )
        ax.set(xticks=[100, 300], xlabel="Training epochs")
        ax.set_xlim(80, 320)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="0.91")
        ax.set_axisbelow(True)
    axes[0].set(title="A   Planted protein effect", ylabel="Donor RMSE (nats)", ylim=(0, None))
    axes[1].set(title="B   Zero donor effect", ylabel="Donor RMSE (nats, log scale)", yscale="log")
    for i, seed in enumerate(DATA_SEEDS):
        axes[0].plot([], [], ("o-", "^--")[i], color="0.4", label=f"Data seed {seed}")
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    fig.savefig(directory / "protein_epochs.png", dpi=180)
    plt.close(fig)


def training_sensitivity(primary, directory):
    longer = run(fit_seeds=(0,), epochs=300)
    longer = longer.loc[longer.fit_seed.notna()].copy()
    longer["epochs"] = 300
    longer.to_csv(directory / "protein_epochs.tsv", sep="\t", index=False)
    plot_epochs(primary, longer, directory)


if __name__ == "__main__":
    torch.set_num_threads(1)
    directory = Path(__file__).resolve().parent
    results = run()
    results.to_csv(directory / "protein_check.tsv", sep="\t", index=False)
    plot(results, directory)
    training_sensitivity(results, directory)
