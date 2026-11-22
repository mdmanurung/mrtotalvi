"""Check the two paired abundance summaries against exact Gaussian density ratios."""

from pathlib import Path
from unittest.mock import patch

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scvi
import torch
from scipy.special import logsumexp

from mrtotalvi import MrTotalVI


def paired_model():
    obs = pd.DataFrame(
        [(f"d{i}_{arm}", f"d{i}", arm) for i in range(5) for arm in ("A", "B") for _ in range(3)],
        columns=["sample", "donor", "arm"],
        index=[f"q{i}" for i in range(30)],
    )
    data = ad.AnnData(np.ones((30, 3), dtype=np.float32), obs=obs)
    data.obsm["protein"] = np.ones((30, 2), dtype=np.float32)
    MrTotalVI.setup_anndata(data, protein_expression_obsm_key="protein", sample_key="sample")
    model = MrTotalVI(data, sample_key="sample", hierarchy_mode="centered_v2", n_latent_u=2)
    model.is_trained_ = True  # The fixture supplies the exact posterior to both public methods.
    return model


def evaluate(model, first_ratio):
    ratios = np.array([first_ratio, -2, -2, -2, -2], dtype=float)
    # At zero, log N(0; mu_A, I) - log N(0; mu_B, I) = (||mu_B||² - ||mu_A||²)/2.
    locations = {}
    for i, r in enumerate(ratios):
        locations[f"d{i}_A"] = [np.sqrt(max(-2 * r, 0)), 0]
        locations[f"d{i}_B"] = [np.sqrt(max(2 * r, 0)), 0]
    loc = torch.tensor([locations[name] for name in model.sample_order], dtype=torch.float32)

    def posterior(x, y, sample_index, **kwargs):
        means = loc[sample_index.long().flatten()]
        return torch.distributions.Normal(means, torch.ones_like(means))

    with patch.object(model.module.qu, "forward", posterior):
        da = model.differential_abundance(
            sample_cov_keys=["arm"],
            donor_key="donor",
            omit_original_sample=False,
            compute_log_enrichment=True,
        ).sel(cell_name="q0")
        local = model.local_sample_enrichment(indices=[0], group_key="arm", contrast=("A", "B"), donor_key="donor").sel(
            cell_name="q0", draw=0
        )
    donors = [f"d{i}" for i in range(5)]
    lp = da.log_probs
    recovered = lp.sel(sample=[f"{d}_A" for d in donors]).values - lp.sel(sample=[f"{d}_B" for d in donors]).values
    np.testing.assert_allclose(recovered, ratios, atol=2e-6)
    np.testing.assert_allclose(local.donor_log_ratio.sel(donor=donors), ratios, atol=2e-6)
    da_score = da.arm_log_enrichs.sel(arm_enrichment="A").item()
    paired_mean = local.donor_log_ratio_mean.item()
    formula = logsumexp(ratios / 2) - logsumexp(-ratios / 2)
    np.testing.assert_allclose(da_score, formula, atol=2e-6)
    np.testing.assert_allclose(paired_mean, ratios.mean(), atol=2e-6)
    return {
        "first_donor": first_ratio,
        "other_donors": -2,
        "grouped_da": da_score,
        "grouped_formula": formula,
        "paired_mean": paired_mean,
        "mean_formula": ratios.mean(),
    }


def plot(results, directory):
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(9, 3.8), layout="constrained", gridspec_kw={"width_ratios": [1, 1.65]}
    )
    left.axhline(0, color="0.6", lw=0.8)
    point = results.loc[results.first_donor == 6].iloc[0]
    left.bar(
        ["d0", "d1", "d2", "d3", "d4"],
        [point.first_donor] + [point.other_donors] * 4,
        color=["#31688e"] + ["#92b6c7"] * 4,
    )
    left.set(title="A   Five donor pairs", ylabel="A versus B log density ratio (nats)", xlabel="Donor")
    right.axhline(0, color="0.6", lw=0.8)
    right.axvspan(2 * np.arcsinh(4 * np.sinh(1)), 8, color="#e9e9e9", label="Opposite signs")
    for formula, measured, color, label, marker in [
        ("grouped_formula", "grouped_da", "#b55109", "Grouped DA", "o"),
        ("mean_formula", "paired_mean", "#13806e", "Mean of donor log ratios", "s"),
    ]:
        right.plot(results.first_donor, results[formula], color=color, lw=1.5, label=label)
        right.scatter(results.first_donor, results[measured], marker=marker, s=19, color=color)
    right.set(
        title="B   Vary one donor, keep four at −2",
        xlabel="Donor d0 log ratio (nats)",
        ylabel="Paired summary (nats)",
    )
    right.legend(frameon=False, fontsize=8, loc="upper left")
    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(directory / "paired_da_check.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    scvi.settings.seed = 0
    torch.set_num_threads(1)
    directory = Path(__file__).resolve().parent
    model = paired_model()
    results = pd.DataFrame(evaluate(model, strength) for strength in range(11))
    results.to_csv(directory / "paired_da_check.tsv", sep="\t", index=False)
    plot(results, directory)
