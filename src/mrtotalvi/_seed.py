"""Training-seed aggregation for MrTotalVI counterfactual datasets."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from collections.abc import Mapping


# Attributes that define the estimand. A mismatch means the seeds did not estimate the
# same quantity. ``random_state`` and storage details legitimately differ and are not listed.
_ESTIMAND_ATTRS = (
    "hierarchy_mode",
    "u_encoder_mode",
    "inference_mode",
    "batch_policy",
    "panel_policy",
    "library_policy",
    "observed_library_estimand",
    "specified_batch",
    "centering",
    "raw_residual_penalty",
    "density",
    "group_density",
    "self_exclusion",
    "specified_panel",
    "group_key",
    "contrast",
    "donor_key",
    "specified_library_size",
    "marginal_reference",
    "reference_cells",
    "max_reference_cells_per_sample",
)

# Settings added to result attrs after results were already being saved. A result that
# lacks one cannot be checked against a result that has it, so the pair is reported as
# unverifiable instead of as a mismatch.
_LATER_SETTING_ATTRS = frozenset(_ESTIMAND_ATTRS[_ESTIMAND_ATTRS.index("specified_panel") :])

# Outputs that depend on a setting, and the attribute that records it. Results written
# before the attribute existed lack it, so their settings cannot be compared.
_SETTING_FOR_OUTPUT = {
    "group_log_density": "group_key",
    "log_ratio": "contrast",
    "donor_log_ratio": "donor_key",
}


def combine_mrtotalvi_seed_results(
    results: Mapping[int, xr.Dataset],
) -> xr.Dataset:
    """Combine seed-specific datasets without pooling uncertainty sources."""
    if not results:
        raise ValueError("results must be a non-empty mapping.")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in results):
        raise TypeError("Every results key must be an integer training seed.")
    if any(not isinstance(dataset, xr.Dataset) for dataset in results.values()):
        raise TypeError("Every results value must be an xarray.Dataset.")

    seeds = sorted(results)
    datasets = [results[seed] for seed in seeds]
    template = datasets[0]
    if "draw" not in template.dims:
        raise ValueError("Every seed result must retain a draw dimension.")
    if len(seeds) < 2:
        warnings.warn(
            "Only one training seed was given, so every between-seed SD is NaN.",
            UserWarning,
            stacklevel=2,
        )
    if template.sizes["draw"] < 2:
        warnings.warn(
            "Each seed result has fewer than 2 draws, so every within-seed SD is NaN.",
            UserWarning,
            stacklevel=2,
        )
    for i, first in enumerate(datasets):
        for j in range(i + 1, len(datasets)):
            if first.equals(datasets[j]):
                warnings.warn(
                    f"Seed results for training seeds {seeds[i]} and {seeds[j]} are "
                    "identical. The same model under two keys gives a between-seed SD of 0, "
                    "which is not a measure of training variation.",
                    UserWarning,
                    stacklevel=2,
                )
    unverified = sorted(
        {
            setting
            for dataset in datasets
            for output, setting in _SETTING_FOR_OUTPUT.items()
            if output in dataset.data_vars and setting not in dataset.attrs
        }
        | {
            key
            for key in _LATER_SETTING_ATTRS
            if len({key in dataset.attrs for dataset in datasets}) > 1
        }
        | {
            "specified_panel"
            for dataset in datasets
            if dataset.attrs.get("panel_policy") == "specified"
            and "specified_panel" not in dataset.attrs
        }
    )
    if unverified:
        warnings.warn(
            f"Seed results lack the {unverified} attribute(s), so those settings cannot be "
            "compared across seeds. Regenerate them with the current package to check.",
            UserWarning,
            stacklevel=2,
        )
    for dataset in datasets[1:]:
        if set(dataset.data_vars) != set(template.data_vars):
            raise ValueError("Seed results must contain identical data variables.")
        if {
            name: variable.dims for name, variable in dataset.data_vars.items()
        } != {
            name: variable.dims for name, variable in template.data_vars.items()
        }:
            raise ValueError("Seed result variable dimensions must match exactly.")
        if dataset.attrs.get("schema_version") != template.attrs.get(
            "schema_version"
        ):
            raise ValueError("Seed result schema_version attributes must match.")
        mismatched = [
            key
            for key in _ESTIMAND_ATTRS
            if not (
                key in _LATER_SETTING_ATTRS
                and (key not in dataset.attrs or key not in template.attrs)
            )
            and dataset.attrs.get(key) != template.attrs.get(key)
        ]
        if mismatched:
            raise ValueError(
                f"Seed results define different estimands (attributes {mismatched}); "
                "they cannot share one between-seed SD."
            )
        try:
            xr.align(template, dataset, join="exact", copy=False)
        except ValueError as error:
            raise ValueError("Seed result coordinates must match exactly.") from error

    combined = xr.concat(
        datasets,
        dim=xr.IndexVariable("training_seed", seeds),
        data_vars="all",
        coords="minimal",
        compat="equals",
        join="exact",
        combine_attrs="drop",
    )
    additions = {}
    skipped_latent: list[str] = []
    partial_variables: list[str] = []
    for variable_name, variable in combined.data_vars.items():
        if "draw" not in variable.dims or not np.issubdtype(
            variable.dtype,
            np.floating,
        ):
            continue
        within_mean_name = f"{variable_name}_within_seed_posterior_mean"
        within_sd_name = f"{variable_name}_within_seed_posterior_sd"
        between_mean_name = f"{variable_name}_between_seed_mean"
        between_sd_name = f"{variable_name}_between_seed_sd"
        conflicts = {
            within_mean_name,
            within_sd_name,
            between_mean_name,
            between_sd_name,
        } & set(combined.data_vars)
        if conflicts:
            raise ValueError(
                "Seed summary variable name conflict(s): "
                f"{sorted(conflicts)}."
            )
        within_mean = variable.mean("draw")
        additions[within_mean_name] = within_mean
        additions[within_sd_name] = variable.std("draw", ddof=1)
        # Latent axes of independently trained seeds are not aligned (any rotation
        # or sign flip fits equally well), so a between-seed mean/SD of latent
        # coordinates has no meaning; summarise only feature-space variables.
        if {"latent_dim", "latent_u_dim"} & set(variable.dims):
            skipped_latent.append(variable_name)
            continue
        additions[between_mean_name] = within_mean.mean("training_seed")
        additions[between_sd_name] = within_mean.std("training_seed", ddof=1)
        n_partial = int(
            (within_mean.isnull().any("training_seed") & within_mean.notnull().any("training_seed"))
            .sum()
        )
        if n_partial:
            partial_variables.append(f"{variable_name} ({n_partial})")

    if partial_variables:
        warnings.warn(
            "Some entries are NaN in only some seeds; their between-seed mean and SD use the "
            f"remaining seeds (variable and entry count): {', '.join(partial_variables)}.",
            UserWarning,
            stacklevel=2,
        )
    combined = combined.assign(additions)
    combined.attrs = dict(template.attrs)
    combined.attrs["uncertainty_separation"] = (
        "within-seed posterior draws and between-seed training variation "
        "are reported separately and never pooled"
    )
    combined.attrs["training_seed_order"] = ",".join(map(str, seeds))
    combined.attrs["between_seed_skipped_latent_variables"] = ",".join(skipped_latent)
    return combined
