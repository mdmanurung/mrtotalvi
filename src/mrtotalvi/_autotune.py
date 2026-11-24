"""Hyperparameter search for MrTotalVI through scvi-tools autotune (ADR-0013)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._model import MrTotalVI

if TYPE_CHECKING:
    from typing import Any, Literal

    from anndata import AnnData
    from scvi.autotune import AutotuneExperiment

GUARD_METRICS = ("residual_active_units", "residual_share")


def run_autotune(
    adata: AnnData,
    metrics: str | list[str],
    mode: Literal["min", "max"],
    search_space: dict[str, dict[str, Any]],
    num_samples: int,
    **kwargs,
) -> AutotuneExperiment:
    """Run a hyperparameter sweep of :class:`~mrtotalvi.MrTotalVI`.

    A thin wrapper of :func:`scvi.autotune.run_autotune` with ``model_cls=MrTotalVI``.
    It adds two things. First, it sets ``model_params["sample_key"]`` to the key
    registered in :meth:`~mrtotalvi.MrTotalVI.setup_anndata`, so the search space need
    not repeat it. Second, when the primary metric is a logged Lightning metric (a name
    ending in ``_validation`` or ``_train``), it also reports ``residual_active_units``
    and ``residual_share`` for the same stage. These are not optimised. They flag
    trials whose donor residual ``eps`` may have collapsed, which ADR-0012 found the
    ELBO does not reveal. The flag is weak (AUROC 0.82 for ``residual_share`` on the
    ADR-0013 simulations), so check the donor effect of the chosen fit; read the
    columns from ``experiment.result_grid.get_dataframe()``. scib-metrics targets
    (for example ``"iLISI"``) are passed through unchanged.

    Needs the optional extra: ``pip install "mrtotalvi[autotune]"``.

    Parameters
    ----------
    adata
        AnnData set up with :meth:`~mrtotalvi.MrTotalVI.setup_anndata`.
    metrics
        Metric or metrics to track; the first is optimised.
    mode
        ``"min"`` or ``"max"`` for the first metric.
    search_space
        ``{"model_params": {...}, "train_params": {...}}`` as in scvi-tools.
    num_samples
        Number of configurations to sample.
    **kwargs
        Passed to :func:`scvi.autotune.run_autotune` (scheduler, searcher, seed,
        resources, logging_dir and the rest), with its defaults.

    Returns
    -------
    The :class:`~scvi.autotune.AutotuneExperiment`; its ``result_grid`` holds the trials.
    """
    import scvi.autotune

    manager = MrTotalVI._get_most_recent_anndata_manager(adata, required=True)
    sample_key = manager.registry["setup_args"]["sample_key"]
    model_params = dict(search_space.get("model_params", {}))
    if model_params.setdefault("sample_key", sample_key) != sample_key:
        raise ValueError(
            f"model_params sample_key={model_params['sample_key']!r} differs from the "
            f"sample_key registered in setup_anndata ({sample_key!r})."
        )
    search_space = {**search_space, "model_params": model_params}

    metrics = [metrics] if isinstance(metrics, str) else list(metrics)
    stage = next((s for s in ("validation", "train") if metrics[0].endswith(f"_{s}")), None)
    if stage is not None:
        metrics += [f"{g}_{stage}" for g in GUARD_METRICS if f"{g}_{stage}" not in metrics]

    return scvi.autotune.run_autotune(MrTotalVI, adata, metrics, mode, search_space, num_samples, **kwargs)
