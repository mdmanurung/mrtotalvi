"""KL weight and free-bits floor for a local latent (EXPERIMENTAL).

One helper for the residual latent ``w`` (Design 1) and the protein-only latent ``z_p`` (Design 2),
so both read the same knobs (G152 to G156, G166). The floor follows Kingma et al. 2016 (C.8):
a unit, or a group of units, costs ``max(lambda, KL)`` instead of ``KL``, so below the floor the
gradient is zero. The floor removes a penalty; it does not force use.
"""

from __future__ import annotations

import math

import torch

FREE_BITS_GRANULARITY = ("per_dim", "group")
FREE_BITS_STATISTIC = ("batch_mean", "per_cell")
FREE_BITS_SCHEDULE = ("constant", "after_warmup")

# scvi anneals ``kl_weight`` from 0 to the plan's ``max_kl_weight`` (default 1.0). "after_warmup" starts the
# floor when ``kl_weight`` reaches that cap; the module passes the cap it was trained with.
DEFAULT_WARMUP_TARGET = 1.0


def free_bits_active(schedule: str, kl_weight: float, warmup_target: float = DEFAULT_WARMUP_TARGET) -> bool:
    """``True`` when the floor applies at this ``kl_weight``.

    ``"after_warmup"`` applies it once ``kl_weight`` has reached ``warmup_target`` (the plan's
    ``max_kl_weight``). A relative 1e-6 slack absorbs float32 rounding of the scvi tensor.
    """
    return schedule == "constant" or float(kl_weight) >= warmup_target * (1 - 1e-6)


def kl_with_free_bits(
    kl: torch.Tensor,
    free_bits: float = 0.0,
    granularity: str = "per_dim",
    statistic: str = "batch_mean",
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-cell KL cost with a free-bits floor; ``kl`` is ``(batch, dims)``, the result ``(batch,)``.

    ``free_bits = 0`` returns ``kl.sum(-1)`` unchanged. Otherwise a unit (``granularity="per_dim"``)
    or the group of all units (``"group"``, whose KL is the sum over its units) costs
    ``lambda`` nats when its KL is below ``lambda``. ``statistic="batch_mean"`` compares the
    minibatch mean KL with ``lambda`` (Kingma 2016); ``"per_cell"`` clamps every cell.

    ``weights`` are the per-cell loss weights the caller applies later (donor weights under
    ``scale_observations``). The ``batch_mean`` statistic is then the weighted mean with the weights
    scaled to mean 1, so the threshold and the weighted loss use the same weights and the loss stays
    continuous at the floor. A floored unit gives each cell the constant ``lambda``, so the weighted
    cost of the unit is ``mean(weights) * lambda``, equal to its weighted cost at the threshold.
    Without ``weights`` the mean is the plain batch mean and the mean of the result is
    ``sum_d max(lambda, mean_batch KL_d)``.
    """
    if free_bits == 0.0:
        return kl.sum(dim=-1)
    if granularity == "group":
        kl = kl.sum(dim=-1, keepdim=True)
    if statistic == "per_cell":
        return kl.clamp(min=free_bits).sum(dim=-1)
    if weights is None:
        stat = kl.detach().mean(dim=0)
    else:
        scaled = (weights / weights.mean()).detach().to(kl.dtype)
        stat = (scaled.unsqueeze(-1) * kl.detach()).mean(dim=0)
    keep = stat >= free_bits
    return torch.where(keep, kl, torch.full_like(kl, free_bits)).sum(dim=-1)


def check_free_bits_options(
    weight: float, free_bits: float, granularity: str, statistic: str, schedule: str, name: str
):
    """Raise ``ValueError`` for a bad option; ``name`` prefixes the argument names in the message."""
    for value, label in ((weight, "kl_weight"), (free_bits, "free_bits")):
        if not (
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
        ):
            raise ValueError(f"{name}_{label} must be a finite number >= 0; got {value!r}.")
    for value, allowed, label in (
        (granularity, FREE_BITS_GRANULARITY, "free_bits_granularity"),
        (statistic, FREE_BITS_STATISTIC, "free_bits_statistic"),
        (schedule, FREE_BITS_SCHEDULE, "free_bits_schedule"),
    ):
        if value not in allowed:
            raise ValueError(f"{name}_{label} must be one of {set(allowed)}; got {value!r}.")
