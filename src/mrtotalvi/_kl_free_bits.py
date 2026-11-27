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

# scvi anneals ``kl_weight`` from 0 to its ``max_kl_weight`` (default 1.0). "after_warmup" starts the
# floor when ``kl_weight`` reaches that value.
WARMUP_DONE_KL_WEIGHT = 1.0


def free_bits_active(schedule: str, kl_weight: float) -> bool:
    """``True`` when the floor applies at this ``kl_weight``."""
    return schedule == "constant" or kl_weight >= WARMUP_DONE_KL_WEIGHT


def kl_with_free_bits(
    kl: torch.Tensor,
    free_bits: float = 0.0,
    granularity: str = "per_dim",
    statistic: str = "batch_mean",
) -> torch.Tensor:
    """Per-cell KL cost with a free-bits floor; ``kl`` is ``(batch, dims)``, the result ``(batch,)``.

    ``free_bits = 0`` returns ``kl.sum(-1)`` unchanged. Otherwise a unit (``granularity="per_dim"``)
    or the group of all units (``"group"``, whose KL is the sum over its units) costs
    ``lambda`` nats when its KL is below ``lambda``. ``statistic="batch_mean"`` compares the
    minibatch mean KL with ``lambda`` (Kingma 2016); ``"per_cell"`` clamps every cell.

    With ``batch_mean`` the floor is a mask on the unweighted batch mean, so the mean of the result
    is ``sum_d max(lambda, mean_batch KL_d)`` and a donor weight on the cells does not change which
    units are floored.
    """
    if free_bits == 0.0:
        return kl.sum(dim=-1)
    if granularity == "group":
        kl = kl.sum(dim=-1, keepdim=True)
    if statistic == "per_cell":
        return kl.clamp(min=free_bits).sum(dim=-1)
    keep = kl.detach().mean(dim=0) >= free_bits
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
