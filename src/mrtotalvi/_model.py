"""MrTotalVI — TotalVI with per-donor hierarchical latent space."""

from __future__ import annotations

import inspect
import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
import xarray as xr
from scvi import REGISTRY_KEYS, settings
from scvi.data import AnnDataManager, fields
from scvi.data._utils import _check_if_view
from scvi.model._totalvi import TOTALVI
from scvi.model._utils import _get_var_names_from_manager, get_max_epochs_heuristic
from scvi.model.base import EmbeddingMixin
from scvi.model.base._save_load import _load_saved_files
from scvi.module._constants import MODULE_KEYS
from scvi.utils import setup_anndata_dsp
from tqdm import tqdm

from ._ambient import AmbientProfile
from ._contracts import (
    authoritative_protein_names,
    ordered_indices_sha256,
    resolve_u_prior,
    resolve_u_prior_supervision,
    take_matrix_rows,
    validate_anndata_counts,
    validate_sample_metadata,
)
from ._counterfactual import (
    get_counterfactual_expression as _get_counterfactual_expression,
)
from ._counterfactual import get_counterfactual_latent as _get_counterfactual_latent
from ._counterfactual import local_sample_enrichment as _local_sample_enrichment
from ._module import MrTotalVAE
from ._stats import (
    _as_indices,
    _differential_expression,
    _loader_order,
    _restore_request_order,
)
from ._stats import (
    differential_abundance as _differential_abundance,
)
from ._stats import (
    get_aggregated_posterior as _get_aggregated_posterior,
)
from ._stats import (
    get_outlier_cell_sample_pairs as _get_outlier_cell_sample_pairs,
)

if TYPE_CHECKING:
    from typing import Literal

    import numpy.typing as npt
    from anndata import AnnData

# Defaults taken from released MrVI (scvi.external.mrvi: DEFAULT_QZ_ATTENTION_KWARGS and
# Adam eps=1e-8, weight_decay=1e-8), plus MrVI's two-layer u encoder (encoder_n_layers=2),
# lr=2e-3 and no plateau schedule (candidate p05, user 2026-11-18).
# True only while ``load`` rebuilds the model in this thread; see the empty-donor check in __init__.
_LOADING_CHECKPOINT: ContextVar[bool] = ContextVar("mrtotalvi_loading_checkpoint", default=False)

DEFAULT_QZ_KWARGS: dict = {"stop_gradients_mlp": True, "dropout_rate": 0.03}
DEFAULT_QU_KWARGS: dict = {"n_layers": 2}
DEFAULT_PLAN_KWARGS: dict = {"eps": 1e-8, "weight_decay": 1e-8}


@contextmanager
def _eval_mode(module):
    """Run ``module`` in eval mode and restore the caller's training flag, so BatchNorm stats stay put."""
    was_training = module.training
    module.eval()
    try:
        yield
    finally:
        module.train(was_training)


class MrTotalVI(EmbeddingMixin, TOTALVI):
    """TotalVI with an MrVI-style hierarchical donor latent space.

    Grafts MrVI's per-sample attention residual onto TotalVI, enabling cell-
    and donor-level variability to be **jointly modelled** rather than treating
    donor identity as a nuisance covariate.

    * A mode-selectable u-encoder (:class:`~._components.EncoderXU_TotalVI`)
      produces the base ``u``. The backward-compatible default is
      sample-conditioned; ``u_encoder_mode="sample_blind"`` bypasses biological
      sample conditioning while retaining registered technical covariates.
    * A donor-specific residual ``eps`` is computed via
      :class:`~._components.EncoderUZ` (attention over a per-donor embedding
      table), giving ``z = z_base + eps``.
    * The decoder is **unchanged**; ``z`` drops in with the same shape as
      TotalVI's ``z``.

    ``hierarchy_mode="centered_v2"`` evaluates every registered sample residual
    and subtracts their equal-sample mean before factual gathering or public
    counterfactual decoding. These outputs are registered-sample model
    transformations, not causal interventions.

    Parameters
    ----------
    adata
        AnnData registered via :meth:`~MrTotalVI.setup_anndata`.
    sample_key
        Key in ``adata.obs`` identifying the donor/sample for each cell.
    n_latent
        Dimensionality of ``z``, the decoder input. Default ``30`` (released MrVI).
    n_latent_u
        Dimensionality of ``u``. Default ``10`` (released MrVI): ``u`` is projected to
        ``n_latent`` by a linear layer inside :class:`~._components.EncoderUZ`. ``None``
        makes ``u`` and ``z`` isomorphic (``n_latent_u = n_latent``, no projection),
        the setting used by checkpoints saved with ``n_latent_u=None``.
    n_latent_sample
        Dimension of the per-donor embedding in :class:`~._components.EncoderUZ`.
    z_u_prior
        Whether the ``eps`` residual has a prior term in the KL. Required by
        ``hierarchy_mode="centered_v2"``.
    z_u_prior_scale
        Log-scale of the prior on ``eps`` (the donor residual).
        ``0.0`` → ``p(eps) = N(0, 1)``. The loss clamps the log-scale at ``-4.0`` from
        below (``pz_scale.clamp(min=-4.0)``), so a value under ``-4`` acts as ``-4``, and a
        learnable scale receives no gradient once it is below ``-4``.
    learn_z_u_prior_scale
        Whether ``pz_scale`` is a learnable parameter. With ``use_map=True`` the
        ``eps`` term is ``-log p(eps)`` at a point, which can be negative, so
        :meth:`get_elbo` values are comparable only within one configuration (same
        ``use_map``, ``hierarchy_mode``, prior scales and KL weights). With
        ``use_map=True`` the optimizer also shrinks ``pz_scale`` and ``eps`` together
        until the ``-4`` clamp stops the scale, which removes most of the donor residual
        (one synthetic run: ``eps`` RMS 0.003 against 0.085 by default after 600 epochs);
        a warning says so. Upstream MrVI has the same option.
    u_prior
        Resolved prior enum: exactly ``"standard"``, ``"mog"``, or ``"vamp"``.
        New calls default to ``"mog"``.
    u_prior_mixture
        Deprecated checkpoint-migration input. Only the exact combinations in
        the 0.2 migration table are accepted; new calls should leave it ``None``.
    u_prior_supervision
        Resolved supervision mode. Omitted/``None`` with zero weight resolves
        to ``"none"``. ``"labels"`` is explicit opt-in and requires registered
        labels plus a finite positive weight.
    u_prior_label_weight
        Label-conditioned mixture-logit weight. Defaults to ``0.0``.
    u_prior_init_seed
        Deterministic seed for training-only Vamp data initialization.
    hierarchy_mode
        ``"legacy"`` preserves historical numerics. ``"centered_v2"`` opts
        into full-registered-universe residual centering and requires
        ``use_map=True`` and ``z_u_prior=True``.
    u_encoder_mode
        ``"sample_conditioned"`` preserves the historical encoder.
        ``"sample_blind"`` bypasses conditional-normalization affine embeddings
        and the explicit sample embedding without changing checkpoint topology.
    sample_embedding_prior_scale
        ``None`` (default) is off and leaves loss and parameters unchanged. A positive
        ``sigma`` adds the Gaussian shrinkage log-prior
        ``sum_s ||e_s - mean_s e_s||^2 / (2 sigma^2)`` on the per-sample embedding table
        that feeds ``eps`` (``module.qz.embedding``), so ``eps`` cannot fit arbitrary
        per-sample offsets without data support. Centring on the across-sample mean leaves a
        shared offset unpenalised. The sum is a global term: the loss (a per-cell mean) gains
        ``penalty / n_cells``, with ``n_cells`` the number of cells in the registered AnnData,
        so the penalty is not multiplied by the batch size and is not annealed by
        ``kl_weight``. With the legacy ``scale_observations=True`` each cell's data term is
        also divided by its donor's cell count while the penalty is not, so one ``sigma``
        shrinks relatively more strongly, by about the donor's cell count, than without it (``hierarchy_mode="centered_v2"`` weights average
        to 1 and are unaffected). The validation metric ``elbo_validation`` reports the
        penalty as ``kl_global / n_obs_validation`` (the scvi-tools convention), about
        ``n_cells / n_obs_validation`` times its per-cell size in the loss, and
        :meth:`get_elbo` leaves it out.
    u_encoder_sample_embedding_prior_scale
        Same prior with its own ``sigma`` on the u-encoder sample tables (``qu.sample_embed``
        and the gamma/beta tables of both conditional normalisations). Requires
        ``u_encoder_mode="sample_conditioned"``. ``None`` (default) is off.
    kl_u_weight
        Static scalar weight applied to ``KL(q_u ‖ p_u)`` before the global
        ``kl_weight`` annealing.  Default ``1.0`` preserves prior behaviour.
    u_prior_scale
        Log-scale of the ``u`` prior. ``0.0`` gives a unit-scale prior.
    u_prior_mixture_k
        Number of mixture components when ``u_prior="mog"`` (default ``20``).
    use_map
        If ``True`` (default), ``eps`` is a deterministic (MAP) output of
        :class:`~._components.EncoderUZ`; if ``False`` it is sampled from a learned
        Normal. ``hierarchy_mode="centered_v2"`` requires ``True``.
    scale_observations
        Legacy MrVI option, default ``False``. If ``True``, each cell's loss is divided
        by the number of cells of its sample. This shrinks gradients by about that
        count, so it trains properly only with a small Adam ``eps``; the
        :meth:`train` default ``eps=1e-8`` is chosen for this. ``centered_v2``
        uses its own normalised sample weights.
    qz_kwargs
        Keyword arguments for :class:`~._components.EncoderUZ`, merged over
        ``DEFAULT_QZ_KWARGS`` (``stop_gradients_mlp=True``, ``dropout_rate=0.03``, the
        values of upstream MrVI's ``DEFAULT_QZ_ATTENTION_KWARGS``). Keys you pass win.
    qu_kwargs
        Keyword arguments for :class:`~._components.EncoderXU_TotalVI`, merged over
        ``DEFAULT_QU_KWARGS`` (``n_layers=2``, as upstream MrVI's ``encoder_n_layers``).
        Keys you pass win. Checkpoints saved before this default load with their
        one-layer ``qu``.
    spec_defaults
        ``None`` (default) applies the ``qz_kwargs`` and ``qu_kwargs`` defaults above.
        ``False`` skips both: it rebuilds the ``qz`` made before ``spec_defaults``
        existed (no stop-gradient or dropout) and the one-layer ``qu``. The resolved
        value is saved with the model; checkpoints that lack it load with ``False``.
    kl_z_weight
        Static scalar weight applied to ``KL(q_z ‖ p_z)`` before the global
        ``kl_weight`` annealing.  Default ``1.0`` preserves prior behaviour.
    residual_prior_mode
        ``centered_v2`` residual prior. ``"raw_mean"`` (default, ADR-0007) averages
        ``-log p(eps)`` over every registered sample's raw residual, which makes a
        sample-private residual about ``S - 1`` times cheaper than in ``legacy``.
        ``"factual"`` scores only the cell's centered factual residual, as upstream MrVI
        scores its ``eps``. It changes training; it is opt-in and unvalidated on real data.
        It requires ``hierarchy_mode="centered_v2"``. Checkpoints without it load with
        ``"raw_mean"``.
    anneal_protein_likelihood
        ``True`` (default, upstream TOTALVI) multiplies the protein likelihood by
        ``kl_weight`` during warm-up, so it gives no gradient at ``kl_weight=0``.
        ``False`` anneals only the KL terms, as the TOTALVI paper describes. It changes
        training; it is opt-in. Checkpoints without it load with ``True``.
    init_prior_from_data
        If ``True`` and ``u_prior="vamp"``, run k-means on a random subsample
        (≤10 000 cells) of the raw encoder input from the frozen training split
        only. The seed and ordered training-index digest are persisted before
        optimization. Non-Vamp use raises. Without it the pseudo-inputs start as
        ``0.01 * randn``, so all Vamp components start nearly identical.
    freeze_prior_after_init
        If ``True`` and ``u_prior="vamp"``, freeze the VampPrior pseudo-input
        parameters after data-driven initialisation so they do not drift during
        training.  This configuration is an unvalidated candidate; no
        differential-abundance stability improvement is established.
    use_batch_norm
        Where to apply batch normalisation (``"encoder"``, ``"decoder"``,
        ``"both"``, ``"none"``).  Defaults to ``"none"``; layer normalisation
        is preferred for sample-level models to avoid confounding donor effects
        with batch statistics.
    use_layer_norm
        Where to apply layer normalisation.  Defaults to ``"both"``.
    batch_representation
        How the batch covariate is fed to the networks.  ``"one-hot"`` (default) is the
        historical behaviour and is bit-for-bit unchanged.  ``"embedding"`` replaces the
        one-hot encoding with a single learned embedding table shared by the u-encoder and
        the decoder, so their input width grows by the embedding dimension rather than by
        ``n_batch`` — useful when there are many batches.  Per-batch parameter tables
        (dispersion, per-batch efficiency, protein background prior, library-size priors)
        remain one-hot indexed in both modes, matching :class:`~scvi.module.VAE`.
        Retrieve the learned vectors with :meth:`get_batch_representation`.

        ``EXPERIMENTAL``: existing checkpoints are unaffected, but ``"embedding"`` changes
        the architecture, so a model trained with it cannot be loaded as ``"one-hot"``.
    batch_embedding_kwargs
        Keyword arguments passed to :class:`~scvi.nn.Embedding` when
        ``batch_representation="embedding"``, e.g. ``{"embedding_dim": 5}``.
    protein_noise_model
        Protein likelihood (ADR-0011). ``"totalvi"`` (default) keeps the stock totalVI
        mixture. ``"ambient"`` adds a free-ambient term ``A = c * a`` (``a`` the ambient
        composition, ``c`` a per-cell burden) to both mixture means. ``"ambient_isotype"``
        replaces the decoded background with ``A`` plus a nonspecific term driven by the
        isotype controls, fits the isotype controls with background only, and drops the
        totalVI background prior. Both opt-in modes need ``ambient_profile``;
        ``"ambient_isotype"`` also needs ``isotype_names``.
    protein_background_source
        ``"z"`` (default) decodes the protein background from ``z``. ``"z_base"`` decodes
        it from ``z_base(u)``, so a sample counterfactual does not move it.
    protein_encoder_input
        ``"raw"`` (default) or ``"ambient_zscore"``, which feeds the ``u`` encoder the
        DSB step-I standardized protein counts (needs ``ambient_profile``). It made
        results worse on the ADR-0011 simulations.
    ambient_profile
        An :class:`~mrtotalvi.preprocessing.AmbientProfile` from
        :func:`~mrtotalvi.preprocessing.estimate_adt_ambient`, or a dict of one profile per
        registered batch. Required by every opt-in protein setting.
    isotype_names
        Names of the isotype-control features on the protein axis. Required by
        ``protein_noise_model="ambient_isotype"``; their outputs are ``NaN`` or dropped
        (see the user guide).
    seed_background_prior_from_ambient
        Seed the learnable totalVI background prior per protein from the empty-droplet
        ``log1p`` mean and SD; training may still move it. Needs one pooled profile. It
        made results worse on the ADR-0011 simulations.
    ambient_burden_prior
        ``(loc, scale)`` of the normal prior on the log free-ambient burden ``log c``.
    n_nuisance
        Dimension of the per-cell nonspecific-binding latent in
        ``protein_noise_model="ambient_isotype"``.
    native_form
        Native signal in ``"ambient_isotype"``: ``"scale"`` (default) ``S = exp(alpha) f``,
        or ``"excess"`` ``S = exp(alpha) (f - 1)``.
    background_mixing_reg
        Weight of totalVI's background-mixing pull in the opt-in modes.
    learn_ambient_profile
        ``EXPERIMENTAL``. Learn the ambient composition as ``softmax(logits)``, started
        at ``ambient_profile``, with the cell-free droplets' multinomial likelihood of
        their summed counts as a global penalty. Needs an opt-in protein noise model and
        a profile with ``droplet_counts``. On the ADR-0012 simulation it gave no clear
        gain (native log1p RMSE 0.267 vs 0.282 with per-batch profiles, 0.342 vs 0.341
        pooled; 5 seeds).
    **model_kwargs
        Additional keyword arguments forwarded to :class:`~._module.MrTotalVAE`
        (and transitively to :class:`~scvi.module._totalvae.TOTALVAE`).

    Notes
    -----
    ``latent_distribution`` is **forced** to ``"normal"``.  Under ``"ln"``,
    the additive u→z hierarchy is mathematically invalid.

    Three separate categorical axes are kept distinct:
    * ``sample_key`` → :class:`~._components.EncoderUZ` embedding table.
    * ``batch_key``  → decoder ``cat_list``.
    * ``panel_key``  → protein background prior.

    The protein background prior follows scvi-tools: ``empirical_protein_background_prior``
    defaults to ``None``, which turns it on above 10 proteins. It is computed once at
    construction from all registered cells, including any you hold out later. For
    held-out evaluation, pass ``empirical_protein_background_prior=False`` or construct
    the model on the training cells only. The inherited initializer SD shrinks with the
    number of sampled cells, as ``sqrt(sum(s_i^2)) / n``.

    See Also
    --------
    :class:`~._module.MrTotalVAE`
    """

    _module_cls = MrTotalVAE

    def __init__(
        self,
        adata: AnnData,
        sample_key: str,
        n_latent: int = 30,
        n_latent_sample: int = 16,
        n_latent_u: int | None = 10,
        z_u_prior: bool = True,
        z_u_prior_scale: float = 0.0,
        u_prior_scale: float = 0.0,
        u_prior_mixture: bool | None = None,
        u_prior_mixture_k: int = 20,
        u_prior_label_weight: float = 0.0,
        u_prior: str = "mog",
        u_prior_supervision: Literal["none", "labels"] | None = None,
        u_prior_init_seed: int = 0,
        learn_z_u_prior_scale: bool = False,
        qz_kwargs: dict | None = None,
        qu_kwargs: dict | None = None,
        use_map: bool = True,
        hierarchy_mode: Literal["legacy", "centered_v2"] = "legacy",
        u_encoder_mode: Literal["sample_conditioned", "sample_blind"] = "sample_conditioned",
        scale_observations: bool = False,
        sample_embedding_prior_scale: float | None = None,
        u_encoder_sample_embedding_prior_scale: float | None = None,
        kl_u_weight: float = 1.0,
        kl_z_weight: float = 1.0,
        residual_prior_mode: Literal["raw_mean", "factual"] = "raw_mean",
        anneal_protein_likelihood: bool = True,
        init_prior_from_data: bool = False,
        freeze_prior_after_init: bool = False,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        batch_representation: Literal["one-hot", "embedding"] = "one-hot",
        batch_embedding_kwargs: dict | None = None,
        spec_defaults: bool | None = None,
        protein_noise_model: Literal["totalvi", "ambient", "ambient_isotype"] = "totalvi",
        protein_background_source: Literal["z", "z_base"] = "z",
        protein_encoder_input: Literal["raw", "ambient_zscore"] = "raw",
        ambient_profile: AmbientProfile | None = None,
        isotype_names: list[str] | None = None,
        seed_background_prior_from_ambient: bool = False,
        ambient_burden_prior: tuple[float, float] = (0.0, 2.0),
        n_nuisance: int = 1,
        native_form: Literal["scale", "excess"] = "scale",
        background_mixing_reg: float = 1e-3,
        learn_ambient_profile: bool = False,
        **model_kwargs,
    ) -> None:
        # MrVI-matching encoder defaults. New models save the resolved True/False, so
        # ``None`` during ``load`` means a checkpoint that predates the flag: rebuild the old
        # architecture. Outside ``load``, ``None`` means "on".
        if spec_defaults is None:
            spec_defaults = not _LOADING_CHECKPOINT.get()
        spec_defaults = bool(spec_defaults)
        if spec_defaults:
            qz_kwargs = {**DEFAULT_QZ_KWARGS, **(qz_kwargs or {})}
            # New models save the resolved qu_kwargs. During load, take the saved value as is:
            # checkpoints from before the two-layer default saved None (one layer).
            if not _LOADING_CHECKPOINT.get():
                qu_kwargs = {**DEFAULT_QU_KWARGS, **(qu_kwargs or {})}
        resolved_u_prior, resolved_u_prior_mixture = resolve_u_prior(
            u_prior,
            u_prior_mixture,
        )
        manager = self._get_most_recent_anndata_manager(adata, required=True)
        has_registered_labels = manager.registry["setup_args"].get("labels_key") is not None
        resolved_supervision, resolved_label_weight = resolve_u_prior_supervision(
            u_prior_supervision,
            u_prior_label_weight,
            has_registered_labels=has_registered_labels,
            legacy_checkpoint_hint=u_prior_mixture is not None,
            resolved_prior=resolved_u_prior,
        )
        resolved_u_prior_mixture_k = int(u_prior_mixture_k)
        registered_n_labels = int(manager.summary_stats.get("n_labels", 0))
        if u_prior_mixture is not None and resolved_u_prior == "vamp" and registered_n_labels > 1:
            # Historical labelled Vamp checkpoints used one global mixture
            # with K equal to the registered label count. Preserve both K and
            # the unconditioned (one-dimensional) categorical weights.
            resolved_u_prior_mixture_k = registered_n_labels
        if not isinstance(u_prior_init_seed, int) or isinstance(u_prior_init_seed, bool):
            raise TypeError("u_prior_init_seed must be an integer.")
        if init_prior_from_data and resolved_u_prior != "vamp":
            raise ValueError("init_prior_from_data=True requires u_prior='vamp'.")
        if freeze_prior_after_init and (resolved_u_prior != "vamp" or not init_prior_from_data):
            raise ValueError("freeze_prior_after_init=True requires data-initialized u_prior='vamp'.")
        if hierarchy_mode not in {"legacy", "centered_v2"}:
            raise ValueError("hierarchy_mode must be one of {'legacy', 'centered_v2'}.")
        if learn_z_u_prior_scale and use_map:
            warnings.warn(
                "learn_z_u_prior_scale=True with use_map=True shrinks pz_scale and eps "
                "together until the -4 clamp, which removes most of the donor residual. "
                "Keep the default fixed scale unless you mean to.",
                UserWarning,
                stacklevel=2,
            )
        if residual_prior_mode == "factual" and hierarchy_mode != "centered_v2":
            raise ValueError("residual_prior_mode='factual' requires hierarchy_mode='centered_v2'.")
        if u_encoder_mode not in {"sample_conditioned", "sample_blind"}:
            raise ValueError("u_encoder_mode must be one of {'sample_conditioned', 'sample_blind'}.")
        if hierarchy_mode == "centered_v2" and not use_map:
            raise ValueError("hierarchy_mode='centered_v2' requires use_map=True.")
        if hierarchy_mode == "centered_v2" and not z_u_prior:
            raise ValueError("hierarchy_mode='centered_v2' requires z_u_prior=True.")
        if model_kwargs.get("latent_distribution", "normal") != "normal":
            raise ValueError(
                "MrTotalVI requires latent_distribution='normal'. "
                "Under 'ln' (softmax normalisation), the additive u→z "
                "hierarchy is mathematically invalid."
            )
        model_kwargs["latent_distribution"] = "normal"
        if model_kwargs.get("gene_dispersion", "gene") == "gene-label":
            raise ValueError(
                "gene_dispersion='gene-label' is not supported: TOTALVI builds the "
                "model but training fails because the label tensor is not passed to "
                "the generative step. Use 'gene' or 'gene-batch'."
            )
        if model_kwargs.get("gene_dispersion", "gene") == "gene-cell":
            raise ValueError(
                "gene_dispersion='gene-cell' is not supported: TOTALVI has no gene-cell branch and "
                "the first forward pass fails with UnboundLocalError. Use 'gene' or 'gene-batch'."
            )
        if model_kwargs.get("protein_dispersion", "protein") == "protein-cell":
            raise ValueError(
                "protein_dispersion='protein-cell' is not supported: TOTALVI has no protein-cell branch "
                "and the first forward pass fails with UnboundLocalError. Use 'protein' or 'protein-batch'."
            )
        if model_kwargs.get("protein_dispersion", "protein") == "protein-label":
            raise ValueError(
                "protein_dispersion='protein-label' is not supported: TOTALVI builds "
                "the model but the first forward pass fails because no label tensor "
                "reaches inference. Use 'protein' or 'protein-batch'."
            )
        for axis, supported in (("gene", ("gene", "gene-batch")), ("protein", ("protein", "protein-batch"))):
            value = model_kwargs.get(f"{axis}_dispersion", axis)
            if value not in supported:
                raise ValueError(f"{axis}_dispersion must be one of {supported}; got {value!r}.")

        # Call TOTALVI.__init__. n_sample / n_latent_sample / z_u_prior_scale /
        # learn_z_u_prior_scale are explicit kwargs here but are absent from
        # TOTALVI's signature — Python routes them into **model_kwargs so they
        # reach MrTotalVAE.__init__ as explicit parameters (not via **kwargs
        # pass-through: MrTotalVAE names them explicitly in its own __init__).
        # MrTotalVAE always builds with n_sample=0 until _setup_hierarchy below
        # so that summary_stats.n_sample (populated by super()) is the source of truth.
        super().__init__(
            adata,
            n_latent=n_latent,
            n_latent_u=n_latent_u,
            z_u_prior=z_u_prior,
            u_prior_scale=u_prior_scale,
            u_prior_mixture=resolved_u_prior_mixture,
            u_prior_mixture_k=resolved_u_prior_mixture_k,
            u_prior_label_weight=resolved_label_weight,
            u_prior=resolved_u_prior,
            u_prior_supervision=resolved_supervision,
            qz_kwargs=qz_kwargs,
            qu_kwargs=qu_kwargs,
            hierarchy_mode=hierarchy_mode,
            u_encoder_mode=u_encoder_mode,
            kl_u_weight=kl_u_weight,
            kl_z_weight=kl_z_weight,
            residual_prior_mode=residual_prior_mode,
            anneal_protein_likelihood=anneal_protein_likelihood,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            batch_representation=batch_representation,
            batch_embedding_kwargs=batch_embedding_kwargs,
            **model_kwargs,
        )

        # At this point self.summary_stats is populated from the registry.
        n_sample = self.summary_stats.n_sample

        counts = adata.obs["_scvi_sample"].value_counts().reindex(range(n_sample), fill_value=0).sort_index()
        missing = counts[counts == 0].index.tolist()
        self._empty_sample_positions = missing
        # A checkpoint load rebuilds the model on the AnnData passed to ``load``, which
        # may hold a subset of the training donors. The weights come from the checkpoint,
        # so an empty donor is allowed there. A fresh model with scale_observations=True
        # fails here because a donor weight would divide by zero cells. Other fresh models
        # fail in ``train``.
        if missing and scale_observations and not _LOADING_CHECKPOINT.get():
            raise ValueError(
                f"Donors at index positions {missing} have no cells in the registered "
                "AnnData, and scale_observations=True weights each cell by its donor's "
                "cell count. Filter these donors from the data before setup_anndata, or "
                "set scale_observations=False."
            )
        n_obs_per_sample = torch.tensor(counts.values, dtype=torch.float32)

        self.module._setup_hierarchy(
            n_sample=n_sample,
            n_latent_sample=n_latent_sample,
            z_u_prior_scale=z_u_prior_scale,
            learn_z_u_prior_scale=learn_z_u_prior_scale,
            use_map=use_map,
            hierarchy_mode=hierarchy_mode,
            u_encoder_mode=u_encoder_mode,
            scale_observations=scale_observations,
            n_obs_per_sample=n_obs_per_sample,
            n_labels=self.summary_stats.get("n_labels", 0),
            prior_centroids=None,
            freeze_prior_after_init=False,
        )

        self.module._set_sample_embedding_prior(
            sample_embedding_prior_scale,
            u_encoder_sample_embedding_prior_scale,
            float(n_obs_per_sample.sum()),
        )

        self._configure_protein_noise(
            protein_noise_model=protein_noise_model,
            protein_background_source=protein_background_source,
            protein_encoder_input=protein_encoder_input,
            ambient_profile=ambient_profile,
            isotype_names=isotype_names,
            seed_background_prior_from_ambient=seed_background_prior_from_ambient,
            ambient_burden_prior=ambient_burden_prior,
            n_nuisance=n_nuisance,
            native_form=native_form,
            background_mixing_reg=background_mixing_reg,
            learn_ambient_profile=learn_ambient_profile,
        )

        registered_sample_key = self.adata_manager.registry.get("setup_args", {}).get("sample_key")
        if registered_sample_key is not None and registered_sample_key != sample_key:
            raise ValueError(
                f"sample_key={sample_key!r} differs from the sample_key registered in "
                f"setup_anndata ({registered_sample_key!r}). Training uses the registered "
                "column, while DA and aggregated posteriors would group by the argument."
            )

        # Sample-level metadata for coordinate labelling
        self._sample_key = sample_key
        self.sample_key = sample_key
        self.hierarchy_mode = hierarchy_mode
        self.u_encoder_mode = u_encoder_mode
        self.resolved_u_prior = resolved_u_prior
        self.u_prior_supervision = resolved_supervision
        self.u_prior_label_weight = resolved_label_weight
        self.u_prior_init_seed = u_prior_init_seed
        self._init_prior_from_data = bool(init_prior_from_data)
        self._freeze_prior_after_init = bool(freeze_prior_after_init)
        self.vamp_training_indices_sha256_ = None
        # Training cells per donor in ``sample_order``; set by train(), None when unknown.
        self.n_train_cells_per_sample_ = None
        self.vamp_initialization_seed_ = u_prior_init_seed if init_prior_from_data else None
        self.sample_order = self.adata_manager.get_state_registry(REGISTRY_KEYS.SAMPLE_KEY).categorical_mapping
        self.label_order = self.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY).categorical_mapping
        # One row per registered sample, in ``sample_order`` (the embedding row order),
        # as in MrVI. ``drop_duplicates`` alone follows the order of first appearance.
        self.sample_info = (
            self.adata.obs[[sample_key]]
            .drop_duplicates()
            .assign(_order=lambda d: d[sample_key].map({s: i for i, s in enumerate(self.sample_order)}))
            .sort_values("_order")
            .drop(columns="_order")
            .reset_index(drop=True)
        )

        # Overwrite init_params_ from TOTALVI with MrTotalVI's full local scope
        self.init_params_ = self._get_init_params(locals())
        self.init_params_["non_kwargs"].update(
            {
                "u_prior": resolved_u_prior,
                "u_prior_mixture": None,
                "u_prior_supervision": resolved_supervision,
                "u_prior_label_weight": resolved_label_weight,
                "u_prior_mixture_k": resolved_u_prior_mixture_k,
                "u_prior_init_seed": u_prior_init_seed,
            }
        )
        self._refresh_model_summary()

    def _refresh_model_summary(self) -> None:
        """Render the resolved scientific semantics, including checkpoint metadata."""
        self._model_summary_string = (
            "MrTotalVI Model\n"
            f"  hierarchy_mode: {self.hierarchy_mode}, "
            f"u_encoder_mode: {self.u_encoder_mode}\n"
            f"  n_latent: {self.module.n_latent}, "
            f"n_latent_u: {self.module.n_latent_u}, "
            f"n_latent_sample: {self.module._n_latent_sample}\n"
            f"  n_sample: {self.summary_stats.n_sample}, "
            f"z_u_prior: {self.module.z_u_prior}\n"
            f"  u_prior: {self.resolved_u_prior}, "
            f"u_prior_mixture_k: {self.module.resolved_u_prior_mixture_k}\n"
            f"  u_prior_supervision: {self.u_prior_supervision}, "
            f"u_prior_label_weight: {self.u_prior_label_weight}\n"
            f"  vamp_initialization_seed: {self.vamp_initialization_seed_}, "
            "vamp_training_indices_sha256: "
            f"{self.vamp_training_indices_sha256_}"
        )

    def _initialize_vamp_from_training_indices(self, train_indices) -> None:
        """Initialize Vamp pseudo-inputs from the frozen training split only."""
        if not self._init_prior_from_data:
            return

        from sklearn.cluster import KMeans

        ordered_indices = np.asarray(train_indices, dtype=np.int64).reshape(-1)
        if ordered_indices.size == 0:
            raise ValueError("Cannot initialize VampPrior from an empty training split.")
        if len(np.unique(ordered_indices)) != ordered_indices.size:
            raise ValueError("VampPrior training indices must be unique.")
        if (ordered_indices < 0).any() or (ordered_indices >= self.adata.n_obs).any():
            raise ValueError("VampPrior training indices are out of AnnData bounds.")

        digest = ordered_indices_sha256(ordered_indices)
        if self.vamp_training_indices_sha256_ is not None:
            if self.vamp_training_indices_sha256_ != digest:
                raise RuntimeError(
                    "VampPrior was already initialized from a different training "
                    "boundary; continuing would silently change checkpoint semantics."
                )
            if self._freeze_prior_after_init:
                self.module.u_vamp_pseudo.requires_grad_(False)
            return

        rng = np.random.default_rng(self.u_prior_init_seed)
        selected = rng.choice(
            ordered_indices,
            min(ordered_indices.size, 10_000),
            replace=False,
        )
        n_components = self.module.resolved_u_prior_mixture_k
        if selected.size < n_components:
            raise ValueError(
                "The frozen training split has fewer cells than VampPrior components: "
                f"{selected.size} < {n_components}."
            )

        genes = take_matrix_rows(
            self.adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY),
            selected,
        ).astype(np.float32, copy=False)
        if self.module.n_input_proteins:
            proteins = take_matrix_rows(
                self.adata_manager.get_from_registry(REGISTRY_KEYS.PROTEIN_EXP_KEY),
                selected,
            ).astype(np.float32, copy=False)
            combined = np.hstack([genes, proteins])
        else:
            combined = genes

        kmeans = KMeans(
            n_clusters=n_components,
            random_state=self.u_prior_init_seed,
            n_init="auto",
        )
        kmeans.fit(combined)
        centroids = torch.as_tensor(kmeans.cluster_centers_, dtype=torch.float32)
        safe = centroids.clamp(min=1e-6, max=20.0)
        pseudo = torch.where(
            centroids > 20.0,
            centroids,
            torch.log(torch.expm1(safe)),
        )
        if not torch.isfinite(pseudo).all():
            raise RuntimeError("Training-only VampPrior initialization produced non-finite values.")
        with torch.no_grad():
            self.module.u_vamp_pseudo.copy_(
                pseudo.to(
                    device=self.module.u_vamp_pseudo.device,
                    dtype=self.module.u_vamp_pseudo.dtype,
                )
            )
        if self._freeze_prior_after_init:
            self.module.u_vamp_pseudo.requires_grad_(False)

        self.vamp_training_indices_sha256_ = digest
        self.vamp_initialization_seed_ = self.u_prior_init_seed
        self._refresh_model_summary()

    @classmethod
    def load_query_data(cls, *args, **kwargs):
        """Not supported: reference-query surgery.

        The inherited scArches path freezes the per-sample embedding tables, so
        samples new to the query keep random, untrained embeddings. See
        ``docs/user_guide.md`` (ArchesMixin / scArches surgery).
        """
        raise NotImplementedError(
            "MrTotalVI does not support reference-query surgery (load_query_data): "
            "new samples would keep random, frozen embeddings. "
            "See docs/user_guide.md (ArchesMixin / scArches surgery)."
        )

    def minify_adata(self, *args, **kwargs):
        """Not supported: minified AnnData needs latent keys MrTotalVI does not store."""
        raise NotImplementedError(
            "MrTotalVI does not support minify_adata: the minified-data path needs "
            "the TOTALVI latent posterior keys, which MrTotalVI does not store. "
            "See docs/user_guide.md (ArchesMixin / scArches surgery)."
        )

    @classmethod
    def load(
        cls,
        dir_path,
        *args,
        hierarchy_mode_override: Literal["legacy", "centered_v2"] | None = None,
        u_encoder_mode_override: Literal["sample_conditioned", "sample_blind"] | None = None,
        allow_semantic_override: bool = False,
        **kwargs,
    ):
        """Load a checkpoint with explicit, auditable mode overrides.

        Missing mode metadata is handled by the constructor defaults. A
        differing override requires ``allow_semantic_override=True`` because it
        changes model meaning without changing tensor topology.
        """
        if hierarchy_mode_override not in {None, "legacy", "centered_v2"}:
            raise ValueError("hierarchy_mode_override must be one of {None, 'legacy', 'centered_v2'}.")
        if u_encoder_mode_override not in {
            None,
            "sample_conditioned",
            "sample_blind",
        }:
            raise ValueError("u_encoder_mode_override must be one of {None, 'sample_conditioned', 'sample_blind'}.")

        # scvi-tools only warns on a gene mismatch and re-registers the passed adata, so the
        # trained names come from the checkpoint itself.
        bound = inspect.signature(TOTALVI.load).bind_partial(dir_path, *args, **kwargs).arguments
        query = bound.get("adata")
        if query is not None:
            _, trained_names, _, _ = _load_saved_files(
                dir_path,
                False,
                prefix=bound.get("prefix"),
                map_location="cpu",
                backup_url=bound.get("backup_url"),
            )
            if trained_names is not None and list(query.var_names) != list(trained_names):
                raise ValueError(
                    "adata.var_names do not match the genes the checkpoint was trained on; "
                    "subset and reorder the genes to the trained order."
                )

        token = _LOADING_CHECKPOINT.set(True)
        try:
            model = super().load(dir_path, *args, **kwargs)
        finally:
            _LOADING_CHECKPOINT.reset(token)
        model._check_feature_names(model.adata)
        module = model.module
        if (
            module.sample_embedding_prior_scale is not None or module.u_encoder_sample_embedding_prior_scale is not None
        ) and module.n_obs_per_sample is not None:
            n_trained = float(module.n_obs_per_sample.sum())
            if abs(n_trained - module._n_obs_total) > 0.5:
                warnings.warn(
                    "The loaded AnnData has "
                    f"{module._n_obs_total:.0f} cells, but the checkpoint was trained on "
                    f"{n_trained:.0f}. The sample-embedding penalty is divided by the cell "
                    "count of the AnnData passed to load, so continued training uses a "
                    "different penalty strength than the original run.",
                    UserWarning,
                    stacklevel=2,
                )
        loaded_hierarchy_mode = model.hierarchy_mode
        loaded_u_encoder_mode = model.u_encoder_mode
        resolved_hierarchy_mode = loaded_hierarchy_mode if hierarchy_mode_override is None else hierarchy_mode_override
        resolved_u_encoder_mode = loaded_u_encoder_mode if u_encoder_mode_override is None else u_encoder_mode_override
        changed = resolved_hierarchy_mode != loaded_hierarchy_mode or resolved_u_encoder_mode != loaded_u_encoder_mode
        if changed and not allow_semantic_override:
            raise ValueError(
                "A differing MrTotalVI mode override changes checkpoint semantics. "
                "Pass allow_semantic_override=True to make the change explicit."
            )
        if resolved_hierarchy_mode == "centered_v2" and not model.module._use_map:
            raise ValueError("hierarchy_mode='centered_v2' requires use_map=True.")
        if resolved_hierarchy_mode == "centered_v2" and not model.module.z_u_prior:
            raise ValueError("hierarchy_mode='centered_v2' requires z_u_prior=True.")
        if changed:
            warnings.warn(
                "Applying an explicit MrTotalVI semantic override while loading: "
                f"hierarchy_mode {loaded_hierarchy_mode!r} -> "
                f"{resolved_hierarchy_mode!r}; u_encoder_mode "
                f"{loaded_u_encoder_mode!r} -> {resolved_u_encoder_mode!r}.",
                UserWarning,
                stacklevel=2,
            )

        model.loaded_hierarchy_mode = loaded_hierarchy_mode
        model.loaded_u_encoder_mode = loaded_u_encoder_mode
        model.resolved_hierarchy_mode = resolved_hierarchy_mode
        model.resolved_u_encoder_mode = resolved_u_encoder_mode
        model.hierarchy_mode = resolved_hierarchy_mode
        model.u_encoder_mode = resolved_u_encoder_mode
        model.module.hierarchy_mode = resolved_hierarchy_mode
        model.module.u_encoder_mode = resolved_u_encoder_mode
        model.module.qu.u_encoder_mode = resolved_u_encoder_mode
        if resolved_u_encoder_mode == "sample_blind" and module.u_encoder_sample_embedding_prior_scale is not None:
            # The sample_blind encoder has no tables for this penalty, and the
            # constructor rejects the pair, so drop the penalty from the module and from
            # init_params_ (a re-saved checkpoint would otherwise fail to load).
            warnings.warn(
                "u_encoder_mode_override='sample_blind' removes the u-encoder sample "
                "tables, so u_encoder_sample_embedding_prior_scale="
                f"{module.u_encoder_sample_embedding_prior_scale} is switched off. "
                "Continued training uses a different objective.",
                UserWarning,
                stacklevel=2,
            )
            module.u_encoder_sample_embedding_prior_scale = None
            for group in model.init_params_.values():
                if isinstance(group, dict) and ("u_encoder_sample_embedding_prior_scale" in group):
                    group["u_encoder_sample_embedding_prior_scale"] = None
        if model._freeze_prior_after_init and model.vamp_training_indices_sha256_ is not None:
            model.module.u_vamp_pseudo.requires_grad_(False)
        model.init_params_["non_kwargs"]["hierarchy_mode"] = resolved_hierarchy_mode
        model.init_params_["non_kwargs"]["u_encoder_mode"] = resolved_u_encoder_mode
        # residual_prior_mode="factual" exists only for centered_v2, and legacy never reads
        # it; keep it only where the constructor accepts it, so the model saves and reloads.
        if (
            resolved_hierarchy_mode != "centered_v2"
            and getattr(model.module, "residual_prior_mode", "raw_mean") != "raw_mean"
        ):
            warnings.warn(
                "residual_prior_mode='factual' applies only to centered_v2; the legacy "
                "override resets it to 'raw_mean'.",
                UserWarning,
                stacklevel=2,
            )
            model.module.residual_prior_mode = "raw_mean"
            for group in model.init_params_.values():
                if isinstance(group, dict) and "residual_prior_mode" in group:
                    group["residual_prior_mode"] = "raw_mean"
        model._refresh_model_summary()
        return model

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        max_epochs: int | None = None,
        lr: float = 2e-3,
        accelerator: str = "auto",
        devices: int | list[int] | str = "auto",
        train_size: float | None = None,
        validation_size: float | None = None,
        shuffle_set_split: bool = True,
        batch_size: int = 256,
        early_stopping: bool = True,
        check_val_every_n_epoch: int | None = None,
        reduce_lr_on_plateau: bool = False,
        n_steps_kl_warmup: int | None = None,
        n_epochs_kl_warmup: int | None = None,
        adversarial_classifier: bool | None = None,
        datasplitter_kwargs: dict | None = None,
        plan_kwargs: dict | None = None,
        external_indexing: list[np.ndarray] | None = None,
        **kwargs,
    ) -> None:
        """Train MrTotalVI after freezing any data-derived prior boundary.

        Adam ``eps=1e-8`` and ``weight_decay=1e-8`` follow released MrVI (TOTALVI: 0.01
        and 1e-6); set them through ``plan_kwargs`` (keys you pass there win). ``lr=2e-3``
        and no plateau schedule also follow MrVI (TOTALVI: ``lr=4e-3`` with a schedule);
        both are arguments of this method and overwrite the same keys in ``plan_kwargs``. KL warm-up,
        early stopping, batch size and clipping keep their TOTALVI defaults.

        Early stopping monitors ``elbo_validation`` (the scvi-tools default), which is
        cell-weighted: its reconstruction and KL terms are plain per-cell sums. With
        ``scale_observations=True`` the training loss weights cells by donor, so
        ``elbo_validation`` is not the quantity being minimised. The training plan then also
        logs ``donor_balanced_loss_train`` and ``donor_balanced_loss_validation``, equal to
        the weighted loss the module trains on. They are for monitoring only and do not
        change the loss, the monitor or when training stops.

        Data-initialized Vamp pseudo-inputs are resolved from the exact training
        indices before the first optimizer step. Pass ``accelerator='cpu'``
        explicitly to suppress the no-GPU warning.
        """
        if self._empty_sample_positions:
            raise ValueError(
                f"Donors at index positions {self._empty_sample_positions} have no cells "
                "in the AnnData of this model, so their embeddings cannot be trained. "
                "Train on an AnnData that holds every registered donor. A loaded model may "
                "still be used for inference on a subset."
            )
        if accelerator == "auto":
            if torch.cuda.is_available():
                accelerator = "gpu"
            else:
                warnings.warn(
                    "No CUDA device found — MrTotalVI will train on CPU. "
                    "Pass accelerator='cpu' to suppress this warning, or "
                    "run on a GPU node.",
                    UserWarning,
                    stacklevel=2,
                )

        resolved_external_indexing = external_indexing
        if self._init_prior_from_data:
            split_kwargs = dict(datasplitter_kwargs or {})
            splitter = self._data_splitter_cls(
                self.adata_manager,
                train_size=train_size,
                validation_size=validation_size,
                shuffle_set_split=shuffle_set_split,
                batch_size=batch_size or settings.batch_size,
                external_indexing=external_indexing,
                **split_kwargs,
            )
            splitter.setup()
            resolved_external_indexing = [
                np.asarray(splitter.train_idx, dtype=np.int64),
                np.asarray(splitter.val_idx, dtype=np.int64),
                np.asarray(splitter.test_idx, dtype=np.int64),
            ]
            self._initialize_vamp_from_training_indices(splitter.train_idx)

        self._warn_inherited_training_schedule(
            max_epochs=max_epochs,
            train_size=train_size,
            batch_size=batch_size,
            n_steps_kl_warmup=n_steps_kl_warmup,
            n_epochs_kl_warmup=n_epochs_kl_warmup,
            adversarial_classifier=adversarial_classifier,
        )

        # TOTALVI.train() overwrites these plan_kwargs keys with its own arguments
        # (scvi/model/_totalvi.py:321-329). Keep that precedence, but say so.
        effective = {
            "lr": lr,
            "adversarial_classifier": (
                self._use_adversarial_classifier if adversarial_classifier is None else adversarial_classifier
            ),
            "reduce_lr_on_plateau": reduce_lr_on_plateau,
            "n_epochs_kl_warmup": n_epochs_kl_warmup,
            "n_steps_kl_warmup": (int(0.75 * self.adata.n_obs) if n_steps_kl_warmup is None else n_steps_kl_warmup),
        }
        ignored = {
            key: (value, effective[key])
            for key, value in (plan_kwargs or {}).items()
            if key in effective and value != effective[key]
        }
        if ignored:
            details = ", ".join(f"{key}={given!r} (used {used!r})" for key, (given, used) in ignored.items())
            warnings.warn(
                f"plan_kwargs values ignored, train() arguments win: {details}. Pass these to train() directly.",
                UserWarning,
                stacklevel=2,
            )

        plan_kwargs = {**DEFAULT_PLAN_KWARGS, **(plan_kwargs or {})}

        # scvi-tools leaves the module in eval mode after a fit and after load, and
        # Lightning >= 2.6 no longer switches it back at fit start. A second train() call
        # would then run without dropout and skip the u-invariance penalty.
        self.module.train()
        result = super().train(
            max_epochs=max_epochs,
            lr=lr,
            accelerator=accelerator,
            devices=devices,
            train_size=train_size,
            validation_size=validation_size,
            shuffle_set_split=shuffle_set_split,
            batch_size=batch_size,
            early_stopping=early_stopping,
            check_val_every_n_epoch=check_val_every_n_epoch,
            reduce_lr_on_plateau=reduce_lr_on_plateau,
            n_steps_kl_warmup=n_steps_kl_warmup,
            n_epochs_kl_warmup=n_epochs_kl_warmup,
            adversarial_classifier=adversarial_classifier,
            datasplitter_kwargs=datasplitter_kwargs,
            plan_kwargs=plan_kwargs,
            external_indexing=resolved_external_indexing,
            **kwargs,
        )
        self._record_train_cells_per_sample()
        return result

    def _record_train_cells_per_sample(self) -> None:
        """Count training cells per donor from the resolved split and warn on empty donors."""
        train_indices = getattr(self, "train_indices_", None)
        if train_indices is None:
            self.n_train_cells_per_sample_ = None
            return
        sample_codes = np.asarray(self.adata.obs["_scvi_sample"], dtype=np.int64)
        self.n_train_cells_per_sample_ = np.bincount(
            sample_codes[np.asarray(train_indices, dtype=np.int64)],
            minlength=len(self.sample_order),
        )
        untrained = np.asarray(self.sample_order)[self.n_train_cells_per_sample_ == 0]
        if untrained.size:
            warnings.warn(
                f"Donors {untrained.tolist()} have no training cells, so their observed "
                "response never entered the likelihood. Counterfactual outputs for these "
                "targets are not backed by training data; see train_support.",
                UserWarning,
                stacklevel=3,
            )

    def _warn_inherited_training_schedule(
        self,
        *,
        max_epochs: int | None,
        train_size: float | None,
        batch_size: int | None,
        n_steps_kl_warmup: int | None,
        n_epochs_kl_warmup: int | None,
        adversarial_classifier: bool | None,
    ) -> None:
        """Warn about two TOTALVI training defaults that act on the u/z hierarchy."""
        use_adversary = self._use_adversarial_classifier if adversarial_classifier is None else adversarial_classifier
        if use_adversary:
            warnings.warn(
                "The TOTALVI adversarial batch classifier is on and acts on z = u + eps, "
                "not on u alone (TOTALVI turns it on when a protein is missing "
                "in some batch). It "
                "can penalise donor information carried by eps when donors are nested "
                "in batches. Pass adversarial_classifier=False to turn it off.",
                UserWarning,
                stacklevel=3,
            )

        n_obs = self.adata.n_obs
        if max_epochs is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                epochs = get_max_epochs_heuristic(n_obs)
        else:
            epochs = max_epochs
        steps_per_epoch = int(np.ceil(n_obs * (train_size or 0.9) / (batch_size or settings.batch_size)))
        # The plan tests truthiness, so n_epochs_kl_warmup=0 falls back to steps. The
        # epoch schedule reaches 1 at epoch index n_epochs_kl_warmup, and the last
        # epoch index is max_epochs - 1.
        if n_epochs_kl_warmup:
            needed = n_epochs_kl_warmup * steps_per_epoch
            available = (epochs - 1) * steps_per_epoch
        else:
            needed = n_steps_kl_warmup if n_steps_kl_warmup is not None else int(0.75 * n_obs)
            available = epochs * steps_per_epoch
        if needed > available:
            warnings.warn(
                f"KL warm-up needs about {needed} steps but training provides about "
                f"{available} (max_epochs={epochs}), so kl_weight stops near "
                f"{available / needed:.2f} instead of 1. This down-weights the u and eps "
                "priors. Pass max_epochs, n_steps_kl_warmup or n_epochs_kl_warmup to "
                "make the schedule finish. The estimate ignores early stopping, "
                "external_indexing and validation_size.",
                UserWarning,
                stacklevel=3,
            )

    # ------------------------------------------------------------------
    # Data registration
    # ------------------------------------------------------------------

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        protein_expression_obsm_key: str,
        sample_key: str,
        labels_key: str | None = None,
        protein_names_uns_key: str | None = None,
        batch_key: str | None = None,
        panel_key: str | None = None,
        layer: str | None = None,
        size_factor_key: str | None = None,
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        **kwargs,
    ):
        """%(summary)s.

        Parameters
        ----------
        %(param_adata)s
        protein_expression_obsm_key
            Key in ``adata.obsm`` for raw, finite, non-negative, integer-like
            protein counts. Every value is validated before registration.
        sample_key
            Key in ``adata.obs`` identifying the donor/sample for each cell.
            Each unique value becomes one row in the per-sample embedding table.
        labels_key
            Optional label metadata. Registration alone never changes the
            objective; label supervision requires explicit constructor opt-in.
        protein_names_uns_key
            Key in ``adata.uns`` for protein names.
        %(param_batch_key)s
        panel_key
            Key in ``adata.obs`` for the panel used to measure proteins.
        %(param_layer)s
        %(param_size_factor_key)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s

        Returns
        -------
        %(returns)s
        """
        validate_anndata_counts(
            adata,
            layer=layer,
            protein_expression_obsm_key=protein_expression_obsm_key,
        )
        # Pure check: a DataFrame/uns conflict or duplicate names raise. Non-string labels
        # stay allowed here, as before; paths that need strings check them later.
        authoritative_protein_names(
            adata,
            protein_expression_obsm_key=protein_expression_obsm_key,
            protein_names_uns_key=protein_names_uns_key,
            required=False,
            require_strings=False,
        )
        # Add integer index column required by compute_local_statistics
        adata.obs["_indices"] = np.arange(adata.n_obs).astype(int)

        setup_method_args = cls._get_setup_method_args(**locals())

        # The panel field mirrors TOTALVI: insert at position 0 when panel_key provided
        if panel_key is not None:
            batch_field = fields.CategoricalObsField("panel", panel_key)
        else:
            batch_field = fields.CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key)

        anndata_fields = [
            fields.LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            fields.CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            fields.CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            fields.NumericalObsField(REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False),
            fields.CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
            fields.NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
            fields.ProteinObsmField(
                REGISTRY_KEYS.PROTEIN_EXP_KEY,
                protein_expression_obsm_key,
                use_batch_mask=True,
                batch_field=batch_field,
                colnames_uns_key=protein_names_uns_key,
                is_count_data=True,
            ),
            # MrTotalVI-specific: donor/sample axis
            fields.CategoricalObsField(REGISTRY_KEYS.SAMPLE_KEY, sample_key),
            # Cell indices for batched counterfactual inference
            fields.NumericalObsField(REGISTRY_KEYS.INDICES_KEY, "_indices"),
        ]
        if panel_key is not None:
            # Mirrors TOTALVI: panel field also registered at position 0
            anndata_fields.insert(0, fields.CategoricalObsField("panel", panel_key))

        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    # ------------------------------------------------------------------
    # u-space statistical APIs
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # ADR-0011: ambient- and isotype-aware protein noise model
    # ------------------------------------------------------------------

    def _configure_protein_noise(
        self,
        protein_noise_model,
        protein_background_source,
        protein_encoder_input,
        ambient_profile,
        isotype_names,
        seed_background_prior_from_ambient,
        ambient_burden_prior,
        n_nuisance,
        native_form="scale",
        background_mixing_reg=1e-3,
        learn_ambient_profile=False,
    ) -> None:
        names = [str(n) for n in self.adata_manager.get_state_registry(REGISTRY_KEYS.PROTEIN_EXP_KEY).column_names]
        needs_profile = (
            protein_noise_model != "totalvi" or protein_encoder_input != "raw" or seed_background_prior_from_ambient
        )
        if needs_profile and ambient_profile is None:
            raise ValueError("This protein noise configuration needs `ambient_profile`.")
        if isinstance(ambient_profile, dict):
            # One profile per registered batch (ADR-0011 C4 alternative), stacked in
            # the batch registry order.
            if seed_background_prior_from_ambient:
                raise ValueError("seed_background_prior_from_ambient needs a single pooled profile.")

            # Batch categories may be numeric; match profile keys by their string form,
            # with whole-number floats written as ints so 1, 1.0 and np.int64(1) agree.
            def batch_name(value) -> str:
                if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_)):
                    if float(value).is_integer():
                        return str(int(value))
                return str(value)

            batches = [
                batch_name(b)
                for b in self.adata_manager.get_state_registry(REGISTRY_KEYS.BATCH_KEY).categorical_mapping
            ]
            by_name = {batch_name(k): v for k, v in ambient_profile.items()}
            if len(by_name) != len(ambient_profile):
                raise ValueError("ambient_profile has keys that collide as strings (for example 1 and '1').")
            not_profiles = sorted(k for k, v in by_name.items() if not isinstance(v, AmbientProfile))
            if not_profiles:
                raise TypeError(f"ambient_profile values must be AmbientProfile objects; check batches {not_profiles}.")
            unknown = sorted(set(by_name) - set(batches))
            if unknown:
                raise ValueError(f"ambient_profile has keys that match no registered batch: {unknown}.")
            missing = sorted(set(batches) - set(by_name))
            if missing:
                raise ValueError(f"No ambient profile for batches {missing}.")
            parts = [by_name[b].aligned(names) for b in batches]
            stacked = AmbientProfile(
                feature_names=names,
                profile=np.stack([p.profile for p in parts]),
                log1p_mean=np.stack([p.log1p_mean for p in parts]),
                log1p_sd=np.stack([p.log1p_sd for p in parts]),
                dsb_log_mean=np.stack([p.dsb_log_mean for p in parts]),
                dsb_log_sd=np.stack([p.dsb_log_sd for p in parts]),
                n_cell_free=sum(p.n_cell_free for p in parts),
                settings={"per_batch": batches},
                droplet_counts=(
                    None
                    if any(p.droplet_counts is None for p in parts)
                    else np.stack([p.droplet_counts for p in parts])
                ),
            )
            profile = stacked
        else:
            profile = ambient_profile.aligned(names) if ambient_profile is not None else None
        if profile is not None:
            for stat in ("profile", "log1p_mean", "log1p_sd", "dsb_log_mean", "dsb_log_sd"):
                if not np.all(np.isfinite(getattr(profile, stat))):
                    raise ValueError(f"ambient_profile.{stat} must be finite.")
        isotype_mask = None
        if isotype_names is not None:
            if isinstance(isotype_names, str):
                raise TypeError(
                    f"isotype_names must be a list of protein names, not a string; pass [{isotype_names!r}]."
                )
            unknown = sorted(set(map(str, isotype_names)) - set(names))
            if unknown:
                raise ValueError(f"Isotype names not on the protein axis: {unknown}.")
            isotype_mask = np.isin(names, [str(n) for n in isotype_names])
        self.module.configure_protein_noise(
            protein_noise_model=protein_noise_model,
            protein_background_source=protein_background_source,
            protein_encoder_input=protein_encoder_input,
            ambient_profile=None if profile is None else profile.profile,
            dsb_log_mean=None if profile is None else profile.dsb_log_mean,
            dsb_log_sd=None if profile is None else profile.dsb_log_sd,
            isotype_mask=isotype_mask,
            ambient_burden_prior=tuple(ambient_burden_prior),
            n_nuisance=n_nuisance,
            native_form=native_form,
            background_mixing_reg=background_mixing_reg,
            learn_ambient_profile=learn_ambient_profile,
            ambient_droplet_counts=None if profile is None else getattr(profile, "droplet_counts", None),
        )
        self.isotype_names = (
            None if isotype_mask is None else [n for n, m in zip(names, isotype_mask, strict=True) if m]
        )
        if protein_noise_model == "ambient_isotype":
            if seed_background_prior_from_ambient:
                warnings.warn(
                    "seed_background_prior_from_ambient has no effect with protein_noise_model="
                    "'ambient_isotype': this model drops the totalVI background prior.",
                    UserWarning,
                    stacklevel=3,
                )
            panels = self.module.protein_batch_mask or {}
            blind = sorted(
                code for code, measured in panels.items() if not np.asarray(measured, bool)[isotype_mask].any()
            )
            if blind:
                warnings.warn(
                    f"Panels {blind} (panel codes) measure no isotype control, so their nonspecific "
                    "binding h is inferred from target counts alone. Treat their native protein with care.",
                    UserWarning,
                    stacklevel=3,
                )
        if seed_background_prior_from_ambient:
            # C1-C3: seed the learnable totalVI background prior per protein from the
            # empty-droplet log1p mean and SD; training may still move it.
            alpha = self.module.background_pro_alpha
            mean = torch.as_tensor(profile.log1p_mean, dtype=alpha.dtype)
            log_sd = torch.log(torch.as_tensor(np.maximum(profile.log1p_sd, 1e-2), dtype=alpha.dtype))
            with torch.no_grad():
                if alpha.ndim == 2:
                    alpha.copy_(mean[:, None].expand_as(alpha))
                    self.module.background_pro_log_beta.copy_(log_sd[:, None].expand_as(alpha))
                else:
                    alpha.copy_(mean)
                    self.module.background_pro_log_beta.copy_(log_sd)

    def _isotype_nan(self, values, protein_list=None):
        """Set isotype control columns to NaN in ``ambient_isotype`` mode (ADR-0011)."""
        if self.module.protein_noise_model != "ambient_isotype" or not self.isotype_names or values is None:
            return values
        names = np.asarray(self.adata_manager.get_state_registry(REGISTRY_KEYS.PROTEIN_EXP_KEY).column_names)
        if protein_list is not None:
            names = names[np.isin(names, list(protein_list))]
        is_iso = np.isin(names.astype(str), self.isotype_names)
        if hasattr(values, "iloc"):
            values = values.astype(float)
            values.loc[:, is_iso] = np.nan
        else:
            values = np.array(values, dtype=float)
            # Sample stacks are (cells, proteins, samples); the protein axis is 1.
            values[:, is_iso] = np.nan
        return values

    def _protein_available(self, adata, indices) -> np.ndarray | None:
        """Measured proteins per requested cell, ``(cells, proteins)`` bool in request order.

        ``None`` when every protein is measured in every batch or panel. The mask table
        is keyed by the codes of the panel field (``module.panel_key``), which is the
        batch field when no ``panel_key`` was registered.
        """
        table = self.module.protein_batch_mask
        if table is None:
            return None
        manager = self.get_anndata_manager(adata)
        codes = np.asarray(manager.get_from_registry(self.module.panel_key)).reshape(-1).astype(int)
        return np.stack([table[str(code)] for code in codes[_as_indices(adata, indices)]]).astype(bool)

    def _unavailable_nan(self, values, adata, indices, protein_list=None):
        """Set proteins not measured in a cell's panel to NaN (``values`` in request order)."""
        available = self._protein_available(adata, indices)
        if available is None or values is None:
            return values
        if protein_list is not None:
            names = np.asarray(self.adata_manager.get_state_registry(REGISTRY_KEYS.PROTEIN_EXP_KEY).column_names)
            available = available[:, np.isin(names, list(protein_list))]
        if hasattr(values, "iloc"):
            return values.astype(float).where(available)
        values = np.asarray(values, dtype=float)
        # Sample stacks are (cells, proteins, samples); the protein axis is 1.
        return np.where(available[..., None] if values.ndim == 3 else available, values, np.nan)

    @torch.inference_mode()
    def get_protein_noise_decomposition(
        self,
        adata: AnnData | None = None,
        indices: npt.ArrayLike | None = None,
        n_samples: int = 25,
        batch_size: int = 256,
    ) -> dict[str, pd.DataFrame]:
        """Expected protein components on the raw count scale (ADR-0011).

        Returns DataFrames ``native``, ``ambient``, ``nonspecific``, ``observed`` and
        ``foreground_probability`` (cells x proteins), each the mean over ``n_samples``
        draws of ``z`` from its posterior. ``observed`` is the fitted expected total count
        (the sum of the three components, the mixture mean), not the measured count.
        ``foreground_probability`` is the decoder mixture weight ``1 - pi``, not the
        posterior probability of foreground given the observed counts ``y``. Isotype
        controls are dropped from the protein columns. Proteins not measured in a
        cell's panel are ``NaN``.
        """
        self._check_if_trained(warn=False)
        if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)) or n_samples < 1:
            raise ValueError(f"n_samples must be an integer >= 1; got {n_samples!r}.")
        return self._protein_decomposition(adata, indices, n_samples, batch_size)

    @torch.inference_mode()
    def _protein_decomposition(
        self, adata, indices, n_samples, batch_size, drop_isotypes=True
    ) -> dict[str, pd.DataFrame]:
        """Body of :meth:`get_protein_noise_decomposition`, without the trained-model check."""
        adata = self._validate_anndata(adata)
        indices = _as_indices(adata, indices)
        # Backed AnnData loads each cell once in ascending order; ``inverse`` maps the
        # loaded rows back to the request order.
        loader_indices, inverse = _loader_order(adata, indices)
        names = np.asarray(self.adata_manager.get_state_registry(REGISTRY_KEYS.PROTEIN_EXP_KEY).column_names)
        keep = np.ones(names.size, dtype=bool)
        if drop_isotypes and getattr(self, "isotype_names", None):
            # self.isotype_names holds str names; registry names keep their dtype.
            keep = ~np.isin(names.astype(str), self.isotype_names)
        loader = self._make_data_loader(adata=adata, indices=loader_indices, batch_size=batch_size)
        chunks: dict[str, list] = {}
        with _eval_mode(self.module):
            for tensors in loader:
                out = self.module.protein_components(tensors, n_samples=n_samples)
                for key, value in out.items():
                    chunks.setdefault(key, []).append(value.cpu().numpy())
        index = adata.obs_names[indices]
        available = self._protein_available(adata, indices)
        result = {}
        for key, parts in chunks.items():
            values = np.concatenate(parts)
            if inverse is not None:
                values = values[inverse]
            if available is not None:
                values = np.where(available, values, np.nan)
            result[key] = pd.DataFrame(values[:, keep], index=index, columns=names[keep])
        return result

    def get_aggregated_posterior(
        self,
        adata: AnnData | None = None,
        sample: str | int | None = None,
        indices: npt.ArrayLike | None = None,
        batch_size: int = 256,
    ):
        """Compute the aggregated posterior mixture over ``u``."""
        return _get_aggregated_posterior(
            self,
            adata=adata,
            sample_key=self.sample_key,
            sample=sample,
            indices=indices,
            batch_size=batch_size,
        )

    def differential_abundance(
        self,
        adata: AnnData | None = None,
        sample_cov_keys: list[str] | None = None,
        sample_subset: list[str] | None = None,
        compute_log_enrichment: bool = False,
        omit_original_sample: bool = True,
        donor_key: str | None = None,
        batch_size: int = 128,
        n_mc_samples: int = 1,
        density_estimator: Literal["posterior", "knn", "scale_floor"] = "posterior",
        knn_k: int = 50,
        scale_floor: float | None = None,
    ) -> xr.Dataset:
        """Compute descriptive MrVI-style abundance scores over ``u``.

        Parameters
        ----------
        adata
            AnnData to compute DA on.  Defaults to the training data.
        sample_cov_keys
            Sample-level covariate column names for grouping.
        sample_subset
            Restrict to these sample names only.
        compute_log_enrichment
            If ``True``, also compute log enrichment vs. the complementary group.
        omit_original_sample
            Exclude a cell's own sample when computing the aggregated log-prob. With
            ``False`` each cell is scored leave-one-cell-out: its own component is removed
            from its own sample's density (a deliberate departure from upstream MrVI). A
            group with no reference sample for a cell scores NaN, with a warning.
        donor_key
            Column in ``adata.obs`` identifying the donor.  When set, per-donor
            log_probs are mean-centred before covariate aggregation, blocking the
            donor effect for repeated-measures designs.  Requires
            ``omit_original_sample=False`` and at least two analysed samples per
            donor; otherwise a ``ValueError`` is raised.  The returned
            ``log_probs`` stay uncentered; only the covariate aggregates use the
            centered scores.
        batch_size
            Dataloader batch size.
        n_mc_samples
            MC draws from ``q(u|x)``; ``> 1`` scores ``log E_q[p(u)]``, a smoothed
            alternative to the plug-in ``log p(E[u])``, not a bias correction.
            Outputs from ``1`` and ``> 1`` sit on different scales; do not mix them.
            See :func:`~mrtotalvi._stats.differential_abundance`.
            Default ``1`` is deterministic.
        density_estimator
            ``"posterior"`` (default, unchanged MrVI aggregated posterior), ``"knn"``
            (neighbour-count density ratio in ``u``) or ``"scale_floor"`` (aggregated
            posterior with kernel scales raised to ``scale_floor``). The non-default
            estimators are opt-in calibrations for the noise of per-cell posterior
            scales; see :func:`~mrtotalvi._stats.differential_abundance` for how each
            maps onto ``log_probs`` and the covariate outputs.
        knn_k
            Neighbours per query cell for ``"knn"``.
        scale_floor
            Minimum kernel scale for ``"scale_floor"`` (about 0.4 calibrated a
            planted-effect test); ``None`` for the other estimators.
        """
        adata = self._validate_anndata(adata)
        selected_samples, _ = validate_sample_metadata(
            adata.obs,
            sample_key=self.sample_key,
            covariate_keys=list(sample_cov_keys or []),
            donor_key=donor_key,
            sample_subset=sample_subset,
            authoritative_order=self.sample_order,
        )
        warnings.warn(
            "differential_abundance() returns descriptive, non-inferential "
            "model scores. Use a separately validated replicate-aware method "
            "for biological abundance inference.",
            UserWarning,
            stacklevel=2,
        )
        result = _differential_abundance(
            self,
            adata=adata,
            sample_key=self.sample_key,
            sample_cov_keys=sample_cov_keys,
            sample_subset=sample_subset,
            compute_log_enrichment=compute_log_enrichment,
            omit_original_sample=omit_original_sample,
            donor_key=donor_key,
            batch_size=batch_size,
            n_mc_samples=n_mc_samples,
            validated_sample_order=selected_samples,
            density_estimator=density_estimator,
            knn_k=knn_k,
            scale_floor=scale_floor,
        )
        if density_estimator != "posterior":
            result.attrs["density_estimator"] = density_estimator
        result.attrs.update(
            {
                "interpretation": "descriptive_non_inferential",
                "biological_inference_supported": False,
                "sample_order_contract": "declared_subset_order",
            }
        )
        return result

    def get_outlier_cell_sample_pairs(
        self,
        adata: AnnData | None = None,
        subsample_size: int | None = 5_000,
        quantile_threshold: float = 0.05,
        admissibility_threshold: float = 0.0,
        batch_size: int = 256,
        random_state: int | None = None,
    ) -> xr.Dataset:
        """Compute admissible cell-sample pairs using aggregated posteriors over ``u``.

        ``random_state`` seeds the per-sample subsample. ``None`` uses
        :attr:`scvi.settings.seed` when set; otherwise the result is not reproducible.
        """
        return _get_outlier_cell_sample_pairs(
            self,
            adata=adata,
            sample_key=self.sample_key,
            subsample_size=subsample_size,
            quantile_threshold=quantile_threshold,
            admissibility_threshold=admissibility_threshold,
            batch_size=batch_size,
            random_state=random_state,
        )

    def get_counterfactual_latent(
        self,
        adata: AnnData | None = None,
        indices: npt.ArrayLike | None = None,
        *,
        target_samples=None,
        inference_mode="latent_mean",
        n_draws=1,
        quantiles=(0.025, 0.5, 0.975),
        reference_indices=None,
        support_quantile=0.05,
        admissibility_threshold=0.0,
        batch_size=256,
        target_chunk_size=None,
        random_state=0,
        zarr_path=None,
        zarr_chunks=None,
    ) -> xr.Dataset:
        """Return centered latent transformations for registered samples.

        The returned dataset contains raw posterior draws for ``u``, ``z_base``,
        ``eps_raw``, ``eps_centered``, and ``z``, plus separate support and
        admissibility indicators. ``posterior_mc`` also adds posterior means and
        quantiles without dropping the ``draw`` dimension.

        Notes
        -----
        This method requires ``hierarchy_mode="centered_v2"``. Centering always
        uses the full registered sample universe, even when ``target_samples``
        requests a subset. Targets cannot extrapolate to unregistered samples,
        and outputs are model transformations rather than causal interventions.
        Public streaming/Zarr export is quarantined. Requests estimated above
        512 MiB must be reduced through explicit subsetting.
        """
        if zarr_path is not None or zarr_chunks is not None:
            raise NotImplementedError(
                "Public streaming/Zarr export is quarantined for MrTotalVI. "
                "Use bounded in-memory requests with explicit cell, target, and "
                "feature subsetting."
            )
        return _get_counterfactual_latent(
            self,
            adata=adata,
            indices=indices,
            target_samples=target_samples,
            inference_mode=inference_mode,
            n_draws=n_draws,
            quantiles=quantiles,
            reference_indices=reference_indices,
            support_quantile=support_quantile,
            admissibility_threshold=admissibility_threshold,
            batch_size=batch_size,
            target_chunk_size=target_chunk_size,
            random_state=random_state,
            zarr_path=None,
            zarr_chunks=None,
        )

    def get_counterfactual_expression(
        self,
        adata: AnnData | None = None,
        indices: npt.ArrayLike | None = None,
        *,
        target_samples=None,
        gene_list=None,
        protein_list=None,
        inference_mode="latent_mean",
        n_draws=1,
        quantiles=(0.025, 0.5, 0.975),
        batch_policy="observed",
        specified_batch=None,
        panel_policy="observed",
        specified_panel=None,
        library_policy="observed",
        specified_library_size=None,
        marginal_reference_indices=None,
        batch_size=256,
        target_chunk_size=None,
        feature_chunk_size=None,
        random_state=0,
        zarr_path=None,
        zarr_chunks=None,
    ) -> xr.Dataset:
        """Return deterministic RNA and protein expectations by registered sample.

        ``batch_policy``, ``panel_policy``, and ``library_policy`` make the
        decoder context explicit. ``observed`` holds the factual context fixed;
        ``specified`` requires registered labels or positive library sizes; and
        ``sample_balanced_marginal`` weights biological samples equally while
        retaining empirical joint technical contexts within each sample.

        Protein component means use decoder parameters analytically and never
        use the decoder's stochastic background-rate sample. Unavailable
        proteins are marked by ``protein_available=False`` and all associated
        protein estimands are ``NaN``.

        Notes
        -----
        This method requires ``hierarchy_mode="centered_v2"`` and is limited to
        registered target samples. It returns non-causal model transformations.
        Public streaming/Zarr export is quarantined. Requests estimated above
        512 MiB must be reduced through explicit subsetting.
        """
        if zarr_path is not None or zarr_chunks is not None:
            raise NotImplementedError(
                "Public streaming/Zarr export is quarantined for MrTotalVI. "
                "Use bounded in-memory requests with explicit cell, target, and "
                "feature subsetting."
            )
        return _get_counterfactual_expression(
            self,
            adata=adata,
            indices=indices,
            target_samples=target_samples,
            gene_list=gene_list,
            protein_list=protein_list,
            inference_mode=inference_mode,
            n_draws=n_draws,
            quantiles=quantiles,
            batch_policy=batch_policy,
            specified_batch=specified_batch,
            panel_policy=panel_policy,
            specified_panel=specified_panel,
            library_policy=library_policy,
            specified_library_size=specified_library_size,
            marginal_reference_indices=marginal_reference_indices,
            batch_size=batch_size,
            target_chunk_size=target_chunk_size,
            feature_chunk_size=feature_chunk_size,
            random_state=random_state,
            zarr_path=None,
            zarr_chunks=None,
        )

    def local_sample_enrichment(
        self,
        adata: AnnData | None = None,
        indices: npt.ArrayLike | None = None,
        *,
        target_samples=None,
        reference_adata=None,
        reference_indices=None,
        inference_mode="latent_mean",
        n_draws=1,
        quantiles=(0.025, 0.5, 0.975),
        group_key=None,
        contrast=None,
        donor_key=None,
        max_reference_cells_per_sample=None,
        batch_size=256,
        reference_chunk_size=None,
        random_state=0,
    ) -> xr.Dataset:
        """Return descriptive local densities over registered samples.

        Each target density is an equal-component mixture of reference
        posteriors over ``u``. A query cell is removed only from its factual
        sample mixture. ``group_key`` aggregates sample densities with
        equal-sample ``logmeanexp``; ``contrast`` is numerator minus denominator;
        and ``donor_key`` requires exact numerator/denominator pairing.

        These outputs are descriptive and non-inferential. A factual singleton
        has zero retained references, ``finite_support=False``, and ``NaN``
        density. A ``NaN`` sample carries no weight in ``group_log_density``
        or in the donor mean and median; only a group or cell with no usable
        sample is ``NaN``.

        With ``max_reference_cells_per_sample`` the retained reference cells are fixed by
        a hash of the cell names; they do not follow ``random_state``.
        """
        return _local_sample_enrichment(
            self,
            adata=adata,
            indices=indices,
            target_samples=target_samples,
            reference_adata=reference_adata,
            reference_indices=reference_indices,
            inference_mode=inference_mode,
            n_draws=n_draws,
            quantiles=quantiles,
            group_key=group_key,
            contrast=contrast,
            donor_key=donor_key,
            max_reference_cells_per_sample=max_reference_cells_per_sample,
            batch_size=batch_size,
            reference_chunk_size=reference_chunk_size,
            random_state=random_state,
        )

    def differential_expression(
        self,
        adata: AnnData | None = None,
        sample_cov_keys: list[str] | None = None,
        sample_subset: list[str] | None = None,
        indices: npt.ArrayLike | None = None,
        batch_size: int = 128,
        normalize_design_matrix: bool = True,
        mc_samples: int = 50,
        filter_inadmissible_samples: bool = False,
        store_lfc: bool = False,
        donor_key: str | None = None,
        delta: float | None = 0.3,
        lambd: float = 0.0,
        store_baseline: bool = False,
        eps_lfc: float = 1e-6,
        use_vmap: bool = False,
        **filter_samples_kwargs,
    ) -> xr.Dataset:
        """Refuse public cell-level DE/LFC for every MrTotalVI hierarchy.

        Biological differential expression requires a donor-pseudobulk method
        such as PyDESeq2, edgeR, or dreamlet. Historical latent-space machinery
        is retained only in a private reproducibility method and is not a
        calibrated inferential API.
        """
        if use_vmap:
            raise NotImplementedError("use_vmap=True is not implemented for MrTotalVI statistics.")
        raise RuntimeError(
            "MrTotalVI.differential_expression() is disabled: legacy and "
            "centered-v2 cell-level p-values/LFC are not validated for biological "
            "inference. Use donor-pseudobulk PyDESeq2, edgeR, or dreamlet."
        )

    def _legacy_differential_expression_for_reproducibility(
        self,
        adata: AnnData | None = None,
        sample_cov_keys: list[str] | None = None,
        sample_subset: list[str] | None = None,
        indices: npt.ArrayLike | None = None,
        batch_size: int = 128,
        normalize_design_matrix: bool = True,
        mc_samples: int = 50,
        filter_inadmissible_samples: bool = False,
        store_lfc: bool = False,
        donor_key: str | None = None,
        delta: float | None = 0.3,
        lambd: float = 0.0,
        store_baseline: bool = False,
        eps_lfc: float = 1e-6,
        use_vmap: bool = False,
        **filter_samples_kwargs,
    ) -> xr.Dataset:
        """Run the historical non-inferential estimator for private reproduction.

        Fits a per-cell weighted least-squares linear model on the sample-
        specific residual ``eps_d = z_d - z_base`` (the donor latent shift),
        following Boyeau et al. (2023).

        Parameters
        ----------
        adata
            AnnData to compute DE on.  Defaults to the training data.
        sample_cov_keys
            Sample-level covariate column names in ``adata.obs``.
        sample_subset
            Restrict DE to these sample names only.
        indices
            Cell indices to process (default: all cells).
        batch_size
            Dataloader batch size.
        normalize_design_matrix
            Min-max normalize each covariate column to [0, 1].
        mc_samples
            Monte-Carlo draws from ``q(u)``; ``1`` uses the posterior mean.
        filter_inadmissible_samples
            Weight out outlier cell-sample pairs via aggregated-posterior
            admissibility scores (see :meth:`get_outlier_cell_sample_pairs`).
        store_lfc
            If ``True``, compute gene/protein log2-fold-changes (LFC) and store
            ``lfc``, ``lfc_std``, and optionally ``pde`` / ``baseline_expression``
            in the returned dataset.  Feature axis is split into ``gene`` and
            ``protein`` coordinates by this wrapper.
        donor_key
            Column in ``adata.obs`` identifying the donor.  Adds donor dummy
            columns as nuisance covariates (repeated-measures approximation).
        delta
            Effect-size threshold for PDE (``P(|lfc| >= delta)``).  Used only
            when ``store_lfc=True``.  Pass ``None`` to skip PDE.
        lambd
            L2 regularisation for ``X^T M X`` inversion.
        store_baseline
            When ``store_lfc=True``, also store the batch-marginalized baseline
            expression (null decode) as ``baseline_expression``.
        eps_lfc
            Small offset added before log2 to avoid log(0).  Default ``1e-6``.
        use_vmap
            ``True`` is unsupported and fails before validation or inference.
        **filter_samples_kwargs
            Forwarded to :meth:`get_outlier_cell_sample_pairs`.
        """
        if use_vmap:
            raise NotImplementedError("use_vmap=True is not implemented for MrTotalVI statistics.")
        if self.hierarchy_mode == "centered_v2":
            raise RuntimeError(
                "The historical differential-expression implementation is not available for centered_v2 models."
            )
        adata = self._validate_anndata(adata)
        selected_samples, _ = validate_sample_metadata(
            adata.obs,
            sample_key=self.sample_key,
            covariate_keys=list(sample_cov_keys or []),
            donor_key=donor_key,
            sample_subset=sample_subset,
            authoritative_order=self.sample_order,
        )
        ds = _differential_expression(
            self,
            adata=adata,
            sample_cov_keys=list(sample_cov_keys) if sample_cov_keys else [],
            sample_subset=sample_subset,
            indices=indices,
            batch_size=batch_size,
            normalize_design_matrix=normalize_design_matrix,
            mc_samples=mc_samples,
            filter_inadmissible_samples=filter_inadmissible_samples,
            store_lfc=store_lfc,
            donor_key=donor_key,
            delta=delta,
            lambd=lambd,
            store_baseline=store_baseline,
            eps_lfc=eps_lfc,
            use_vmap=use_vmap,
            validated_sample_order=selected_samples,
            **filter_samples_kwargs,
        )
        if store_lfc and "feature" in ds.coords:
            n_genes = self.summary_stats.n_vars
            ds = ds.assign_coords(feature=(["gene"] * n_genes + ["protein"] * (ds.sizes["feature"] - n_genes)))
        ds.attrs.update(
            {
                "interpretation": "historical_private_non_inferential",
                "biological_inference_supported": False,
            }
        )
        return ds

    # ------------------------------------------------------------------
    # Latent representation
    # ------------------------------------------------------------------

    def get_normalized_expression(
        self,
        adata=None,
        indices=None,
        n_samples_overall=None,
        transform_batch=None,
        gene_list=None,
        protein_list=None,
        library_size=1,
        **kwargs,
    ):
        """TOTALVI ``get_normalized_expression`` with the library scaling applied once.

        scvi-tools 1.5.2 (``scvi/model/_totalvi.py:540``) multiplies the running RNA sum by
        ``library_size`` inside the ``transform_batch`` loop, so two batches at
        ``library_size=1000`` gave row sums near 500500. This runs the upstream method at
        ``library_size=1``, where that loop is exact, and scales the RNA output once.
        Protein output and ``library_size`` of ``1`` or ``"latent"`` are unchanged, except
        that with ``protein_noise_model="ambient_isotype"`` isotype control columns of the
        protein output are ``NaN`` (their foreground probability is not identified), and
        that proteins not measured in a cell's batch or panel are ``NaN`` (as in
        :meth:`get_protein_noise_decomposition`).

        With ``protein_noise_model`` ``"ambient"`` or ``"ambient_isotype"`` the protein
        output is the ``native`` component of :meth:`get_protein_noise_decomposition` (raw
        count scale, mean over ``n_samples`` draws) unless ``include_protein_background=True``,
        which returns the full fitted mean as before. The native path does not support
        ``transform_batch``, ``scale_protein``, ``sample_protein_mixing`` or
        ``return_mean=False`` and raises ``NotImplementedError`` for them; pass
        ``include_protein_background=True`` to use them. ``"totalvi"`` mode is unchanged.
        For backed AnnData, rows come back in the requested order (upstream returns them
        sorted but labels them in request order). See
        :meth:`scvi.model.TOTALVI.get_normalized_expression`.
        """
        resolved = self.adata if adata is None else adata
        if n_samples_overall is not None:
            # Same draw as upstream (with replacement); done here so backed data loads the drawn rows.
            indices = np.random.choice(_as_indices(resolved, indices), n_samples_overall)
        native = self.module.protein_noise_model != "totalvi" and not kwargs.get("include_protein_background", False)
        native_protein = None
        if native:
            unsupported = [
                name
                for name, used in (
                    ("transform_batch", transform_batch is not None),
                    ("scale_protein", bool(kwargs.get("scale_protein", False))),
                    ("sample_protein_mixing", bool(kwargs.get("sample_protein_mixing", False))),
                    ("return_mean=False", kwargs.get("return_mean", True) is False),
                )
                if used
            ]
            if unsupported:
                raise NotImplementedError(
                    f"With protein_noise_model={self.module.protein_noise_model!r} the protein output is the "
                    f"native component, which does not support {unsupported}. Pass "
                    "include_protein_background=True for the full fitted mean."
                )
            # Drawn first, so the draws match a get_protein_noise_decomposition call under the same seed.
            native_protein = self._native_protein(
                resolved, indices, protein_list, kwargs.get("n_samples", 1), kwargs.get("batch_size") or 256
            )
        loader_indices, inverse = _loader_order(resolved, indices)
        request = dict(
            adata=adata,
            indices=loader_indices,
            transform_batch=transform_batch,
            gene_list=gene_list,
            protein_list=protein_list,
            **kwargs,
        )
        with _eval_mode(self.module):
            library_arg = library_size if library_size is None or isinstance(library_size, str) else 1
            # Upstream indexes the protein-name tuple with a list when it builds DataFrames, so a
            # protein subset is requested as arrays and labelled here.
            frame_subset = (
                protein_list is not None
                and not kwargs.get("return_numpy")
                and not (kwargs.get("n_samples", 1) > 1 and kwargs.get("return_mean", True) is False)
            )
            if frame_subset:
                rna, protein = super().get_normalized_expression(
                    library_size=library_arg, **{**request, "return_numpy": True}
                )
                rows = np.arange(resolved.n_obs) if loader_indices is None else loader_indices
                rows = resolved.obs_names[rows]
                manager = self.get_anndata_manager(resolved)
                genes = _get_var_names_from_manager(manager)
                if gene_list is not None:
                    genes = genes[[gene in gene_list for gene in genes]]
                proteins = np.asarray(self.protein_state_registry.column_names)
                proteins = proteins[[name in protein_list for name in proteins]]
                rna = pd.DataFrame(rna, columns=genes, index=rows)
                protein = pd.DataFrame(protein, columns=proteins, index=rows)
            else:
                rna, protein = super().get_normalized_expression(library_size=library_arg, **request)
            if (
                library_arg == 1
                and library_size is not None
                and not isinstance(library_size, str)
                and library_size != 1
            ):
                rna = rna * library_size
        if inverse is not None:
            names = np.asarray(resolved.obs_names)[_as_indices(resolved, indices)]
            rna = _restore_request_order(rna, inverse, names)
            protein = _restore_request_order(protein, inverse, names)
        if native_protein is not None:
            protein = native_protein.to_numpy() if kwargs.get("return_numpy") else native_protein
        protein = self._isotype_nan(protein, protein_list)
        return rna, self._unavailable_nan(protein, resolved, indices, protein_list)

    def _native_protein(self, adata, indices, protein_list, n_samples, batch_size):
        """Native protein component as a frame with the registry columns, request order."""
        # Isotype columns stay; `_isotype_nan` blanks them in "ambient_isotype" mode only.
        native = self._protein_decomposition(adata, indices, n_samples, batch_size, drop_isotypes=False)["native"]
        names = np.asarray(self.adata_manager.get_state_registry(REGISTRY_KEYS.PROTEIN_EXP_KEY).column_names)
        if protein_list is not None:
            names = names[np.isin(names, list(protein_list))]
        return native.reindex(columns=names)

    def _in_request_order(self, method, adata, indices, **kwargs):
        """Run an inherited ``method`` loading each backed cell once, then restore request order."""
        resolved = self.adata if adata is None else adata
        loader_indices, inverse = _loader_order(resolved, indices)
        output = method(adata=adata, indices=loader_indices, **kwargs)
        if inverse is None:
            return output
        names = np.asarray(resolved.obs_names)[_as_indices(resolved, indices)]
        return _restore_request_order(output, inverse, names)

    def get_protein_foreground_probability(self, adata=None, indices=None, **kwargs):
        """TOTALVI ``get_protein_foreground_probability`` in request order; isotype columns are ``NaN``.

        Rows come back in the requested order on backed AnnData. With
        ``protein_noise_model="ambient_isotype"`` the loss is the same for any mixing logit of
        an isotype control, so its foreground probability is not identified. Those columns are
        ``NaN``, as in :meth:`get_protein_noise_decomposition`. Proteins not measured in a
        cell's batch or panel are ``NaN`` in every mode. Pass
        arguments after ``indices`` by keyword. See
        :meth:`scvi.model.TOTALVI.get_protein_foreground_probability`.
        """
        resolved = self.adata if adata is None else adata
        protein_list = kwargs.get("protein_list")
        wants_frame = not kwargs.get("return_numpy")
        # Upstream indexes a tuple with a list when it builds a DataFrame for a protein subset,
        # so ask for arrays and label them here.
        with _eval_mode(self.module):  # upstream leaves dropout and BatchNorm as the caller set them
            out = self._in_request_order(
                super().get_protein_foreground_probability, adata, indices, **{**kwargs, "return_numpy": True}
            )
        if wants_frame and out.ndim == 2:
            names = np.asarray(self.protein_state_registry.column_names)
            if protein_list is not None:
                names = names[np.isin(names, list(protein_list))]
            rows = resolved.obs_names[_as_indices(resolved, indices)]
            out = pd.DataFrame(out, index=rows, columns=names)
        out = self._isotype_nan(out, protein_list)
        return self._unavailable_nan(out, resolved, indices, protein_list)

    def get_latent_library_size(self, adata=None, indices=None, **kwargs):
        """TOTALVI ``get_latent_library_size`` with rows in the requested order on backed data."""
        return self._in_request_order(super().get_latent_library_size, adata, indices, **kwargs)

    def posterior_predictive_sample(self, adata=None, indices=None, **kwargs):
        """TOTALVI ``posterior_predictive_sample`` with rows in the requested order on backed data."""
        return self._in_request_order(super().posterior_predictive_sample, adata, indices, **kwargs)

    @torch.inference_mode()
    def get_feature_correlation_matrix(
        self,
        adata=None,
        indices=None,
        n_samples: int = 10,
        batch_size: int = 64,
        rna_size_factor: int = 1000,
        transform_batch=None,
        correlation_type: Literal["spearman", "pearson"] = "spearman",
        log_transform: bool = False,
        silent: bool = True,
    ):
        """TOTALVI ``get_feature_correlation_matrix`` that samples from ``adata``.

        The scvi-tools 1.5.2 body (``scvi/model/_totalvi.py:1046-1131``) validates
        ``adata`` but calls ``_get_denoised_samples`` without it, so correlations always
        came from the training data. This copy passes ``adata`` through. With
        ``protein_noise_model="ambient_isotype"`` the isotype rows and columns are ``NaN``,
        because a control's mixing gate is not identified. See
        :meth:`scvi.model.TOTALVI.get_feature_correlation_matrix`.
        """
        from collections.abc import Iterable

        import pandas as pd
        from scipy.stats import spearmanr
        from scvi.model._utils import (
            _get_batch_code_from_category,
            _get_var_names_from_manager,
        )
        from scvi.utils import track

        adata = self._validate_anndata(adata)
        adata_manager = self.get_anndata_manager(adata, required=True)
        if not isinstance(transform_batch, Iterable):
            transform_batch = [transform_batch]
        transform_batch = _get_batch_code_from_category(adata_manager, transform_batch)

        # Backed AnnData rejects repeated indices; load each cell once and expand back.
        loader_indices, inverse = _loader_order(adata, indices)
        corr_mats = []
        for b in track(transform_batch, disable=silent):
            with _eval_mode(self.module):
                denoised_data = self._get_denoised_samples(
                    adata=adata,
                    n_samples=n_samples,
                    batch_size=batch_size,
                    rna_size_factor=rna_size_factor,
                    transform_batch=b,
                    indices=loader_indices,
                )
            if inverse is not None:
                denoised_data = denoised_data[inverse]
            # Stack the posterior samples as extra rows: (cells * samples, features).
            flattened = np.concatenate([denoised_data[:, :, i] for i in range(n_samples)], axis=0).astype(np.float64)
            if log_transform is True:
                # Upstream reads self.n_genes, which neither TOTALVI nor MrTotalVI defines.
                n_genes = self.summary_stats.n_vars
                flattened[:, :n_genes] = np.log(flattened[:, :n_genes] + 1e-8)
                flattened[:, n_genes:] = np.log1p(flattened[:, n_genes:])
            if correlation_type == "pearson":
                corr_matrix = np.corrcoef(flattened, rowvar=False)
            else:
                corr_matrix, _ = spearmanr(flattened, axis=0)
            corr_mats.append(corr_matrix)
        corr_matrix = np.mean(np.stack(corr_mats), axis=0)
        names = np.concatenate(
            [
                np.asarray(_get_var_names_from_manager(adata_manager)),
                self.protein_state_registry.column_names,
            ]
        )
        corr = pd.DataFrame(corr_matrix, index=names, columns=names)
        if self.module.protein_noise_model == "ambient_isotype" and self.isotype_names:
            # A control's sampled mixing gate is not identified, so its correlations are not either.
            is_iso = np.isin(np.asarray(names).astype(str), self.isotype_names)
            corr.iloc[is_iso, :] = np.nan
            corr.iloc[:, is_iso] = np.nan
        return corr

    def get_latent_representation(
        self,
        adata: AnnData | None = None,
        indices: npt.ArrayLike | None = None,
        give_mean: bool = True,
        batch_size: int | None = None,
        give_z: bool = True,
        **kwargs,
    ) -> npt.NDArray:
        """Compute the latent representation of the data.

        Parameters
        ----------
        adata
            AnnData object. Defaults to the object used to initialise the model.
        indices
            Indices of observations to use.
        give_mean
            If ``True``, uses the posterior mean of ``u`` as the query to
            :class:`~.EncoderUZ`.  When ``give_z=True`` and ``give_mean=True``,
            returns ``z_mean = z_base(u_mean) + eps(u_mean, d)``.
        batch_size
            Minibatch size.
        give_z
            If ``True`` (default), returns the sample-aware ``z = z_base + eps``
            using the cell's actual donor index. If ``False``, returns ``u``.
            Under the legacy default encoder, ``u`` is sample-conditioned; it
            is sample-blind only when explicitly configured as such.

        Returns
        -------
        If ``give_z=True``, an array of shape ``(n_obs, n_latent)``.
        If ``give_z=False``, an array of shape ``(n_obs, n_latent_u)``.

        Notes
        -----
        This method does **not** delegate to
        :meth:`super().get_latent_representation` for the ``give_z=True`` path.
        The base class returns ``qz.loc`` when ``give_mean=True``, and ``qz`` is
        the distribution over ``u`` — silently discarding the donor residual.
        Both paths are implemented manually to avoid this silent wrong-output bug.
        """
        if kwargs:
            raise TypeError(
                "MrTotalVI.get_latent_representation() got unsupported keyword "
                f"arguments {sorted(kwargs)}; mc_samples, return_dist and dataloader "
                "from the base class are not implemented."
            )
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        loader_indices, inverse = _loader_order(adata, indices)
        scdl = self._make_data_loader(adata=adata, indices=loader_indices, batch_size=batch_size)

        was_training = self.module.training
        self.module.eval()
        results = []
        try:
            for tensors in scdl:
                inf_inputs = self.module._get_inference_input(tensors)
                with torch.inference_mode():
                    out = self.module.inference(**inf_inputs)

                if not give_z:
                    # Return u (sample-conditioned base, posterior mean or sample)
                    rep = out["qz"].loc if give_mean else out["u"]
                else:
                    # give_z=True: return z = z_base + eps using real sample_index.
                    # When give_mean=False, out[Z_KEY] is already z_base + eps from
                    # the reparameterised sample.  When give_mean=True we re-run qz
                    # with u_mean so the returned z uses the posterior mean of u.
                    if give_mean:
                        sample_index = inf_inputs["sample_index"]
                        u_mean = out["qz"].loc
                        with torch.inference_mode():
                            if self.hierarchy_mode == "centered_v2":
                                _, _, _, z_all = self.module._all_sample_residuals(u_mean)
                                rep = self.module._gather_sample(
                                    z_all,
                                    sample_index,
                                )
                            else:
                                z_base, eps, _ = self.module.qz(
                                    u_mean,
                                    sample_index,
                                    deterministic=True,
                                )
                                rep = z_base + eps
                    else:
                        rep = out[MODULE_KEYS.Z_KEY]

                results.append(rep.detach().cpu().numpy())
        finally:
            self.module.train(was_training)
        latent = np.concatenate(results, axis=0)
        return latent if inverse is None else latent[inverse]

    # ------------------------------------------------------------------
    # Counterfactual representations (internal)
    # ------------------------------------------------------------------

    def _iter_cf_representations(
        self,
        adata: AnnData,
        indices: npt.ArrayLike | None,
        batch_size: int,
        use_mean: bool = True,
    ):
        """Yield ``(reps, cell_positions)`` per minibatch of counterfactual ``z``.

        For each cell, runs the encoder once to get ``qz``, then loops over
        all ``n_sample`` donors.  For each donor ``d``:

        .. code-block::

            cf_sample = d * ones(n_cells)
            z_base, eps = module.qz(u, cf_sample)
            z[cell, d, :] = z_base + eps

        ``reps`` has shape ``(n_cells_in_minibatch, n_sample, n_latent)`` and
        ``cell_positions`` holds the row positions of those cells in ``adata``.
        ``use_mean=True`` uses ``qz.loc`` (posterior mean of the encoder) as ``u``;
        otherwise the sampled ``u`` from inference.
        """
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        n_sample = self.summary_stats.n_sample

        was_training = self.module.training
        self.module.eval()
        try:
            with torch.inference_mode():
                for tensors in tqdm(scdl, desc="Counterfactual representations"):
                    cell_idx = tensors[REGISTRY_KEYS.INDICES_KEY].long().flatten()

                    # Full TotalVI + MrTotalVI encoder pass
                    inf_inputs = self.module._get_inference_input(tensors)
                    base_out = self.module.inference(**inf_inputs)

                    # u: sample-conditioned base (conditioned on real donor)
                    if use_mean:
                        u = base_out["qz"].loc  # posterior mean of qu: (batch, n_latent)
                    else:
                        u = base_out["u"]  # reparameterised sample from qu

                    n_cells = u.shape[0]
                    dev = u.device

                    if self.hierarchy_mode == "centered_v2":
                        _, _, _, z_all = self.module._all_sample_residuals(u)
                        batch_reps = z_all.detach().cpu().numpy()
                    else:
                        # Preserve the legacy counterfactual path exactly.
                        cf_zs = []
                        for d in range(n_sample):
                            cf_sample = torch.full(
                                (n_cells, 1),
                                d,
                                dtype=torch.long,
                                device=dev,
                            )
                            z_base, eps, _ = self.module.qz(u, cf_sample, deterministic=use_mean)
                            cf_zs.append((z_base + eps).detach().cpu())

                        batch_reps = torch.stack(cf_zs, dim=0).permute(1, 0, 2).numpy()
                    yield batch_reps, cell_idx.numpy()
        finally:
            self.module.train(was_training)

    def _n_selected_cells(self, adata: AnnData, indices: npt.ArrayLike | None) -> int:
        """Number of cells ``indices`` selects (all cells when ``None``)."""
        if indices is None:
            return adata.n_obs
        return len(np.arange(adata.n_obs)[np.asarray(indices)])

    @staticmethod
    def _check_output_size(n_bytes: float, max_gb: float | None, what: str, hint: str) -> None:
        """Raise before allocating an output larger than ``max_gb`` GiB."""
        if max_gb is not None and n_bytes > max_gb * 2**30:
            raise ValueError(
                f"{what} would need {n_bytes / 2**30:.1f} GiB, above max_gb={max_gb}. "
                f"{hint} Pass max_gb=None to allow it."
            )

    def _compute_cf_representations(
        self,
        adata: AnnData,
        indices: npt.ArrayLike | None,
        batch_size: int,
        use_mean: bool = True,
    ) -> tuple[npt.NDArray, list[str]]:
        """Counterfactual ``z`` for every cell x donor combination.

        Returns ``(reps, cell_names)`` with ``reps`` of shape ``(n_cells, n_sample,
        n_latent)``; see :meth:`_iter_cf_representations`.
        """
        # Backed AnnData loads each cell once in ascending order; ``inverse`` maps the
        # loaded rows back to the request order.
        loader_indices, inverse = _loader_order(adata, indices)
        all_reps = []
        all_positions = []
        for batch_reps, cell_positions in self._iter_cf_representations(
            adata, loader_indices, batch_size, use_mean=use_mean
        ):
            all_positions.append(cell_positions)
            all_reps.append(batch_reps)
        reps = np.concatenate(all_reps, axis=0)
        positions = np.concatenate(all_positions)
        if inverse is not None:
            reps, positions = reps[inverse], positions[inverse]
        return reps, adata.obs_names[positions].tolist()

    # ------------------------------------------------------------------
    # Public counterfactual API
    # ------------------------------------------------------------------

    def _validate_anndata(self, adata=None, copy_if_view: bool = True):
        """Validate ``adata`` after resetting its row positions in ``obs["_indices"]``.

        ``_indices`` is written once in ``setup_anndata``. A subset, reordered or
        concatenated copy keeps the parent's positions, which then index the wrong
        rows of ``adata.obs_names`` when cell names are attached to outputs.
        """
        if adata is not None and adata is not self.adata:
            # Same in-place view handling as the base class (raises if a view and
            # copy_if_view=False), so the caller's object is registered once.
            _check_if_view(adata, copy_if_view=copy_if_view)
            adata.obs["_indices"] = np.arange(adata.n_obs).astype(int)
            self._check_feature_names(adata)
        return super()._validate_anndata(adata, copy_if_view=copy_if_view)

    def _check_feature_names(self, adata) -> None:
        """Raise if RNA or protein names differ from the registered names and order.

        The decoder reads features by position, so an equal-width permutation gives a
        wrong latent without an error; scvi-tools only warns about RNA names on load.
        """
        registered_rna = self.adata_manager.get_state_registry(REGISTRY_KEYS.X_KEY).get("column_names")
        if registered_rna is not None and list(adata.var_names) != list(registered_rna):
            raise ValueError(
                "adata.var_names do not match the registered RNA names and order; "
                "subset and reorder the genes to the registry."
            )
        key = self.adata_manager.registry["setup_args"]["protein_expression_obsm_key"]
        matrix = adata.obsm[key] if key in adata.obsm else None
        if isinstance(matrix, pd.DataFrame):
            registered = list(self.adata_manager.get_state_registry(REGISTRY_KEYS.PROTEIN_EXP_KEY).column_names)
            if list(matrix.columns) != registered:
                raise ValueError(
                    f"adata.obsm['{key}'] columns do not match the registered protein names "
                    "and order; reorder the protein matrix to the registry."
                )
        self._check_protein_names(adata)

    def _check_protein_names(self, adata) -> None:
        """Raise if ``adata.uns`` protein names differ from the registered order.

        scvi-tools compares protein names only for a DataFrame in obsm. With a numpy
        matrix the names live in ``uns``, and a reordered axis would be read against the
        registered order without an error, on transfer and on ``load``.
        """
        names_key = self.adata_manager.registry["setup_args"].get("protein_names_uns_key")
        if names_key is None:
            return
        if names_key not in adata.uns:
            warnings.warn(
                f"adata.uns['{names_key}'] is missing, so the protein axis order cannot be "
                "checked against the registry (ad.concat drops uns by default).",
                UserWarning,
                stacklevel=3,
            )
            return
        registered = list(self.adata_manager.get_state_registry(REGISTRY_KEYS.PROTEIN_EXP_KEY).column_names)
        if list(adata.uns[names_key]) != registered:
            raise ValueError(
                f"adata.uns['{names_key}'] does not match the registered protein names "
                "and order; reorder the protein matrix to the registry."
            )

    def _admissibility_rows(self, admissibility, cell_names) -> np.ndarray:
        """Boolean ``(n_cell, n_sample)`` support rows, aligned by cell and sample name."""
        if not isinstance(admissibility, xr.DataArray) or set(admissibility.dims) != {
            "cell_name",
            "sample",
        }:
            raise ValueError("admissibility must be an xarray.DataArray over (cell_name, sample).")
        if admissibility.dtype != bool:
            raise ValueError("admissibility must be boolean.")
        for dim in ("cell_name", "sample"):
            if admissibility.indexes[dim].has_duplicates:
                raise ValueError(f"admissibility has duplicate {dim} labels.")
        missing_cells = np.setdiff1d(np.asarray(cell_names), admissibility["cell_name"].values)
        missing_samples = np.setdiff1d(np.asarray(self.sample_order), admissibility["sample"].values)
        if missing_cells.size or missing_samples.size:
            raise ValueError(
                "admissibility must cover every requested cell and registered sample; "
                f"missing {missing_cells.size} cell(s) and samples {missing_samples.tolist()}."
            )
        return (
            admissibility.sel(cell_name=list(cell_names), sample=list(self.sample_order))
            .transpose("cell_name", "sample")
            .values
        )

    def get_local_sample_representation(
        self,
        adata: AnnData | None = None,
        indices: npt.ArrayLike | None = None,
        batch_size: int = 256,
        use_mean: bool = True,
        max_gb: float | None = 4.0,
        *,
        admissibility: xr.DataArray | None = None,
    ) -> xr.DataArray:
        """Compute the local per-donor latent representation.

        For each cell, returns a ``(n_sample, n_latent)`` matrix of
        counterfactual ``z`` values — one per registered donor.

        Parameters
        ----------
        adata
            AnnData object. Defaults to the object used to initialise the model.
        indices
            Indices of observations to use.
        batch_size
            Minibatch size.
        use_mean
            If ``True``, uses the posterior mean of the encoder as ``u``.
        max_gb
            Raise before encoding when the output would exceed this many GiB
            (``n_cell * n_sample * n_latent`` float32). ``None`` disables the check.
            Pass fewer ``indices`` or use :meth:`get_local_sample_distances` with
            ``groupby`` to stay under it.
        admissibility
            Optional boolean ``(cell_name, sample)`` support mask, for example
            ``get_outlier_cell_sample_pairs()["is_admissible"]``. A cell's
            representation for a sample it has no support in is set to NaN. ``None``
            (default) masks nothing, as upstream MrVI.

        Returns
        -------
        :class:`~xarray.DataArray` of shape ``(n_cell, n_sample, n_latent)``.
        """
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        self._check_output_size(
            4 * self._n_selected_cells(adata, indices) * self.summary_stats.n_sample * self.module.n_latent,
            max_gb,
            "The local sample representation",
            "Pass fewer indices.",
        )

        reps, cell_names = self._compute_cf_representations(adata, indices, batch_size, use_mean=use_mean)
        if admissibility is not None:
            reps = reps.copy()
            reps[~self._admissibility_rows(admissibility, cell_names)] = np.nan
        return xr.DataArray(
            reps,
            dims=["cell_name", "sample", "latent_dim"],
            coords={
                "cell_name": cell_names,
                "sample": self.sample_order,
            },
            name="sample_representations",
        )

    def get_local_sample_distances(
        self,
        adata: AnnData | None = None,
        indices: npt.ArrayLike | None = None,
        batch_size: int = 256,
        use_mean: bool = True,
        norm: Literal["l2", "l1"] = "l2",
        groupby: str | list[str] | None = None,
        keep_cell: bool = True,
        max_gb: float | None = 4.0,
        *,
        admissibility: xr.DataArray | None = None,
    ) -> xr.DataArray | xr.Dataset:
        """Compute cell-specific pairwise donor distance matrices.

        For each cell, computes a symmetric ``(n_sample, n_sample)`` distance
        matrix over the counterfactual donor representations. Distances are
        computed per minibatch, so the ``(n_cell, n_sample, n_latent)``
        representation is never held in memory.

        Parameters
        ----------
        adata
            AnnData object. Defaults to the object used to initialise the model.
        indices
            Indices of observations to use.
        batch_size
            Minibatch size.
        use_mean
            If ``True``, uses the posterior mean of the encoder as ``u``.
        norm
            ``"l2"`` (Euclidean) or ``"l1"`` (Manhattan).
        groupby
            One or more columns of ``adata.obs``, as in
            :meth:`scvi.external.MRVI.get_local_sample_distances`. For each column the
            result holds the mean distance matrix over the cells of each group, as
            ``<column>_distances``.
        keep_cell
            With ``groupby``, also keep the per-cell matrices (variable ``"cell"``).
            ``False`` keeps only the group means, which needs memory for one
            ``(n_sample, n_sample)`` matrix per group. It must be ``True`` without
            ``groupby``.
        max_gb
            Raise before encoding when the per-cell output (``n_cell * n_sample^2``
            float32) would exceed this many GiB. ``None`` disables the check. Use
            ``groupby`` with ``keep_cell=False``, or fewer ``indices``, to stay under it.
        admissibility
            Optional boolean ``(cell_name, sample)`` support mask, for example
            ``get_outlier_cell_sample_pairs()["is_admissible"]``. A cell's distance
            between two samples is NaN unless it has support in both. Group means then
            average each sample pair over the cells with support in that pair, and a
            ``<column>_n_cells`` variable holds those counts. ``None`` (default) masks
            nothing, as upstream MrVI.

        Returns
        -------
        :class:`~xarray.DataArray` of shape ``(n_cell, n_sample, n_sample)`` without
        ``groupby``. With ``groupby``, an :class:`~xarray.Dataset` with ``"cell"`` (if
        ``keep_cell``) and one ``<column>_distances`` variable of shape ``(<column>, sample_x, sample_y)`` per column.
        """
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        if norm not in ("l2", "l1"):
            raise ValueError(f"Unsupported norm '{norm}'. Choose 'l2' or 'l1'.")
        p = 2.0 if norm == "l2" else 1.0
        group_keys = [groupby] if isinstance(groupby, str) else list(groupby or [])
        for key in group_keys:
            if key not in adata.obs:
                raise KeyError(f"groupby column {key!r} is not in adata.obs.")
            if adata.obs[key].isna().any():
                raise ValueError(f"groupby column {key!r} has missing labels; fill or drop them first.")
        if not keep_cell and not group_keys:
            raise ValueError("keep_cell=False needs groupby; there would be no output.")

        n_sample = self.summary_stats.n_sample
        n_cells = self._n_selected_cells(adata, indices)
        if keep_cell:
            self._check_output_size(
                4 * n_cells * n_sample * n_sample,
                max_gb,
                "The per-cell distance matrices",
                "Pass groupby with keep_cell=False for group means, or fewer indices.",
            )
        # Backed AnnData loads each cell once in ascending order; ``inverse`` maps the
        # loaded rows back to the request order, and a repeated cell counts once per
        # request in the group means, as it does for in-memory data.
        loader_indices, inverse = _loader_order(adata, indices)
        weight = None if inverse is None else np.bincount(inverse)
        if keep_cell:
            n_loaded = n_cells if inverse is None else weight.size
            dists = np.empty((n_loaded, n_sample, n_sample), dtype=np.float32)
        positions_seen: list[np.ndarray] = []
        sums = {key: {} for key in group_keys}  # key -> group -> [sum matrix, n cells]
        start = 0
        for batch_reps, positions in self._iter_cf_representations(
            adata, loader_indices, batch_size, use_mean=use_mean
        ):
            chunk = torch.as_tensor(batch_reps, dtype=torch.float32)
            batch_d = torch.cdist(chunk, chunk, p=p, compute_mode="donot_use_mm_for_euclid_dist").numpy()
            batch_names = adata.obs_names[positions].tolist()
            positions_seen.append(positions)
            w = np.ones(len(positions)) if weight is None else weight[start : start + len(positions)]
            # A pair counts for a cell only where the cell has support in both samples.
            pair_ok = np.ones(batch_d.shape, dtype=bool)
            if admissibility is not None:
                support = self._admissibility_rows(admissibility, batch_names)
                pair_ok = support[:, :, None] & support[:, None, :]
                batch_d = np.where(pair_ok, batch_d, np.nan)
            if keep_cell:
                dists[start : start + len(positions)] = batch_d
            start += len(positions)
            for key in group_keys:
                labels = adata.obs[key].to_numpy()[positions]
                for label in np.unique(labels):
                    in_group = labels == label
                    w_g = w[in_group][:, None, None]
                    total = (np.where(pair_ok[in_group], batch_d[in_group], 0.0) * w_g).sum(axis=0, dtype=np.float64)
                    count = (pair_ok[in_group] * w_g).sum(axis=0).astype(np.int64)
                    prior = sums[key].get(label, [0.0, 0])
                    sums[key][label] = [prior[0] + total, prior[1] + count]

        positions = np.concatenate(positions_seen)
        if inverse is not None:
            positions = positions[inverse]
            if keep_cell:
                dists = dists[inverse]
        cell_names = adata.obs_names[positions].tolist()
        coords = {"sample_x": self.sample_order, "sample_y": self.sample_order}
        if not group_keys:
            return xr.DataArray(
                dists,
                dims=["cell_name", "sample_x", "sample_y"],
                coords={"cell_name": cell_names, **coords},
                name="sample_distances",
            )
        data_vars = {}
        if keep_cell:
            data_vars["cell"] = (["cell_name", "sample_x", "sample_y"], dists)
        for key in group_keys:
            labels = list(sums[key])
            with np.errstate(invalid="ignore", divide="ignore"):
                means = np.stack([sums[key][g][0] / sums[key][g][1] for g in labels])
            data_vars[f"{key}_distances"] = ([key, "sample_x", "sample_y"], means.astype(np.float32))
            if admissibility is not None:
                counts = np.stack([sums[key][g][1] for g in labels]).astype(np.int64)
                data_vars[f"{key}_n_cells"] = ([key, "sample_x", "sample_y"], counts)
        group_coords = {key: list(sums[key]) for key in group_keys}
        if keep_cell:
            group_coords["cell_name"] = cell_names
        return xr.Dataset(data_vars, coords={**coords, **group_coords})
