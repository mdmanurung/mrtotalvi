"""MrTotalVAE — TotalVI with MrVI-style hierarchical donor latent space."""

from __future__ import annotations

import contextlib
import inspect
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scvi import REGISTRY_KEYS
from scvi.distributions._negative_binomial import log_nb_positive
from scvi.module._constants import MODULE_KEYS
from scvi.module._totalvae import TOTALVAE
from scvi.module.base import EmbeddingModuleMixin, LossOutput, auto_move_data
from scvi.nn import DecoderTOTALVI
from torch.distributions import Normal, kl_divergence

from ._components import (
    PROTEIN_RESIDUAL_HEADS,
    AmbientNuisanceEncoder,
    BatchEmbeddingDecoderAdapter,
    DecoderTOTALVIResidual,
    EncoderUZ,
    EncoderXU_TotalVI,
    ProteinResidualEncoder,
    init_u_prior,
)
from ._components import (
    build_u_prior as _build_u_prior,
)
from ._components import (
    kl_u as _kl_u,
)
from ._kl_free_bits import check_free_bits_options, free_bits_active, kl_with_free_bits

# Active-unit cut-off of Burda et al. (2016, ICLR): a latent dimension is active when the
# variance of its posterior mean across cells exceeds 0.01.
RESIDUAL_ACTIVE_THRESHOLD = 0.01

# Registry key of the per-cell matrix q(z_p | y) reads with protein_latent_encoder_input="precomputed".
PROTEIN_LATENT_INPUT_KEY = "protein_latent_input"


def residual_collapse_metrics(eps: torch.Tensor, z: torch.Tensor) -> dict[str, torch.Tensor]:
    """Minibatch collapse diagnostics of the donor residual ``eps`` (ADR-0013).

    Pass posterior means: the loss passes the mean of ``q(eps)`` when ``use_map=False``.

    ``residual_active_units`` counts the ``eps`` dimensions whose variance across the
    cells of the minibatch exceeds :data:`RESIDUAL_ACTIVE_THRESHOLD` (Burda et al. 2016).
    ``residual_share`` is ``mean ||eps||^2 / mean ||z||^2``. Both fall to 0 when the
    residual collapses and ``z`` carries no donor effect; the ELBO does not show this.
    Lightning averages them over the minibatches of an epoch.
    """
    with torch.no_grad():
        active = (eps.var(dim=0, unbiased=False) > RESIDUAL_ACTIVE_THRESHOLD).sum().to(eps.dtype)
        share = eps.pow(2).sum(-1).mean() / z.pow(2).sum(-1).mean().clamp_min(1e-12)
    return {"residual_active_units": active, "residual_share": share}


def _with_w(decoder, head: str, z: torch.Tensor) -> torch.Tensor:
    """Decoder input of a protein head: ``[z, 0]`` if the head reads ``w``, else ``z``."""
    return decoder.with_w(head, z) if hasattr(decoder, "with_w") else z


POSTERIOR_SPLITS = ("excess", "whole")
POSTERIOR_BACKGROUNDS = ("sampled", "lognormal_mean")


def _check_posterior_options(split, background) -> None:
    """Raise ``ValueError`` for an option outside the posterior readout's allowed values."""
    for name, value, allowed in (
        ("split", split, POSTERIOR_SPLITS),
        ("background", background, POSTERIOR_BACKGROUNDS),
    ):
        if value not in allowed:
            raise ValueError(f"{name} must be one of {allowed}; got {value!r}.")


def _posterior_native(y, m1, fore_scale, theta, mixing, split):
    """Per-draw ``r * y * share``, the foreground responsibility ``r`` and the mixture log-likelihood.

    ``m1`` is the background-component mean (efficiency included), ``m2 = m1 * fore_scale``
    the foreground mean, ``theta`` the NB inverse dispersion and ``mixing`` the logit of
    ``pi = P(background)``. ``r = (1-pi) NB(y; m2) / [pi NB(y; m1) + (1-pi) NB(y; m2)]``.
    ``share`` is ``(f-1)/f`` for ``split="excess"`` (binomial thinning of the foreground
    count) and 1 for ``"whole"``.
    """
    log_nb1 = log_nb_positive(y, m1, theta)
    log_nb2 = log_nb_positive(y, m1 * fore_scale, theta)
    log_bg = F.logsigmoid(mixing) + log_nb1
    log_fg = F.logsigmoid(-mixing) + log_nb2
    log_lik = torch.logaddexp(log_bg, log_fg)
    resp = torch.exp(log_fg - log_lik)
    share = (fore_scale - 1.0) / fore_scale if split == "excess" else torch.ones_like(fore_scale)
    return resp * y * share, resp, log_lik


class MrTotalVAE(EmbeddingModuleMixin, TOTALVAE):
    r"""TotalVI VAE with an MrVI-style u→z hierarchical latent space.

    Grafts MrVI's hierarchical design onto TotalVI as faithfully as the
    multimodal input allows. A mode-selectable u-encoder
    (:class:`~.EncoderXU_TotalVI`) replaces TotalVI's stock encoder for the
    base representation ``u``; a donor-specific residual ``eps`` is computed
    by attending over a per-sample embedding table via :class:`~.EncoderUZ`.
    The legacy default gives:

    .. math::
        u \sim q_u(x_{\text{rna}},\, x_{\text{prot}},\, d)
        \quad\text{(sample-conditioned)}

    .. math::
        z = z_{\text{base}}(u) + \varepsilon,\quad
        \varepsilon \sim \text{AttentionBlock}(u,\; e_d)

    The decoder input is ``z`` with ``n_latent`` dimensions, as in stock TotalVI, so
    the hierarchy itself needs no decoder changes. With ``n_latent_u=None`` (the module
    default) ``z_base = u``; the model default ``n_latent_u=10`` projects ``u`` to
    ``n_latent`` inside :class:`~._components.EncoderUZ`. Setting ``batch_representation="embedding"`` is
    separate and does change the decoder: it is rebuilt without the batch category,
    widened by the embedding dimension, and wrapped in
    :class:`~._components.BatchEmbeddingDecoderAdapter`. See that parameter.

    Two-level KL loss:

    * ``kl_u = KL(q_u \\| p_u)`` — from the sample-conditioned u-encoder, where ``p_u``
      is by default a learned mixture-of-Gaussians prior (``u_prior="mog"``).
    * ``kl_z = -\\log p(\\varepsilon) = -\\log N(0, \\exp(\\text{pz\\_scale}))`` — eps residual.

    Parameters
    ----------
    n_input_genes
        Number of input genes.
    n_input_proteins
        Number of input proteins.
    n_sample
        Number of donors/samples.  Pass ``0`` as a placeholder; call
        :meth:`_setup_hierarchy` before training.
    n_latent_sample
        Dimension of the per-sample embedding in :class:`~.EncoderUZ`.
    z_u_prior_scale
        Log-scale of the prior on ``eps``.  ``0.0`` → ``p(eps) = N(0,1)``.
    learn_z_u_prior_scale
        If ``True``, ``pz_scale`` is a learnable :class:`~torch.nn.parameter.Parameter`;
        otherwise it is a fixed buffer (:meth:`~torch.nn.Module.register_buffer`).
    hierarchy_mode
        ``"legacy"`` preserves historical execution. ``"centered_v2"``
        centers raw residuals over the full registered sample universe.
    u_encoder_mode
        ``"sample_conditioned"`` preserves historical execution;
        ``"sample_blind"`` bypasses biological sample-conditioning parameters.
    batch_representation
        ``"one-hot"`` (default) preserves historical execution: batch is one-hot encoded
        into the u-encoder input and into every decoder :class:`~scvi.nn.FCLayers` block.
        ``"embedding"`` replaces both with a single learned embedding table, so the input
        width grows by ``batch_embedding_kwargs["embedding_dim"]`` instead of by
        ``n_batch``. Useful when the number of batches is large. Per-batch *parameter
        tables* — gene/protein dispersion, ``log_per_batch_efficiency``, the protein
        background prior and the library-size priors — stay one-hot indexed either way,
        matching :class:`~scvi.module.VAE`.
    batch_embedding_kwargs
        Keyword arguments passed to :class:`~scvi.nn.Embedding` when
        ``batch_representation="embedding"``, e.g. ``{"embedding_dim": 5}``.
    n_latent_protein_residual
        **EXPERIMENTAL.** ``0`` (default) is off and leaves every output unchanged. ``n > 0`` adds an
        ``n``-dimensional protein residual latent ``w``. See :class:`~mrtotalvi.MrTotalVI`.
    protein_residual_heads
        Which protein heads read ``[z, w]``: ``"fore_mixing"`` or ``"all"``. See
        :class:`~mrtotalvi.MrTotalVI`.
    protein_residual_encoder_input
        ``"protein"`` (default) or ``"protein_sample"``. See :class:`~mrtotalvi.MrTotalVI`.
    n_latent_protein_only
        **EXPERIMENTAL.** ``0`` (default) is off and leaves every output unchanged. ``n > 0`` adds the
        protein-only latent ``z_p`` with encoder ``q(z_p | y)``. See :class:`~mrtotalvi.MrTotalVI`, which
        also documents the ``protein_latent_*`` options.
    **kwargs
        All remaining keyword arguments forwarded verbatim to
        :class:`~scvi.module._totalvae.TOTALVAE`.

    Notes
    -----
    ``latent_distribution`` is **forced** to ``"normal"``.  Under ``"ln"``
    (softmax normalisation), ``u`` is simplex-constrained and the additive
    residual hierarchy is mathematically invalid.
    """

    def __init__(
        self,
        n_input_genes: int,
        n_input_proteins: int,
        n_sample: int = 0,
        n_latent_sample: int = 16,
        z_u_prior_scale: float = 0.0,
        learn_z_u_prior_scale: bool = False,
        n_latent_u: int | None = None,
        z_u_prior: bool = True,
        u_prior_scale: float = 0.0,
        u_prior_mixture: bool = True,
        u_prior_mixture_k: int = 20,
        u_prior_label_weight: float = 0.0,
        u_prior: str = "mog",
        u_prior_supervision: str = "none",
        qz_kwargs: dict | None = None,
        qu_kwargs: dict | None = None,
        use_map: bool = True,
        hierarchy_mode: str = "legacy",
        u_encoder_mode: str = "sample_conditioned",
        scale_observations: bool = False,
        kl_u_weight: float = 1.0,
        kl_z_weight: float = 1.0,
        residual_prior_mode: str = "raw_mean",
        anneal_protein_likelihood: bool = True,
        protein_likelihood_weight: float = 1.0,
        n_latent_protein_residual: int = 0,
        protein_residual_heads: str = "fore_mixing",
        protein_residual_encoder_input: str = "protein",
        n_latent_protein_only: int = 0,
        protein_latent_combination: str = "replace",
        protein_latent_heads: str = "fore_mixing",
        protein_latent_prior: str = "standard",
        protein_latent_encoder_input: str = "log1p",
        protein_latent_conditioning: str = "blind",
        protein_latent_consistency_weight: float = 0.0,
        protein_latent_kl_weight: float = 1.0,
        protein_latent_free_bits: float = 0.0,
        protein_latent_free_bits_granularity: str = "per_dim",
        protein_latent_free_bits_statistic: str = "batch_mean",
        protein_latent_free_bits_schedule: str = "constant",
        protein_latent_decode_draw: str = "draw",
        batch_representation: str = "one-hot",
        batch_embedding_kwargs: dict | None = None,
        u_invariance_weight: float = 0.0,
        u_invariance_mode: str = "counterfactual",
        sample_embedding_prior_post_ln: bool = False,
        **kwargs,
    ) -> None:
        if kwargs.get("latent_distribution", "normal") != "normal":
            raise ValueError(
                "MrTotalVAE requires latent_distribution='normal'. "
                "Under 'ln' (softmax), u is simplex-constrained and the "
                "additive residual hierarchy is invalid."
            )
        kwargs["latent_distribution"] = "normal"
        if (
            isinstance(n_latent_protein_residual, bool)
            or int(n_latent_protein_residual) != n_latent_protein_residual
            or n_latent_protein_residual < 0
        ):
            raise ValueError(
                f"n_latent_protein_residual must be a whole number >= 0; got {n_latent_protein_residual!r}."
            )
        if protein_residual_encoder_input not in {"protein", "protein_sample"}:
            raise ValueError("protein_residual_encoder_input must be one of {'protein', 'protein_sample'}.")
        if protein_residual_heads not in PROTEIN_RESIDUAL_HEADS:
            raise ValueError(f"protein_residual_heads must be one of {sorted(PROTEIN_RESIDUAL_HEADS)}.")
        if u_prior not in {"standard", "mog", "vamp"}:
            raise ValueError("u_prior must be exactly one of {'standard', 'mog', 'vamp'}.")
        if bool(u_prior_mixture) != (u_prior != "standard"):
            raise ValueError(
                "u_prior_mixture contradicts u_prior; pass the resolved prior enum "
                "through MrTotalVI instead of constructing a contradictory module."
            )
        if u_prior_supervision not in {"none", "labels"}:
            raise ValueError("u_prior_supervision must be one of {'none', 'labels'}.")
        if u_prior_supervision == "none" and float(u_prior_label_weight) != 0.0:
            raise ValueError("u_prior_supervision='none' requires u_prior_label_weight=0.0.")

        n_continuous_cov = kwargs.get("n_continuous_cov", 0)
        n_cats_per_cov = kwargs.get("n_cats_per_cov", None)
        # ``qu`` follows this module's own default (False). The parent TOTALVAE encoder
        # keeps TOTALVAE's default (True) unless the caller passed the flag, so it
        # is built for batch + covariates and must be fed them: ``self.encode_covariates``
        # stays the parent's value and ``qu`` reads ``self.qu_encode_covariates``.
        qu_encode_covariates = kwargs.get("encode_covariates", False)

        super().__init__(n_input_genes, n_input_proteins, **kwargs)

        # ADR-0011 protein noise model. The defaults reproduce stock totalVI; the
        # opt-in modes are configured by ``configure_protein_noise``.
        self.protein_noise_model = "totalvi"
        self.protein_background_source = "z"
        self.protein_encoder_input = "raw"
        # Protein residual latent w (EXPERIMENTAL); read by ``_build_decoder`` below.
        self.n_latent_protein_residual = int(n_latent_protein_residual)
        self.protein_residual_heads = protein_residual_heads
        self.protein_residual_encoder_input = protein_residual_encoder_input
        # Protein-only latent z_p (EXPERIMENTAL); read by ``_build_decoder`` below.
        self._set_protein_latent_options(
            n_latent_protein_only,
            protein_latent_combination,
            protein_latent_heads,
            protein_latent_prior,
            protein_latent_encoder_input,
            protein_latent_conditioning,
            protein_latent_consistency_weight,
            protein_latent_kl_weight,
            protein_latent_free_bits,
            protein_latent_free_bits_granularity,
            protein_latent_free_bits_statistic,
            protein_latent_free_bits_schedule,
            protein_latent_decode_draw,
            u_invariance_weight=u_invariance_weight,
        )

        # Store hyperparams so _setup_hierarchy can use them if called deferred
        self._n_sample = n_sample
        self._n_latent_sample = n_latent_sample
        self._z_u_prior_scale = z_u_prior_scale
        self._learn_z_u_prior_scale = learn_z_u_prior_scale
        self._n_latent_u_requested = n_latent_u
        self.z_u_prior = bool(z_u_prior)
        self.u_prior_scale_value = float(u_prior_scale)
        self.u_prior_mixture = bool(u_prior_mixture)
        self.u_prior_mixture_k = int(u_prior_mixture_k)
        self.u_prior_label_weight = float(u_prior_label_weight)
        self.u_prior_type = u_prior
        self.u_prior_supervision = u_prior_supervision
        self.qz_kwargs = qz_kwargs or {}
        self.qu_kwargs = qu_kwargs or {}
        self._use_map = use_map
        self.hierarchy_mode = hierarchy_mode
        self.u_encoder_mode = u_encoder_mode
        self._scale_observations = scale_observations
        # Per-term KL weights: static scalars applied as kl_u_weight*kl_u + kl_z_weight*kl_z
        # before the global kl_weight annealing. Defaults (1.0, 1.0) reproduce prior behavior.
        # A separate per-term annealing schedule would require training-plan changes.
        self.kl_u_weight = float(kl_u_weight)
        self.kl_z_weight = float(kl_z_weight)
        # centered_v2 residual prior: "raw_mean" (ADR-0007) averages -log p over every
        # sample's raw residual; "factual" scores only the cell's centered factual eps,
        # as upstream MrVI does for its eps.
        if residual_prior_mode not in {"raw_mean", "factual"}:
            raise ValueError("residual_prior_mode must be one of {'raw_mean', 'factual'}.")
        self.residual_prior_mode = residual_prior_mode
        # True (upstream TOTALVI) scales the protein likelihood by kl_weight during
        # warm-up; False anneals only the KL terms, as the TOTALVI paper describes.
        self.anneal_protein_likelihood = bool(anneal_protein_likelihood)
        # Scales the protein reconstruction and the protein background terms (EXPERIMENTAL).
        self.protein_likelihood_weight = float(protein_likelihood_weight)
        if not (math.isfinite(self.protein_likelihood_weight) and self.protein_likelihood_weight > 0):
            raise ValueError(f"protein_likelihood_weight must be finite and > 0; got {protein_likelihood_weight!r}.")
        # Counterfactual u-invariance penalty (see u_invariance_penalty); 0 disables it.
        self.u_invariance_weight = float(u_invariance_weight)
        if self.u_invariance_weight < 0:
            raise ValueError("u_invariance_weight must be >= 0.")
        if u_invariance_mode not in {"counterfactual", "dropout_consistency"}:
            raise ValueError("u_invariance_mode must be 'counterfactual' or 'dropout_consistency'.")
        self.u_invariance_mode = u_invariance_mode
        if self.u_invariance_weight > 0 and (hierarchy_mode != "legacy" or u_encoder_mode != "sample_blind"):
            raise ValueError("u_invariance_weight > 0 needs hierarchy_mode='legacy' and u_encoder_mode='sample_blind'.")
        # Apply the eps sample-embedding prior after layer_norm_embed (closes the LayerNorm
        # loophole: a raw-table prior can be undone by shrinking the free across-sample mean).
        self.sample_embedding_prior_post_ln = bool(sample_embedding_prior_post_ln)
        self.n_continuous_cov = int(n_continuous_cov)
        self.n_cats_per_cov = list(n_cats_per_cov or [])
        self.qu_encode_covariates = bool(qu_encode_covariates)
        if (
            u_prior == "vamp"
            and self.qu_encode_covariates
            and (self.n_continuous_cov > 0 or len(self.n_cats_per_cov) > 0)
        ):
            raise ValueError(
                "u_prior='vamp' cannot be combined with encode_covariates=True and "
                "registered categorical or continuous covariates: the VampPrior "
                "pseudo-inputs have no covariate values to feed qu. Use "
                "u_prior='mog' or 'standard', or encode_covariates=False."
            )

        # Opt-in shrinkage priors on per-sample embedding tables (None = off).
        self.sample_embedding_prior_scale: float | None = None
        self.u_encoder_sample_embedding_prior_scale: float | None = None
        self._n_obs_total: float | None = None

        self.batch_representation = batch_representation
        self._batch_dim: int | None = None
        if batch_representation == "embedding":
            self.init_embedding(REGISTRY_KEYS.BATCH_KEY, self.n_batch, **(batch_embedding_kwargs or {}))
            self._batch_dim = self.get_embedding(REGISTRY_KEYS.BATCH_KEY).embedding_dim
            self._rebuild_decoder_for_batch_embedding(self._batch_dim, kwargs)
        elif batch_representation != "one-hot":
            raise ValueError("`batch_representation` must be one of 'one-hot', 'embedding'.")

        # Opt-in protein residual latent w (EXPERIMENTAL). Off builds nothing, so the default
        # module has the same parameters and random draws as before.
        if self.n_latent_protein_residual > 0:
            if batch_representation != "embedding":
                self.decoder = self._build_decoder(None, kwargs)
            self.qw = ProteinResidualEncoder(
                n_input_proteins,
                self.n_latent_protein_residual,
                n_hidden=self._decoder_arg("n_hidden", kwargs),
            )

        # Opt-in protein-only latent z_p (EXPERIMENTAL). Off builds nothing.
        if self.n_latent_protein_only > 0:
            if batch_representation != "embedding":
                self.decoder = self._build_decoder(None, kwargs)
            self.qzp = ProteinResidualEncoder(
                n_input_proteins,
                self.n_latent_protein_only,
                n_hidden=self._decoder_arg("n_hidden", kwargs),
                log1p_input=self.protein_latent_encoder_input == "log1p",
            )
            if self.protein_latent_prior == "mog":
                # The package's MoG prior (k=20, learned means, scales and weights), on its own holder
                # so z_p's parameters do not share names with the u prior.
                self.pzp = nn.Module()
                init_u_prior(
                    self.pzp,
                    self.n_latent_protein_only,
                    u_prior_mixture=True,
                    u_prior_mixture_k=20,
                    u_prior_type="mog",
                    u_prior_supervision="none",
                )

        if n_sample > 0:
            self._setup_hierarchy(
                n_sample,
                n_latent_sample,
                z_u_prior_scale,
                learn_z_u_prior_scale,
            )

    # ------------------------------------------------------------------
    # Batch representation
    # ------------------------------------------------------------------

    def _rebuild_decoder_for_batch_embedding(self, batch_dim: int, kwargs: dict) -> None:
        """Replace the inherited decoder with one that takes a batch embedding.

        :class:`~scvi.module.TOTALVAE` builds its decoder with the batch as a one-hot
        category. Under ``batch_representation="embedding"`` the batch category is dropped
        and the decoder input is widened by ``batch_dim`` instead, then wrapped in
        :class:`~._components.BatchEmbeddingDecoderAdapter` so the call site in
        :meth:`~scvi.module.TOTALVAE.generative` is unchanged.

        Only the decoder's *conditioning* moves to the embedding. The per-batch parameter
        tables — gene/protein dispersion, ``log_per_batch_efficiency``, the protein
        background prior and the library-size priors — remain indexed by one-hot batch,
        matching :class:`~scvi.module.VAE`, since those are lookup tables rather than
        network inputs.

        Defaults are read from :meth:`~scvi.module.TOTALVAE.__init__`'s own signature
        rather than restated here, so this stays correct if the parent's defaults change.
        """
        decoder = self._build_decoder(batch_dim, kwargs)
        self.decoder = BatchEmbeddingDecoderAdapter(
            decoder,
            lambda batch_index: self.compute_embedding(REGISTRY_KEYS.BATCH_KEY, batch_index),
        )

    @staticmethod
    def _decoder_arg(name: str, kwargs: dict):
        """A TOTALVAE constructor argument: the caller's value, else the parent's own default."""
        if name in kwargs:
            return kwargs[name]
        return inspect.signature(TOTALVAE.__init__).parameters[name].default

    def _build_decoder(self, batch_dim: int | None, kwargs: dict) -> DecoderTOTALVI:
        """Build the decoder as :class:`~scvi.module.TOTALVAE` does, for one-hot or embedded batch.

        ``batch_dim=None`` keeps the batch as a one-hot category of every layer. An integer widens
        the input by the embedding size and drops the batch category. With
        ``n_latent_protein_residual > 0`` the result is a :class:`~._components.DecoderTOTALVIResidual`.
        """
        _arg = lambda name: self._decoder_arg(name, kwargs)
        n_cats_per_cov = _arg("n_cats_per_cov")
        use_batch_norm = _arg("use_batch_norm")
        use_layer_norm = _arg("use_layer_norm")
        cat_list = list([] if n_cats_per_cov is None else n_cats_per_cov)
        if batch_dim is None:
            cat_list = [self.n_batch, *cat_list]
        extra = {}
        if self.n_latent_protein_residual > 0:
            decoder_class = DecoderTOTALVIResidual
            extra = {"n_residual": self.n_latent_protein_residual, "heads": self.protein_residual_heads}
        elif self.n_latent_protein_only > 0:
            decoder_class = DecoderTOTALVIResidual
            extra = {
                "n_residual": self.n_latent_protein_only,
                "heads": self.protein_latent_heads,
                # z_p takes the place of z in the heads it feeds (replace) or follows it (concat).
                "replace_n_z": _arg("n_latent") if self.protein_latent_combination == "replace" else None,
                "require_residual": True,
            }
        else:
            decoder_class = DecoderTOTALVI
        return decoder_class(
            _arg("n_latent") + _arg("n_continuous_cov") + (batch_dim or 0),
            self.n_input_genes,
            self.n_input_proteins,
            n_layers=_arg("n_layers_decoder"),
            n_cat_list=cat_list,
            n_hidden=_arg("n_hidden"),
            dropout_rate=_arg("dropout_rate_decoder"),
            use_batch_norm=use_batch_norm in ("decoder", "both"),
            use_layer_norm=use_layer_norm in ("decoder", "both"),
            scale_activation="softplus" if _arg("use_size_factor_key") else "softmax",
            **extra,
            **(_arg("extra_decoder_kwargs") or {}),
        )

    def _batch_representation_for(self, batch_index: torch.Tensor | None) -> torch.Tensor | None:
        """Embedding for ``batch_index``, or ``None`` when batch is one-hot encoded."""
        if getattr(self, "_batch_dim", None) is None or batch_index is None:
            return None
        return self.compute_embedding(REGISTRY_KEYS.BATCH_KEY, batch_index)

    # ------------------------------------------------------------------
    # Hierarchy setup
    # ------------------------------------------------------------------

    def _setup_hierarchy(
        self,
        n_sample: int,
        n_latent_sample: int | None = None,
        z_u_prior_scale: float | None = None,
        learn_z_u_prior_scale: bool | None = None,
        use_map: bool | None = None,
        hierarchy_mode: str | None = None,
        u_encoder_mode: str | None = None,
        scale_observations: bool | None = None,
        n_obs_per_sample: torch.Tensor | None = None,
        n_labels: int | None = None,
        prior_centroids: torch.Tensor | None = None,
        freeze_prior_after_init: bool = False,
    ) -> None:
        """Build / replace the u→z hierarchy after the base TOTALVAE is initialised.

        Safe to call from :meth:`~MrTotalVI.__init__` after ``super().__init__``,
        e.g. once the registry summary stats are available.

        Parameters
        ----------
        n_sample
            Total number of donors registered in the AnnData.
        n_latent_sample, z_u_prior_scale, learn_z_u_prior_scale
            Override the values stored in ``__init__``; ``None`` → use stored value.
        use_map
            If ``False``, treat eps as stochastic (split attention output into mean and
            log-scale; reparameterise).  ``None`` → use value from ``__init__``.
        scale_observations
            If ``True``, weight each cell's ELBO contribution by
            ``1 / n_cells_in_that_sample`` so high-cell-count donors do not dominate.
            ``None`` → use value from ``__init__``.
        n_obs_per_sample
            Integer tensor of shape ``(n_sample,)`` with per-donor cell counts.
            Required (and used only) when ``scale_observations=True``.
        prior_centroids
            Optional ``(K, dim)`` tensor of cluster centroids for data-driven prior
            initialization. See :func:`init_u_prior` for semantics.
        freeze_prior_after_init
            Passed through to :func:`init_u_prior`. See its docstring for semantics.
        """
        if n_latent_sample is None:
            n_latent_sample = self._n_latent_sample
        if z_u_prior_scale is None:
            z_u_prior_scale = self._z_u_prior_scale
        if learn_z_u_prior_scale is None:
            learn_z_u_prior_scale = self._learn_z_u_prior_scale
        if use_map is None:
            use_map = self._use_map
        if hierarchy_mode is None:
            hierarchy_mode = self.hierarchy_mode
        if u_encoder_mode is None:
            u_encoder_mode = self.u_encoder_mode
        if scale_observations is None:
            scale_observations = self._scale_observations

        if hierarchy_mode not in {"legacy", "centered_v2"}:
            raise ValueError("hierarchy_mode must be one of {'legacy', 'centered_v2'}.")
        if u_encoder_mode not in {"sample_conditioned", "sample_blind"}:
            raise ValueError("u_encoder_mode must be one of {'sample_conditioned', 'sample_blind'}.")
        if hierarchy_mode == "centered_v2" and not use_map:
            raise ValueError("hierarchy_mode='centered_v2' requires use_map=True.")
        if hierarchy_mode == "centered_v2" and not self.z_u_prior:
            raise ValueError("hierarchy_mode='centered_v2' requires z_u_prior=True.")

        self._n_sample = n_sample
        self._use_map = use_map
        self.hierarchy_mode = hierarchy_mode
        self.u_encoder_mode = u_encoder_mode
        self._scale_observations = scale_observations
        if n_labels is not None:
            self.n_labels = int(n_labels)

        n_latent_u = self.n_latent if self._n_latent_u_requested is None else int(self._n_latent_u_requested)

        # Mode-selectable u-encoder: the legacy default is sample-conditioned.
        # Explicitly registered technical covariates remain available when
        # encode_covariates=True, including in sample-blind mode.
        self.qu = EncoderXU_TotalVI(
            n_input_genes=self.n_input_genes,
            n_input_proteins=self.n_input_proteins,
            n_latent=n_latent_u,
            n_sample=n_sample,
            u_encoder_mode=u_encoder_mode,
            n_batch=self.n_batch,
            n_continuous_cov=self.n_continuous_cov,
            n_cats_per_cov=self.n_cats_per_cov,
            encode_covariates=self.qu_encode_covariates,
            batch_dim=getattr(self, "_batch_dim", None),
            **self.qu_kwargs,
        )

        # Isomorphic dims (n_latent_u=None) → z_base = u, decoder unchanged
        if self.n_latent_protein_residual > 0 and self.protein_residual_encoder_input == "protein_sample":
            # The donor table needs n_sample, which is known only here; replace the sample-blind encoder.
            self.qw = ProteinResidualEncoder(
                self.n_input_proteins,
                self.n_latent_protein_residual,
                n_hidden=self.qw.fc1.out_features,
                n_sample=n_sample,
            )

        if self.n_latent_protein_only > 0 and self.protein_latent_conditioning == "donor":
            self.qzp = ProteinResidualEncoder(
                self.n_input_proteins,
                self.n_latent_protein_only,
                n_hidden=self.qzp.fc1.out_features,
                n_sample=n_sample,
                log1p_input=self.protein_latent_encoder_input == "log1p",
            )

        self.qz = EncoderUZ(
            n_latent=self.n_latent,
            n_sample=n_sample,
            n_latent_u=None if n_latent_u == self.n_latent else n_latent_u,
            n_latent_sample=n_latent_sample,
            use_map=use_map,
            **self.qz_kwargs,
        )
        init_u_prior(
            self,
            n_latent_u=n_latent_u,
            n_labels=getattr(self, "n_labels", 0),
            u_prior_scale=self.u_prior_scale_value,
            u_prior_mixture=self.u_prior_mixture,
            u_prior_mixture_k=self.u_prior_mixture_k,
            u_prior_label_weight=self.u_prior_label_weight,
            u_prior_type=getattr(self, "u_prior_type", "mog"),
            u_prior_supervision=getattr(self, "u_prior_supervision", "none"),
            u_vamp_pseudo_dim=self.n_input_genes + self.n_input_proteins,
            prior_centroids=prior_centroids,
            freeze_prior_after_init=freeze_prior_after_init,
        )

        if learn_z_u_prior_scale:
            self.pz_scale = nn.Parameter(torch.full((self.n_latent,), float(z_u_prior_scale)))
        else:
            self.register_buffer(
                "pz_scale",
                torch.full((self.n_latent,), float(z_u_prior_scale)),
            )

        # Persistent per-sample cell counts restore observation reweighting.
        if scale_observations and n_obs_per_sample is not None:
            self.register_buffer("n_obs_per_sample", n_obs_per_sample.float(), persistent=True)
        else:
            self.n_obs_per_sample = None

    # ------------------------------------------------------------------
    # Opt-in shrinkage prior on per-sample embedding tables
    # ------------------------------------------------------------------

    def _set_sample_embedding_prior(
        self,
        sample_embedding_prior_scale: float | None,
        u_encoder_sample_embedding_prior_scale: float | None,
        n_obs_total: float | None,
    ) -> None:
        """Validate and store the shrinkage settings (``None`` switches a prior off)."""
        for name, value in (
            ("sample_embedding_prior_scale", sample_embedding_prior_scale),
            ("u_encoder_sample_embedding_prior_scale", u_encoder_sample_embedding_prior_scale),
        ):
            if value is not None and (isinstance(value, bool) or not np.isfinite(float(value)) or float(value) <= 0.0):
                raise ValueError(f"{name} must be None or a finite positive float.")
        if u_encoder_sample_embedding_prior_scale is not None and (self.u_encoder_mode != "sample_conditioned"):
            raise ValueError(
                "u_encoder_sample_embedding_prior_scale requires "
                "u_encoder_mode='sample_conditioned'; the sample_blind u encoder has "
                "no sample tables."
            )
        active = sample_embedding_prior_scale is not None or u_encoder_sample_embedding_prior_scale is not None
        if active and (n_obs_total is None or float(n_obs_total) <= 0.0):
            raise ValueError("A sample-embedding prior needs n_obs_total > 0.")
        self.sample_embedding_prior_scale = (
            None if sample_embedding_prior_scale is None else float(sample_embedding_prior_scale)
        )
        self.u_encoder_sample_embedding_prior_scale = (
            None if u_encoder_sample_embedding_prior_scale is None else float(u_encoder_sample_embedding_prior_scale)
        )
        self._n_obs_total = None if n_obs_total is None else float(n_obs_total)

    @staticmethod
    def _centered_sq_norm(table: torch.Tensor) -> torch.Tensor:
        """``sum_s ||e_s - mean_s e_s||^2`` for a ``(n_sample, dim)`` table."""
        return ((table - table.mean(dim=0, keepdim=True)) ** 2).sum()

    def sample_embedding_penalty(self) -> torch.Tensor | None:
        r"""Gaussian shrinkage log-prior penalty on per-sample tables, or ``None`` if off.

        .. math::
            \sum_{T} \sum_s \lVert e_{T,s} - \bar e_T \rVert^2 / (2\sigma^2)

        The tables ``T`` are the eps attention embedding ``qz.embedding`` (scale
        ``sample_embedding_prior_scale``) and, when the second scale is set and the
        u encoder is sample-conditioned, ``qu.sample_embed`` plus the gamma/beta tables of
        both conditional normalisations (scale ``u_encoder_sample_embedding_prior_scale``).
        Centring on the across-sample mean leaves a shared offset unpenalised. This is the
        full-data negative log prior (constants dropped); the loss divides it by the number
        of cells.
        """
        terms = []
        if self.sample_embedding_prior_scale is not None:
            sigma = self.sample_embedding_prior_scale
            table = self.qz.embedding.weight
            if self.sample_embedding_prior_post_ln:
                table = self.qz.layer_norm_embed(table)
            terms.append(self._centered_sq_norm(table) / (2.0 * sigma**2))
        if self.u_encoder_sample_embedding_prior_scale is not None and self.u_encoder_mode == "sample_conditioned":
            sigma = self.u_encoder_sample_embedding_prior_scale
            tables = [
                self.qu.sample_embed.weight,
                self.qu.cond_norm1.gamma_embedding.weight,
                self.qu.cond_norm1.beta_embedding.weight,
                self.qu.cond_norm2.gamma_embedding.weight,
                self.qu.cond_norm2.beta_embedding.weight,
            ]
            terms.append(sum(self._centered_sq_norm(t) for t in tables) / (2.0 * sigma**2))
        if not terms:
            return None
        return sum(terms)

    def _vamp_component_dist(self) -> Normal:
        """Run VampPrior pseudoinputs through qu to get K component distributions.

        Pseudoinputs are in raw (rna+protein) input space; Softplus constrains
        them to ≥ 0 so that log1p inside EncoderXU_TotalVI stays well-defined.
        Reference donor (index 0) is used — the learnable pseudoinputs absorb
        the manifold variation; sample conditioning is deliberately excluded
        from the prior.
        """
        K = self.resolved_u_prior_mixture_k
        pseudo = F.softplus(self.u_vamp_pseudo)  # (K, n_input_genes + n_input_proteins)
        x_p = pseudo[:, : self.n_input_genes]
        y_p = pseudo[:, self.n_input_genes :]
        sample_idx = torch.zeros(K, 1, device=pseudo.device, dtype=torch.long)
        batch_idx = torch.zeros(K, 1, device=pseudo.device, dtype=torch.long) if self.qu_encode_covariates else None
        return self.qu(
            x_p,
            y_p,
            sample_idx,
            batch_index=batch_idx,
            batch_rep=self._batch_representation_for(batch_idx),
        )

    # ------------------------------------------------------------------
    # DataLoader → inference plumbing
    # ------------------------------------------------------------------

    def _get_inference_input(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Extend TotalVI's inference input with the sample index.

        The ``SAMPLE_KEY`` tensor is added unconditionally; it is present in
        every DataLoader batch once :meth:`MrTotalVI.setup_anndata` has
        registered it.  Minified-mode inference is not supported for
        MrTotalVAE (the hierarchy requires the full encoder path).
        """
        base = super()._get_inference_input(tensors)
        base["sample_index"] = tensors[REGISTRY_KEYS.SAMPLE_KEY]
        if self.n_latent_protein_only > 0 and self.protein_latent_encoder_input == "precomputed":
            # The supplied per-cell matrix (for example full dsb); the package does not transform it.
            base["protein_latent_input"] = tensors[PROTEIN_LATENT_INPUT_KEY]
        return base

    def _all_sample_residuals(
        self,
        u: torch.Tensor,
        target_chunk_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate and center raw residuals over every registered sample."""
        n_sample = int(self._n_sample)
        if target_chunk_size is None:
            target_chunk_size = n_sample
        if isinstance(target_chunk_size, bool) or not isinstance(target_chunk_size, int) or target_chunk_size < 1:
            raise ValueError("target_chunk_size must be a positive integer or None.")

        sample_dim = 1 if u.ndim == 2 else 2
        raw_chunks: list[torch.Tensor] = []
        z_base: torch.Tensor | None = None
        for start in range(0, n_sample, target_chunk_size):
            stop = min(start + target_chunk_size, n_sample)
            targets = torch.arange(start, stop, device=u.device, dtype=torch.long)
            n_targets = targets.numel()
            if u.ndim == 2:
                n_cells, n_latent_u = u.shape
                expanded_u = (
                    u.unsqueeze(1).expand(n_cells, n_targets, n_latent_u).reshape(n_cells * n_targets, n_latent_u)
                )
                expanded_targets = targets.unsqueeze(0).expand(n_cells, n_targets).reshape(n_cells * n_targets, 1)
                chunk_z_base, chunk_raw, _ = self.qz(expanded_u, expanded_targets)
                chunk_z_base = chunk_z_base.reshape(n_cells, n_targets, self.n_latent)
                chunk_raw = chunk_raw.reshape(n_cells, n_targets, self.n_latent)
            elif u.ndim == 3:
                n_draws, n_cells, n_latent_u = u.shape
                expanded_u = (
                    u.unsqueeze(2)
                    .expand(n_draws, n_cells, n_targets, n_latent_u)
                    .reshape(n_draws, n_cells * n_targets, n_latent_u)
                )
                expanded_targets = targets.unsqueeze(0).expand(n_cells, n_targets).reshape(n_cells * n_targets, 1)
                chunk_z_base, chunk_raw, _ = self.qz(expanded_u, expanded_targets)
                chunk_z_base = chunk_z_base.reshape(n_draws, n_cells, n_targets, self.n_latent)
                chunk_raw = chunk_raw.reshape(n_draws, n_cells, n_targets, self.n_latent)
            else:
                raise ValueError("u must have shape (cell, latent) or (draw, cell, latent).")

            if z_base is None:
                z_base = chunk_z_base.select(sample_dim, 0)
            raw_chunks.append(chunk_raw)

        if z_base is None:  # pragma: no cover - construction rejects an empty registry
            raise RuntimeError("The registered sample universe is empty.")
        eps_raw = torch.cat(raw_chunks, dim=sample_dim)
        eps_centered = eps_raw - eps_raw.mean(dim=sample_dim, keepdim=True)
        z_all = z_base.unsqueeze(sample_dim) + eps_centered
        return z_base, eps_raw, eps_centered, z_all

    @staticmethod
    def _gather_sample(
        values: torch.Tensor,
        sample_index: torch.Tensor,
    ) -> torch.Tensor:
        """Gather one registered sample per cell from an all-sample tensor."""
        sample_index = sample_index.to(device=values.device, dtype=torch.int64).flatten()
        if values.ndim == 3:
            gather_index = sample_index[:, None, None].expand(-1, 1, values.shape[-1])
            return values.gather(1, gather_index).squeeze(1)
        gather_index = sample_index[None, :, None, None].expand(values.shape[0], -1, 1, values.shape[-1])
        return values.gather(2, gather_index).squeeze(2)

    # ------------------------------------------------------------------
    # Inference: u → (z_base, eps) → z
    # ------------------------------------------------------------------

    @auto_move_data
    def _regular_inference(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        batch_index: torch.Tensor | None = None,
        panel_index: torch.Tensor | None = None,
        label: torch.Tensor | None = None,
        n_samples: int = 1,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        sample_index: torch.Tensor | None = None,
        cf_sample: torch.Tensor | None = None,
        target_chunk_size: int | None = None,
        protein_latent_input: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict]:
        """Compute inference quantities with the hierarchical u→z decomposition.

        Calls TotalVI's ``_regular_inference`` to obtain library size, protein
        background priors, and dispersion parameters (all of which are parallel
        to ``z`` and require no ``z`` dependency), then runs the configured
        u-encoder to replace TotalVI's ``qz`` and ``z``:

        .. code-block::

            qu = EncoderXU_TotalVI(x, y, real_sample)  # sample-conditioned Normal
            u  = qu.rsample()                           # batch or MC by batch
            z_base, eps = qz(u, cf_sample)              # attention over per-sample embed
            z  = z_base + eps                           # → decoder input

        ``QZ_KEY`` in the returned dict is replaced with ``qu``. Our ``loss()``
        override computes ``kl_u = KL(qu || p(u))`` with the configured ``u`` prior
        (standard, MoG or Vamp) and adds ``kl_z`` for ``eps``.

        Parameters
        ----------
        sample_index
            Integer donor index, shape ``(batch, 1)`` or ``(batch,)``.
            Used by ``qu``; always the **real** donor (never counterfactual).
        cf_sample
            Counterfactual donor override for ``eps`` only.  If not ``None``,
            replaces ``sample_index`` in the :class:`~.EncoderUZ` forward pass.
            ``qu`` always uses ``sample_index`` (real donor).
        """
        # TotalVI encoder pass — we keep ql, library_gene, back_mean_prior, dispersions.
        # Its qz and z are discarded and replaced below.
        out = super()._regular_inference(
            x,
            y,
            batch_index=batch_index,
            panel_index=panel_index,
            label=label,
            n_samples=n_samples,
            cont_covs=cont_covs,
            cat_covs=cat_covs,
        )

        if self.n_latent_protein_residual > 0:
            # q(w | y): protein counts only, plus the real donor when protein_residual_encoder_input is
            # 'protein_sample' (never the counterfactual donor). One draw per
            # Monte-Carlo draw of z, so the loss averages over w as it does over u.
            qw = self.qw(y, sample_index)
            out["qw"] = qw
            out["w"] = qw.rsample((n_samples,)) if n_samples > 1 else qw.rsample()

        if self.n_latent_protein_only > 0:
            # q(z_p | y): protein counts only (plus the real donor when donor-conditioned, never the
            # counterfactual donor). One draw per Monte-Carlo draw of z. Readouts with
            # protein_latent_decode_draw="mean" decode the posterior mean instead.
            qzp = self._protein_latent_dist(y, batch_index, sample_index, protein_latent_input)
            out["qzp"] = qzp
            if self._zp_mean_decode and not self.training:
                zp = qzp.loc if n_samples == 1 else qzp.loc.unsqueeze(0).expand(n_samples, *qzp.loc.shape)
            else:
                zp = qzp.rsample((n_samples,)) if n_samples > 1 else qzp.rsample()
            out["zp"] = zp

        if not hasattr(self, "qz"):
            # Hierarchy not yet built (n_sample=0 placeholder). Return TotalVI outputs.
            return out

        # Pass technical covariates only when encode_covariates=True. Biological
        # sample conditioning is controlled separately by u_encoder_mode.
        qu_batch_index = batch_index if self.qu_encode_covariates else None
        qu = self.qu(
            x,
            self._qu_protein_input(y, batch_index),
            sample_index,
            batch_index=qu_batch_index,
            cont_covs=cont_covs if self.qu_encode_covariates else None,
            cat_covs=cat_covs if self.qu_encode_covariates else None,
            batch_rep=self._batch_representation_for(qu_batch_index),
        )  # Normal: params (batch, n_latent_u)
        if n_samples > 1:
            u = qu.rsample((n_samples,))  # (n_samples, batch, n_latent)
        else:
            u = qu.rsample()  # (batch, n_latent)

        # Replace TotalVI's qz with qu so base loss computes KL(qu, N(0,1)) as kl_u
        out[MODULE_KEYS.QZ_KEY] = qu
        out["qu"] = qu

        # eps residual: cf_sample enables counterfactual donor substitution
        sample_index_cf = sample_index if cf_sample is None else cf_sample
        if self.hierarchy_mode == "legacy":
            z_base, eps, eps_dist = self.qz(u, sample_index_cf)
            z = z_base + eps
        else:
            z_base, eps_raw_all, eps_centered_all, z_all = self._all_sample_residuals(
                u,
                target_chunk_size=target_chunk_size,
            )
            eps = self._gather_sample(eps_centered_all, sample_index_cf)
            z = self._gather_sample(z_all, sample_index_cf)
            eps_dist = None
            out["eps_raw_all"] = eps_raw_all
            out["eps_centered_all"] = eps_centered_all
            out["z_all"] = z_all

        out[MODULE_KEYS.Z_KEY] = z
        out["u"] = u
        out["z_base"] = z_base
        if self.protein_noise_model != "totalvi":
            qc, qh = self.nuisance_encoder(y, getattr(self, "isotype_index", None), batch_index)
            out["qc"] = qc
            out["log_c"] = qc.rsample((n_samples,)) if n_samples > 1 else qc.rsample()
            out["qh"] = qh
            if qh is not None:
                out["h"] = qh.rsample((n_samples,)) if n_samples > 1 else qh.rsample()
        out["eps"] = eps
        out["eps_dist"] = eps_dist  # None when use_map=True
        return out

    def build_u_prior(
        self,
        u: torch.Tensor,
        label_index: torch.Tensor | None = None,
    ):
        """Build the configured prior over ``u``."""
        return _build_u_prior(self, u, label_index)

    def kl_u(
        self,
        qu: Normal,
        sampled_u: torch.Tensor,
        label_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute ``KL(q(u|x,s) || p(u))`` under the configured prior."""
        return _kl_u(self, qu, sampled_u, label_index)

    def _loss_with_mc_samples(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | dict],
        generative_outputs: dict[str, torch.Tensor | dict],
        pro_recons_weight: float,
        kl_weight: float,
    ) -> LossOutput:
        """Compute the TOTALVAE loss when hierarchy outputs carry MC samples."""
        qz = inference_outputs[MODULE_KEYS.QZ_KEY]
        ql = inference_outputs[MODULE_KEYS.QL_KEY]
        px_ = generative_outputs["px_"]
        py_ = generative_outputs["py_"]
        per_batch_efficiency = generative_outputs["per_batch_efficiency"]

        x = tensors[REGISTRY_KEYS.X_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        panel_index = tensors[self.panel_key]
        y = tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY]

        if self.protein_batch_mask is not None:
            pro_batch_mask_minibatch = torch.zeros_like(y)
            for b in torch.unique(panel_index):
                b_indices = (panel_index == b).reshape(-1)
                pro_batch_mask_minibatch[b_indices] = torch.tensor(
                    self.protein_batch_mask[str(int(b.item()))].astype("float32"),
                    device=y.device,
                )
        else:
            pro_batch_mask_minibatch = None

        reconst_loss_gene, reconst_loss_protein = self.get_reconstruction_loss(
            x,
            y,
            px_,
            py_,
            pro_batch_mask_minibatch,
            per_batch_efficiency,
        )

        # NOTE: This kl_div_z is a *preliminary* placeholder using the pre-hierarchical
        # KL(q_z, N(0,1)).  Both this value and the `loss` computed below are discarded
        # by the calling loss() method, which recomputes them using kl_u + kl_z after
        # the two-level KL is available.  It is kept here to satisfy LossOutput's
        # interface contract (kl_local["kl_div_z"] must be present).
        kl_div_z = kl_divergence(qz, Normal(0, 1)).sum(dim=1)
        if not self.use_observed_lib_size:
            n_batch = self.library_log_means.shape[1]
            local_library_log_means = F.linear(
                F.one_hot(batch_index.squeeze(-1), n_batch).float(),
                self.library_log_means,
            )
            local_library_log_vars = F.linear(
                F.one_hot(batch_index.squeeze(-1), n_batch).float(),
                self.library_log_vars,
            )
            kl_div_l_gene = kl_divergence(
                ql,
                Normal(local_library_log_means, torch.sqrt(local_library_log_vars)),
            ).sum(dim=1)
        else:
            kl_div_l_gene = torch.zeros_like(kl_div_z)

        kl_div_back_pro_full = kl_divergence(
            Normal(py_["back_alpha"], py_["back_beta"]),
            inference_outputs["back_mean_prior"],
        )
        lkl_back_pro_full = -torch.distributions.LogNormal(
            torch.tensor([0.0], device=x.device),
            torch.tensor([1.0], device=x.device),
        ).log_prob(per_batch_efficiency)
        # totalVI's fixed 1e-3 pull of the mixing toward background; opt-in models may
        # reweight it (``background_mixing_reg``).
        reg = getattr(self, "background_mixing_reg", 1e-3)
        lkl_protein_expressed = -reg * torch.distributions.Bernoulli(logits=py_["mixing"]).log_prob(
            torch.ones_like(py_["mixing"])
        )
        if self.protein_noise_model == "ambient_isotype":
            # The background is A + N with its own priors; the totalVI background
            # prior no longer describes any term (ADR-0011).
            kl_div_back_pro_full = torch.zeros_like(kl_div_back_pro_full)
            lkl_protein_expressed = lkl_protein_expressed * self.target_mask
        if pro_batch_mask_minibatch is not None:
            while pro_batch_mask_minibatch.ndim < kl_div_back_pro_full.ndim:
                pro_batch_mask_minibatch = pro_batch_mask_minibatch.unsqueeze(0)
            kl_div_back_pro_full = pro_batch_mask_minibatch.bool() * kl_div_back_pro_full
            if self.protein_noise_model != "totalvi":
                # A protein the panel never measured has no count to pull its gate; the stock
                # totalvi mode keeps the upstream penalty unchanged.
                lkl_protein_expressed = pro_batch_mask_minibatch.bool() * lkl_protein_expressed
        kl_div_back_pro = (
            kl_div_back_pro_full.sum(dim=-1) + lkl_back_pro_full.sum(dim=-1) + lkl_protein_expressed.sum(dim=-1)
        )

        # loss() rebuilds the training loss from the per-cell terms below, so this
        # scalar only keeps the LossOutput complete for direct callers.
        protein_weight = kl_weight if self.anneal_protein_likelihood else 1.0
        w_pro = self.protein_likelihood_weight
        loss = torch.mean(
            reconst_loss_gene
            + protein_weight * pro_recons_weight * w_pro * reconst_loss_protein
            + kl_weight * kl_div_z
            + kl_div_l_gene
            + kl_weight * w_pro * kl_div_back_pro
        )
        # Average the Monte-Carlo dimension so every per-cell term is (batch,).
        # Otherwise LossOutput reads n_samples as n_obs_minibatch, and compute_elbo
        # cannot stack terms of different shapes. The mean over (mc, batch) equals the
        # mean over batch of the MC mean, so the scalar loss is unchanged.
        if reconst_loss_gene.ndim > 1:
            reconst_loss_gene = reconst_loss_gene.mean(dim=0)
        if reconst_loss_protein.ndim > 1:
            reconst_loss_protein = reconst_loss_protein.mean(dim=0)
        if kl_div_back_pro.ndim > 1:
            kl_div_back_pro = kl_div_back_pro.mean(dim=0)
        kl_div_back_pro = kl_div_back_pro + self._nuisance_kl(inference_outputs)
        reconst_losses = {
            "reconst_loss_gene": reconst_loss_gene,
            "reconst_loss_protein": reconst_loss_protein,
        }
        kl_local = {
            "kl_div_z": kl_div_z,
            "kl_div_l_gene": kl_div_l_gene,
            "kl_div_back_pro": kl_div_back_pro,
        }
        extra_metrics = (
            {
                "z": inference_outputs[MODULE_KEYS.Z_KEY],
                "batch": tensors[REGISTRY_KEYS.BATCH_KEY],
                "labels": tensors[REGISTRY_KEYS.LABELS_KEY],
            }
            if self.extra_payload_autotune
            else {}
        )
        return LossOutput(
            loss=loss,
            reconstruction_loss=reconst_losses,
            kl_local=kl_local,
            extra_metrics=extra_metrics,
            n_obs_minibatch=x.shape[0],
        )

    @torch.inference_mode()
    def sample(self, tensors, n_samples=1):
        """Sample counts from the generative model, with the protein efficiency applied.

        The loss fits protein counts with means ``per_batch_efficiency * rate_back`` and
        ``per_batch_efficiency * rate_fore``. The inherited TOTALVAE.sample drops the
        factor, so its protein samples sit on a different scale from the fitted
        distribution (stock TOTALVI shares this). This override uses the fitted means.
        """
        from scvi.distributions import NegativeBinomial, NegativeBinomialMixture, ZeroInflatedNegativeBinomial

        _, generative_outputs = self.forward(
            tensors,
            inference_kwargs={"n_samples": n_samples},
            compute_loss=False,
        )
        px_ = generative_outputs["px_"]
        py_ = generative_outputs["py_"]
        efficiency = generative_outputs["per_batch_efficiency"]
        if self.gene_likelihood == "zinb":
            rna_dist = ZeroInflatedNegativeBinomial(mu=px_["rate"], theta=px_["r"], zi_logits=px_["dropout"])
        else:
            rna_dist = NegativeBinomial(mu=px_["rate"], theta=px_["r"])
        protein_dist = NegativeBinomialMixture(
            mu1=efficiency * py_["rate_back"],
            mu2=efficiency * py_["rate_fore"],
            theta1=py_["r"],
            mixture_logits=py_["mixing"],
        )
        return rna_dist.sample().cpu(), protein_dist.sample().cpu()

    def marginal_ll(self, *args, **kwargs):
        """Not defined for MrTotalVAE.

        The inherited TOTALVAE estimator scores ``z = u + eps`` under N(0, I) and
        evaluates ``q(u | x)`` at ``z``. That ignores the MoG/Vamp prior on ``u``
        and the ``eps`` terms, so the value is not an importance-sampled
        ``log p(x, y)`` for this model.
        """
        raise NotImplementedError(
            "MrTotalVI does not implement get_marginal_ll: the inherited TOTALVI "
            "estimator uses a N(0, I) prior on z and omits the u prior and eps "
            "terms, so it is not a valid marginal likelihood for this model. Use "
            "get_elbo or get_reconstruction_error for model comparison."
        )

    # ------------------------------------------------------------------
    # Loss: two-level KL (kl_u + kl_z)
    # ------------------------------------------------------------------

    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | dict],
        generative_outputs: dict[str, torch.Tensor | dict],
        pro_recons_weight: float = 1.0,
        kl_weight: float = 1.0,
    ) -> LossOutput:
        r"""Extend TotalVI's loss with the second-level KL for the sample residual.

        TotalVI's existing ``kl_div_z`` term is repurposed as ``kl_u =
        KL(q_u \| p_u)`` where ``p_u`` is the configured prior (default: a
        learned mixture-of-Gaussians, see :attr:`u_prior_mixture`).  This
        method adds:

        .. math::
            kl_z = -\log p(\varepsilon) = -\log N(0,\; \exp(\text{pz\_scale}))(\varepsilon)

        and folds it into both ``loss`` and ``kl_local["kl_div_z"]``.

        The ``LogNormal(0, 1)`` prior on the batch-protein efficiency is applied per cell, as
        in stock TotalVI. Each cell adds ``-log p(efficiency)`` for its batch, so the shared
        parameter's prior weight grows with the batch's cell count.
        """
        if "eps" in inference_outputs and (
            inference_outputs[MODULE_KEYS.Z_KEY].ndim > 2 or self.protein_noise_model != "totalvi"
        ):
            loss_out = self._loss_with_mc_samples(
                tensors,
                inference_outputs,
                generative_outputs,
                pro_recons_weight,
                kl_weight,
            )
        else:
            loss_out = super().loss(
                tensors,
                inference_outputs,
                generative_outputs,
                pro_recons_weight,
                kl_weight,
            )

        protein_weight = kl_weight if self.anneal_protein_likelihood else 1.0
        w_pro = self.protein_likelihood_weight
        if "eps" not in inference_outputs:
            # Hierarchy not built (n_sample=0 placeholder). The parent loss annealed the
            # protein term with kl_weight; add the rest back when it should not.
            if not self.anneal_protein_likelihood:
                loss_out.loss = loss_out.loss + (1 - kl_weight) * pro_recons_weight * (
                    loss_out.reconstruction_loss["reconst_loss_protein"].mean()
                )
            if w_pro != 1.0:
                # The parent loss (scvi) weights only the reconstruction; the weighted protein
                # terms stay annealed with kl_weight as in the hierarchy path.
                loss_out.loss = loss_out.loss + (w_pro - 1.0) * (
                    protein_weight * pro_recons_weight * loss_out.reconstruction_loss["reconst_loss_protein"].mean()
                    + kl_weight * loss_out.kl_local["kl_div_back_pro"].mean()
                )
            kl_w = self._kl_w(inference_outputs)
            if kl_w is not None:
                # p(w) = N(0, I), weight 1, annealed with kl_weight like kl_z (G82).
                return LossOutput(
                    loss=loss_out.loss + kl_weight * kl_w.mean(),
                    reconstruction_loss=loss_out.reconstruction_loss,
                    kl_local={**loss_out.kl_local, "kl_div_w": kl_w},
                    extra_metrics={**loss_out.extra_metrics, **self._residual_metrics(inference_outputs, kl_w)},
                )
            zp_terms = self._zp_loss(inference_outputs, kl_weight)
            if zp_terms is not None:
                term, zp_local, zp_metrics = zp_terms
                return LossOutput(
                    loss=loss_out.loss + term.mean(),
                    reconstruction_loss=loss_out.reconstruction_loss,
                    kl_local={**loss_out.kl_local, **zp_local},
                    extra_metrics={**loss_out.extra_metrics, **zp_metrics},
                )
            return loss_out

        label_index = tensors[REGISTRY_KEYS.LABELS_KEY]
        kl_u = self.kl_u(
            inference_outputs.get("qu", inference_outputs[MODULE_KEYS.QZ_KEY]),
            inference_outputs["u"],
            label_index,
        )

        if self.z_u_prior:
            peps = Normal(0.0, torch.exp(self.pz_scale.clamp(min=-4.0)))
            if self.hierarchy_mode == "centered_v2" and self.residual_prior_mode == "factual":
                kl_z = -peps.log_prob(inference_outputs["eps"]).sum(dim=-1)
                if kl_z.ndim > 1:
                    kl_z = kl_z.mean(dim=0)
            elif self.hierarchy_mode == "centered_v2":
                eps_raw_all = inference_outputs["eps_raw_all"]
                sample_dim = 1 if eps_raw_all.ndim == 3 else 2
                kl_z = -peps.log_prob(eps_raw_all).sum(dim=-1).mean(dim=sample_dim)
                if kl_z.ndim > 1:
                    kl_z = kl_z.mean(dim=0)
            else:
                eps = inference_outputs["eps"]
                eps_dist = inference_outputs.get("eps_dist")
                if eps_dist is not None:
                    # use_map=False: analytic KL(q(eps) || p(eps)) includes entropy
                    kl_z = kl_divergence(eps_dist, peps).sum(dim=-1)
                else:
                    # use_map=True: deterministic eps uses cross-entropy -log p(eps)
                    kl_z = -peps.log_prob(eps).sum(dim=-1)
                # kl_z shape: (batch,) or (n_samples, batch) → reduce mc dim
                if kl_z.ndim > 1:
                    kl_z = kl_z.mean(dim=0)
        else:
            kl_z = torch.zeros_like(kl_u)

        # Update kl_local: kl_div_z now = kl_u_weight*kl_u + kl_z_weight*kl_z
        kl_local = dict(loss_out.kl_local)
        kl_local["kl_div_z"] = self.kl_u_weight * kl_u + self.kl_z_weight * kl_z
        # Protein residual latent: p(w) = N(0, I), weight 1, annealed with kl_weight like kl_z (G82).
        kl_w = self._kl_w(inference_outputs)
        if kl_w is not None:
            kl_local["kl_div_w"] = kl_w
        kl_w_term = 0.0 if kl_w is None else kl_weight * kl_w
        # Protein-only latent z_p: weighted KL with the free-bits floor, plus the optional consistency KL.
        zp_terms = self._zp_loss(inference_outputs, kl_weight)
        if zp_terms is not None:
            kl_w_term = kl_w_term + zp_terms[0]
            kl_local.update(zp_terms[1])

        if self._scale_observations and self.n_obs_per_sample is not None:
            # Weight each cell's ELBO by 1/n_cells_in_that_sample so high-cell-count
            # donors do not dominate training.  Reconstruct the full per-cell ELBO from
            # individual component tensors (parent loss() already called torch.mean()).
            sample_index = tensors[REGISTRY_KEYS.SAMPLE_KEY].to(torch.int64).flatten()
            prefactors = self.n_obs_per_sample[sample_index]  # (batch,)
            per_cell = (
                loss_out.reconstruction_loss["reconst_loss_gene"]
                + protein_weight * pro_recons_weight * w_pro * loss_out.reconstruction_loss["reconst_loss_protein"]
                + kl_weight * kl_local["kl_div_z"]
                + kl_w_term
                + kl_local["kl_div_l_gene"]
                + kl_weight * w_pro * kl_local["kl_div_back_pro"]
            )
            if self.hierarchy_mode == "centered_v2":
                weights = self.n_obs_per_sample.sum() / (self._n_sample * prefactors)
                loss = (per_cell * weights).mean()
            else:
                loss = (per_cell / prefactors).mean()
        else:
            if self._scale_observations:
                import warnings

                warnings.warn(
                    "scale_observations=True but n_obs_per_sample is None "
                    "(buffer was not restored after load). "
                    "Call _setup_hierarchy(n_obs_per_sample=...) to restore weighting.",
                    UserWarning,
                    stacklevel=2,
                )
            loss = (
                loss_out.reconstruction_loss["reconst_loss_gene"]
                + protein_weight * pro_recons_weight * w_pro * loss_out.reconstruction_loss["reconst_loss_protein"]
                + kl_weight * kl_local["kl_div_z"]
                + kl_w_term
                + kl_local["kl_div_l_gene"]
                + kl_weight * w_pro * kl_local["kl_div_back_pro"]
            ).mean()

        if kl_w is not None:
            loss_out.extra_metrics.update(self._residual_metrics(inference_outputs, kl_w))
        if zp_terms is not None:
            loss_out.extra_metrics.update(zp_terms[2])

        if self.u_invariance_weight > 0 and self.training:
            u_pen = self.u_invariance_penalty(tensors, inference_outputs)
            loss = loss + self.u_invariance_weight * u_pen
            loss_out.extra_metrics["u_invariance"] = u_pen.detach()

        # Collapse guard for autotune (ADR-0013): logged as residual_*_<train|validation>.
        # Active units are defined on posterior means (Burda et al. 2016). With use_map=False
        # eps is a draw, so read the mean of q(eps) and the matching z = z_base + mean;
        # otherwise a collapsed q(eps) = N(0, s) would read as active noise.
        eps_dist = inference_outputs.get("eps_dist")
        if eps_dist is not None:
            eps = eps_dist.loc
            z = inference_outputs["z_base"] + eps
        else:
            eps, z = inference_outputs["eps"], inference_outputs[MODULE_KEYS.Z_KEY]
        # With several Monte Carlo draws of u, the first draw stands for the cell.
        if eps.ndim == 3:
            eps, z = eps[0], z[0]
        # The variance across one cell is 0, which would bias the epoch mean low.
        if eps.shape[0] >= 2:
            loss_out.extra_metrics.update(residual_collapse_metrics(eps, z))

        penalty = self.sample_embedding_penalty()
        droplet = self.ambient_droplet_penalty()
        if droplet is not None:
            penalty = droplet if penalty is None else penalty + droplet
        # With donor weighting, elbo_validation stays cell-weighted (it is the early-stopping
        # monitor). Log the loss the module trains on as a second, donor-balanced metric;
        # scvi logs it as donor_balanced_loss_<train|validation>.
        weighted = self._scale_observations and self.n_obs_per_sample is not None
        if weighted:
            full = loss if penalty is None else loss + penalty / self._n_obs_total
            loss_out.extra_metrics["donor_balanced_loss"] = full.detach()
        if penalty is None:
            return LossOutput(
                loss=loss,
                reconstruction_loss=loss_out.reconstruction_loss,
                kl_local=kl_local,
                extra_metrics=loss_out.extra_metrics,
            )
        # ``loss`` is a per-cell mean, so the full-data objective is N * loss. The
        # penalty is a global term (independent of the minibatch) added once to the
        # full-data objective, hence ``loss + penalty / N`` with N the number of cells,
        # not the minibatch size. It is not scaled by ``kl_weight``: the warm-up exists to
        # stop local latents collapsing, and this prior should act from the first step.
        # Passing the sum as ``kl_global`` lets the training plan report it as
        # ``kl_global / n_obs_training``, the scvi-tools convention for global terms.
        return LossOutput(
            loss=loss + penalty / self._n_obs_total,
            reconstruction_loss=loss_out.reconstruction_loss,
            kl_local=kl_local,
            kl_global=penalty,
            extra_metrics=loss_out.extra_metrics,
        )

    def u_invariance_penalty(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | dict],
    ) -> torch.Tensor:
        r"""Counterfactual re-encoding penalty that makes ``u`` sample-invariant.

        Under the generative model ``x = f(z_base(u) + eps(u, s))``, ``u`` does not depend on the
        sample, so two decodes of the same ``u`` under different samples should map back to the
        same ``u``. For each cell, with ``s`` its own sample and ``s'`` drawn uniformly,

        .. math::
            P = \operatorname{mean}_i \lVert \mu_u(\bar x(u_i, s'_i)) - \mu_u(\bar x(u_i, s_i))\rVert^2

        where :math:`\bar x` are the deterministic RNA and protein means. Both decodes run
        without gradients, so only the (sample-blind) ``u`` encoder is trained by ``P``; ``eps``
        cannot shrink to satisfy it. Comparing two decoded means avoids a noisy-count versus
        denoised-mean input mismatch.
        """
        u = inference_outputs["u"]
        if u.ndim != 2:
            raise ValueError("u_invariance_penalty supports a single posterior draw only.")
        x = tensors[REGISTRY_KEYS.X_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        sample_index = tensors[REGISTRY_KEYS.SAMPLE_KEY].to(torch.int64)
        cont_covs = tensors.get(REGISTRY_KEYS.CONT_COVS_KEY)
        cat_covs = tensors.get(REGISTRY_KEYS.CAT_COVS_KEY)
        library = x.sum(dim=-1, keepdim=True)
        n_sample = int(self._n_sample)
        if self.u_invariance_mode == "dropout_consistency":
            # Ablation: same sample twice, decoder and eps net in train mode, so the penalty
            # is pure dropout-noise consistency (the behaviour of the first prototype).
            cf_sample = sample_index
            eval_decodes = False
        else:
            # Another sample for every cell (never the own one), deterministic decodes.
            offset = torch.randint(1, max(n_sample, 2), sample_index.shape, device=sample_index.device)
            cf_sample = (sample_index + offset) % n_sample
            eval_decodes = True
        u_detached = u.detach()
        nets = (self.decoder, self.qz)
        was_training = [net.training for net in nets]
        if eval_decodes:
            # Eval mode: no dropout masks and no BatchNorm running-stat updates.
            for net in nets:
                net.eval()
        try:
            decoded = []
            for s in (sample_index, cf_sample):
                with torch.no_grad():
                    z_base, eps, _ = self.qz(u_detached, s, deterministic=True)
                dec = self._deterministic_decoder_parameters(
                    z_base + eps, library, batch_index, cont_covs=cont_covs, cat_covs=cat_covs
                )
                # Clone out of inference mode so the encoder pass below can be differentiated.
                decoded.append((dec["rna_rate"].clone(), dec["protein_total_mean"].clone()))
        finally:
            for net, flag in zip(nets, was_training, strict=True):
                net.train(flag)
        qu_batch_index = batch_index if self.qu_encode_covariates else None
        means = [
            self.qu(
                rna,
                protein,
                sample_index,
                batch_index=qu_batch_index,
                cont_covs=cont_covs if self.qu_encode_covariates else None,
                cat_covs=cat_covs if self.qu_encode_covariates else None,
                batch_rep=self._batch_representation_for(qu_batch_index),
            ).loc
            for rna, protein in decoded
        ]
        return (means[1] - means[0]).square().sum(dim=-1).mean()

    # ------------------------------------------------------------------
    # ADR-0011: ambient- and isotype-aware protein noise model (opt-in)
    # ------------------------------------------------------------------

    def configure_protein_noise(
        self,
        protein_noise_model: str = "totalvi",
        protein_background_source: str = "z",
        protein_encoder_input: str = "raw",
        ambient_profile: np.ndarray | None = None,
        dsb_log_mean: np.ndarray | None = None,
        dsb_log_sd: np.ndarray | None = None,
        isotype_mask: np.ndarray | None = None,
        ambient_burden_prior: tuple[float, float] = (0.0, 2.0),
        n_nuisance: int = 1,
        nuisance_h_input: str = "isotypes_total",
        native_form: str = "scale",
        background_mixing_reg: float = 1e-3,
        learn_ambient_profile: bool = False,
        ambient_droplet_counts: np.ndarray | None = None,
    ) -> None:
        """Switch on an opt-in protein noise model (ADR-0011).

        ``"totalvi"`` keeps the stock mixture. ``"ambient"`` adds the free-ambient
        term ``A = c * a`` to both mixture means. ``"ambient_isotype"`` replaces the
        decoded background with ``A + N(h)``, fits isotype controls with background
        only, and drops the totalVI background prior. ``protein_background_source=
        "z_base"`` decodes the background from ``z_base(u)`` so a sample switch does
        not move it. ``protein_encoder_input="ambient_zscore"`` feeds ``qu`` the DSB
        step-I standardized protein counts.

        ``learn_ambient_profile=True`` makes the composition ``a`` a parameter,
        ``softmax(logits)`` per profile row, started at the given profile, and adds the
        cell-free droplets' multinomial likelihood of their summed counts as a global
        term (see :meth:`ambient_droplet_penalty`).
        """
        if learn_ambient_profile and protein_noise_model == "totalvi":
            raise ValueError("learn_ambient_profile needs protein_noise_model 'ambient' or 'ambient_isotype'.")
        if protein_noise_model not in {"totalvi", "ambient", "ambient_isotype"}:
            raise ValueError("protein_noise_model must be 'totalvi', 'ambient' or 'ambient_isotype'.")
        if protein_background_source not in {"z", "z_base"}:
            raise ValueError("protein_background_source must be 'z' or 'z_base'.")
        if protein_encoder_input not in {"raw", "ambient_zscore"}:
            raise ValueError("protein_encoder_input must be 'raw' or 'ambient_zscore'.")
        non_default = (
            protein_noise_model != "totalvi" or protein_background_source != "z" or protein_encoder_input != "raw"
        )
        if non_default and self.n_latent_protein_residual > 0:
            raise NotImplementedError(
                "n_latent_protein_residual > 0 is only implemented for protein_noise_model='totalvi' with "
                "protein_background_source='z' and protein_encoder_input='raw'; got "
                f"protein_noise_model={protein_noise_model!r}, protein_background_source="
                f"{protein_background_source!r}, protein_encoder_input={protein_encoder_input!r}."
            )
        if self.n_latent_protein_only > 0:
            if protein_noise_model != "totalvi":
                raise NotImplementedError(
                    "n_latent_protein_only > 0 is only implemented for protein_noise_model='totalvi'; "
                    f"got {protein_noise_model!r}."
                )
            if protein_background_source == "z_base" and self.protein_latent_heads == "all":
                raise ValueError(
                    "protein_background_source='z_base' needs a background head that does not read z_p: "
                    "use protein_latent_heads='fore_mixing'."
                )
        if non_default and self.u_invariance_weight > 0:
            raise ValueError("u_invariance_weight > 0 is only supported with the default protein model.")
        if protein_encoder_input == "ambient_zscore" and self.u_prior_type == "vamp":
            raise ValueError(
                "protein_encoder_input='ambient_zscore' is blocked with u_prior='vamp': the "
                "pseudo-inputs live in raw count space."
            )
        if nuisance_h_input not in {"isotypes_total", "isotypes_batch"}:
            raise ValueError("nuisance_h_input must be 'isotypes_total' or 'isotypes_batch'.")
        if nuisance_h_input != "isotypes_total" and protein_noise_model != "ambient_isotype":
            raise ValueError("nuisance_h_input applies to protein_noise_model='ambient_isotype' only.")
        if native_form not in {"scale", "excess"}:
            raise ValueError("native_form must be 'scale' or 'excess'.")
        self.native_form = native_form
        if protein_noise_model == "totalvi" and background_mixing_reg != 1e-3:
            raise ValueError("background_mixing_reg applies to the opt-in protein noise models only.")
        # A negative weight lets -reg * log(pi) fall without bound as pi -> 0.
        if not (np.isfinite(background_mixing_reg) and background_mixing_reg >= 0):
            raise ValueError("background_mixing_reg must be finite and >= 0.")
        if isinstance(n_nuisance, bool) or int(n_nuisance) != n_nuisance or n_nuisance < 1:
            raise ValueError("n_nuisance must be a positive whole number.")
        self.background_mixing_reg = float(background_mixing_reg)
        P = self.n_input_proteins
        self.protein_noise_model = protein_noise_model
        self.protein_background_source = protein_background_source
        self.protein_encoder_input = protein_encoder_input

        if protein_noise_model != "totalvi":
            if ambient_profile is None:
                raise ValueError(f"protein_noise_model={protein_noise_model!r} needs an ambient profile.")
            profile = np.atleast_2d(np.asarray(ambient_profile, dtype=np.float32))
            if profile.shape[1] != P or not np.all(np.isfinite(profile)) or profile.min() < 0:
                raise ValueError("ambient_profile must be finite, non-negative, with one column per protein.")
            if profile.shape[0] not in {1, self.n_batch}:
                raise ValueError("ambient_profile needs one pooled row or one row per batch.")
            if not np.all(profile.sum(axis=1) > 0):
                raise ValueError(
                    "ambient_profile needs a positive row sum for every row; an all-zero row has no composition."
                )
            profile = profile / profile.sum(axis=1, keepdims=True)
            self.register_buffer("ambient_profile", torch.as_tensor(profile))
            self.learn_ambient_profile = bool(learn_ambient_profile)
            if self.learn_ambient_profile:
                if ambient_droplet_counts is None:
                    raise ValueError(
                        "learn_ambient_profile needs the profile's droplet_counts (summed cell-free ADT counts)."
                    )
                counts = np.atleast_2d(np.asarray(ambient_droplet_counts, dtype=np.float64))
                if counts.shape != profile.shape or not np.all(np.isfinite(counts)) or counts.min() < 0:
                    raise ValueError("droplet_counts must be finite, non-negative and match the profile shape.")
                p_hat = counts / counts.sum(axis=1, keepdims=True)
                # sum_p N_p log p_hat_p, so the penalty is N * KL(p_hat || a) and 0 at the MLE.
                entropy_term = np.where(counts > 0, counts * np.log(np.where(p_hat > 0, p_hat, 1.0)), 0.0).sum()
                # float64: N is ~1e7-1e8 per capture, so the two sums cancel to a few units
                self.register_buffer("ambient_droplet_counts", torch.as_tensor(counts, dtype=torch.float64))
                self.register_buffer("ambient_droplet_entropy", torch.tensor(float(entropy_term), dtype=torch.float64))
                self.ambient_logits = nn.Parameter(torch.log(torch.as_tensor(profile).clamp(min=1e-8)))
            loc, scale = ambient_burden_prior
            if not (np.isfinite(loc) and np.isfinite(scale) and scale > 0):
                raise ValueError("ambient_burden_prior must be a finite (loc, scale) with scale > 0.")
            self.ambient_burden_prior_loc = float(loc)
            self.ambient_burden_prior_scale = float(scale)
            n_iso = 0
            target = np.ones(P, dtype=np.float32)
            if protein_noise_model == "ambient_isotype":
                mask = np.zeros(P, dtype=bool) if isotype_mask is None else np.asarray(isotype_mask, dtype=bool)
                if mask.shape != (P,) or not mask.any():
                    # C11: fail closed, never invent controls.
                    raise ValueError("protein_noise_model='ambient_isotype' needs at least one isotype control.")
                n_iso = int(mask.sum())
                target = (~mask).astype(np.float32)
                self.register_buffer("isotype_index", torch.as_tensor(np.flatnonzero(mask), dtype=torch.long))
                self.nuisance_eta = nn.Parameter(torch.zeros(max(self.n_batch, 1), P))
                self.nuisance_lambda = nn.Parameter(0.1 * torch.ones(int(n_nuisance), P))
            self.register_buffer("target_mask", torch.as_tensor(target))
            self.nuisance_encoder = AmbientNuisanceEncoder(
                P, n_isotypes=n_iso, n_nuisance=int(n_nuisance), h_input=nuisance_h_input, n_batch=self.n_batch
            )

        zp_zscore = self.n_latent_protein_only > 0 and self.protein_latent_encoder_input == "ambient_zscore"
        if protein_encoder_input == "ambient_zscore" or zp_zscore:
            if dsb_log_mean is None or dsb_log_sd is None:
                raise ValueError("an 'ambient_zscore' protein encoder input needs DSB log mean and SD.")
            mean = np.atleast_2d(np.asarray(dsb_log_mean, dtype=np.float32))
            sd = np.atleast_2d(np.asarray(dsb_log_sd, dtype=np.float32))
            if mean.shape != sd.shape or mean.shape[1] != P:
                raise ValueError("DSB statistics need one column per protein.")
            if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(sd))):
                raise ValueError("DSB statistics must be finite.")
            self.register_buffer("dsb_log_mean", torch.as_tensor(mean))
            if not np.all(sd > 1e-3):
                low = np.flatnonzero(~np.all(sd > 1e-3, axis=0))
                raise ValueError(
                    "protein_encoder_input='ambient_zscore' needs an empty-droplet log SD above 1e-3 "
                    f"for every protein; proteins at columns {low.tolist()} have SD "
                    f"{np.round(sd.min(axis=0)[low], 6).tolist()}, from few or no counts in cell-free "
                    "droplets, so their z-scores would be undefined or extreme."
                )
            self.register_buffer("dsb_log_sd", torch.as_tensor(sd))
            # Only q(u) reads the pre-transformed input through ``protein_input_pretransformed``;
            # q(z_p | y) z-scores in ``_protein_latent_dist``.
            self.qu.protein_input_pretransformed = protein_encoder_input == "ambient_zscore"

    def _context_rows(self, table: torch.Tensor, batch_index: torch.Tensor | None) -> torch.Tensor:
        if table.shape[0] == 1 or batch_index is None:
            return table[0]
        return table[batch_index.reshape(-1).to(torch.long)]

    def ambient_composition(self) -> torch.Tensor:
        """Ambient composition table ``(rows, proteins)``: learnt softmax or the fixed profile."""
        if getattr(self, "learn_ambient_profile", False):
            return torch.softmax(self.ambient_logits, dim=-1)
        return self.ambient_profile

    def ambient_droplet_penalty(self) -> torch.Tensor | None:
        r"""Negative droplet log-likelihood of the learnt composition, or ``None`` if off.

        .. math::
            \sum_b \sum_p N_{bp} (\log \hat p_{bp} - \log a_{bp}) = \sum_b N_b\, KL(\hat p_b \| a_b)

        ``N`` are the summed cell-free counts per profile row. It is a full-data term,
        added once to the objective like the sample-embedding prior.
        """
        if not getattr(self, "learn_ambient_profile", False):
            return None
        log_a = torch.log_softmax(self.ambient_logits.double(), dim=-1)
        nll = self.ambient_droplet_entropy - (self.ambient_droplet_counts * log_a).sum()
        return nll.to(self.ambient_logits.dtype)

    def _qu_protein_input(self, y: torch.Tensor, batch_index: torch.Tensor | None) -> torch.Tensor:
        if getattr(self, "protein_encoder_input", "raw") != "ambient_zscore":
            return y
        return self._zscore_protein(y, batch_index)

    def _set_protein_latent_options(
        self,
        n_latent_protein_only,
        combination,
        heads,
        prior,
        encoder_input,
        conditioning,
        consistency_weight,
        kl_weight,
        free_bits,
        granularity,
        statistic,
        schedule,
        decode_draw,
        *,
        u_invariance_weight,
    ) -> None:
        """Validate and store the ``protein_latent_*`` options (EXPERIMENTAL, see :class:`~mrtotalvi.MrTotalVI`)."""
        n = n_latent_protein_only
        if isinstance(n, bool) or int(n) != n or n < 0:
            raise ValueError(f"n_latent_protein_only must be a whole number >= 0; got {n!r}.")
        for value, allowed, name in (
            (combination, {"replace", "concat"}, "protein_latent_combination"),
            (heads, set(PROTEIN_RESIDUAL_HEADS), "protein_latent_heads"),
            (prior, {"standard", "mog"}, "protein_latent_prior"),
            (encoder_input, {"log1p", "ambient_zscore", "precomputed"}, "protein_latent_encoder_input"),
            (conditioning, {"blind", "donor"}, "protein_latent_conditioning"),
            (decode_draw, {"draw", "mean"}, "protein_latent_decode_draw"),
        ):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {sorted(allowed)}; got {value!r}.")
        if not (isinstance(consistency_weight, (int, float)) and math.isfinite(consistency_weight)) or (
            consistency_weight < 0
        ):
            raise ValueError(
                f"protein_latent_consistency_weight must be a finite number >= 0; got {consistency_weight!r}."
            )
        check_free_bits_options(kl_weight, free_bits, granularity, statistic, schedule, "protein_latent")
        self.n_latent_protein_only = int(n)
        self.protein_latent_combination = combination
        self.protein_latent_heads = heads
        self.protein_latent_prior = prior
        self.protein_latent_encoder_input = encoder_input
        self.protein_latent_conditioning = conditioning
        self.protein_latent_consistency_weight = float(consistency_weight)
        self.protein_latent_kl_weight = float(kl_weight)
        self.protein_latent_free_bits = float(free_bits)
        self.protein_latent_free_bits_granularity = granularity
        self.protein_latent_free_bits_statistic = statistic
        self.protein_latent_free_bits_schedule = schedule
        self.protein_latent_decode_draw = decode_draw
        # Set by readouts that decode the posterior mean of z_p (see ``_zp_mean_decode_if_set``).
        self._zp_mean_decode = False
        if self.n_latent_protein_only == 0:
            return
        if self.n_latent_protein_residual > 0:
            raise NotImplementedError(
                "n_latent_protein_only > 0 cannot be combined with n_latent_protein_residual > 0: both feed "
                "the protein heads and the choice between them is not made here. Use one of them."
            )
        if u_invariance_weight > 0:
            raise NotImplementedError(
                "u_invariance_weight > 0 is not implemented with n_latent_protein_only > 0: its counterfactual "
                "decode would need a z_p."
            )
        if self.protein_latent_consistency_weight > 0:
            if self.n_latent_protein_only != self.n_latent:
                raise ValueError(
                    "protein_latent_consistency_weight > 0 compares q(z | x, y) with q(z_p | y) dimension by "
                    f"dimension, so n_latent_protein_only must equal n_latent ({self.n_latent}); got "
                    f"{self.n_latent_protein_only}. No projection between the two is defined."
                )
        if prior == "mog" and self.protein_latent_free_bits > 0 and granularity == "per_dim":
            raise NotImplementedError(
                "protein_latent_free_bits > 0 with protein_latent_prior='mog' has no per-dimension KL; "
                "use protein_latent_free_bits_granularity='group'."
            )

    def _zscore_protein(self, y: torch.Tensor, batch_index: torch.Tensor | None) -> torch.Tensor:
        """DSB step-I standardised protein counts, with the empty-droplet statistics of the batch."""
        mean = self._context_rows(self.dsb_log_mean, batch_index)
        sd = self._context_rows(self.dsb_log_sd, batch_index)
        return (torch.log(y + 10.0) - mean) / sd

    def _protein_latent_dist(
        self, y: torch.Tensor, batch_index: torch.Tensor | None, sample_index, latent_input: torch.Tensor | None = None
    ) -> Normal:
        """``q(z_p | y)``; ``sample_index`` is read only by a donor-conditioned encoder.

        With ``protein_latent_encoder_input="precomputed"`` the encoder reads ``latent_input`` (the
        supplied per-cell matrix, no gradient) instead of ``y``.
        """
        if self.protein_latent_encoder_input == "ambient_zscore":
            y = self._zscore_protein(y, batch_index)
        elif self.protein_latent_encoder_input == "precomputed":
            if latent_input is None:
                raise RuntimeError("protein_latent_encoder_input='precomputed' needs the registered matrix.")
            y = latent_input.detach().to(y.dtype)
        return self.qzp(y, sample_index)

    @contextlib.contextmanager
    def _zp_mean_decode_if_set(self):
        """Decode the posterior mean of ``z_p`` inside the block when ``protein_latent_decode_draw="mean"``."""
        if self.n_latent_protein_only == 0 or self.protein_latent_decode_draw != "mean":
            yield
            return
        previous, self._zp_mean_decode = self._zp_mean_decode, True
        try:
            yield
        finally:
            self._zp_mean_decode = previous

    def forward(self, *args, **kwargs):
        """Readouts (``compute_loss=False``) honour ``protein_latent_decode_draw``; the loss always draws."""
        if kwargs.get("compute_loss", True):
            return super().forward(*args, **kwargs)
        with self._zp_mean_decode_if_set():
            return super().forward(*args, **kwargs)

    def _kl_zp(self, inference_outputs) -> torch.Tensor | None:
        """``KL(q(z_p | y) || p(z_p))`` per cell, shape ``(batch,)``; ``None`` when ``z_p`` is off.

        Closed form for ``N(0, I)``; a Monte-Carlo estimate on the draws of ``z_p`` for the MoG prior.
        """
        qzp = inference_outputs.get("qzp")
        if qzp is None:
            return None
        if self.protein_latent_prior == "mog":
            return _kl_u(self.pzp, qzp, inference_outputs["zp"], None)
        return self._kl_zp_dims(qzp).sum(dim=-1)

    @staticmethod
    def _kl_zp_dims(qzp: Normal) -> torch.Tensor:
        """Per-dimension ``KL(q(z_p | y) || N(0, 1))``, shape ``(batch, n_latent_protein_only)``."""
        return kl_divergence(qzp, Normal(torch.zeros_like(qzp.loc), torch.ones_like(qzp.scale)))

    def _consistency_kl(self, inference_outputs) -> torch.Tensor | None:
        """``KL(q(z | u, x, y) || q(z_p | y))`` per cell, closed form; ``None`` when off.

        ``q(z | u)`` is ``N(z_base + mean(eps), scale(eps))`` given the draw of ``u``, so the term is
        conditional on ``u`` and needs ``use_map=False`` with the legacy hierarchy. Both sides
        receive gradient.
        """
        if self.protein_latent_consistency_weight == 0.0 or "qzp" not in inference_outputs:
            return None
        eps_dist = inference_outputs.get("eps_dist")
        if eps_dist is None:
            raise RuntimeError("protein_latent_consistency_weight > 0 needs q(eps) (use_map=False, legacy hierarchy).")
        qz = Normal(inference_outputs["z_base"] + eps_dist.loc, eps_dist.scale)
        kl = kl_divergence(qz, inference_outputs["qzp"]).sum(dim=-1)
        return kl.mean(dim=0) if kl.ndim > 1 else kl

    def _zp_loss(self, inference_outputs, kl_weight: float):
        """Loss term, ``kl_local`` entries and logged metrics of ``z_p``; ``None`` when it is off.

        The term per cell is ``kl_weight * beta * cost + consistency_weight * KL_consistency``, where
        ``cost`` is the KL with the free-bits floor (:func:`~mrtotalvi._kl_free_bits.kl_with_free_bits`).
        ``kl_div_zp`` keeps the true KL, so logs and ``elbo_validation`` stay comparable. The
        consistency weight is constant: it does not follow the ``kl_weight`` warm-up.
        """
        qzp = inference_outputs.get("qzp")
        if qzp is None:
            return None
        floor = (
            self.protein_latent_free_bits
            if free_bits_active(self.protein_latent_free_bits_schedule, kl_weight)
            else 0.0
        )
        if self.protein_latent_prior == "mog":
            kl, dims = self._kl_zp(inference_outputs), None
            cost = kl_with_free_bits(kl.unsqueeze(-1), floor, "group", self.protein_latent_free_bits_statistic)
        else:
            dims = self._kl_zp_dims(qzp)
            kl = dims.sum(dim=-1)
            cost = kl_with_free_bits(
                dims, floor, self.protein_latent_free_bits_granularity, self.protein_latent_free_bits_statistic
            )
        term = kl_weight * self.protein_latent_kl_weight * cost
        kl_local = {"kl_div_zp": kl}
        metrics = {"kl_zp": kl.detach().mean()}
        loc = qzp.loc.detach()
        # The variance across one cell is 0, which would bias the epoch mean low.
        if loc.shape[0] >= 2:
            metrics["protein_latent_active_units"] = (
                (loc.var(dim=0, unbiased=False) > RESIDUAL_ACTIVE_THRESHOLD).sum().to(loc.dtype)
            )
        if dims is not None:
            for index, value in enumerate(dims.detach().mean(dim=0)):
                metrics[f"kl_zp_dim_{index:02d}"] = value
        consistency = self._consistency_kl(inference_outputs)
        if consistency is not None:
            term = term + self.protein_latent_consistency_weight * consistency
            kl_local["kl_div_zp_consistency"] = consistency
            metrics["kl_zp_consistency"] = consistency.detach().mean()
        return term, kl_local, metrics

    def _kl_w(self, inference_outputs) -> torch.Tensor | None:
        """``KL(q(w | y) || N(0, I))`` per cell, shape ``(batch,)``; ``None`` when ``w`` is off."""
        qw = inference_outputs.get("qw")
        if qw is None:
            return None
        return kl_divergence(qw, Normal(torch.zeros_like(qw.loc), torch.ones_like(qw.scale))).sum(dim=-1)

    def _residual_metrics(self, inference_outputs, kl_w: torch.Tensor) -> dict[str, torch.Tensor]:
        """Logged each epoch (G96): mean ``kl_w`` and the active units of ``w``.

        A unit is active when the variance of its posterior mean across the minibatch exceeds
        :data:`RESIDUAL_ACTIVE_THRESHOLD` (Burda et al. 2016). ``w`` collapsing gives 0 active units
        and a ``kl_w`` near 0; ``w`` absorbing the protein signal gives a large ``kl_w``.
        """
        metrics = {"kl_w": kl_w.detach().mean()}
        loc = inference_outputs["qw"].loc.detach()
        # The variance across one cell is 0, which would bias the epoch mean low.
        if loc.shape[0] >= 2:
            active = (loc.var(dim=0, unbiased=False) > RESIDUAL_ACTIVE_THRESHOLD).sum().to(loc.dtype)
            metrics["protein_residual_active_units"] = active
        return metrics

    def _nuisance_kl(self, inference_outputs) -> torch.Tensor | float:
        if getattr(self, "protein_noise_model", "totalvi") == "totalvi":
            return 0.0
        qc = inference_outputs["qc"]
        prior_c = Normal(
            torch.full_like(qc.loc, self.ambient_burden_prior_loc),
            torch.full_like(qc.scale, self.ambient_burden_prior_scale),
        )
        kl = kl_divergence(qc, prior_c).sum(dim=-1)
        qh = inference_outputs.get("qh")
        if qh is not None:
            kl = kl + kl_divergence(qh, Normal(torch.zeros_like(qh.loc), torch.ones_like(qh.scale))).sum(dim=-1)
        return kl

    def _require_default_protein_model(self, what: str) -> None:
        if (
            getattr(self, "protein_noise_model", "totalvi") != "totalvi"
            or getattr(self, "protein_background_source", "z") != "z"
            or getattr(self, "protein_encoder_input", "raw") != "raw"
        ):
            raise NotImplementedError(f"{what} is not implemented for the opt-in ADR-0011 protein noise models yet.")

    def _get_generative_input(self, tensors, inference_outputs):
        base = super()._get_generative_input(tensors, inference_outputs)
        if (
            getattr(self, "protein_background_source", "z") == "z_base"
            or getattr(self, "protein_noise_model", "totalvi") != "totalvi"
        ):
            base["z_base"] = inference_outputs.get("z_base")
            base["log_c"] = inference_outputs.get("log_c")
            base["h"] = inference_outputs.get("h")
        if self.n_latent_protein_residual > 0:
            # The cell's own draw of q(w): factual decodes read it. Counterfactual and DE decodes
            # never pass it, so they read w = 0.
            base["w"] = inference_outputs.get("w")
        if self.n_latent_protein_only > 0:
            # The cell's own draw of q(z_p). A decode without it raises: z_p = 0 is not "no information".
            base["zp"] = inference_outputs.get("zp")
        return base

    def _residual_decoder(self) -> DecoderTOTALVIResidual | None:
        """The decoder whose protein heads read ``w`` or ``z_p``, or ``None`` when both are off."""
        if self.n_latent_protein_residual == 0 and self.n_latent_protein_only == 0:
            return None
        decoder = self.decoder
        return decoder.decoder if isinstance(decoder, BatchEmbeddingDecoderAdapter) else decoder

    @contextlib.contextmanager
    def _decoder_latent(self, value: torch.Tensor | None):
        """Hand ``w`` or ``z_p`` to the decoder inside the block (the call site in scvi is fixed)."""
        core = self._residual_decoder()
        if core is None:
            yield
            return
        core.set_residual(value)
        try:
            yield
        finally:
            core.set_residual(None)

    def _snapshot_background_bn(self):
        """Copy the BatchNorm buffers of the protein background decoder."""
        return [
            (m, {k: getattr(m, k).detach().clone() for k in ("running_mean", "running_var", "num_batches_tracked")})
            for m in self.decoder.py_back_decoder.modules()
            if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.track_running_stats
        ]

    def _background_head(self, z_base, batch_index, cont_covs, cat_covs, transform_batch):
        """Protein background parameters at ``z_base``; runs only the background head.

        Mirrors ``TOTALVAE.generative`` and ``DecoderTOTALVI.forward`` for ``back_alpha``,
        ``back_beta`` and ``rate_back``, without running the RNA and other protein heads.
        """
        if cont_covs is None:
            decoder_input = z_base
        elif z_base.dim() != cont_covs.dim():
            decoder_input = torch.cat([z_base, cont_covs.unsqueeze(0).expand(z_base.size(0), -1, -1)], dim=-1)
        else:
            decoder_input = torch.cat([z_base, cont_covs], dim=-1)
        categorical_input = torch.split(cat_covs, 1, dim=1) if cat_covs is not None else ()
        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch
        dec = self.decoder
        # Under `batch_representation="embedding"` the batch enters through `decoder_input`
        # (see `_deterministic_decoder_parameters`), not as a categorical argument.
        if isinstance(dec, BatchEmbeddingDecoderAdapter):
            decoder_input = dec.batch_input(decoder_input, batch_index)
            cat_list = categorical_input
        else:
            cat_list = (batch_index, *categorical_input)
        back = dec.py_back_decoder(decoder_input, *cat_list)
        back_cat_z = torch.cat([back, decoder_input], dim=-1)
        alpha = dec.py_back_mean_log_alpha(back_cat_z, *cat_list)
        beta = dec.activation_function_bg(dec.py_back_mean_log_beta(back_cat_z, *cat_list)) + 1e-8
        rate_back = torch.exp(Normal(alpha, beta).rsample())
        return {"back_alpha": alpha, "back_beta": beta, "rate_back": rate_back}

    def generative(
        self,
        z,
        library_gene,
        batch_index,
        label,
        cont_covs=None,
        cat_covs=None,
        size_factor=None,
        transform_batch=None,
        z_base=None,
        log_c=None,
        h=None,
        w=None,
        zp=None,
    ):
        """TOTALVAE generative step, extended by the opt-in ADR-0011 protein model.

        ``w`` is the protein residual latent (only with ``n_latent_protein_residual > 0``). ``None``
        decodes at ``w = 0``, which is what counterfactual and DE decodes use. ``zp`` is the
        protein-only latent (only with ``n_latent_protein_only > 0``); the decode needs it.
        """
        mode = getattr(self, "protein_noise_model", "totalvi")
        source = getattr(self, "protein_background_source", "z")
        # With source "z_base" the z pass's background BatchNorm update is dropped, so that head
        # advances once, on z_base (see ``_background_head``).
        bn_state = self._snapshot_background_bn() if source == "z_base" and self.training else None
        with self._decoder_latent(w if w is not None else zp):
            out = super().generative(
                z, library_gene, batch_index, label, cont_covs, cat_covs, size_factor, transform_batch
            )
        if bn_state is not None:
            for bn, saved in bn_state:
                for name, value in saved.items():
                    getattr(bn, name).copy_(value)
        if mode == "totalvi" and source == "z":
            return out
        py_ = out["py_"]
        if source == "z_base":
            if z_base is None:
                raise NotImplementedError("protein_background_source='z_base' needs z_base from inference.")
            if z_base.ndim < z.ndim:
                z_base = z_base.expand_as(z)
            py_.update(self._background_head(z_base, batch_index, cont_covs, cat_covs, transform_batch))
        fore_scale = py_["fore_scale"]
        back_mean = torch.exp(py_["back_alpha"] + 0.5 * py_["back_beta"].square())
        b_idx = batch_index if transform_batch is None else torch.full_like(batch_index, int(transform_batch))
        zeros = torch.zeros_like(fore_scale)
        if mode == "totalvi":
            py_["rate_fore"] = py_["rate_back"] * fore_scale
            ambient, nonspecific, native = zeros, back_mean, back_mean * (fore_scale - 1.0)
        else:
            if log_c is None:
                raise NotImplementedError(f"protein_noise_model={mode!r} needs the ambient burden from inference.")
            ambient = torch.exp(log_c) * self._context_rows(self.ambient_composition(), b_idx)
            if mode == "ambient":
                py_["rate_back"] = ambient + py_["rate_back"]
                py_["rate_fore"] = ambient + (py_["rate_back"] - ambient) * fore_scale
                nonspecific, native = back_mean, back_mean * (fore_scale - 1.0)
            else:
                if h is None:
                    raise NotImplementedError("protein_noise_model='ambient_isotype' needs h from inference.")
                eta = self.nuisance_eta[b_idx.reshape(-1).to(torch.long)]
                nonspecific = F.softplus(eta + h @ self.nuisance_lambda)
                # "scale": S = exp(alpha) f (f >= 1, so S never vanishes);
                # "excess": S = exp(alpha) (f - 1), which can reach zero.
                shift = 1.0 if getattr(self, "native_form", "scale") == "excess" else 0.0
                native = torch.exp(py_["back_alpha"]) * (fore_scale - shift) * self.target_mask
                py_["rate_back"] = ambient + nonspecific
                py_["rate_fore"] = ambient + nonspecific + native
        py_["ambient_mean"] = ambient
        py_["nonspecific_mean"] = nonspecific
        py_["native_mean"] = native
        eff = out["per_batch_efficiency"]
        out["py_norm_"] = {**py_, "rate_back": eff * py_["rate_back"], "rate_fore": eff * py_["rate_fore"]}
        return out

    @torch.inference_mode()
    def protein_components(self, tensors: dict[str, torch.Tensor], n_samples: int = 1) -> dict[str, torch.Tensor]:
        """Posterior expected protein components on the raw count scale (ADR-0011).

        Returns ``native``, ``ambient``, ``nonspecific``, ``observed`` (their sum) and
        ``foreground_probability``, each ``(cells, proteins)``, averaged over
        ``n_samples`` posterior draws. ``efficiency`` multiplies every component (C7).
        """
        # generative() and the z_base background head read batch tensors straight from
        # ``tensors``; only inference() moves its inputs, so move them all here.
        tensors = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in tensors.items()}
        inf_in = self._get_inference_input(tensors)
        with self._zp_mean_decode_if_set():
            inf_out = self.inference(**inf_in, n_samples=n_samples)
        gen_in = self._get_generative_input(tensors, inf_out)
        gen_out = self.generative(**gen_in)
        py_ = gen_out["py_"]
        eff = gen_out["per_batch_efficiency"]
        if "native_mean" not in py_:
            back_mean = torch.exp(py_["back_alpha"] + 0.5 * py_["back_beta"].square())
            py_ = {
                **py_,
                "ambient_mean": torch.zeros_like(back_mean),
                "nonspecific_mean": back_mean,
                "native_mean": back_mean * (py_["fore_scale"] - 1.0),
            }
        fg = 1.0 - torch.sigmoid(py_["mixing"])
        res = {
            "native": eff * fg * py_["native_mean"],
            "ambient": eff * py_["ambient_mean"].expand_as(fg),
            "nonspecific": eff * py_["nonspecific_mean"].expand_as(fg),
            "foreground_probability": fg,
        }
        res["observed"] = res["native"] + res["ambient"] + res["nonspecific"]
        if n_samples > 1:
            res = {k: v.mean(dim=0) for k, v in res.items()}
        return res

    @torch.inference_mode()
    def protein_posterior_readout(
        self,
        tensors: dict[str, torch.Tensor],
        n_samples: int = 25,
        split: str = "excess",
        background: str = "sampled",
    ):
        """EXPERIMENTAL. Posterior expected native protein count given ``y``, ``E[native | y, z]``.

        Only for ``protein_noise_model="totalvi"``. See
        :meth:`MrTotalVI.get_protein_noise_decomposition` for the options. Returns
        ``(cells, proteins)``, the plain mean over the ``n_samples`` draws. With
        ``n_latent_protein_residual > 0`` each draw decodes the cell's own draw of ``q(w)``.
        """
        _check_posterior_options(split, background)
        if self.protein_noise_model != "totalvi":
            raise NotImplementedError(
                "The posterior protein readout is only implemented for protein_noise_model='totalvi'; "
                f"got {self.protein_noise_model!r}."
            )
        tensors = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in tensors.items()}
        with self._zp_mean_decode_if_set():
            inf_out = self.inference(**self._get_inference_input(tensors), n_samples=n_samples)
        gen_out = self.generative(**self._get_generative_input(tensors, inf_out))
        py_, eff = gen_out["py_"], gen_out["per_batch_efficiency"]
        y = tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY]
        if background == "sampled":
            rate_back = py_["rate_back"]
        else:
            rate_back = torch.exp(py_["back_alpha"] + 0.5 * py_["back_beta"].square())
        m1 = eff * rate_back
        res, _, _ = _posterior_native(y, m1, py_["fore_scale"], py_["r"], py_["mixing"], split)
        return res.mean(dim=0) if res.ndim == 3 else res

    def _deterministic_decoder_parameters(
        self,
        z: torch.Tensor,
        library_size: torch.Tensor,
        batch_index: torch.Tensor,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        zp: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Decode RNA and protein means without using stochastic ``rate_back``.

        ``zp`` is the cell's own protein-only latent, required with ``n_latent_protein_only > 0``
        (G165: counterfactual and DE decodes keep the cell's own ``z_p``; callers pass its posterior
        mean). With ``w`` the heads read ``w = 0`` (G90).
        """
        with self._decoder_latent(zp):
            return self._deterministic_decode(z, library_size, batch_index, cont_covs, cat_covs)

    @torch.inference_mode()
    def _deterministic_decode(
        self,
        z: torch.Tensor,
        library_size: torch.Tensor,
        batch_index: torch.Tensor,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        self._require_default_protein_model("Deterministic counterfactual decoding")
        if cont_covs is None:
            decoder_input = z
        else:
            decoder_input = torch.cat([z, cont_covs], dim=-1)
        categorical_input = tuple(torch.split(cat_covs, 1, dim=1)) if cat_covs is not None else ()
        decoder = self.decoder

        # Under `batch_representation="embedding"` the wrapped decoder was built without
        # the batch category, so the embedding is folded into `decoder_input` here and
        # `batch_index` is dropped from the categorical arguments.
        if isinstance(decoder, BatchEmbeddingDecoderAdapter):
            decoder_input = decoder.batch_input(decoder_input, batch_index)
            conditioning = categorical_input
        else:
            conditioning = (batch_index, *categorical_input)

        px = decoder.px_decoder(decoder_input, *conditioning)
        px_cat_z = torch.cat([px, decoder_input], dim=-1)
        rna_scale = decoder.px_scale_activation(
            decoder.px_scale_decoder(
                px_cat_z,
                *conditioning,
            )
        )

        # Counterfactual and DE decodes read w = 0 (G90): a head that reads w gets zeros appended.
        back_input = _with_w(decoder, "back", decoder_input)
        py_back = decoder.py_back_decoder(
            back_input,
            *conditioning,
        )
        py_back_cat_z = torch.cat([py_back, back_input], dim=-1)
        back_alpha = decoder.py_back_mean_log_alpha(
            py_back_cat_z,
            *conditioning,
        )
        back_beta = (
            decoder.activation_function_bg(
                decoder.py_back_mean_log_beta(
                    py_back_cat_z,
                    *conditioning,
                )
            )
            + 1e-8
        )

        fore_input = _with_w(decoder, "fore", decoder_input)
        py_fore = decoder.py_fore_decoder(
            fore_input,
            *conditioning,
        )
        py_fore_cat_z = torch.cat([py_fore, fore_input], dim=-1)
        fore_scale = (
            decoder.py_fore_scale_decoder(
                py_fore_cat_z,
                *conditioning,
            )
            + 1
            + 1e-8
        )

        mixing_input = _with_w(decoder, "mixing", decoder_input)
        mixing_hidden = decoder.sigmoid_decoder(
            mixing_input,
            *conditioning,
        )
        mixing_cat_z = torch.cat([mixing_hidden, mixing_input], dim=-1)
        mixing = decoder.py_background_decoder(
            mixing_cat_z,
            *conditioning,
        )
        efficiency = torch.exp(
            F.linear(
                F.one_hot(
                    batch_index.squeeze(-1).to(torch.int64),
                    self.n_batch,
                ).float(),
                self.log_per_batch_efficiency,
            )
        )
        background = efficiency * torch.exp(back_alpha + 0.5 * back_beta.square())
        foreground = background * fore_scale
        foreground_probability = 1.0 - torch.sigmoid(mixing)
        background_contribution = (1.0 - foreground_probability) * background
        foreground_contribution = foreground_probability * foreground
        rna_rate = library_size * rna_scale
        rna_extra = {}
        if self.gene_likelihood == "zinb":
            # rna_rate stays the NB component mean; the expected count also needs the zero probability.
            dropout_cat_z = torch.cat([mixing_hidden, decoder_input], dim=-1)
            zero_probability = torch.sigmoid(decoder.px_dropout_decoder_gene(dropout_cat_z, *conditioning))
            rna_extra = {
                "rna_zero_probability": zero_probability,
                "rna_expected_count": (1.0 - zero_probability) * rna_rate,
            }
        return {
            "rna_scale": rna_scale,
            "rna_rate": rna_rate,
            **rna_extra,
            "protein_background_component_mean": background,
            "protein_foreground_component_mean": foreground,
            "protein_foreground_probability": foreground_probability,
            "protein_background_contribution": background_contribution,
            "protein_foreground_contribution": foreground_contribution,
            "protein_total_mean": background_contribution + foreground_contribution,
            "protein_batch_efficiency": efficiency,
        }

    @torch.inference_mode()
    def _infer_lfc_aux(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        batch_index: torch.Tensor,
        panel_index: torch.Tensor | None = None,
        label: torch.Tensor | None = None,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        sample_index: torch.Tensor | None = None,
        cf_sample: torch.Tensor | None = None,
        **_,
    ) -> dict:
        """Return inference outputs that are constant across CRN MC draws.

        ``library_gene`` depends only on ``x`` and ``batch_index``, not on
        ``u_anchor``, so it is safe to compute once per batch value and cache.
        Called by ``_stats.py`` before the MC loop to avoid running the full
        encoder ``mc_samples*(1+n_fixed)`` times per batch value.
        """
        n_cells = x.shape[0]
        if label is None:
            label = torch.zeros(n_cells, 1, dtype=torch.long, device=x.device)
        if panel_index is None:
            panel_index = batch_index
        out = self._regular_inference(
            x,
            y,
            batch_index=batch_index,
            panel_index=panel_index,
            label=label,
            n_samples=1,
            cont_covs=cont_covs,
            cat_covs=cat_covs,
            sample_index=sample_index,
            cf_sample=cf_sample,
        )
        return {"library_gene": out["library_gene"]}

    @torch.inference_mode()
    def compute_h_from_x_eps(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        sample_index: torch.Tensor,
        batch_index: torch.Tensor,
        extra_eps: torch.Tensor,
        label: torch.Tensor | None = None,
        panel_index: torch.Tensor | None = None,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        cf_sample: torch.Tensor | None = None,
        u_anchor: torch.Tensor | None = None,
        _lfc_aux: dict | None = None,
        zp: torch.Tensor | None = None,
        protein_latent_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ``concat([px_scale, py_scale_det])`` at ``z = z_base + extra_eps``.

        This is the decoder hook consumed by the LFC block in
        :func:`~mrtotalvi._stats.differential_expression` when
        ``store_lfc=True``.  It mirrors MRVI's ``compute_h_from_x_eps`` but
        handles the TotalVI multimodal output (RNA + protein) and uses a
        deterministic protein background reconstruction (D-021) so that the
        x_1/x_0 contrast differs only via ``extra_eps``, not via independent
        background draws.

        Parameters
        ----------
        x
            RNA count matrix, shape ``(batch, n_genes)``.
        y
            Protein count matrix, shape ``(batch, n_proteins)``.
        sample_index
            Integer donor index, shape ``(batch, 1)`` or ``(batch,)``.
        batch_index
            Batch integer index for the counterfactual decode, shape
            ``(batch, 1)``.  Can differ from the cell's own batch to obtain
            batch-averaged LFC.
        extra_eps
            Counterfactual eps shift, shape ``(batch, n_latent)``.
            Set to the regression beta vector for the active covariate when
            computing x_1; set to the null (zeros or ``eps_mean``) for x_0.
        label
            Cell-type label tensor.  Defaults to zeros when ``None``.
        panel_index
            Protein panel index.  Defaults to ``batch_index`` when ``None``.
        cont_covs
            Continuous covariate tensor (passed through to inference).
        cat_covs
            Categorical covariate tensor (passed through to inference).
        cf_sample
            Counterfactual sample override for the eps encoder.  ``None``
            uses ``sample_index`` (real donor).
        zp
            Posterior mean of the cell's own ``z_p``, with ``n_latent_protein_only > 0``. ``None``
            encodes ``y`` here with the passed ``batch_index``, which differs from the cell's batch
            when that is a counterfactual batch and ``ambient_zscore`` uses per-batch statistics.

        Returns
        -------
        :class:`torch.Tensor`
            Shape ``(batch, n_genes + n_proteins)``.  RNA columns first,
            protein columns second.  The ``gene``/``protein`` coordinate
            split is applied at the model level (B3) when assembling the
            xarray output.
        """
        n_cells = x.shape[0]
        if label is None:
            label = torch.zeros(n_cells, 1, dtype=torch.long, device=x.device)
        if panel_index is None:
            panel_index = batch_index

        sample_index_cf = sample_index if cf_sample is None else cf_sample

        if u_anchor is not None and _lfc_aux is not None:
            # CRN fast path: skip inference — library_gene is constant per batch
            # value and was pre-computed by _infer_lfc_aux in _stats.py.
            u = u_anchor
            library_gene = _lfc_aux["library_gene"]
        else:
            # Full inference path: needed when u_anchor is None (legacy biased
            # path) or when no aux cache is available.
            if u_anchor is None:
                import warnings

                warnings.warn(
                    "compute_h_from_x_eps called without u_anchor: falling back to "
                    "qu.mean, which produces Jensen-biased LFC estimates. Pass "
                    "u_anchor (a sample from q(u|x)) for unbiased CRN estimation.",
                    UserWarning,
                    stacklevel=2,
                )
            out = self._regular_inference(
                x,
                y,
                batch_index=batch_index,
                panel_index=panel_index,
                label=label,
                n_samples=1,
                cont_covs=cont_covs,
                cat_covs=cat_covs,
                sample_index=sample_index,
                cf_sample=cf_sample,
            )
            qu = out["qu"]
            u = u_anchor if u_anchor is not None else qu.mean
            library_gene = out["library_gene"]

        z_base, _, _ = self.qz(u, sample_index_cf)
        z = z_base + extra_eps  # counterfactual latent

        if self.n_latent_protein_only > 0 and zp is None:
            zp = self._protein_latent_dist(y, batch_index, sample_index, protein_latent_input).loc

        gen_out = self.generative(
            z=z,
            library_gene=library_gene,
            batch_index=batch_index,
            label=label,
            cont_covs=cont_covs,
            cat_covs=cat_covs,
            zp=zp,
        )

        px_ = gen_out["px_"]
        py_ = gen_out["py_"]

        # RNA: softmax scale is deterministic given z.
        px_scale = px_["scale"]  # (batch, n_genes)

        # Protein: deterministic background reconstruction (D-021).
        # py_["scale"] is stochastic because DecoderTOTALVI calls
        # Normal(back_alpha, back_beta).rsample() internally.  Using that
        # stochastic draw in both x_1 and x_0 introduces independent
        # background noise that does NOT cancel in the LFC.  Fix: use the
        # log-mean back_alpha directly (exp of the Gaussian mean) as a
        # deterministic background stand-in.
        rate_back_det = torch.exp(py_["back_alpha"])  # (batch, n_proteins)
        rate_fore_det = rate_back_det * py_["fore_scale"]  # fore_scale is deterministic
        # py_["mixing"] is a raw logit; sigmoid → mixing probability.
        protein_mixing_det = torch.sigmoid(py_["mixing"])
        py_scale_det = F.normalize((1.0 - protein_mixing_det) * rate_fore_det, p=1, dim=-1)  # (batch, n_proteins)

        return torch.cat([px_scale, py_scale_det], dim=-1)  # (batch, n_genes + n_proteins)
