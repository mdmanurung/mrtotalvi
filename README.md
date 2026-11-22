# mrtotalvi

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/mdmanurung/mrtotalvi/test.yaml?branch=main
[badge-docs]: https://app.readthedocs.org/projects/mrtotalvi/badge/

MrTotalVI models CITE-seq RNA and protein counts in a sample-aware latent space.
It combines the TOTALVI likelihood ([Gayoso et al. 2021][totalvi]) with the two-level latent space of MrVI ([Boyeau et al. 2025][mrvi]):

- `u` is the cell state, from an encoder conditioned on sample.
- `z = z_base(u) + eps(u, sample)` adds a sample-specific residual from an attention module.
- The TOTALVI decoder reads `z`, batch and library size.

It builds on [scvi-tools][] (>= 1.5.2).

## Installation

mrtotalvi needs Python 3.12 or newer. It is not on PyPI yet, so install it from GitHub:

```bash
pip install git+https://github.com/mdmanurung/mrtotalvi.git
```

See [installation][] for uv and GPU setups.

## Usage

`adata` holds raw RNA counts in the `counts` layer and protein counts in `obsm["protein"]`.

```python
from mrtotalvi import MrTotalVI

MrTotalVI.setup_anndata(
    adata,
    layer="counts",
    protein_expression_obsm_key="protein",
    sample_key="donor_timepoint",
    batch_key="batch",
)
model = MrTotalVI(adata, sample_key="donor_timepoint")
model.train(n_epochs_kl_warmup=40)  # set warm-up here; train() overwrites plan_kwargs values

u = model.get_latent_representation(give_z=False)  # sample-free cell state
z = model.get_latent_representation(give_z=True)  # cell state with the sample residual
```

The [user guide][] covers priors, counterfactual samples and the statistical APIs.
The [API reference][] lists every method.
Users of the `mdmanurung/scvi-tools` fork should read the [migration notes][].

## Release notes and bugs

See the [changelog][]. Report bugs on the [issue tracker][].

## Citation

A mrtotalvi paper is not out yet. Please cite the two methods it combines:

- Gayoso A, Steier Z, Lopez R, et al. Joint probabilistic modeling of single-cell multi-omic data with totalVI. *Nature Methods* 18, 272–282 (2021). [doi:10.1038/s41592-020-01050-x][totalvi]
- Boyeau P, Hong J, Gayoso A, et al. Deep generative modeling of sample-level heterogeneity in single-cell genomics. *Nature Methods* 22, 2264–2274 (2025). [doi:10.1038/s41592-025-02808-x][mrvi]

## License

BSD 3-Clause. The code derives from the scvi-tools TOTALVI and MrVI implementations; see `LICENSE`.

[totalvi]: https://doi.org/10.1038/s41592-020-01050-x
[mrvi]: https://doi.org/10.1038/s41592-025-02808-x
[scvi-tools]: https://scvi-tools.org/
[issue tracker]: https://github.com/mdmanurung/mrtotalvi/issues
[tests]: https://github.com/mdmanurung/mrtotalvi/actions/workflows/test.yaml
[documentation]: https://mrtotalvi.readthedocs.io
[installation]: https://mrtotalvi.readthedocs.io/page/installation.html
[user guide]: https://mrtotalvi.readthedocs.io/page/user_guide.html
[migration notes]: https://mrtotalvi.readthedocs.io/page/migration.html
[changelog]: https://mrtotalvi.readthedocs.io/page/changelog.html
[api reference]: https://mrtotalvi.readthedocs.io/page/api.html
