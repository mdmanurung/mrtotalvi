# Installation

mrtotalvi needs Python 3.12 or newer.
We recommend [uv][], which installs Python for you and keeps each project’s dependencies in their own virtual environment.

## Development version

mrtotalvi is not on PyPI yet.
Install the latest development version from GitHub:

:::::{tab-set}
::::{tab-item} uv
:sync: uv

```bash
uv add git+https://github.com/mdmanurung/mrtotalvi.git
```
::::

::::{tab-item} Pip
:sync: pip

```bash
pip install git+https://github.com/mdmanurung/mrtotalvi.git
```
::::
:::::

The `lamin` extra adds [lamindb][] support: `pip install "mrtotalvi[lamin] @ git+https://github.com/mdmanurung/mrtotalvi.git"`.

mrtotalvi depends on scvi-tools, which pulls in PyTorch.
For a GPU or CPU-only build, install PyTorch first by following the [PyTorch install guide][pytorch].

To work on mrtotalvi itself, see {doc}`contributing`.

[uv]: https://docs.astral.sh/uv/
[lamindb]: https://docs.lamin.ai/
[pytorch]: https://pytorch.org/get-started/locally/
