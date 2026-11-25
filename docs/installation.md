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

## Optional extras

  | Extra      | Adds                                                                                                                        | Install                                                                               |
  | ---------- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
  | `autotune` | hyperparameter search with {func}`mrtotalvi.run_autotune` (Ray Tune, hyperopt, scib-metrics through `scvi-tools[autotune]`) | `pip install "mrtotalvi[autotune] @ git+https://github.com/mdmanurung/mrtotalvi.git"` |
  | `lamin`    | [lamindb][] support                                                                                                         | `pip install "mrtotalvi[lamin] @ git+https://github.com/mdmanurung/mrtotalvi.git"`    |

With uv, add the extra in brackets: `uv add "mrtotalvi[autotune] @ git+https://github.com/mdmanurung/mrtotalvi.git"`.
On a shared machine, start Ray with a CPU cap first, for example `ray.init(num_cpus=4)`, and pass `ignore_reinit_error=True` to `run_autotune`. Left alone, Ray claims every CPU on the node and can stall at startup.

## PyTorch

mrtotalvi depends on scvi-tools, which pulls in PyTorch.
For a GPU or CPU-only build, install PyTorch first by following the [PyTorch install guide][pytorch].

To work on mrtotalvi itself, see {doc}`contributing`.

[uv]: https://docs.astral.sh/uv/
[lamindb]: https://docs.lamin.ai/
[pytorch]: https://pytorch.org/get-started/locally/
