"""MrTotalVI — TotalVI with per-donor hierarchical latent space (MrVI-style)."""

from importlib.metadata import PackageNotFoundError, version

from ._model import MrTotalVI
from ._module import MrTotalVAE
from ._seed import combine_mrtotalvi_seed_results

try:
    __version__ = version("mrtotalvi")
except PackageNotFoundError:  # source checkout without install
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    "MrTotalVI",
    "MrTotalVAE",
    "combine_mrtotalvi_seed_results",
]
