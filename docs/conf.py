"""Sphinx configuration for the mrtotalvi docs."""

from importlib.metadata import version as _version

project = "mrtotalvi"
author = "mdmanurung"
release = _version("mrtotalvi")

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
myst_heading_anchors = 3
exclude_patterns = ["_build", "reviews/**"]
html_theme = "furo"
html_title = "mrtotalvi"
# ADR-0008 links a local .scratch file that is not in the repo.
suppress_warnings = ["myst.xref_missing"]
