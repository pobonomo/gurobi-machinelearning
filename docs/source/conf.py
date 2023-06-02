# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from pathlib import Path

from sphinx_pyproject import SphinxConfig

sys.path.insert(0, os.path.abspath("../../src/"))
# -- Project information -----------------------------------------------------
config = SphinxConfig("../../pyproject.toml", globalns=globals())

project = "Gurobi Machine Learning"
copyright = "2023, Gurobi Optimization, LLC. All Rights Reserved."
html_logo = "_static/gurobi-logo-title.png"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.extlinks",
    "sphinx.ext.imgconverter",
    "sphinx_rtd_theme",
    "sphinx_copybutton",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinx.ext.autosectionlabel",
    "nbsphinx",
    "sphinx_design",
]


def get_versions(file: Path, acc=None):
    if acc is None:
        acc = dict()
    new_dict = {x.split("==")[0]: x.split("==")[1] for x in file.read_text().split()}
    return {**new_dict, **acc}


root_path = Path().resolve().parent.parent
dep_versions = get_versions(root_path / "requirements.tox.txt")
dep_versions = get_versions(root_path / "requirements.keras.txt", dep_versions)
dep_versions = get_versions(root_path / "requirements.pytorch.txt", dep_versions)
dep_versions = get_versions(root_path / "requirements.sklearn.txt", dep_versions)
dep_versions = get_versions(root_path / "requirements.pandas.txt", dep_versions)


VARS_SHAPE = """See :py:func:`add_predictor_constr <gurobi_ml.add_predictor_constr>` for acceptable values for input_vars and output_vars"""
CLASS_SHORT = """Stores the changes to :gurobipy:`model` for representing an instance into it.\n    Inherits from :class:`AbstractPredictorConstr <gurobi_ml.modeling.base_predictor_constr.AbstractPredictorConstr>`.\n"""


rst_epilog = f"""
.. |GurobiVersion| replace:: {dep_versions["gurobipy"]}
.. |NumpyVersion| replace:: {dep_versions["numpy"]}
.. |ScipyVersion| replace:: {dep_versions["scipy"]}
.. |PandasVersion| replace:: {dep_versions["pandas"]}
.. |TorchVersion| replace:: {dep_versions["torch"]}
.. |SklearnVersion| replace:: {dep_versions["scikit-learn"]}
.. |TensorflowVersion| replace:: {dep_versions["tensorflow"]}
.. |VariablesDimensionsWarn| replace:: {VARS_SHAPE}
.. |ClassShort| replace:: {CLASS_SHORT}
"""

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

intersphinx_mapping = {
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "torch": ("https://pytorch.org/docs/master/", None),
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"


myst_enable_extensions = [
    "dollarmath",
]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
autodoc_member_order = "groupwise"
autodoc_mock_imports = ["torch", "tensorflow"]
nbsphinx_custom_formats = {
    ".md": ["jupytext.reads", {"fmt": "myst"}],
}
nbsphinx_allow_errors = False
bibtex_bibfiles = ["refs.bib"]

extlinks_detect_hardcoded_links = True
extlinks = {
    "issue": ("https://github.com/Gurobi/gurobi-machinelearning/issues/%s", "issue %s"),
    "gurobipy": (
        "https://www.gurobi.com/documentation/current/refman/py_%s.html",
        "gurobipy %s",
    ),
    "pypi": ("https://pypi.org/project/%s/", "%s"),
}

# -- Options for LaTeX output -----------------------------------------------------
latex_logo = "_static/gurobi.png"

latex_elements = {
    "preamble": r"""
    \newcommand\sphinxbackoftitlepage{%
Copyright(c), 2023, Gurobi Optimization, LLC. All Rights Reserved.
}
    """,
}
