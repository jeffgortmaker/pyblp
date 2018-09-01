"""Sphinx configuration."""

from pathlib import Path

import pyblp


# get the location of the source directory
source_path = Path(__file__).resolve().parent

# configure locations of other configuration files
html_static_path = ['static']
templates_path = ['templates']
exclude_patterns = templates_path

# configure project information
language = 'en'
project = 'pyblp'
author = 'Jeff Gortmaker'
copyright = f'2018, {author}'
release = version = pyblp.__version__

# configure build information
nitpicky = True
tls_verify = False
master_doc = 'index'
source_suffix = '.rst'

# configure extensions
extensions = [
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon'
]
intersphinx_mapping = {
    'matplotlib': ('https://matplotlib.org/', None),
    'python': ('https://docs.python.org/3.6/', None),
    'sympy': ('https://docs.sympy.org/latest/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'patsy': ('https://patsy.readthedocs.io/en/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'pandas': ('http://pandas.pydata.org/pandas-docs/stable/', None),

}
math_number_all = True
autosummary_generate = True
ipython_savefig_dir = 'images'
numpydoc_show_class_members = False
ipython_execlines = Path(source_path / 'include.py').read_text().splitlines()

# add a line to be executed by IPython that stores the location of the source directory so that files can be saved
ipython_execlines.append(f'source_path = \'{source_path.as_posix()}\'')

# configure theme information
html_theme = 'sphinx_rtd_theme'


def setup(app):
    """Configure extra resources."""
    app.add_javascript('override.js')
    app.add_stylesheet('override.css')
