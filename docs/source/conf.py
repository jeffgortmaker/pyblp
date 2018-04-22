"""Sphinx configuration."""

from pathlib import Path

import pyblp


# configure project information
language = 'en'
project = 'pyblp'
author = 'Jeff Gortmaker'
copyright = f'2018, {author}'
release = version = pyblp.__version__

# configure build information
master_doc = 'index'
source_suffix = '.rst'
htmlhelp_basename = 'blpdoc'

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
ipython_execlines = Path(Path(__file__).resolve().parent / '_static' / 'header.py').read_text().splitlines()
intersphinx_mapping = {
    'matplotlib': ('https://matplotlib.org/', None),
    'python': ('https://docs.python.org/3.6/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'pandas': ('http://pandas.pydata.org/pandas-docs/stable/', None)
}
math_number_all = True
autosummary_generate = True
html_show_sourcelink = False
ipython_savefig_dir = 'images'
numpydoc_show_class_members = False

# configure locations of other configuration files
html_static_path = ['_static']
templates_path = ['_templates']
exclude_patterns = templates_path

# configure theme information
html_theme = 'sphinx_rtd_theme'
def setup(app):
    app.add_javascript('override.js')
    app.add_stylesheet('override.css')
