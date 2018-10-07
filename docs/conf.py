"""Sphinx configuration."""

import ast
import os
from pathlib import Path
import re
from typing import Any, Optional, Tuple

import astunparse
import pyblp
import sphinx.application


# get the location of the source directory
source_path = Path(__file__).resolve().parent

# define a function that reads a file in this directory
read = lambda p: Path(Path(__file__).resolve().parent / p).read_text()

# configure locations of other configuration files
html_static_path = ['static']
templates_path = ['templates']
exclude_patterns = ['_build', '_downloads', 'notebooks', 'templates', '**.ipynb_checkpoints']

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

# configure extensions
extensions = [
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'nbsphinx'
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
numpydoc_show_class_members = False
autosectionlabel_prefix_document = True
nbsphinx_prolog = read('templates/nbsphinx_prolog.rst')

# configure HTM information
html_theme = 'sphinx_rtd_theme'

# configure LaTeX information
latex_show_urls = 'footnote'
latex_elements = {
    'preamble': read('static/preamble.tex')
}


def process_notebooks() -> None:
    """Copy notebook files to _notebooks and _downloads, replacing domains with Markdown equivalents."""

    # construct a RTD URL that will be used in absolute links
    rtd_version = os.environ.get('READTHEDOCS_VERSION', 'latest')
    rtd_url = f'https://{project}.readthedocs.io/{language}/{rtd_version}/'

    # load each notebook
    for notebook_path in Path(source_path / 'notebooks').glob('**/*.ipynb'):
        download = notebook = notebook_path.read_text()

        # extract parts of the path relative to the notebooks directory and construct the directory's relative location
        relative_parts = notebook_path.relative_to(source_path).parts[1:]
        relative_location = '../' * len(relative_parts)

        # extract the text, document, and section associated with supported domains
        for role, content in re.findall(':([a-z]+):`([^`]+)`', notebook):
            domain = f':{role}:`{content}`'
            if role == 'ref':
                document, text = content.split(':', 1)
                section = re.sub(r'-+', '-', re.sub('[^0-9a-zA-Z]+', '-', text)).strip('-').lower()
            elif role in {'mod', 'func', 'class', 'meth', 'attr', 'exc'}:
                text = f'`{content}`'
                section = f'{project}.{content}'
                document = f'_api/{project}.{content}'
                if role == 'mod':
                    section = f'module-{section}'
                elif role == 'attr':
                    document = document.rsplit('.', 1)[0]
            else:
                raise NotImplementedError(f"The domain '{domain}' is not supported.")

            # reStructuredText does not support linked code (API links will be formatted with Javascript overrides)
            download_text = text
            notebook_text = text.strip('`')

            # replace the domain with Markdown equivalents
            download = download.replace(domain, f'[{download_text}]({rtd_url}{document}.html#{section})')
            notebook = notebook.replace(domain, f'[{notebook_text}]({relative_location}{document}.rst#{section})')

        # save the updated notebook files
        for updated, location in [(download, '_downloads'), (notebook, '_notebooks')]:
            updated_path = source_path / Path(location, *relative_parts)
            updated_path.parent.mkdir(parents=True, exist_ok=True)
            updated_path.write_text(updated)


def process_signature(*args: Any) -> Optional[Tuple[str, str]]:
    """Strip type hints from signatures."""
    signature = args[5]
    if signature is None:
        return None
    assert isinstance(signature, str)
    node = ast.parse(f'def f{signature}: pass').body[0]
    assert isinstance(node, ast.FunctionDef)
    node.returns = None
    if node.args.args:
        for arg in node.args.args:
            arg.annotation = None
    return astunparse.unparse(node).splitlines()[2][5:-1], ''


def setup(app: sphinx.application.Sphinx) -> None:
    """Process notebooks, configure extra resources, and strip type hints."""
    process_notebooks()
    app.add_javascript('override.js')
    app.add_stylesheet('override.css')
    app.connect('autodoc-process-signature', process_signature)
