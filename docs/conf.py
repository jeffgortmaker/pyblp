"""Sphinx configuration."""

import ast
import copy
import os
import json
from pathlib import Path
import re
import shutil
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

# construct a RTD URL that will be used in absolute links and make it available to included files
rtd_version = os.environ.get('READTHEDOCS_VERSION', 'latest')
rtd_url = f'https://{project}.readthedocs.io/{language}/{rtd_version}'

# configure extensions
extensions = [
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.extlinks',
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
    'pandas': ('http://pandas.pydata.org/pandas-docs/stable/', None)
}
extlinks = {
    'rtd': (f'{rtd_url}/%s', None)
}
math_number_all = True
autosummary_generate = True
numpydoc_show_class_members = False
autosectionlabel_prefix_document = True
nbsphinx_prolog = read('templates/nbsphinx_prolog.rst')
nbsphinx_epilog = read('templates/nbsphinx_epilog.rst')

# configure HTML information
html_theme = 'sphinx_rtd_theme'

# configure LaTeX information
latex_elements = {
    'preamble': read('static/preamble.tex')
}


def clean_directories() -> None:
    """Clean directories that will be generated."""
    for name in ['_api', '_downloads', '_notebooks']:
        shutil.rmtree(source_path / name, ignore_errors=True)


def process_notebooks() -> None:
    """Copy notebook files to _notebooks and _downloads, resetting executing counts and replacing domains with Markdown
    equivalents.
    """
    for notebook_path in Path(source_path / 'notebooks').glob('**/*.ipynb'):
        notebook = json.loads(notebook_path.read_text())
        download = copy.deepcopy(notebook)

        # extract parts of the path relative to the notebooks directory and construct the directory's relative location
        relative_parts = notebook_path.relative_to(source_path).parts[1:]
        relative_location = '../' * len(relative_parts)

        # manipulate notebook cells
        for notebook_cell, download_cell in zip(notebook['cells'], download['cells']):
            # reset download execution counts
            for data in [download_cell] + download_cell.get('outputs', []):
                if 'execution_count' in data:
                    data['execution_count'] = 1

            # replace supported Sphinx domains with Markdown equivalents
            if notebook_cell['cell_type'] == 'markdown':
                for source_index, notebook_source in enumerate(notebook_cell['source']):
                    for role, content in re.findall(':([a-z]+):`([^`]+)`', notebook_source):
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

                        # replace the domain with Markdown equivalents (reStructuredText doesn't support linked code)
                        notebook_cell['source'][source_index] = notebook_cell['source'][source_index].replace(
                            domain,
                            f'[{text.strip("`")}]({relative_location}{document}.rst#{section})'
                        )
                        download_cell['source'][source_index] = download_cell['source'][source_index].replace(
                            domain,
                            f'[{text}]({rtd_url}/{document}.html#{section})'
                        )

        # save the updated notebook files
        for updated, location in [(download, '_downloads'), (notebook, '_notebooks')]:
            updated_path = source_path / Path(location, *relative_parts)
            updated_path.parent.mkdir(parents=True, exist_ok=True)
            updated_path.write_text(json.dumps(updated, indent=1, sort_keys=True, separators=(', ', ': ')))


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
    """Clean directories, process notebooks, configure extra resources, and strip type hints."""
    clean_directories()
    process_notebooks()
    app.add_javascript('override.js')
    app.add_stylesheet('override.css')
    app.connect('autodoc-process-signature', process_signature)
