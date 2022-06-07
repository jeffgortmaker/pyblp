"""Sets up the package."""

import re
from pathlib import Path

from setuptools import find_packages, setup


# define a function that reads a file in this directory
read = lambda p: Path(Path(__file__).resolve().parent / p).read_text()

# parse the current version
version_match = re.search(r'^__version__ = \'([^\']*)\'', read('pyblp/version.py'), re.M)
assert version_match is not None

# set up the package
setup(
    name='pyblp',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=read('requirements.txt').splitlines(),
    extras_require={
        'tests': ['pytest', 'pytest-xdist', 'linearmodels', 'dill', 'pathos'],
        'docs': [
            'sphinx==2.0.0', 'pandas', 'ipython', 'matplotlib', 'astunparse', 'sphinx-rtd-theme==0.4.3',
            'nbsphinx==0.5.0', 'jinja2~=2.11', 'docutils==0.17',
        ],
    },
    include_package_data=True,
    description="BLP demand estimation with Python 3",
    long_description=read('README.rst').split('description-start')[1].strip(),
    version=version_match.group(1),
    author="Jeff Gortmaker, Christopher Conlon",
    author_email="jeff@jeffgortmaker.com",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering"
    ],
    url="https://github.com/jeffgortmaker/pyblp",
    project_urls={
        "Documentation": "http://pyblp.readthedocs.io/en/latest",
        "Tracker": "https://github.com/jeffgortmaker/pyblp/issues"
    }
)
