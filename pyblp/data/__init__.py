r"""Locations of example data that are included in the package for convenience.

Attributes
----------
NEVO_PRODUCTS_LOCATION : `str`
    Location of a CSV file containing the fake cereal product data from :ref:`references:Nevo (2000)`. The file includes
    the same pre-computed excluded instruments used in the original paper.
NEVO_AGENTS_LOCATION : `str`
    Location of a CSV file containing the fake cereal agent data. Included in the file are Monte Carlo weights and
    draws along with demographics from the original paper.
BLP_PRODUCTS_LOCATION : `str`
    Location of a CSV file containing the automobile product data extracted by
    :ref:`references:Andrews, Gentzkow, and Shapiro (2017)` from the original GAUSS code for
    :ref:`references:Berry, Levinsohn, and Pakes (1999)`, which is commonly assumed to be the same data used in
    :ref:`references:Berry, Levinsohn, and Pakes (1995)`.

    The file also includes a set of excluded instruments. First, "sums of characteristics" BLP instruments from the
    original paper were computed with :func:`~pyblp.build_blp_instruments`. The examples section in the documentation
    for this function shows how to construct these instruments from scratch. As in the original paper, the "rival"
    instrument constructed from the ``trend`` variable was excluded due to collinearity issues, and the ``mpd``
    variable was added to the set of excluded instruments for supply.

BLP_AGENTS_LOCATION : `str`
    Location of a CSV file containing the automobile agent data. Included in the file are the importance sampling
    weights and draws along with the income demographic from the original paper.

Examples
--------
.. toctree::

   /_notebooks/api/data.ipynb

"""

from pathlib import Path


_DATA_PATH = Path(__file__).resolve().parent
NEVO_PRODUCTS_LOCATION = str(_DATA_PATH / 'nevo_products.csv')
NEVO_AGENTS_LOCATION = str(_DATA_PATH / 'nevo_agents.csv')
BLP_PRODUCTS_LOCATION = str(_DATA_PATH / 'blp_products.csv')
BLP_AGENTS_LOCATION = str(_DATA_PATH / 'blp_agents.csv')
