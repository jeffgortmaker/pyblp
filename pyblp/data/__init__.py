r"""Locations of example data that are included in the package for convenience.

Attributes
----------
NEVO_PRODUCTS_LOCATION : `str`
    Location of a CSV file containing the fake cereal product data from :ref:`references:Nevo (2000)`. The file includes
    the same pre-computed excluded instruments used in the original paper.
NEVO_AGENTS_LOCATION : `str`
    Location of a CSV file containing the fake cereal agent data. Included in the file are Monte Carlo weights and
    draws along with demographics, which are used by :ref:`references:Nevo (2000)` to solve the fake cereal problem.
BLP_PRODUCTS_LOCATION : `str`
    Location of a CSV file containing the automobile product data extracted by
    :ref:`references:Andrews, Gentzkow, and Shapiro (2017)` from the original GAUSS code for
    :ref:`references:Berry, Levinsohn, and Pakes (1999)`, which is commonly assumed to be the same data used in
    :ref:`references:Berry, Levinsohn, and Pakes (1995)`.

    The file also includes a set of excluded instruments. First, "sums of characteristics" BLP instruments from the
    original paper were computed with :func:`~pyblp.build_blp_instruments`. As in the original paper, the ``mpd``
    variable was added to the set of excluded instruments for supply. Due to a collinearity problem with these original
    instruments, each set of excluded instruments was then interacted up to the second degree, standardized, replaced
    with the minimum set of principal components that explained at least 99% of the variance, and standardized again.

BLP_AGENTS_LOCATION : `str`
    Location of a CSV file containing automobile agent data. Included in the file are 200 Monte Carlo weights and draws
    for each market, which, unlike in the fake cereal data, are not the same draws used in the original paper.

    Also included is an income demographic, which consists of draws from lognormal distributions. The log of income has
    a common standard deviation, ``1.72``, and the following market-varying means:

    ====  =======
    Year  Mean
    ====  =======
    1971  2.01156
    1972  2.06526
    1973  2.07843
    1974  2.05775
    1975  2.02915
    1976  2.05346
    1977  2.06745
    1978  2.09805
    1979  2.10404
    1980  2.07208
    1981  2.06019
    1982  2.06561
    1983  2.07672
    1984  2.10437
    1985  2.12608
    1986  2.16426
    1987  2.18071
    1988  2.18856
    1989  2.21250
    1990  2.18377
    ====  =======

    These numbers were extracted also extracted by :ref:`references:Andrews, Gentzkow, and Shapiro (2017)` from the
    original GAUSS code for :ref:`references:Berry, Levinsohn, and Pakes (1999)`.

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
