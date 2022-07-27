r"""Locations of example data that are included in the package for convenience.

Attributes
----------
NEVO_PRODUCTS_LOCATION : `str`
    Location of a CSV file containing the fake cereal product data from :ref:`references:Nevo (2000a)`. The file
    includes the same pre-computed excluded instruments used in the original paper. The data are from Aviv Nevo's Matlab
    code, which was archived on Eric Rasmusen's website.
NEVO_AGENTS_LOCATION : `str`
    Location of a CSV file containing the agent data from :ref:`references:Nevo (2000a)`. Included in the file are Monte
    Carlo weights and draws along with demographics from the original paper. The data are from Aviv Nevo's Matlab code,
    which was archived on Eric Rasmusen's website.
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
    Location of a CSV file containing the agent data from :ref:`references:Berry, Levinsohn, and Pakes (1999)`. Included
    in the file are the importance sampling weights and draws along with the income demographic from the original paper.
    These data are also from the replication code of :ref:`references:Andrews, Gentzkow, and Shapiro (2017)`.
PETRIN_PRODUCTS_LOCATION : `str`
    Location of a CSV file containing the automobile product data from :ref:`references:Petrin (2002)`. The file
    includes the same pre-computed excluded instruments used in the original paper. The data are from Amil Petrin's
    GAUSS code, available on his website.
PETRIN_AGENTS_LOCATION : `str`
    Location of a CSV file containing agent data similar to that used by :ref:`references:Petrin (2002)`. The file
    includes 1,000 scrambled Halton draws in each market, along with demographics resampled from the Consumer
    Expenditure Survey (CEX) used by the original paper. The original paper used pseudo Monte Carlo draws and
    importance sampling. The demographics that were resampled are from Amil Petrin's GAUSS code, available on his
    website.
PETRIN_VALUES_LOCATION : `str`
    Location of a CSV file containing micro moment values matched by :ref:`references:Petrin (2002)`. These are the
    rounded values reported in Table 6a of the working paper version of the original paper.
PETRIN_COVARIANCES_LOCATION : `str`
    Location of a CSV file containing micro moment sample covariances used by :ref:`references:Petrin (2002)`. The data
    are from Amil Petrin's GAUSS code, available on his website.

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
PETRIN_PRODUCTS_LOCATION = str(_DATA_PATH / 'petrin_products.csv')
PETRIN_AGENTS_LOCATION = str(_DATA_PATH / 'petrin_agents.csv')
PETRIN_VALUES_LOCATION = str(_DATA_PATH / 'petrin_values.csv')
PETRIN_COVARIANCES_LOCATION = str(_DATA_PATH / 'petrin_covariances.csv')
