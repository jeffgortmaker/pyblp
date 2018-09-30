"""Locations of example data that are included in the package for convenience.

Attributes
----------
NEVO_PRODUCTS_LOCATION : `str`
    Location of a CSV file containing the fake cereal product data from :ref:`Nevo (2000) <n00>`. The file includes
    the same pre-computed excluded instruments used in the original paper.
NEVO_AGENTS_LOCATION : `str`
    Location of a CSV file containing the fake cereal agent data. Included in the file are Monte Carlo weights and
    draws along with demographics, which collectively are used by :ref:`Nevo (2000) <n00>` to solve the fake cereal
    problem.
BLP_PRODUCTS_LOCATION : `str`
    Location of a CSV file containing the automobile product data extracted by
    :ref:`Andrews, Gentzkow, and Shapiro (2017) <ags17>` from the original GAUSS code for
    :ref:`Berry, Levinsohn, and Pakes (1999) <blp99>`, which is commonly assumed to be the same data used in
    :ref:`Berry, Levinsohn, and Pakes (1995) <blp95>`.

    The file also includes a set of :ref:`Chamberlain's (1987) <c87>` optimal excluded instruments for the automobile
    problem from :ref:`Berry, Levinsohn, and Pakes (1995) <blp95>`, which are used to solve the problem in
    :doc:`Examples </examples>`. These instruments were computed according to the following procedure:

        1. Traditional excluded BLP instruments from the original paper were computed with
           :func:`~pyblp.build_blp_instruments`. As in the original paper, the ``mpd`` variable was added to the set of
           excluded supply-side instruments.
        2. Each set of excluded instruments was interacted up to the second degree, standardized, and replaced with the
           minimum set of principal components that explained at least 99% of the variance.
        3. These two sets of principal components were used as excluded demand- and supply-side instruments when solving
           the first GMM stage of a :class:`~pyblp.Problem` configured as in :doc:`Examples </examples>` (but with
           non-optimal instruments).
        4. The :meth:`~pyblp.ProblemResults.compute_optimal_instruments` method was used to estimate the optimal
           excluded instruments for the problem.

BLP_AGENTS_LOCATION : `str`
    Location of a CSV file containing automobile agent data. Included in the file are 200 Monte Carlo weights and draws
    for each market, which, unlike in the fake cereal data, are not the same draws used in the original paper.

    Also included is an income demographic, which consists of draws from lognormal distributions with common standard
    deviation ``1.72`` and the following market-varying means:

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

    These numbers were extracted also extracted from the original GAUSS code for
    :ref:`Berry, Levinsohn, and Pakes (1999) <blp99>`.

Examples
--------
Any number of functions can be used to load these data into memory. In this example, we'll first use :mod:`numpy`.

.. ipython:: python

   import numpy as np
   blp_product_data = np.recfromcsv(pyblp.data.BLP_PRODUCTS_LOCATION, encoding='utf-8')
   blp_agent_data = np.recfromcsv(pyblp.data.BLP_AGENTS_LOCATION, encoding='utf-8')

Record arrays can be cumbersome to manipulate. A more flexible alternative, the :class:`pandas.DataFrame`, can be built
by the :func:`pandas.read_csv` function in the :mod:`pandas` package, which, unlike :mod:`numpy`, is not a pyblp
requirement.

.. ipython:: python

   import pandas as pd
   blp_product_data = pd.read_csv(pyblp.data.BLP_PRODUCTS_LOCATION)
   blp_agent_data = pd.read_csv(pyblp.data.BLP_AGENTS_LOCATION)

For more examples, refer to the :doc:`Examples </examples>` section.

"""

from pathlib import Path


_DATA_PATH = Path(__file__).resolve().parent
NEVO_PRODUCTS_LOCATION = str(_DATA_PATH / 'nevo_products.csv')
NEVO_AGENTS_LOCATION = str(_DATA_PATH / 'nevo_agents.csv')
BLP_PRODUCTS_LOCATION = str(_DATA_PATH / 'blp_products.csv')
BLP_AGENTS_LOCATION = str(_DATA_PATH / 'blp_agents.csv')
