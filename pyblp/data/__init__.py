"""Locations of example data that are included in the package for convenience.

Attributes
----------
BLP_PRODUCTS_LOCATION : `str`
    Location of a CSV file containing the automobile product data extracted by
    :ref:`Knittel and Metaxoglou (2014) <km14>` from the original GAUSS code for
    :ref:`Berry, Levinsohn, and Pakes (1999) <blp99>`, which is commonly assumed to be the same data used in
    :ref:`Berry, Levinsohn, and Pakes (1995) <blp95>`.
BLP_AGENTS_LOCATION : `str`
    Location of a CSV file containing the random draws used by :ref:`Knittel and Metaxoglou (2014) <km14>` to accompany
    the automobile product data. Also included is a single demographic column, which consists of income draws from
    lognormal distributions. The standard deviation and market-varying means for these distributions were extracted by
    :ref:`Andrews, Gentzkow, and Shapiro (2017) <ags17>` from the original GAUSS code for
    :ref:`Berry, Levinsohn, and Pakes (1999) <blp99>`.
NEVO_PRODUCTS_LOCATION : `str`
    Location of a CSV file containing the fake cereal product data from :ref:`Nevo (2000) <n00>`.
NEVO_AGENTS_LOCATION : `str`
    Location of a CSV file containing the random draws used by :ref:`Nevo (2000) <n00>` to accompany the fake cereal
    product data.

Examples
--------
Any number of functions can be used to load these data into memory. For example, the following code uses :mod:`numpy`:

.. ipython:: python

   import numpy as np
   blp_product_data = np.recfromcsv(pyblp.data.BLP_PRODUCTS_LOCATION)
   blp_agent_data = np.recfromcsv(pyblp.data.BLP_AGENTS_LOCATION)

Record arrays can be cumbersome to manipulate. A more flexible alternative, the :class:`pandas.DataFrame`, can be built
by the :func:`pandas.read_csv` function in the :mod:`pandas` package, which, unlike :mod:`numpy`, is not a pyblp
requirement:

.. ipython:: python

   import pandas as pd
   blp_product_data = pd.read_csv(pyblp.data.BLP_PRODUCTS_LOCATION)
   blp_agent_data = pd.read_csv(pyblp.data.BLP_AGENTS_LOCATION)

"""

from pathlib import Path


_DATA_PATH = Path(__file__).resolve().parent
BLP_PRODUCTS_LOCATION = str(_DATA_PATH / 'blp_products.csv')
BLP_AGENTS_LOCATION = str(_DATA_PATH / 'blp_agents.csv')
NEVO_PRODUCTS_LOCATION = str(_DATA_PATH / 'nevo_products.csv')
NEVO_AGENTS_LOCATION = str(_DATA_PATH / 'nevo_agents.csv')
