"""Locations of example data that are included in the package for convenience.

Attributes
----------
BLP_PRODUCTS_LOCATION : `str`
    Location of a CSV file containing the automobile product data extracted by
    :ref:`Andrews, Gentzkow, and Shapiro (2017) <ags17>` from the original GAUSS code for
    :ref:`Berry, Levinsohn, and Pakes (1999) <blp99>`, which is commonly assumed to be the same data used in
    :ref:`Berry, Levinsohn, and Pakes (1995) <blp95>`. The file includes some basic pre-computed instruments, which are
    the instruments used in the original paper, excluding a few that give rise to collinearity issues.
BLP_AGENTS_LOCATION : `str`
    Location of a CSV file containing the automobile agent data, which are also extracted from the original GAUSS code.
    Included in the file are importance sampling weights, nodes for integration, and income.

    Instead of being named ``nodes0``, ``nodes1``, ``nodes2``, and so on as expected by :class:`~pyblp.Problem`, each
    column of nodes is associated with a named product characteristic. Usually, it wouldn't matter which nodes were
    associated with which characteristics in :math:`X_2`. However, since nodes in the automobile problem were computed
    according to importance sampling, the ordering of nodes severely impacts estimation results. If you want to use
    these nodes, before passing them to :class:`~pyblp.Problem`, they will need to be renamed. Alternatively, the
    :doc:`Examples </examples>` section demonstrates how use :func:`~pyblp.build_matrix` to build a new ``nodes`` field.

    Before importance sampling, income consists of draws from lognormal distributions with standard deviation ``1.72``
    and the following means:

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

NEVO_PRODUCTS_LOCATION : `str`
    Location of a CSV file containing the fake cereal product data from :ref:`Nevo (2000) <n00>`. The file includes
    the same pre-computed instruments used in the original paper.
NEVO_AGENTS_LOCATION : `str`
    Location of a CSV file containing the fake cereal agent data. Included in the file Monte Carlo weights and draws,
    along with demographics, which collectively are used by :ref:`Nevo (2000) <n00>` to solve the fake cereal problem.

    Unlike for the automobile data, integration nodes are named ``nodes0``, ``nodes1``, ``nodes2``, and so on as
    expected by :class:`~pyblp.Problem`. This is because for a large enough set of Monte Carlo draws, the ordering of
    nodes shouldn't impact estimation results. However, since there are only a small number of agents in these example
    data, their ordering does somewhat impact results. For this reason, when formulating :math:`X_2`, the following
    ordering is the one that is generally used: ``1 + prices + sugar + mushy``.

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
BLP_PRODUCTS_LOCATION = str(_DATA_PATH / 'blp_products.csv')
BLP_AGENTS_LOCATION = str(_DATA_PATH / 'blp_agents.csv')
NEVO_PRODUCTS_LOCATION = str(_DATA_PATH / 'nevo_products.csv')
NEVO_AGENTS_LOCATION = str(_DATA_PATH / 'nevo_agents.csv')
