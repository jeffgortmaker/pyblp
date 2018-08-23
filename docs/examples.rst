.. currentmodule:: pyblp


Examples
========

This section explains how pyblp can be used to solve standard BLP example problems, compute post-estimation outputs, and simulate problems.

To start, we'll import the package, as well as :mod:`numpy`, since we'll be using it to manipulate data. The following lines of code, which are executed before each section of examle code in the documentation, import both packages and limit verbosity. Status updates can be informative, so you'll probably want to keep your own verbosity turned on.

.. literalinclude:: include.py
  :language: ipython

In this section, we'll also import :mod:`matplotlib.pyplot`, which, although not a pyblp dependency, can be used to plot results.

.. ipython:: python

   import matplotlib.pyplot as plt

.. ipython:: python
   :suppress:

   import matplotlib
   matplotlib.use('agg')

   from pathlib import Path
   def savefig(path):
       global source_path
       plt.savefig(Path(source_path) / path, transparent=True, bbox_inches='tight')
       plt.clf()


Configuring Data
----------------

Two sets of example data are included in the package:

- Fake cereal data from :ref:`Nevo (2000) <n00>`.
- Automobile data from :ref:`Berry, Levinsohn, and Pakes (1995) <blp95>`.

Locations of CSV files that contain the data are in the :mod:`pyblp.data` module.


Product Data
~~~~~~~~~~~~

The `product_data` argument of :class:`Problem` should be a structured array-like object with fields that store data. Product data can be a structured :class:`numpy.ndarray`, a :class:`pandas.DataFrame`, or other similar objects. We'll use :mod:`numpy` to read the data.

.. ipython:: python

   nevo_products = np.recfromcsv(pyblp.data.NEVO_PRODUCTS_LOCATION, encoding='utf-8')
   blp_products = np.recfromcsv(pyblp.data.BLP_PRODUCTS_LOCATION, encoding='utf-8')
   nevo_products.dtype.names
   blp_products.dtype.names

Both sets of product data contain market IDs, product IDs, two sets of firm IDs (the second are IDs after a simple merger, which are used later), shares, prices, a number of product characteristics, and pre-computed instruments. The fake cereal product IDs will be used to construct fixed effects and the automobile product IDs are called clustering IDs because they will be used to compute clustered standard errors.


Agent Data
~~~~~~~~~~

The `agent_data` argument of :class:`Problem` should also be a structured array-like object.

.. ipython:: python

   nevo_agents = np.recfromcsv(pyblp.data.NEVO_AGENTS_LOCATION, encoding='utf-8')
   blp_agents = np.recfromcsv(pyblp.data.BLP_AGENTS_LOCATION, encoding='utf-8')
   nevo_agents.dtype.names
   blp_agents.dtype.names

Both sets of agent data contain market IDs, integration weights, integration nodes, and demographics. The fake cereal data contains multiple demographics, whereas income is the only demographic included in the automobile data. The fake cereal data contains simple Monte Carlo draws. Nodes and weights in the automobile data are from importance sampling.

Instead of being named ``nodes0``, ``nodes1``, ``nodes3``, and so on as expected by :class:`Problem`, each column of nodes in the automobile data is associated with a named product characteristic. Usually, it wouldn't matter which nodes were associated with which characteristics. However, since nodes in the automobile problem were computed according to importance sampling, the ordering of nodes severely impacts estimation results. Since there are so few agents, the ordering of nodes also impacts estimation results for the fake cereal problem, but not by as much.

When constructing the automobile ``nodes`` field expected by :class:`Problem`, we'll choose an ordering of characteristics that we'll use again below when formulating :math:`X_2`. The :func:`build_matrix` function is convenient for building matrices according to a :class:`Formulation`. Since NumPy structured arrays are not easily mutable, we'll replace our automobile agents with a :class:`dict`.

.. ipython:: python

   blp_agents = {n: blp_agents[n] for n in blp_agents.dtype.names}
   blp_agents['nodes'] = pyblp.build_matrix(
       pyblp.Formulation('0 + constant_nodes + I(0) + hpwt_nodes + air_nodes + mpd_nodes + space_nodes'),
       blp_agents
   )

We've included an arbitrary column of zeros because in the automobile problem, we'll end up interacting prices only with income, not with unobserved characteristics. 


Solving Problems
----------------

Formulations, product data, and either agent data or an integration configuration are collectively used to initialize a :class:`Problem`. Once initialized, :meth:`Problem.solve` runs the estimation routine. The arguments to :meth:`Problem.solve` configure how estimation is performed. For example, `optimization` and `iteration` configure the optimization and iteration routines that are used by the outer and inner loops of estimation.


The Fake Cereal Problem
~~~~~~~~~~~~~~~~~~~~~~~

Up to four :class:`Formulation` configurations can be used to configure a :class:`Problem`. For the fake cereal problem, we'll specify formulations for :math:`X_1`, :math:`X_2`, and :math:`d`. The formulation for :math:`X_1` consists of prices and product fixed effects, which we will absorb into :math:`X_1` with the `absorb` argument of :class:`Formulation`. Columns in the formulation for :math:`X_2` are in the same order as the columns of nodes formulated above.

.. ipython:: python

   nevo_product_formulations = (
       pyblp.Formulation('0 + prices', absorb='C(product_ids)'),
       pyblp.Formulation('1 + prices + sugar + mushy')
   )
   nevo_product_formulations
   nevo_agent_formulation = pyblp.Formulation('0 + income + income_squared + age + child')
   nevo_agent_formulation
   nevo_problem = pyblp.Problem(
       nevo_product_formulations, 
       nevo_products,
       nevo_agent_formulation,
       nevo_agents
   )
   nevo_problem

.. note::

   If we were interested in estimates for each product, we could replace the formulation for :math:`X_1` with ``0 + prices + C(product_ids)`` and supplement :math:`Z_D` with product indicator variables (the documentation for the :func:`build_matrix` function demonstrates how to do this). Absorption of fixed effects yields the same first-stage results as including them as indicator variables, although results for GMM stages after the first may be slightly different because the two methods can give rise to different weighting matrices.

Inspecting the attributes of the :class:`Problem` instance helps to confirm that the problem has been configured correctly. For example, inspecting :attr:`Problem.products` and :attr:`Problem.agents` confirms that product data were structured correctly and that agent data were built correctly.

.. ipython:: python

   nevo_problem.products
   nevo_problem.agents

The initialized problem can be solved with :meth:`Problem.solve`. We'll use the same starting values as :ref:`Nevo (2000) <n00>`. By passing a diagonal matrix of ones as starting values for :math:`\Sigma`, we're choosing to optimize over only variance terms. Similarly, zeros in the starting values for :math:`\Pi` mean that those parameters will be fixed at zero.

To solve the problem, we'll use a non-default unbounded optimization routine that is similar to the default one in Matlab used by :ref:`Nevo (2000) <n00>`. For the sake of speed in this example, we'll only perform one GMM step.

.. ipython:: python

   nevo_sigma = np.diag([0.3302, 2.4526, 0.0163, 0.2441])
   nevo_pi = [
      [ 5.4819,  0,      0.2037,  0     ],
      [15.8935, -1.2000, 0,       2.6342],
      [-0.2506,  0,      0.0511,  0     ],
      [ 1.2650,  0,     -0.8091,  0     ]
   ]
   nevo_results = nevo_problem.solve(
       nevo_sigma,
       nevo_pi,
       steps=1,
       optimization=pyblp.Optimization('bfgs')
   )
   nevo_results

Results are similar to those in the original paper.


The Automobile Problem
~~~~~~~~~~~~~~~~~~~~~~

Unlike the fake cereal problem, we won't absorb any fixed effects. However, we'll estimate the supply side of the automobile problem, so we need to formulate :math:`X_3` in addition to the three other matrices. Again, columns in the formulation for :math:`X_2` are in the same order as the columns of nodes formulated above.

.. ipython:: python

   blp_product_formulations = (
       pyblp.Formulation('1 + hpwt + air + mpd + space'),
       pyblp.Formulation('1 + prices + hpwt + air + mpd + space'),
       pyblp.Formulation('1 + log(hpwt) + air + log(mpg) + log(space) + trend')
   )
   blp_product_formulations
   blp_agent_formulation = pyblp.Formulation('0 + I(1 / income)')
   blp_agent_formulation

The original specification for the automobile problem includes the term :math:`\log(y_i - p_j)`, in which :math:`y` is income and :math:`p` are prices. Instead of including this term, which gives rise to a host of numerical problems, we'll follow :ref:`Berry, Levinsohn, and Pakes (1999) <blp99>` and uses its first-order linear approximation, :math:`p_j / y_i`. The above formulation for :math:`d` includes a column of :math:`1 / y_i` values, which we'll interact with :math:`p_j`.

Similar to :ref:`Andrews, Gentzkow, and Shapiro (2017) <ags17>`, who replicated the original :ref:`Berry, Levinsohn, and Pakes (1995) <blp95>`, we'll use published estimates as our starting values for :math:`\Sigma`. By passing a column vector of all zeros except for negative the published estimate for the coefficient on :math:`\log(y_i - p_j)` as the starting values for :math:`\Pi`, we're choosing to interact the inverse of income only with prices.

.. ipython:: python

   blp_sigma = np.diag([3.612, 0, 4.628, 1.818, 1.050, 2.056])
   blp_pi = [
       [  0    ],
       [-43.501],
       [  0    ],
       [  0    ],
       [  0    ],
       [  0    ]
   ]
   blp_problem = pyblp.Problem(
       blp_product_formulations, 
       blp_products, 
       blp_agent_formulation, 
       blp_agents
   )

A linear marginal cost specification is the default, so we'll need to use the `costs_type` argument to employ the log-linear specification used by :ref:`Berry, Levinsohn, and Pakes (1995) <blp95>`. A downside of this specification is that nonpositive estimated marginal costs can create problems for the optimization routine when computing :math:`\tilde{c}(\hat{\theta}) = \log c(\hat{\theta})`. Since this specification of the automobile problem suffers from such problems, we'll use the `costs_bounds` argument to bound marginal costs from below by a small number. 

Finally, as in the original paper, we'll use the `se_type` argument to cluster by product IDs, which were specified as ``clustering_ids`` in product data. Again, to speed up this example, we'll stop after one GMM step.

.. ipython:: python

   blp_results = blp_problem.solve(
       blp_sigma,
       blp_pi,
       steps=1,
       costs_type='log',
       costs_bounds=(0.001, None),
       se_type='clustered'
   )
   blp_results

Again, results are similar to those in the original paper.

One thing to note is that unlike our estimation procedure for the fake cereal problem, for which we used the non-default BFGS routine, here we used the default optimization routine, which supports parameter bounds. These bounds are displayed along with other optimization information when verbosity is not turned off. We can also look at them by inspecting their :class:`Results` attributes.

.. ipython::

   blp_results.sigma_bounds
   blp_results.pi_bounds

The default bounds were chosen to reduce the risk of numerical overflow. Without such bounds, optimization routines tend to try out large parameter values, which creates errors. The package tries to intelligently handle such errors, but they can create problems for the optimization routine. Using an optimization routine that supports bounds and choosing reasonable bounds yourself can be very important.


Problem Results
---------------

The :meth:`Problem.solve` method returns an instance of the :class:`Results` class, which, when printed, displays basic estimation results. The results that are displayed are simply formatted information extracted from various class attributes such as :attr:`Results.sigma` and :attr:`Results.sigma_se`.

Additional post-estimation outputs can be computed with :class:`Results` methods.


Elasticities and Diversion Ratios
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can estimate elasticities, :math:`\varepsilon`, and diversion ratios, :math:`\mathscr{D}`, with :meth:`Results.compute_elasticities` and :meth:`Results.compute_diversion_ratios`.

.. ipython:: python

   nevo_elasticities = nevo_results.compute_elasticities()
   blp_elasticities = blp_results.compute_elasticities()
   nevo_ratios = nevo_results.compute_diversion_ratios()
   blp_ratios = blp_results.compute_diversion_ratios()

Post-estimation outputs are computed for each market and stacked. We'll use :func:`matplotlib.pyplot.matshow` and :func:`matplotlib.pyplot.colorbar` to display the matrices associated with a single market.

.. ipython:: python

   single_nevo_market = nevo_products['market_ids'] == 'C01Q1'
   single_blp_market = blp_products['market_ids'] == 1971
   plt.colorbar(plt.matshow(nevo_elasticities[single_nevo_market]));

   @suppress
   savefig('images/nevo_elasticities.png')

.. image:: images/nevo_elasticities.png

.. ipython:: python

   plt.colorbar(plt.matshow(blp_elasticities[single_blp_market]));

   @suppress
   savefig('images/blp_elasticities.png')

.. image:: images/blp_elasticities.png

.. ipython:: python

   plt.colorbar(plt.matshow(nevo_ratios[single_nevo_market]));

   @suppress
   savefig('images/nevo_ratios.png')

.. image:: images/nevo_ratios.png

.. ipython:: python

   plt.colorbar(plt.matshow(blp_ratios[single_blp_market]));

   @suppress
   savefig('images/blp_ratios.png')

.. image:: images/blp_ratios.png

Diagonals in the first two images consist of own elasticities, and diagonals in the last two are diversion ratios to the outside good. The second and fourth images have empty columns because the selected market in the automobile problem has fewer products than other markets, and the extra columns are filled with ``numpy.nan``.

Elasticities and diversion ratios can be computed with respect to variables other than ``prices`` with the `name` argument of :meth:`Results.compute_elasticities` and :meth:`Results.compute_diversion_ratios`. Additionally, the :meth:`Results.compute_long_run_diversion_ratios` can be used to used to understand substitution when products are eliminated from the choice set.

The convenience methods :meth:`Results.extract_diagonals` and :meth:`Results.extract_diagonal_means` can be used to extract information about own elasticities of demand from elasticity matrices.

.. ipython:: python

   nevo_means = nevo_results.extract_diagonal_means(nevo_elasticities)
   blp_means = blp_results.extract_diagonal_means(blp_elasticities)

An alternative to summarizing full elasticity matrices is to use :meth:`Results.compute_aggregate_elasticities` to estimate aggregate elasticities of demand, :math:`E`, in each market, which reflect the change in total sales under a proportional sales tax of some factor.

.. ipython:: python

   nevo_aggregates = nevo_results.compute_aggregate_elasticities(factor=0.1)
   blp_aggregates = blp_results.compute_aggregate_elasticities(factor=0.1)

Since demand for an entire product category is generally less elastic than the average elasticity of individual products, mean own elasticities are generally larger in magnitude than aggregate elasticities.


.. ipython:: python

   plt.hist([nevo_means, blp_means], color=['maroon', 'navy'], bins=50);

   @suppress
   savefig('images/mean_own_elasticities.png')

.. image:: images/mean_own_elasticities.png

.. ipython:: python

   plt.hist([nevo_aggregates, blp_aggregates], color=['maroon', 'navy'], bins=50);

   @suppress
   savefig('images/aggregate_elasticities.png')

.. image:: images/aggregate_elasticities.png


Marginal Costs and Markups
~~~~~~~~~~~~~~~~~~~~~~~~~~

To compute marginal costs, :math:`c`, the `product_data` passed to :class:`Problem` must have had a `firm_ids` field. Since we included firm IDs in both problems, we can use :meth:`Results.compute_costs`.

.. ipython:: python

   nevo_costs = nevo_results.compute_costs()
   blp_costs = blp_results.compute_costs()

Other methods that compute supply-side outputs often compute marginal costs themselves. For example, :meth:`Results.compute_markups` will compute marginal costs when estimating markups, :math:`\mathscr{M}`, but computation can be sped up if we just use our pre-computed values.

.. ipython:: python

   nevo_markups = nevo_results.compute_markups(costs=nevo_costs)
   blp_markups = blp_results.compute_markups(costs=blp_costs)
   plt.hist([nevo_markups, blp_markups], color=['maroon', 'navy'], bins=50);

   @suppress
   savefig('images/markups.png')

.. image:: images/markups.png


Mergers
~~~~~~~

Before computing post-merger outputs, we'll supplement our pre-merger markups with some other outputs. We'll compute Herfindahl-Hirschman Indices, :math:`\text{HHI}`, with :meth:`Results.compute_hhi`; population-normalized gross expected profits, :math:`\pi`, with :meth:`Results.compute_profits`; and population-normalized consumer surpluses, :math:`\text{CS}`, with :meth:`Results.compute_consumer_surpluses`.

.. ipython:: python

   nevo_hhi = nevo_results.compute_hhi()
   blp_hhi = blp_results.compute_hhi()
   nevo_profits = nevo_results.compute_profits(costs=nevo_costs)
   blp_profits = blp_results.compute_profits(costs=blp_costs)
   nevo_cs = nevo_results.compute_consumer_surpluses()
   blp_cs = blp_results.compute_consumer_surpluses()

To compute post-merger outputs, the `firm_ids` field in the `product_data` passed to :class:`Problem` must have had at least two columns. Columns after the first represent changes, such as mergers. Although mergers are commonly what firm ID changes represent, these additional columns can represent any type of change.

Since we included two columns of firm IDs in both problems, we can use :meth:`Results.compute_approximate_prices` or :meth:`Results.compute_prices` to estimate post-merger prices. The first method, which is discussed, for example, in :ref:`Nevo (1997) <n97>`, assumes that shares and their price derivatives are unaffected by the merger. The second method does not make these assumptions and iterates over the :math:`\zeta`-markup equation from :ref:`Morrow and Skerlos (2011) <ms11>` to solve the full system of :math:`J_t` equations and :math:`J_t` unknowns in each market :math:`t`. We'll use the latter, since it is fast enough for the two example problems.

.. ipython:: python

   nevo_changed_prices = nevo_results.compute_prices(costs=nevo_costs)
   blp_changed_prices = blp_results.compute_prices(costs=blp_costs)

If the problems were configured with more than two columns of firm IDs, we could estimate post-merger prices for the other mergers with the `firms_index` argument, which is by default ``1``.

We'll compute post-merger shares with :meth:`Results.compute_shares`.

.. ipython:: python

   nevo_changed_shares = nevo_results.compute_shares(nevo_changed_prices)
   blp_changed_shares = blp_results.compute_shares(blp_changed_prices)

Post-merger prices and shares are used to compute other post-merger outputs. For example, :math:`\text{HHI}` increases.

.. ipython:: python

   nevo_changed_hhi = nevo_results.compute_hhi(firms_index=1, shares=nevo_changed_shares)
   blp_changed_hhi = blp_results.compute_hhi(firms_index=1, shares=blp_changed_shares)
   plt.hist(
       [nevo_changed_hhi - nevo_hhi, blp_changed_hhi - blp_hhi], 
       color=['maroon', 'navy'], 
       bins=50
   );

   @suppress
   savefig('images/hhi_changes.png')

.. image:: images/hhi_changes.png

Markups, :math:`\mathscr{M}`, and profits, :math:`\pi`, generally increase as well.

.. ipython:: python

   nevo_changed_markups = nevo_results.compute_markups(nevo_changed_prices, nevo_costs)
   blp_changed_markups = blp_results.compute_markups(blp_changed_prices, blp_costs)
   plt.hist([nevo_changed_markups - nevo_markups, blp_changed_markups - blp_markups], color=['maroon', 'navy'], bins=50);

   @suppress
   savefig('images/markup_changes.png')

.. image:: images/markup_changes.png

.. ipython:: python

   nevo_changed_profits = nevo_results.compute_profits(
       nevo_changed_prices,
       nevo_changed_shares,
       nevo_costs
   )
   blp_changed_profits = blp_results.compute_profits(
       blp_changed_prices,
       blp_changed_shares,
       blp_costs
   )
   bins = np.linspace(-0.002, 0.002, 50)
   plt.hist(
       [nevo_changed_profits - nevo_profits, blp_changed_profits - blp_profits], 
       color=['maroon', 'navy'], 
       bins=50
   );

   @suppress
   savefig('images/profit_changes.png')

.. image:: images/profit_changes.png

On the other hand, consumer surpluses, :math:`\text{CS}`, generally decrease.

.. ipython:: python

   nevo_changed_cs = nevo_results.compute_consumer_surpluses(nevo_changed_prices)
   blp_changed_cs = blp_results.compute_consumer_surpluses(blp_changed_prices)
   plt.hist(
       [nevo_changed_cs - nevo_cs, blp_changed_cs - blp_cs], 
       color=['maroon', 'navy'], 
       bins=50
   );

   @suppress
   savefig('images/cs_changes.png')

.. image:: images/cs_changes.png


Logit Benchmark
---------------

Comparing results from the full BLP model with results from the simpler Logit model is straightforward. A Logit :class:`Problem` can be created by simply excluding the formulation for :math:`X_2` along with any agent information. We'll set up and solve a simpler version of the fake cereal problem. Since we won't include any nonlinear characteristics or parameters, we don't have to worry about configuring an optimization routine.

.. ipython:: python
   
   nevo_logit_formulation = pyblp.Formulation('0 + prices', absorb='C(product_ids)')
   nevo_logit_formulation
   nevo_logit_problem = pyblp.Problem(nevo_logit_formulation, nevo_products)
   nevo_logit_problem
   nevo_logit_results = nevo_logit_problem.solve()
   nevo_logit_results

Logit :class:`Results` can be to compute the same types of post-estimation outputs as :class:`Results` created by a full BLP problem.


Simulating Problems
-------------------

Before configuring and solving a problem with real data, papers such as :ref:`Armstrong (2016) <a16>` recommend performing Monte Carlo analysis on simulated data to verify that it is possible to accurately estimate model parameters. For example, before configuring and solving the above automobile problem, it may have been a good idea to simulate data according to the assumed models of supply and demand. During such Monte Carlo anaysis, the data would only be used to determine sample sizes and perhaps to choose true parameters that are within reason.

Simulations are configured with the :class:`Simulation` class, which requires much of the same inputs as the :class:`Problem` class. The two main differences are:

    1. Variables in formulations that cannot be loaded from `product_data` or `agent_data` will be drawn from independent uniform distributions.
    2. True parameters along with the distribution of unobserved product characteristics are both specified.

First, we'll use :func:`build_id_data` to build market and firm IDs for a model in which there are :math:`T = 50` markets, and in each market :math:`t`, :math:`J_t = 20` products produced by :math:`F_t = 10` firms.

.. ipython:: python

   simulation_ids = pyblp.build_id_data(T=50, J=20, F=10)

Next, we'll choose configure :class:`Integration` to build agent data according to a level-``5`` Gauss-Hermite product rule.

.. ipython:: python

   simulation_integration = pyblp.Integration('product', 5)
   simulation_integration

We'll then pass these along formulations and parameters to :class:`Simulation`.

.. ipython:: python

   simulation = pyblp.Simulation(
       product_formulations=(
           pyblp.Formulation('1 + prices + x'),
           pyblp.Formulation('0 + y'),
           pyblp.Formulation('1 + x + z')
       ),
       beta=[1, -5, 1],
       sigma=1,
       gamma=[1, 2, 2],
       product_data=simulation_ids,
       agent_formulation=pyblp.Formulation('0 + d'),
       pi=2,
       integration=simulation_integration,
       seed=0
   )
   simulation

When :class:`Simulation` is initialized, it constructs agent data and simulates all product data except for prices and shares.

.. ipython:: python

   simulation.product_data
   simulation.agent_data

The instruments in :attr:`Simulation.product_data` are basic ones computed with :func:`build_blp_instruments` that are functions of all exogenous numerical variables in the problem. For this example, :math:`Z_D` is a constant column followed by all non-price characteristics and two other columns of traditional BLP instruments that are computed for the one demand-side characteristic that doesn't enter :math:`X_3`. Similarly, :math:`Z_S` is a constant column followed by all non-price characteristics and two other columns of BLP instruments that are computed for the one supply-side characteristic that doesn't enter :math:`X_1` or :math:`X_2`.

The :class:`Simulation` can be further configured with other arguments that determine how unobserved product characteristics are simulated and how marginal costs are specified.

Since at this stage, prices and shares are all zeros, we still need to solve the simulation with :meth:`Simulation.solve`. This method computes Bertrand-Nash prices and shares. Just like :meth:`Results.compute_prices`, it iterates over the :math:`\zeta`-markup equation from :ref:`Morrow and Skerlos (2011) <ms11>` to do so.

.. ipython:: python

   simulated_products = simulation.solve()
   simulated_products

Now, we can try to recover the true parameters by creating and solving a :class:`Problem`. To make estimation easy, we'll use the same formulations and the same unobserved agent data. However, we'll choose starting values that are half the true parameters so that the optimization routine has to do some work.

.. ipython:: python

   simulated_problem = pyblp.Problem(
       simulation.product_formulations,
       simulated_products,
       simulation.agent_formulation,
       simulation.agent_data
   )
   simulated_results = simulated_problem.solve(
       0.5 * simulation.sigma, 
       0.5 * simulation.pi, 
       steps=1
   )
   simulated_results
   simulation.beta
   simulation.gamma
   simulation.sigma
   simulation.pi

The parameters seem to have been estimated reasonably well.

In addition to checking that the configuration for a model based on actual data makes sense, the :class:`Simulation` class can also be a helpful tool for better understanding under what general conditions BLP models can be accurately estimated. Simulations are also used extensively in pyblp's test suite.


Multiprocessing
---------------

For large problems or simulations, multiprocessing may be useful if your computing environment has access to many cores. Multiprocessing can be enabled with the :func:`parallel` context manager, which is used in a ``with`` statement. Any methods that perform market-by-market computation will distribute their work among the processes in the context created by :func:`parallel`.

Although the problems in this example are small enough that there are no gains from parallel processing, the following code demonstrates how to compute elasticities with a pool of two processes for the problem that we simulated and solved above.

.. ipython:: python
  
   with pyblp.parallel(2):
       elasticities = simulated_results.compute_elasticities()
   elasticities.shape

Similarly, if we executed :meth:`Problem.solve` or :meth:`Simulation.solve` within a :func:`parallel` context, all of their market-by-market computation would be distributed among the processes in the pool.
