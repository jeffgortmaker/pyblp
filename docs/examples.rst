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

- Automobile data from :ref:`Berry, Levinsohn, and Pakes (1995) <blp95>`.
- Fake cereal data from :ref:`Nevo (2000) <n00>`.

Locations of CSV files that contain the data are in the :mod:`pyblp.data` module.


Product Data
~~~~~~~~~~~~

Many different packages can be used to to load the data. We'll use :mod:`numpy`.

.. ipython:: python

   blp_products = np.recfromcsv(pyblp.data.BLP_PRODUCTS_LOCATION)
   nevo_products = np.recfromcsv(pyblp.data.NEVO_PRODUCTS_LOCATION)

The `product_data` argument in :class:`Problem` initialization is a structured array-like object with fields that store data. Product data can be a structured :class:`numpy.ndarray`, a :class:`pandas.DataFrame`, or other similar objects. We'll use simple :class:`dict` instances, which are more more mutable than the :class:`numpy.ndarray` instances loaded above.

.. ipython:: python

   
   blp_products.dtype.names
   nevo_products.dtype.names
   blp_products = {n: blp_products[n] for n in blp_products.dtype.names}
   nevo_products = {n: nevo_products[n] for n in nevo_products.dtype.names}

Both sets of data contain market IDs, two sets of firm IDs (the second are IDs after a simple merger, which are used later), shares, prices, and a number of product characteristics. The fake cereal data also includes product IDs (``demand_ids``), which will be used to construct fixed effects, as well as pre-computed instruments (``demand_instruments0``, ``demand_instruments1``, and so on).

We'll use the :func:`build_blp_instruments` function to construct both demand- and supply-side instruments for the automobile problem. The function accepts a :class:`Formulation` configuration, which determines which product characteristics will be used to construct traditional BLP instruments. Additionally, we'll add the excluded demand variable, miles per dollar, to the set of supply-side instruments. As in :ref:`Berry, Levinsohn, and Pakes (1995) <blp95>`, even though miles per gallon and the trend will end up being excluded supply variables, we won't include them in the set of demand-side instruments because of collinearity issues.

.. ipython:: python

   blp_demand_formulation = pyblp.Formulation('hpwt + air + mpd + space')
   blp_demand_formulation
   blp_supply_formulation = pyblp.Formulation('log(hpwt) + air + log(mpd) + log(space) + trend')
   blp_supply_formulation
   blp_products['demand_instruments'] = pyblp.build_blp_instruments(
       blp_demand_formulation, 
       blp_products
   )
   blp_products['supply_instruments'] = np.c_[
       blp_products['mpd'],
       pyblp.build_blp_instruments(
           blp_supply_formulation, 
           blp_products
       )
   ]


Agent Data
~~~~~~~~~~

The original specification for the automobile problem includes the term :math:`\log(y_i - p_j)`, in which :math:`y` is income and :math:`p` are prices. Instead of including this term, which gives rise to a host of numerical problems, we'll follow :ref:`Berry, Levinsohn, and Pakes (1999) <blp99>` and use its first-order linear approximation, :math:`p_j / y_i`. To do so, we'll need agent income, :math:`y`. The CSV file includes sample draws from the lognormal distributions that were used in the original paper. Since the draws included for the automobile problem are the ones used by :ref:`Knittel and Metaxoglou (2014) <km14>` and not the draws from the original paper, we might as well configure the package to build our own set of Monte Carlo draws for each market. A small number of draws speeds up estimation for this example.

.. ipython:: python

   blp_agents = np.recfromcsv(pyblp.data.BLP_AGENTS_LOCATION)
   blp_agents.dtype.names
   blp_integration = pyblp.Integration('monte_carlo', 50, seed=1)
   blp_integration

We'll use the agent data for the fake ceral problem as given.

.. ipython:: python

   nevo_agents = np.recfromcsv(pyblp.data.NEVO_AGENTS_LOCATION)
   nevo_agents.dtype.names


Solving Problems
----------------

Formulations configurations, product data, and either agent data or an integration configuration are collectively used to initialize a :class:`Problem`. Once a problem is initialized, :meth:`Problem.solve` performs estimation. The arguments to :meth:`Problem.solve` configure how estimation is performed. For example, `optimization` and `iteration` configure the optimization and iteration routines that are used by the outer and inner loops of estimation.


The Automobile Problem
~~~~~~~~~~~~~~~~~~~~~~

Up to four :class:`Formulation` configurations can be used to configure a :class:`Problem`. We'll uses all four to configure the automobile problem. The required ones configure :math:`X_1` and :math:`X_2`, and the optional ones that we'll use here configure :math:`X_3` (for estimating supply) and :math:`d` (for incorporating one or more demographics). We can re-use the demand- and supply-side formulations defined above.

.. ipython:: python

   blp_product_formulations = (
       blp_demand_formulation,
       pyblp.Formulation('prices + hpwt + air + mpd + space'),
       blp_supply_formulation
   )
   blp_product_formulations
   blp_agent_formulation = pyblp.Formulation('0 + I(1 / income)')
   blp_agent_formulation


The :class:`Integration` configuration will be used by :class:`Problem` to build unobserved agent data.

.. ipython:: python

   blp_problem = pyblp.Problem(
       blp_product_formulations,
       blp_products,
       blp_agent_formulation,
       blp_agents,
       blp_integration
   )
   blp_problem

Inspecting the attributes of the :class:`Problem` instance helps to confirm that the problem has been configured correctly. For example, inspecting :attr:`Problem.products` and :attr:`Problem.agents` confirms that product data were structured correctly and that agent data were built correctly.

.. ipython:: python

   blp_problem.products
   blp_problem.agents

The initialized problem can be solved with :meth:`Problem.solve`. By passing a diagonal matrix of ones as starting values for :math:`\Sigma`, we're choosing to optimize over only variance terms; we're setting the second element to zero because it corresponds to prices and we're interested only in the interaction term, :math:`p_j / y_i`. By passing a column vector of all zeros except for one negative value as starting values for :math:`\Pi`, we're choosing to interact the inverse of income only with prices.

.. ipython:: python

   blp_sigma = np.diag([1, 0, 1, 1, 1, 1])
   blp_pi = np.c_[[0, -10, 0, 0, 0, 0]]

A linear marginal cost specification is the default, so we'll need to use the `linear_costs` argument to employ the log-linear specification used by :ref:`Berry, Levinsohn, and Pakes (1995) <blp95>`. A downside of this specification is that nonpositive estimated marginal costs can create problems for the optimization routine when computing :math:`\tilde{c}(\hat{\theta}) = \log c(\hat{\theta})`. Since this specification of the automobile problem suffers from negative estimates of marginal costs, we'll use the `costs_bounds` argument to bound marginal costs from below by a small number. Unfortunately, doing so introduces error into analytic gradient computation, so we'll configure the optimization routine to compute gradients with finite differences. Supply-side gradient computation is computationally expensive anyways, so this is not that big of a problem. To further speed up this example, we'll use a high objective tolerance and stop after one GMM step.

.. ipython:: python

   blp_results = blp_problem.solve(
       blp_sigma,
       blp_pi,
       steps=1,
       linear_costs=False,
       costs_bounds=(0.1, None),
       optimization=pyblp.Optimization('l-bfgs-b', {'ftol': 1e-3}, compute_gradient=False)
   )
   blp_results

Estimates, which are in the same order as product characteristics configured during :class:`Problem` initialization, are somewhat similar to those in :ref:`Berry, Levinsohn, and Pakes (1995) <blp95>`. Of course, shortcuts used to speed up optimization and divergences from the original configuration give rise to differences.


The Fake Cereal Problem
~~~~~~~~~~~~~~~~~~~~~~~

Unlike the automobile problem, we will not estimate a supply side when solving the fake cereal problem. However, we still need to specify formulations for :math:`X_1`, :math:`X_2`, and :math:`d`. The formulation for :math:`X_1` consists only of prices. However, since there is a ``demand_ids`` field in the fake cereal product data, fixed effects created from the categorical IDs in the field will be absorbed into :math:`X_1` (as well as into :math:`Z_D` and other relevant matrices) through a demeaning procedure. If there were more than one column of demand IDs to absorb, the iterative demeaning algorithm of :ref:`Rios-Avila (2015) <r15>` would be used. If we were interested in parameter estimates for each product, we could include ``C(demand_ids)`` in the formulation for :math:`X_1` and supplement :math:`Z_D` with product indicator variables (the documentation for the :func:`build_matrix` function demonstrates how to do this). Absorption of fixed effects yields the same first-stage results as including them as indicator variables (results for GMM stages after the first may be slightly different because the two methods can give rise to different updated weighting matrices).

.. ipython:: python

   nevo_product_formulations = (
       pyblp.Formulation('0 + prices'),
       pyblp.Formulation('prices + sugar + mushy')
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

Since we initialized the problem without supply-side data, there's no need to choose a marginal cost specification. When configuring :math:`\Sigma` and :math:`\Pi`, we'll use the same starting values as :ref:`Nevo (2000) <n00>`. We'll also use a non-default unbounded optimization routine that is similar to the default for Matlab, and, again, we'll only perform one GMM step for the sake of speed in this example.

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

Often, the above starting values give rise to some warnings during the first few GMM objective evaluations about floating point problems. This is because some optimization routines attempt to evaluate the objective at parameter values that lead to overflow while, for example, computing :math:`\hat{\delta}`. For example, using ``pyblp.Optimization('slsqp')`` displays some warnings when :attr:`options.verbose` is ``True``. The default behavior of :meth:`Problem.solve` is to revert problematic elements in :math:`\hat{\delta}` and its Jacobian before computing the objective value, which allows the optimization routine to continue searching the parameter space. For more information, refer to :meth:`Problem.solve`. In particular, the `sigma_bounds` and `pi_bounds` arguments can be used to bound the parameter space over which the optimization problem searches.

Again, results are similar to those in the original paper. Compared to the automobile problem, results are even closer to the original because our specification and agent data were essentially the same.


Problem Results
---------------

The :meth:`Problem.solve` method returns an instance of the :class:`Results` class, which, when printed, displays basic estimation results. The results that are displayed are simply formatted information extracted from various class attributes such as :attr:`Results.sigma` and :attr:`Results.sigma_se`. Standard errors are either robust or unadjusted, depending on the `se_type` argument of :meth:`Problem.solve`.

Additional post-estimation outputs can be computed with :class:`Results` methods.


Elasticities and Diversion Ratios
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can estimate elasticities, :math:`\varepsilon`, and diversion ratios, :math:`\mathscr{D}`, with :meth:`Results.compute_elasticities` and :meth:`Results.compute_diversion_ratios`.

.. ipython:: python

   blp_elasticities = blp_results.compute_elasticities()
   nevo_elasticities = nevo_results.compute_elasticities()
   blp_ratios = blp_results.compute_diversion_ratios()
   nevo_ratios = nevo_results.compute_diversion_ratios()

Post-estimation outputs are computed for each market and stacked. We'll use :func:`matplotlib.pyplot.matshow` and :func:`matplotlib.pyplot.colorbar` to display the matrices associated with market ID ``1``.

.. ipython:: python

   first_blp_market = blp_products['market_ids'] == 1
   first_nevo_market = nevo_products['market_ids'] == 1
   plt.colorbar(plt.matshow(blp_elasticities[first_blp_market]));

   @suppress
   savefig('images/blp_elasticities.png')

.. image:: images/blp_elasticities.png

.. ipython:: python

   plt.colorbar(plt.matshow(nevo_elasticities[first_nevo_market]));

   @suppress
   savefig('images/nevo_elasticities.png')

.. image:: images/nevo_elasticities.png

.. ipython:: python

   plt.colorbar(plt.matshow(blp_ratios[first_blp_market]));

   @suppress
   savefig('images/blp_ratios.png')

.. image:: images/blp_ratios.png

.. ipython:: python

   plt.colorbar(plt.matshow(nevo_ratios[first_nevo_market]));

   @suppress
   savefig('images/nevo_ratios.png')

.. image:: images/nevo_ratios.png

Diagonals in the first two images consist of own elasticities, and diagonals in the last two are diversion ratios to the outside good. The first and third images have empty columns because market ``1`` in the automobile problem has fewer products than other markets, and the extra columns are filled with ``numpy.nan``.

Elasticities and diversion ratios can be computed with respect to variables other than ``pricess`` with the `name` argument of :meth:`Results.compute_elasticities` and :meth:`Results.compute_diversion_ratios`. Additionally, the :meth:`Results.compute_long_run_diversion_ratios` can be used to used to understand substitution when products are eliminated from the choice set.

Other methods that compute similar matrices are :meth:`Results.compute_elasticities` and :meth:`Results.compute_diversion_ratios`, which estimate :math:`\varepsilon` and :math:`\mathscr{D}` with respect to non-price characteristics. A similar method is :meth:`Results.compute_long_run_diversion_ratios`, which can be used to understand substitution when products are eliminated from the choice set. The convenience methods :meth:`Results.extract_diagonals` and :meth:`Results.extract_diagonal_means` can be used to extract information about own elasticities of demand from elasticity matrices.

.. ipython:: python

   blp_means = blp_results.extract_diagonal_means(blp_elasticities)
   nevo_means = nevo_results.extract_diagonal_means(nevo_elasticities)

An alternative to summarizing full elasticity matrices is to use :meth:`Results.compute_aggregate_elasticities` to estimate aggregate elasticities of demand, :math:`E`, in each market, which reflect the change in total sales under a proportional sales tax of some factor.

.. ipython:: python

   blp_aggregates = blp_results.compute_aggregate_elasticities(factor=0.1)
   nevo_aggregates = nevo_results.compute_aggregate_elasticities(factor=0.1)

Since demand for an entire product category is generally less elastic than the average elasticity of individual products, mean own elasticities are generally larger in magnitude than aggregate elasticities.


.. ipython:: python

   bins = np.linspace(-5, 0, 50)
   plt.hist(blp_means, bins, alpha=0.5, color='maroon');
   plt.hist(nevo_means, bins, alpha=0.5, color='navy');

   @suppress
   savefig('images/mean_own_elasticities.png')

.. image:: images/mean_own_elasticities.png

.. ipython:: python

   plt.hist(blp_aggregates, bins, alpha=0.5, color='maroon');
   plt.hist(nevo_aggregates, bins, alpha=0.5, color='navy');

   @suppress
   savefig('images/aggregate_elasticities.png')

.. image:: images/aggregate_elasticities.png


Marginal Costs and Markups
~~~~~~~~~~~~~~~~~~~~~~~~~~

To compute marginal costs, :math:`c`, the `product_data` passed to :class:`Problem` must have had a `firm_ids` field. Since we included firm IDs in both problems, we can use :meth:`Results.compute_costs`.

.. ipython:: python

   blp_costs = blp_results.compute_costs()
   nevo_costs = nevo_results.compute_costs()

Other methods that compute supply-side outputs often require marginal costs. For example, :meth:`Results.compute_markups` will compute marginal costs when estimating markups, :math:`\mathscr{M}`, but computation can be sped up if we just pass our pre-computed values.

.. ipython:: python

   blp_markups = blp_results.compute_markups(costs=blp_costs)
   nevo_markups = nevo_results.compute_markups(costs=nevo_costs)
   bins = np.linspace(0, 1.5, 50)
   plt.hist(blp_markups, bins, alpha=0.5, color='maroon');
   plt.hist(nevo_markups, bins, alpha=0.5, color='navy');

   @suppress
   savefig('images/markups.png')

.. image:: images/markups.png


Mergers
~~~~~~~

Before computing post-merger outputs, we'll supplement our pre-merger markups with some other outputs. We'll compute Herfindahl-Hirschman Indices, :math:`\text{HHI}`, with :meth:`Results.compute_hhi`; population-normalized gross expected profits, :math:`\pi`, with :meth:`Results.compute_profits`; and population-normalized consumer surpluses, :math:`\text{CS}`, with :meth:`Results.compute_consumer_surpluses`.

.. ipython:: python

   blp_hhi = blp_results.compute_hhi()
   nevo_hhi = nevo_results.compute_hhi()
   blp_profits = blp_results.compute_profits(costs=blp_costs)
   nevo_profits = nevo_results.compute_profits(costs=nevo_costs)
   blp_cs = blp_results.compute_consumer_surpluses()
   nevo_cs = nevo_results.compute_consumer_surpluses()

To compute post-merger outputs, the `firm_ids` field in the `product_data` passed to :class:`Problem` must have had at least two columns. Columns after the first represent changes, such as mergers. Although mergers are commonly what firm ID changes represent, these additional columns can represent any type of change.

Since we included two columns of firm IDs in both problems, we can use :meth:`Results.solve_approximate_merger` or :meth:`Results.solve_merger` to estimate post-merger prices. The first method, which is discussed, for example, in :ref:`Nevo (1997) <n97>`, assumes that shares and their price derivatives are unaffected by the merger. The second method does not make these two assumptions and iterates over the :math:`\zeta`-markup equation from :ref:`Morrow and Skerlos (2011) <ms11>` to solve the full system of :math:`J_t` equations and :math:`J_t` unknowns in each market :math:`t`. We'll use the latter, since it is fast enough for the two example problems.

.. ipython:: python

   blp_changed_prices = blp_results.solve_approximate_merger(costs=blp_costs)
   nevo_changed_prices = nevo_results.solve_merger(costs=nevo_costs)

If the problems were configured with more than two columns of firm IDs, we could estimate post-merger prices for the other mergers with the `firms_index` argument, which is by default ``1``.

We'll compute post-merger shares with :meth:`Results.compute_shares`.

.. ipython:: python

   blp_changed_shares = blp_results.compute_shares(blp_changed_prices)
   nevo_changed_shares = nevo_results.compute_shares(nevo_changed_prices)

Post-merger prices and shares are used to compute other post-merger outputs. For example, :math:`\text{HHI}` increases.

.. ipython:: python

   blp_changed_hhi = blp_results.compute_hhi(firms_index=1, shares=blp_changed_shares)
   nevo_changed_hhi = nevo_results.compute_hhi(firms_index=1, shares=nevo_changed_shares)
   bins = np.linspace(0, 3000, 50)
   plt.hist(blp_changed_hhi - blp_hhi, bins, alpha=0.5, color='maroon');
   plt.hist(nevo_changed_hhi - nevo_hhi, bins, alpha=0.5, color='navy');

   @suppress
   savefig('images/hhi_changes.png')

.. image:: images/hhi_changes.png

Markups, :math:`\mathscr{M}`, and profits, :math:`\pi`, generally increase as well.

.. ipython:: python

   blp_changed_markups = blp_results.compute_markups(blp_changed_prices, blp_costs)
   nevo_changed_markups = nevo_results.compute_markups(nevo_changed_prices, nevo_costs)
   bins = np.linspace(-0.05, 0.25, 50)
   plt.hist(blp_changed_markups - blp_markups, bins, alpha=0.5, color='maroon');
   plt.hist(nevo_changed_markups - nevo_markups, bins, alpha=0.5, color='navy');

   @suppress
   savefig('images/markup_changes.png')

.. image:: images/markup_changes.png

.. ipython:: python

   blp_changed_profits = blp_results.compute_profits(
       blp_changed_prices,
       blp_changed_shares,
       blp_costs
   )
   nevo_changed_profits = nevo_results.compute_profits(
       nevo_changed_prices,
       nevo_changed_shares,
       nevo_costs
   )
   bins = np.linspace(-0.002, 0.002, 50)
   plt.hist(blp_changed_profits - blp_profits, bins, alpha=0.5, color='maroon');
   plt.hist(nevo_changed_profits - nevo_profits, bins, alpha=0.5, color='navy');

   @suppress
   savefig('images/profit_changes.png')

.. image:: images/profit_changes.png

On the other hand, consumer surpluses, :math:`\text{CS}`, generally decrease.

.. ipython:: python

   blp_changed_cs = blp_results.compute_consumer_surpluses(blp_changed_prices)
   nevo_changed_cs = nevo_results.compute_consumer_surpluses(nevo_changed_prices)
   bins = np.linspace(-0.12, 0.01, 50)
   plt.hist(blp_changed_cs - blp_cs, bins, alpha=0.5, color='maroon');
   plt.hist(nevo_changed_cs - nevo_cs, bins, alpha=0.5, color='navy');

   @suppress
   savefig('images/cs_changes.png')

.. image:: images/cs_changes.png


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
           pyblp.Formulation('prices + x'),
           pyblp.Formulation('0 + y'),
           pyblp.Formulation('x + z')
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

Since at this stage, prices and shares are all zeros, we still need to solve the simulation with :meth:`Simulation.solve`. This method computes Bertrand-Nash prices and shares. Just like :meth:`Results.solve_merger`, it iterates over the :math:`\zeta`-markup equation from :ref:`Morrow and Skerlos (2011) <ms11>` to do so.

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
