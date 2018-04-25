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

   blp_data = np.recfromcsv(pyblp.data.BLP_PRODUCTS_LOCATION)
   nevo_data = np.recfromcsv(pyblp.data.NEVO_PRODUCTS_LOCATION)

The `product_data` argument in :class:`Problem` initialization is a structured array-like object with a number of fields. Product data can be a structured :class:`numpy.ndarray`, a :class:`pandas.DataFrame`, or other similar objects. We'll use a simple :class:`dict`, which we can immediately fill with some columns from the raw data that don't require modification.

.. ipython:: python

   blp_product_data = {
       'market_ids': blp_data['market_ids'],
       'shares': blp_data['shares'],
       'prices': blp_data['prices']
   }
   nevo_product_data = {
       'market_ids': nevo_data['market_ids'],
       'shares': nevo_data['shares'],
       'prices': nevo_data['prices']
   }

For both sets of data, matrices of firm IDs consist of a column of baseline IDs, as well as IDs after a simple merger. We'll use the first colum during estimation to compute supply-side moments for the automobile data, and we'll use both columns to compute many post-estimation outputs. The two CSV files include both sets of IDs.

.. ipython:: python

   blp_product_data['firm_ids'] = np.c_[blp_data['firm_ids'], blp_data['changed_firm_ids']]
   nevo_product_data['firm_ids'] = np.c_[nevo_data['firm_ids'], nevo_data['changed_firm_ids']]

Linear and nonlinear characteristics for the automobile problem are a constant column followed by four product characterstics. Cost characteristics are the same but with logged continuous variables, with miles per gallon instead of per dollar, and with a trend.

.. ipython:: python

   blp_product_data['linear_characteristics'] = np.c_[
       np.ones(blp_data.size), 
       blp_data['hpwt'], 
       blp_data['air'], 
       blp_data['mpd'], 
       blp_data['space']
   ]
   blp_product_data['nonlinear_characteristics'] = blp_product_data['linear_characteristics']
   blp_product_data['cost_characteristics'] = np.c_[
       np.ones(blp_data.size), 
       np.log(blp_data['hpwt']), 
       blp_data['air'], 
       np.log(blp_data['mpg']),
       np.log(blp_data['space']),
       blp_data['trend']
   ]

Linear characteristics for the fake cereal problem are simply product indicators. Nonlinear characteristics are a constant column followed by two product characteristics. There are no cost characteristics because we'll only be incorporating supply-side moment into the automobile problem.

.. ipython:: python

   nevo_indicators = pyblp.build_indicators(nevo_data['product_ids'])
   nevo_product_data['linear_characteristics'] = nevo_indicators
   nevo_product_data['nonlinear_characteristics'] = np.c_[
       np.ones(nevo_data.size),
       nevo_data['sugar'],
       nevo_data['mushy']
   ]

Demand-side instruments for the automobile problem are the linear and nonlinear characteristics along with their traditional BLP instrument counterparts constructed by :func:`build_blp_instruments`; as in in :ref:`Berry, Levinsohn, and Pakes (1995) <blp95>`, miles per gallon and the trend are not included due to collinearity. Supply-side instruments are the cost characteristics, their traditional BLP instrument counterparts, and the excluded demand variable, miles per dollar.

.. ipython:: python

   blp_product_data['demand_instruments'] = np.c_[
       blp_product_data['linear_characteristics'],
       pyblp.build_blp_instruments({
           'market_ids': blp_data['market_ids'],
           'firm_ids': blp_data['firm_ids'],
           'characteristics': blp_product_data['linear_characteristics']
       })
   ]
   blp_product_data['supply_instruments'] = np.c_[
       blp_product_data['cost_characteristics'],
       pyblp.build_blp_instruments({
           'market_ids': blp_data['market_ids'],
           'firm_ids': blp_data['firm_ids'],
           'characteristics': blp_product_data['cost_characteristics']
       }),
       blp_data['mpd']
   ]

Instruments for the fake cereal problem are product indicators along with the instruments taken directly from the CSV file.

.. ipython:: python
   
   nevo_instruments = np.column_stack([nevo_data[f'instruments{i}'] for i in range(20)])
   nevo_product_data['demand_instruments'] = np.c_[nevo_indicators, nevo_instruments]


Agent Data
~~~~~~~~~~

The package also includes example agent data. Since the draws included for the automobile problem are the ones used by :ref:`Knittel and Metaxoglou (2014) <km14>` and not the draws from the original paper, we might as well configure the package to build our own set of Monte Carlo draws for each market. A small number of draws speeds up estimation for this example.

.. ipython:: python

   blp_integration = pyblp.Integration('monte_carlo', 200, seed=0)

Agent data from the fake cereal CSV file can be used without modification.

.. ipython:: python

   nevo_agent_data = np.recfromcsv(pyblp.data.NEVO_AGENTS_LOCATION)
   nevo_agent_data

Unlike the product data matrices constructed above, agent data from the CSV file represents multi-column fields with multiple fields, each with an index at the end. The pyblp package can handle both matrix representations.


Solving Problems
----------------

Problem data along with agent data or an integration configuration are used to initialize :class:`Problem` classes. Once a problem is initialized, :meth:`Problem.solve` performs estimation. The arguments to :meth:`Problem.solve` configure how estimation is performed. For example, `optimization` and `iteration` configure the optimization and iteration routines that are used by the outer and inner loops of estimation.


The Automobile Problem
~~~~~~~~~~~~~~~~~~~~~~

The :class:`Integration` configuration will be used by :class:`Problem` to build agent data. By default, :class:`Problem` includes prices, :math:`p`, in :math:`X_2`. We'll use the `nonlinear_prices` argument to include prices only in :math:`X_1` as in :ref:`Berry, Levinsohn, and Pakes (1995) <blp95>`.

.. ipython:: python

   blp_problem = pyblp.Problem(
       blp_product_data,
       integration=blp_integration,
       nonlinear_prices=False
   )

Inspecting the attributes of the :class:`Problem` instance helps to confirm that the problem has been configured correctly. For example, inspecting :attr:`Problem.products` and :attr:`Problem.agents` confirms that product data was structured correctly and that agent data was built correctly.

.. ipython:: python

   blp_problem.products
   blp_problem.agents

The initialized problem can be solved with :meth:`Problem.solve`. By passing an identity matrix as starting values for :math:`\Sigma`, we're choosing to optimize over only variance terms, and we're choosing to have all five nonlinear parameters start at one. Although we'll use the same log-linear marginal cost specification that was used in the original paper, we'll speed up optimization by using the default optimization routine, which computes an analytic gradient. We'll also only perform one GMM step. You can pass ``pyblp.Optimization('nelder-mead', compute_gradient=False)`` to the `optimization` argument if you want to use the original :ref:`Nelder and Mead (1965) <nm65>` routine.

.. ipython:: python

   blp_sigma = np.diag(np.ones(5))
   blp_results = blp_problem.solve(blp_sigma, linear_costs=False, steps=1)
   blp_results

Estimates, which are in the same order as product characteristics configured during :class:`Problem` initialization, are similar to those in :ref:`Berry, Levinsohn, and Pakes (1995) <blp95>`. Of course, divergences from the original configuration create differences. For example, this configuration does not incorporate an interaction between prices and income. To do so, you could include income as a demographic when initializing :class:`Problem`, and, in :meth:`Problem.solve`, allow one or more parameters in :math:`\Pi` to vary.


The Fake Cereal Problem
~~~~~~~~~~~~~~~~~~~~~~~

Unlike the automobile problem, we have included demographics in the agent data for the fake cereal problem and we have not included supply-side information in its product data. Also, we will configure this problem to include prices in :math:`X_2`, which is the default for :class:`Problem`.

.. ipython:: python

   nevo_problem = pyblp.Problem(nevo_product_data, nevo_agent_data)

Since we initialized the problem without supply-side data, there's no need to choose a marginal cost specification. However, since we initialized the problem with demographics, we need to configure not only :math:`\Sigma`, but also :math:`\Pi`. We'll use the same starting values as :ref:`Nevo (2000) <n00>`. We'll also use a non-default :func:`scipy.optimize.minimize` quasi-Newton optimization routine with BFGS hessian approximation, which is similar to the default Matlab optimization routine, and, again, we'll only perform one GMM step for the sake of speed in this example.

.. ipython:: python

   nevo_sigma = np.diag([2.4526, 0.3302, 0.0163, 0.2441])
   nevo_pi = [
      [15.8935, -1.2000, 0,       2.6342],
      [ 5.4819,  0,      0.2037,  0     ],
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

Often, the above starting values give rise to some warnings during the first few GMM objective evaluations about floating point problems. This is because some optimization routines attempt to evaluate the objective at parameter values that lead to overflow while, for example, computing :math:`\hat{\delta}`. For example, using ``pyblp.Optimization('slsqp')`` displays some warnings if :attr:`options.verbose` is ``True``. The default behavior of :meth:`Problem.solve` is to revert problematic elements in :math:`\hat{\delta}` and its Jacobian before computing the objective value, which allows the optimization routine to continue searching the parameter space. For more information, refer to :meth:`Problem.solve`. In particular, the `sigma_bounds` and `pi_bounds` arguments can be used to bound the parameter space over which the optimization problem searches.

Again, results are similar to those in the original paper. Compared to the automobile problem, results are even closer to the original because we didn't simulate our own agent data.


Problem Results
---------------

The :meth:`Problem.solve` method returns an instance of the :class:`Results` class, which, when printed, displays basic estimation results. The results that are displayed are simply formatted information extracted from various class attributes such as :attr:`Results.sigma` and :attr:`Results.sigma_se`. Standard errors are either robust or unadjusted, depending on the `se_type` argument of :meth:`Problem.solve`.

Additional post-estimation outputs can be computed with :class:`Results` methods.


Elasticities and Diversion Ratios
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can estimate elasticities, :math:`\varepsilon`, and diversion ratios, :math:`\mathscr{D}`, with :meth:`Results.compute_price_elasticities` and :meth:`Results.compute_price_diversion_ratios`.

.. ipython:: python

   blp_elasticities = blp_results.compute_price_elasticities()
   nevo_elasticities = nevo_results.compute_price_elasticities()
   blp_ratios = blp_results.compute_price_diversion_ratios()
   nevo_ratios = nevo_results.compute_price_diversion_ratios()

Post-estimation outputs are computed for each market and stacked. We'll use :func:`matplotlib.pyplot.matshow` and :func:`matplotlib.pyplot.colorbar` to display the matrices associated with market ID ``1``.

.. ipython:: python

   first_blp_market = blp_data['market_ids'] == 1
   first_nevo_market = nevo_data['market_ids'] == 1
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

Other methods that compute similar matrices are :meth:`Results.compute_elasticities` and :meth:`Results.compute_diversion_ratios`, which estimate :math:`\varepsilon` and :math:`\mathscr{D}` with respect to non-price characteristics, and :meth:`Results.compute_long_run_diversion_ratios`, which can be used to used to understand substitution when products are eliminated from the choice set.

Additionally, the convenience methods :meth:`Results.extract_diagonals` and :meth:`Results.extract_diagonal_means` can be used to extract information about own elasticities of demand from elasticity matrices.

.. ipython:: python

   blp_means = blp_results.extract_diagonal_means(blp_elasticities)
   nevo_means = nevo_results.extract_diagonal_means(nevo_elasticities)

An alternative to summarizing full elasticity matrices is to use :meth:`Results.compute_aggregate_price_elasticities` to estimate aggregate elasticities of demand, :math:`E`, in each market, which reflect the change in total sales under a proportional sales tax of some factor.

.. ipython:: python

   blp_aggregates = blp_results.compute_aggregate_price_elasticities(factor=0.1)
   nevo_aggregates = nevo_results.compute_aggregate_price_elasticities(factor=0.1)

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

Other methods that compute supply-side outputs require such estimates of :math:`c`. For example, we pass these estimates to :meth:`Results.compute_markups` when estimating markups, :math:`\mathscr{M}`.

.. ipython:: python

   blp_markups = blp_results.compute_markups(blp_costs)
   nevo_markups = nevo_results.compute_markups(nevo_costs)
   bins = np.linspace(0, 2, 50)
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
   blp_profits = blp_results.compute_profits(blp_costs)
   nevo_profits = nevo_results.compute_profits(nevo_costs)
   blp_cs = blp_results.compute_consumer_surpluses()
   nevo_cs = nevo_results.compute_consumer_surpluses()

To compute post-merger outputs, the `firm_ids` field in the `product_data` passed to :class:`Problem` must have had at least two columns. Columns after the first represent changes, such as mergers. Although mergers are commonly what firm ID changes represent, these additional columns can represent any type of change.

Since we included two columns of firm IDs in both problems, we can use :meth:`Results.solve_approximate_merger` or :meth:`Results.solve_merger` to estimate post-merger prices. The first method, which is discussed, for example, in :ref:`Nevo (1997) <n97>`, assumes that shares and their price derivatives are unaffected by the merger. The second method does not make these two assumptions and iterates over the :math:`\zeta`-markup equation from :ref:`Morrow and Skerlos (2011) <ms11>` to solve the full system of :math:`J_t` equations and :math:`J_t` unknowns in each market :math:`t`. We'll use the latter, since it is fast enough for the two example problems.

.. ipython:: python

   blp_changed_prices = blp_results.solve_merger(blp_costs)
   nevo_changed_prices = nevo_results.solve_merger(nevo_costs)

If the problems were configured with more than two columns of firm IDs, we could estimate post-merger prices for the other mergers with the `firm_ids_index` argument, which is by default ``1``.

We'll compute post-merger shares with :meth:`Results.compute_shares`.

.. ipython:: python

   blp_changed_shares = blp_results.compute_shares(blp_changed_prices)
   nevo_changed_shares = nevo_results.compute_shares(nevo_changed_prices)

Post-merger prices and shares are used to compute other post-merger outputs. For example, :math:`\text{HHI}` increases.

.. ipython:: python

   blp_changed_hhi = blp_results.compute_hhi(blp_changed_shares, firm_ids_index=1)
   nevo_changed_hhi = nevo_results.compute_hhi(nevo_changed_shares, firm_ids_index=1)
   bins = np.linspace(0, 3000, 50)
   plt.hist(blp_changed_hhi - blp_hhi, bins, alpha=0.5, color='maroon');
   plt.hist(nevo_changed_hhi - nevo_hhi, bins, alpha=0.5, color='navy');

   @suppress
   savefig('images/hhi_changes.png')

.. image:: images/hhi_changes.png

Markups, :math:`\mathscr{M}`, and profits, :math:`\pi`, generally increase as well.

.. ipython:: python

   blp_changed_markups = blp_results.compute_markups(blp_costs, blp_changed_prices)
   nevo_changed_markups = nevo_results.compute_markups(nevo_costs, nevo_changed_prices)
   bins = np.linspace(-0.05, 0.25, 50)
   plt.hist(blp_changed_markups - blp_markups, bins, alpha=0.5, color='maroon');
   plt.hist(nevo_changed_markups - nevo_markups, bins, alpha=0.5, color='navy');

   @suppress
   savefig('images/markup_changes.png')

.. image:: images/markup_changes.png

.. ipython:: python

   blp_changed_profits = blp_results.compute_profits(
       blp_costs,
       blp_changed_prices,
       blp_changed_shares
   )
   nevo_changed_profits = nevo_results.compute_profits(
       nevo_costs,
       nevo_changed_prices,
       nevo_changed_shares
   )
   bins = np.linspace(-0.001, 0.001, 50)
   plt.hist(blp_changed_profits - blp_profits, bins, alpha=0.5, color='maroon');
   plt.hist(nevo_changed_profits - nevo_profits, bins, alpha=0.5, color='navy');

   @suppress
   savefig('images/profit_changes.png')

.. image:: images/profit_changes.png

On the other hand, consumer surpluses, :math:`\text{CS}`, generally decrease.

.. ipython:: python

   blp_changed_cs = blp_results.compute_consumer_surpluses(blp_changed_prices)
   nevo_changed_cs = nevo_results.compute_consumer_surpluses(nevo_changed_prices)
   bins = np.linspace(-0.06, 0.01, 50)
   plt.hist(blp_changed_cs - blp_cs, bins, alpha=0.5, color='maroon');
   plt.hist(nevo_changed_cs - nevo_cs, bins, alpha=0.5, color='navy');

   @suppress
   savefig('images/cs_changes.png')

.. image:: images/cs_changes.png


Simulating Problems
-------------------

Before configuring and solving a problem concerning real data, papers such as :ref:`Armstrong (2016) <a16>` recommend performing Monte Carlo analysis on simulated data to verify that it is possible to accurately estimate model parameters. For example, before configuring and solving the above automobile problem, it may have been a good idea to simulate data according to the assumed models of supply and demand. During such Monte Carlo anaysis, the data would only be used to determine sample sizes and perhaps to choose true parameters that are within reason.

Simulations are configured with the :class:`Simulation` class, which requires market and firm IDs, a configuration for how to build agent data, and parameter matrices that configure true parameter values and which characteristics enter into which parts of the model.

First, we'll use :func:`build_id_data` to build market and firm IDs for a model in which there are :math:`T = 50` markets, and in each market :math:`t`, :math:`J_t = 20` products produced by :math:`F_t = 10` firms.

.. ipython:: python

   simulation_id_data = pyblp.build_id_data(T=50, J=20, F=10)

Next, we'll choose configure :class:`Integration` to build agent data according to a level-``5`` Gauss-Hermite product rule.

.. ipython:: python

   simulation_integration = pyblp.Integration('product', 5)

We'll then pass these along with parameter matrix configurations to :class:`Simulation`.

.. ipython:: python

   simulation = pyblp.Simulation(
       simulation_id_data,
       simulation_integration,
       gamma=[5, 2, 3, 0],
       beta=[-10, 1, 0, 0, 2],
       sigma=[
           [0, 0, 0, 0, 0],
           [0, 2, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]
       ],
       pi=[
           [0],
           [1],
           [0],
           [0],
           [0]
       ],
       seed=0
   )

For a detailed explanation of how to use parameter matrices to configure all sorts of simulations, refer to :class:`Simulation`. Generally, the first element in `beta` and the first row in `sigma` and `pi` all correspond to prices. The following element in `beta`, the following row in `sigma` and `pi`, and the first element in `gamma` all correspond to a constant column. All following elements and rows correspond to other product characteristics.

The first three elements of `gamma` are nonzero, so :math:`X_3` is a constant column followed by two simulated product characteristics. The first two and the last element of `beta` are nonzero, so :math:`X_1` is prices followed by a constant column and a third simulated product characteristic that isn't in :math:`X_1` becuse the last element of `gamma` is zero. The second and third elements on the diagonal of `sigma` are nonzero, so :math:`X_2` is a constant column followed by the first non-constant product characteristic in :math:`X_2`. There is one column in `pi`, so :math:`d` is a single simulated demographic.

When :class:`Simulation` is initialized, it constructs agent data and simulates all product data except for prices and shares.

.. ipython:: python

   simulation.product_data
   simulation.agent_data

The instruments in :attr:`Simulation.product_data` are canonical ones that are computed with :func:`build_blp_instruments`. For this example, :math:`Z_D` is a constant column followed by all non-price characteristics and two other columns of traditional BLP instruments that are computed for the one demand-side characteristic that doesn't enter :math:`X_3`. Similarly, :math:`Z_S` is a constant column followed by all non-price characteristics and two other columns of BLP instruments that are computed for the one supply-side characteristic that doesn't enter :math:`X_1` or :math:`X_2`.

The :class:`Simulation` can be further configured with other arguments that control how unobserved product characteristics are simulated and how marginal costs are specified. Generally, observed product characteristics and demographics are drawn from the standard uniform distribution, and :math:`\xi` and :math:`\omega` are drawn from a bivariate normal distribution.

Since at this stage, prices and shares are all ``numpy.nan``, we still need to solve the simulation with :meth:`Simulation.solve`. This method computes Bertrand-Nash prices and shares. Just like :meth:`Results.solve_merger`, it iterates over the :math:`\zeta`-markup equation from :ref:`Morrow and Skerlos (2011) <ms11>` to do so.

.. ipython:: python

   simulated_product_data = simulation.solve()
   simulated_product_data

Now, we can try to recover the true parameters by creating and solving a :class:`Problem`. To make estimation easy, we'll use the same agent data and the same parameter sparsity structure. However, we'll choose starting values that are half the true parameters so that the optimization routine has to do some work.

.. ipython:: python
   :okwarning:

   simulated_problem = pyblp.Problem(
       simulated_product_data,
       simulation.agent_data,
       nonlinear_prices=False
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

The parameters seem to have been estimated reasonably well, except for the first elements in :math:`\Sigma` and :math:`\Pi`, which are still within an estimated standard error of their true values.

In addition to checking that the configuration for a model based on actual data makes sense, the :class:`Simulation` class can also be a helpful tool for better understanding under what general conditions BLP models can be accurately estimated. Additionally, it is used extensively in pyblp's test suite.
