Background
==========

The following sections provide a very brief overview of the BLP model and how it is estimated. This goal is to concisely introduce the notation and terminology used throughout the rest of the documentation. For a more in-depth overview, refer to :ref:`references:Conlon and Gortmaker (2020)`.


The Model
---------

There are :math:`t = 1, 2, \dotsc, T` markets, each with :math:`j = 1, 2, \dotsc, J_t` products produced by :math:`f = 1, 2, \dotsc, F_t` firms, for a total of :math:`N` products across all markets. There are :math:`i = 1, 2, \dotsc, I_t` individuals/agents who choose among the :math:`J_t` products and an outside good :math:`j = 0`. These numbers also represent sets. For example, :math:`J_t = \{1, 2, \dots, J_t\}`.


Demand
~~~~~~

Observed demand-side product characteristics are contained in the :math:`N \times K_1` matrix of linear characteristics, :math:`X_1`, and the :math:`N \times K_2` matrix of nonlinear characteristics, :math:`X_2`, which is typically a subset of :math:`X_1`. Unobserved demand-side product characteristics, :math:`\xi`, are a :math:`N \times 1` vector.

In market :math:`t`, observed agent characteristics are a :math:`I_t \times D` matrix called demographics, :math:`d`. Unobserved agent characteristics are a :math:`I_t \times K_2` matrix, :math:`\nu`.

The indirect utility of agent :math:`i` from purchasing product :math:`j` in market :math:`t` is

.. math:: U_{ijt} = \underbrace{\delta_{jt} + \mu_{ijt}}_{V_{ijt}} + \epsilon_{ijt},
   :label: utilities

in which the mean utility is, in vector-matrix form,

.. math:: \delta = \underbrace{X_1^\text{en}\alpha + X_1^\text{ex}\beta^\text{ex}}_{X_1\beta} + \xi.

The :math:`K_1 \times 1` vector of demand-side linear parameters, :math:`\beta`, is partitioned into two components: :math:`\alpha` is a :math:`K_1^\text{en} \times 1` vector of parameters on the :math:`N \times K_1^\text{en}` submatrix of endogenous characteristics, :math:`X_1^\text{en}`, and :math:`\beta^\text{ex}` is a :math:`K_1^\text{ex} \times 1` vector of parameters on the :math:`N \times K_1^\text{ex}` submatrix of exogenous characteristics, :math:`X_1^\text{ex}`. Usually, :math:`X_1^\text{en} = p`, prices, so :math:`\alpha` is simply a scalar.

The agent-specific portion of utility in a single market is, in vector-matrix form,

.. math:: \mu = X_2(\Sigma \nu' + \Pi d').
   :label: mu

The model incorporates both observable (demographics) and unobservable taste heterogeneity though random coefficients. For the unobserved heterogeneity, we let :math:`\nu` denote independent draws from the standard normal distribution. These are scaled by a :math:`K_2 \times K_2` lower-triangular matrix :math:`\Sigma`, which denotes the Cholesky root of the covariance matrix for unobserved taste heterogeneity. The :math:`K_2 \times D` matrix :math:`\Pi` measures how agent tastes vary with demographics.

In the above expression, random coefficients are assumed to be normally distributed. To incorporate one or more lognormal random coefficients, the associated columns in the parenthesized expression can be exponentiated before being pre-multiplied by :math:`X_2`. For example, this allows for the coefficient on price to be lognormal so that demand slopes down for all agents. For lognormal random coefficients, a constant column is typically included in :math:`d` so that its coefficients in :math:`\Pi` parametrize the means of the logs of the random coefficients.

Random idiosyncratic preferences, :math:`\epsilon_{ijt}`, are assumed to be Type I Extreme Value, so that conditional on the heterogeneous coefficients, market shares follow the well-known logit form. Aggregate market shares are obtained by integrating over the distribution of individual heterogeneity. They are approximated with Monte Carlo integration or quadrature rules defined by the :math:`I_t \times K_2` matrix of integration nodes, :math:`\nu`, and an :math:`I_t \times 1` vector of integration weights, :math:`w`:

.. math:: s_{jt} \approx \sum_{i \in I_t} w_{it} s_{ijt},
   :label: shares

where the probability that agent :math:`i` chooses product :math:`j` in market :math:`t` is

.. math:: s_{ijt} = \frac{\exp V_{ijt}}{1 + \sum_{k \in J_t} \exp V_{ikt}}.
   :label: probabilities

There is a one in the denominator because the utility of the outside good is normalized to :math:`U_{i0t} = 0`. The scale of utility is normalized by the variance of :math:`\epsilon_{ijt}`.

   
Supply
~~~~~~

Observed supply-side product characteristics are contained in the :math:`N \times K_3` matrix of supply-side characteristics, :math:`X_3`. Prices cannot be supply-side characteristics, but non-price product characteristics often overlap with the demand-side characteristics in :math:`X_1` and :math:`X_2`. Unobserved supply-side product characteristics, :math:`\omega`, are a :math:`N \times 1` vector.

Firm :math:`f` chooses prices in market :math:`t` to maximize the profits of its products :math:`J_{ft} \subset J_t`:

.. math:: \pi_{ft} = \sum_{j \in J_{ft}} (p_{jt} - c_{jt})s_{jt}.

In a single market, the corresponding multi-product differentiated Bertrand first order conditions are, in vector-matrix form,

.. math:: p - c = \underbrace{\Delta^{-1}s}_{\eta},
   :label: eta

where the multi-product Bertrand markup :math:`\eta` depends on :math:`\Delta`, a :math:`J_t \times J_t` matrix of intra-firm (negative) demand derivatives:

.. math:: \Delta = -\mathscr{H} \odot \frac{\partial s}{\partial p}.

Here, :math:`\mathscr{H}` denotes the market-level ownership or product holdings matrix in the market, where :math:`\mathscr{H}_{jk}` is typically :math:`1` if the same firm produces products :math:`j` and :math:`k`, and :math:`0` otherwise.

To include a supply side, we must specify a functional form for marginal costs:

.. math:: \tilde{c} = f(c) = X_3\gamma + \omega.
   :label: costs

The most common choices are :math:`f(c) = c` and :math:`f(c) = \log(c)`.


Estimation
----------

A demand side is always estimated but including a supply side is optional. With only a demand side, there are three sets of parameters to be estimated: :math:`\beta` (which may include :math:`\alpha`), :math:`\Sigma` and :math:`\Pi`. With a supply side, there is also :math:`\gamma`. The linear parameters, :math:`\beta` and :math:`\gamma`, are typically concentrated out of the problem. The exception is :math:`\alpha`, which cannot be concentrated out when there is a supply side because it is needed to compute demand derivatives and hence marginal costs. Linear parameters that are not concentrated out along with unknown nonlinear parameters in :math:`\Sigma` and :math:`\Pi` are collectively denoted :math:`\theta`.

The GMM problem is

.. math:: \min_\theta q(\theta) = \bar{g}(\theta)'W\bar{g}(\theta),
   :label: objective

in which :math:`q(\theta)` is the GMM objective. By default, PyBLP scales this value by :math:`N` so that objectives across different problem sizes are comparable. This behavior can be disabled. In some of the BLP literature and in earlier versions of this package, the objective was scaled by :math:`N^2`.

Here, :math:`W` is a :math:`M \times M` weighting matrix and :math:`\bar{g}` is a :math:`M \times 1` vector of averaged demand- and supply-side moments:

.. math:: \bar{g} = \begin{bmatrix} \bar{g}_D \\ \bar{g}_S \end{bmatrix} = \frac{1}{N} \begin{bmatrix} \sum_{j,t} Z_{D,jt}'\xi_{jt} \\ \sum_{j,t} Z_{S,jt}'\omega_{jt} \end{bmatrix}
   :label: averaged_moments

where :math:`Z_D` and :math:`Z_S` are :math:`N \times M_D` and :math:`N \times M_S` matrices of demand- and supply-side instruments.

The vector :math:`\bar{g}` contains sample analogues of the demand- and supply-side moment conditions :math:`E[g_{D,jt}] = E[g_{S,jt}] = 0` where

.. math:: \begin{bmatrix} g_{D,jt} & g_{S,jt} \end{bmatrix} = \begin{bmatrix} \xi_{jt}Z_{D,jt} & \omega_{jt}Z_{S,jt} \end{bmatrix}.
   :label: moments

In each GMM stage, a nonlinear optimizer finds the :math:`\hat{\theta}` that minimizes the GMM objective value :math:`q(\theta)`.


The Objective
~~~~~~~~~~~~~

Given a :math:`\theta`, the first step to computing the objective :math:`q(\theta)` is to compute :math:`\delta(\theta)` in each market with the following standard contraction:

.. math:: \delta_{jt} \leftarrow \delta_{jt} + \log s_{jt} - \log s_{jt}(\delta, \theta)
   :label: contraction

where :math:`s` are the market's observed shares and :math:`s(\delta, \theta)` are calculated market shares. Iteration terminates when the norm of the change in :math:`\delta(\theta)` is less than a small number.

With a supply side, marginal costs are then computed according to :eq:`eta`:

.. math:: c_{jt}(\theta) = p_{jt} - \eta_{jt}(\theta).

Concentrated out linear parameters are recovered with linear IV-GMM:

.. math:: \begin{bmatrix} \hat{\beta}^\text{ex} \\ \hat{\gamma} \end{bmatrix} = (X'ZWZ'X)^{-1}X'ZWZ'Y(\theta)
   :label: iv

where

.. math:: X = \begin{bmatrix} X_1^\text{ex} & 0 \\ 0 & X_3 \end{bmatrix}, \quad Z = \begin{bmatrix} Z_D & 0 \\ 0 & Z_S \end{bmatrix}, \quad Y(\theta) = \begin{bmatrix} \delta(\theta) - X_1^\text{en}\hat{\alpha} & 0 \\ 0 & \tilde{c}(\theta) \end{bmatrix}.

With only a demand side, :math:`\alpha` can be concentrated out, so :math:`X = X_1`, :math:`Z = Z_D`, and :math:`Y = \delta(\theta)` recover the full :math:`\hat{\beta}` in :eq:`iv`.

Finally, the unobserved product characteristics (i.e., the structural errors),

.. math:: \begin{bmatrix} \xi(\theta) \\ \omega(\theta) \end{bmatrix} = \begin{bmatrix} \delta(\theta) - X_1\hat{\beta} \\ \tilde{c}(\theta) - X_3\hat{\gamma} \end{bmatrix},

are interacted with the instruments to form :math:`\bar{g}(\theta)` in :eq:`averaged_moments`, which gives the GMM objective :math:`q(\theta)` in :eq:`objective`.


The Gradient
~~~~~~~~~~~~

The gradient of the GMM objective in :eq:`objective` is 

.. math:: \nabla q(\theta) = 2\bar{G}(\theta)'W\bar{g}(\theta)
   :label: gradient

where

.. math:: \bar{G} = \begin{bmatrix} \bar{G}_D \\ \bar{G}_S \end{bmatrix} = \frac{1}{N} \begin{bmatrix} \sum_{j,t} Z_{D,jt}'\frac{\partial\xi_{jt}}{\partial\theta} \\ \sum_{j,t} Z_{S,jt}'\frac{\partial\omega_{jt}}{\partial\theta} \end{bmatrix}.
   :label: averaged_moments_jacobian

Writing :math:`\delta` as an implicit function of :math:`s` in :eq:`shares` gives the demand-side Jacobian:

.. math:: \frac{\partial\xi}{\partial\theta} = \frac{\partial\delta}{\partial\theta} = -\left(\frac{\partial s}{\partial\delta}\right)^{-1}\frac{\partial s}{\partial\theta}.

The supply-side Jacobian is derived from the definition of :math:`\tilde{c}` in :eq:`costs`:

.. math:: \frac{\partial\omega}{\partial\theta} = \frac{\partial\tilde{c}}{\partial\theta_p} = -\frac{\partial\tilde{c}}{\partial c}\frac{\partial\eta}{\partial\theta}.

The second term in this expression is derived from the definition of :math:`\eta` in :eq:`eta`:

.. math:: \frac{\partial\eta}{\partial\theta} = -\Delta^{-1}\left(\frac{\partial\Delta}{\partial\theta}\eta + \frac{\partial\Delta}{\partial\xi}\eta\frac{\partial\xi}{\partial\theta}\right).


Weighting Matrices
~~~~~~~~~~~~~~~~~~

Conventionally, the 2SLS weighting matrix is used in the first stage:

.. math:: W = \begin{bmatrix} (Z_D'Z_D / N)^{-1} & 0 \\ 0 & (Z_S'Z_S / N)^{-1} \end{bmatrix}.
   :label: 2sls_W

With two-step GMM, :math:`W` is updated before the second stage according to 

.. math:: W = S^{-1}.
   :label: W

For heteroscedasticity robust weighting matrices,

.. math:: S = \frac{1}{N}\sum_{j,t} g_{jt}g_{jt}'.
   :label: robust_S

For clustered weighting matrices with :math:`c = 1, 2, \dotsc, C` clusters,

.. math:: S = \frac{1}{N}\sum_{c=1}^C g_cg_c',
   :label: clustered_S

where, letting the set :math:`J_{ct} \subset J_t` denote products in cluster :math:`c` and market :math:`t`,

.. math:: g_c = \sum_{t \in T} \sum_{j \in J_{ct}} g_{jt}.

For unadjusted weighting matrices,

.. math:: S = \frac{1}{N} \begin{bmatrix} \sigma_\xi^2 Z_D'Z_D & \sigma_{\xi\omega} Z_D'Z_S \\ \sigma_{\xi\omega} Z_S'Z_D & \sigma_\omega^2 Z_S'Z_S \end{bmatrix}
   :label: unadjusted_S

where

.. math:: \text{Var}(\xi, \omega) = \begin{bmatrix} \sigma_\xi^2 & \sigma_{\xi\omega} \\ \sigma_{\xi\omega} & \sigma_\omega^2 \end{bmatrix}.


Standard Errors
~~~~~~~~~~~~~~~

The covariance matrix of the estimated parameters is

.. math:: \text{Var}(\hat{\theta}) = (\bar{G}'W\bar{G})^{-1}\bar{G}'WSW\bar{G}(\bar{G}'W\bar{G})^{-1}.
   :label: covariances

Standard errors are the square root of the diagonal of this matrix divided by :math:`N`.

If the weighting matrix was chosen such that :math:`W = S^{-1}`, this simplifies to

.. math:: \text{Var}(\hat{\theta}) = (\bar{G}'W\bar{G})^{-1}.
   :label: unadjusted_covariances

Standard errors extracted from this simpler expression are called unadjusted.


Fixed Effects
-------------

The unobserved product characteristics can be partitioned into

.. math:: \begin{bmatrix} \xi_{jt} \\ \omega_{jt} \end{bmatrix} = \begin{bmatrix} \xi_{k_1} + \xi_{k_2} + \cdots + \xi_{k_{E_D}} + \Delta\xi_{jt} \\ \omega_{\ell_1} + \omega_{\ell_2} + \cdots + \omega_{\ell_{E_S}} + \Delta\omega_{jt} \end{bmatrix}
   :label: fe

where :math:`k_1, k_2, \dotsc, k_{E_D}` and :math:`\ell_1, \ell_2, \dotsc, \ell_{E_S}` index unobserved characteristics that are fixed across :math:`E_D` and :math:`E_S` dimensions. For example, with :math:`E_D = 1` dimension of product fixed effects, :math:`\xi_{jt} = \xi_j + \Delta\xi_{jt}`.

Small numbers of fixed effects can be estimated with dummy variables in :math:`X_1`, :math:`X_3`, :math:`Z_D`, and :math:`Z_S`. However, this approach does not scale with high dimensional fixed effects because it requires constructing and inverting an infeasibly large matrix in :eq:`iv`. 

Instead, fixed effects are typically absorbed into :math:`X`, :math:`Z`, and :math:`Y(\theta)` in :eq:`iv`. With one fixed effect, these matrices are simply de-meaned within each level of the fixed effect. Both :math:`X` and :math:`Z` can be de-meaned just once, but :math:`Y(\theta)` must be de-meaned for each new :math:`\theta`.

This procedure is equivalent to replacing each column of the matrices with residuals from a regression of the column on the fixed effect. The Frish-Waugh-Lovell (FWL) theorem of :ref:`references:Frisch and Waugh (1933)` and :ref:`references:Lovell (1963)` guarantees that using these residualized matrices gives the same results as including fixed effects as dummy variables. When :math:`E_D > 1` or :math:`E_S > 1`, the matrices are residualized with more involved algorithms.

Once fixed effects have been absorbed, estimation is as described above with the structural errors :math:`\Delta\xi` and :math:`\Delta\omega`.


Micro Moments
-------------

In the spirit of :ref:`references:Imbens and Lancaster (1994)`, :ref:`references:Petrin (2002)`, and :ref:`references:Berry, Levinsohn, and Pakes (2004)`, more detailed micro data on individual agent decisions can be used to supplement the standard demand- and supply-side moments :math:`\bar{g}_D` and :math:`\bar{g}_S` in :eq:`averaged_moments` with an additional :math:`m = 1, 2, \ldots, M_M` averaged micro moments, :math:`\bar{g}_M`, for a total of :math:`M = M_D + M_S + M_M` averaged moments:

.. math:: \bar{g} = \begin{bmatrix} \bar{g}_D \\ \bar{g}_S \\ \bar{g}_M \end{bmatrix}.

Each micro moment :math:`m` is approximated in a set :math:`T_m \subset T` of markets in which its micro data are relevant and then averaged across these markets:

.. math:: \bar{g}_{M,m} \approx \frac{1}{T_m} \sum_{t \in T_m} g_{M,mt}.
   :label: averaged_micro_moments

The vector :math:`\bar{g}_M` contains sample analogues of micro moment conditions :math:`E[g_{M,mt}] = 0` where :math:`g_{M,mt}` is typically a function of choice choice probabilities, data in market :math:`t`, and a statistic :math:`\mathcal{V}_{mt}` computed from survey data that the moment aims to match.

Mico moments are computed for each :math:`\theta` and contribute to the GMM objective :math:`q(\theta)` in :eq:`objective`. Their derivatives with respect to :math:`\theta` are added as rows to :math:`\bar{G}` in :eq:`averaged_moments_jacobian`, and blocks are added to both :math:`W` and :math:`S` in :eq:`2sls_W` and :eq:`W`. The covariance between standard moments and micro moments is assumed to be zero, so these matrices are block-diagonal. The covariance between micro moments :math:`m` and :math:`n` in :math:`S` is set to zero if :math:`T_{mn} = T_m \cap T_n = \emptyset` and otherwise is

.. math:: \text{Cov}(\bar{g}_{M,m}, \bar{g}_{M,n}) = \frac{1}{T_{mn}} \sum_{t \in T_{mn}} \text{Cov}(g_{M,mt}, g_{M,nt}).
   :label: averaged_micro_moment_covariances


Random Coefficients Nested Logit
--------------------------------

Incorporating parameters that measure within nesting group correlation gives the random coefficients nested logit (RCNL) model of :ref:`references:Brenkers and Verboven (2006)` and :ref:`references:Grigolon and Verboven (2014)`. There are :math:`h = 1, 2, \dotsc, H` nesting groups and each product :math:`j` is assigned to a group :math:`h(j)`. The set :math:`J_{ht} \subset J_t` denotes the products in group :math:`h` and market :math:`t`.

In the RCNL model, idiosyncratic preferences are partitioned into

.. math:: \epsilon_{ijt} = \bar{\epsilon}_{ih(j)t} + (1 - \rho_{h(j)})\bar{\epsilon}_{ijt}

where :math:`\bar{\epsilon}_{ijt}` is Type I Extreme Value and :math:`\bar{\epsilon}_{ih(j)t}` is distributed such that :math:`\epsilon_{ijt}` is still Type I Extreme Value. 

The nesting parameters, :math:`\rho`, can either be a :math:`H \times 1` vector or a scalar so that for all groups :math:`\rho_h = \rho`. Letting :math:`\rho \to 0` gives the standard BLP model and :math:`\rho \to 1` gives division by zero errors. With :math:`\rho_h \in (0, 1)`, the expression for choice probabilities in :eq:`probabilities` becomes more complicated:

.. math:: s_{ijt} = \frac{\exp[V_{ijt} / (1 - \rho_{h(j)})]}{\exp[V_{ih(j)t} / (1 - \rho_{h(j)})]}\cdot\frac{\exp V_{ih(j)t}}{1 + \sum_{h \in H} \exp V_{iht}}
   :label: nested_probabilities

where 

.. math:: V_{iht} = (1 - \rho_h)\log\sum_{k \in J_{ht}} \exp[V_{ikt} / (1 - \rho_h)].
   :label: inclusive_value

The contraction for :math:`\delta(\theta)` in :eq:`contraction` is also slightly different:

.. math:: \delta_{jt} \leftarrow \delta_{jt} + (1 - \rho_{h(j)})[\log s_{jt} - \log s_{jt}(\delta, \theta)].
   :label: nested_contraction

Otherwise, estimation is as described above with :math:`\rho` included in :math:`\theta`.


Logit and Nested Logit
----------------------

Letting :math:`\Sigma = 0` gives the simpler logit (or nested logit) model where there is a closed-form solution for :math:`\delta`. In the logit model,

.. math:: \delta_{jt} = \log s_{jt} - \log s_{0t},
   :label: logit_delta

and a lack of nonlinear parameters means that nonlinear optimization is often unneeded.

In the nested logit model, :math:`\rho` must be optimized over, but there is still a closed-form solution for :math:`\delta`:

.. math:: \delta_{jt} = \log s_{jt} - \log s_{0t} - \rho_{h(j)}[\log s_{jt} - \log s_{h(j)t}].
   :label: nested_logit_delta

where

.. math:: s_{ht} = \sum_{j \in J_{ht}} s_{jt}.

In both models, a supply side can still be estimated jointly with demand. Estimation is as described above with a representative agent in each market: :math:`I_t = 1` and :math:`w_1 = 1`.


Equilibrium Prices
------------------

Counterfactual evaluation, synthetic data simulation, and optimal instrument generation often involve solving for prices implied by the Bertrand first order conditions in :eq:`eta`. Solving this system with Newton's method is slow and iterating over :math:`p \leftarrow c + \eta(p)` may not converge because it is not a contraction.

Instead, :ref:`references:Morrow and Skerlos (2011)` reformulate the solution to :eq:`eta`:

.. math:: p - c = \underbrace{\Lambda^{-1}(\mathscr{H} \odot \Gamma)'(p - c) - \Lambda^{-1}s}_{\zeta}
   :label: zeta

where :math:`\Lambda` is a diagonal :math:`J_t \times J_t` matrix approximated by

.. math:: \Lambda_{jj} \approx \sum_{i \in I_t} w_{it} s_{ijt}\frac{\partial U_{ijt}}{\partial p_{jt}}

and :math:`\Gamma` is a dense :math:`J_t \times J_t` matrix approximated by

.. math:: \Gamma_{jk} \approx \sum_{i \in I_t} w_{it} s_{ijt}s_{ikt}\frac{\partial U_{ijt}}{\partial p_{jt}}.

Equilibrium prices are computed by iterating over the :math:`\zeta`-markup equation in :eq:`zeta`,

.. math:: p \leftarrow c + \zeta(p),
   :label: zeta_contraction

which, unlike :eq:`eta`, is a contraction. Iteration terminates when the norm of firms' first order conditions, :math:`||\Lambda(p)(p - c - \zeta(p))||`, is less than a small number.

If marginal costs depend on quantity, then they also depend on prices and need to be updated during each iteration: :math:`c_{jt} = c_{jt}(s_{jt}(p))`.
