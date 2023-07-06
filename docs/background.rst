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

In the above expression, random coefficients are assumed to be normally distributed, but this expression supports all elliptical distributions. To incorporate one or more lognormal random coefficients, the associated columns in the parenthesized expression can be exponentiated before being pre-multiplied by :math:`X_2`. For example, this allows for the coefficient on price to be lognormal so that demand slopes down for all agents. For lognormal random coefficients, a constant column is typically included in :math:`d` so that its coefficients in :math:`\Pi` parametrize the means of the logs of the random coefficients. More generally, all log-elliptical distributions are supported. A logit link function is also supported.

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

where the multi-product Bertrand markup :math:`\eta` depends on :math:`\Delta`, a :math:`J_t \times J_t` matrix of intra-firm (negative, transposed) demand derivatives:

.. math:: \Delta = -\mathscr{H} \odot \frac{\partial s}{\partial p}'.

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

.. math:: X = \begin{bmatrix} X_1^\text{ex} & 0 \\ 0 & X_3 \end{bmatrix}, \quad Z = \begin{bmatrix} Z_D & 0 \\ 0 & Z_S \end{bmatrix}, \quad Y(\theta) = \begin{bmatrix} \delta(\theta) - X_1^\text{en}\hat{\alpha} \\ \tilde{c}(\theta) \end{bmatrix}.

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

.. math:: \frac{\partial\omega}{\partial\theta} = \frac{\partial\tilde{c}}{\partial\theta} = -\frac{\partial\tilde{c}}{\partial c}\frac{\partial\eta}{\partial\theta}.

The second term in this expression is derived from the definition of :math:`\eta` in :eq:`eta`:

.. math:: \frac{\partial\eta}{\partial\theta} = -\Delta^{-1}\left(\frac{\partial\Delta}{\partial\theta}\eta + \frac{\partial\Delta}{\partial\xi}\eta\frac{\partial\xi}{\partial\theta}\right).

One thing to note is that :math:`\frac{\partial\xi}{\partial\theta} = \frac{\partial\delta}{\partial\theta}` and :math:`\frac{\partial\omega}{\partial\theta} = \frac{\partial\tilde{c}}{\partial\theta}` need not hold during optimization if we concentrate out linear parameters because these are then functions of :math:`\theta`. Fortunately, one can use orthogonality conditions to show that it is fine to treat these parameters as fixed when computing the gradient.


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

where :math:`\sigma_\xi^2`, :math:`\sigma_\omega^2`, and :math:`\sigma_{\xi\omega}` are estimates of the variances and covariance between the structural errors.

Simulation error can be accounted for by resampling agents :math:`r = 1, \dots, R` times, evaluating each :math:`\bar{g}_r`, and adding the following to :math:`S`:

.. math:: \frac{1}{R - 1} \sum_{r=1}^R (\bar{g}_r - \bar{\bar{g}})(\bar{g}_r - \bar{\bar{g}})', \quad \bar{\bar{g}} = \frac{1}{R} \sum_{r=1}^R \bar{g}_r.
   :label: simulation_S


Standard Errors
~~~~~~~~~~~~~~~

An estimate of the asymptotic covariance matrix of :math:`\sqrt{N}(\hat{\theta} - \theta_0)` is

.. math:: (\bar{G}'W\bar{G})^{-1}\bar{G}'WSW\bar{G}(\bar{G}'W\bar{G})^{-1}.
   :label: covariances

Standard errors are the square root of the diagonal of this matrix divided by :math:`N`.

If the weighting matrix was chosen such that :math:`W = S^{-1}`, this simplifies to

.. math:: (\bar{G}'W\bar{G})^{-1}.
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

More detailed micro data on individual choices can be used to supplement the standard demand- and supply-side moments :math:`\bar{g}_D` and :math:`\bar{g}_S` in :eq:`averaged_moments` with an additional :math:`m = 1, 2, \ldots, M_M` micro moments, :math:`\bar{g}_M`, for a total of :math:`M = M_D + M_S + M_M` moments:

.. math:: \bar{g} = \begin{bmatrix} \bar{g}_D \\ \bar{g}_S \\ \bar{g}_M \end{bmatrix}.

:ref:`references:Conlon and Gortmaker (2023)` provides a standardized framework for incorporating micro moments into BLP-style estimation. What follows is a simplified summary of this framework. Each micro moment :math:`m` is the difference between an observed value :math:`f_m(\bar{v})` and its simulated analogue :math:`f_m(v)`:

.. math:: \bar{g}_{M,m} = f_m(\bar{v}) - f_m(v),
    :label: micro_moment

in which :math:`f_m(\cdot)` is a function that maps a vector of :math:`p = 1, \ldots, P_M` micro moment parts :math:`\bar{v} = (\bar{v}_1, \dots, \bar{v}_{P_M})'` or :math:`v = (v_1, \dots, v_{P_M})'` into a micro statistic. Each sample micro moment part :math:`p` is an average over observations :math:`n \in N_{d_m}` in the associated micro dataset :math:`d_p`:

.. math:: \bar{v}_p = \frac{1}{N_{d_p}} \sum_{n \in N_{d_p}} v_{pi_nj_nt_n}.
    :label: observed_micro_part

Its simulated analogue is

.. math:: v_p = \frac{\sum_{t \in T} \sum_{i \in I_t} \sum_{j \in J_t \cup \{0\}} w_{it} s_{ijt} w_{d_pijt} v_{pijt}}{\sum_{t \in T} \sum_{i \in I_t} \sum_{j \in J_t \cup \{0\}} w_{it} s_{ijt} w_{d_pijt}},
    :label: simulated_micro_part

In which :math:`w_{it} s_{ijt} w_{d_pijt}` is the probability an observation in the micro dataset is for an agent :math:`i` who chooses :math:`j` in market :math:`t`.

The simplest type of micro moment is just an average over the entire sample, with :math:`f_m(v) = v_1`. For example, with :math:`v_{1ijt}` equal to the income for an agent :math:`i` who chooses :math:`j` in market :math:`t`, micro moment :math:`m` would match the average income in dataset :math:`d_p`. Observed values such as conditional expectations, covariances, correlations, or regression coefficients can be matched by choosing the appropriate function :math:`f_m`. For example, with :math:`v_{2ijt}` equal to the interaction between income and an indicator for the choice of the outside option, and with :math:`v_{3ijt}` equal to an indicator for the choiced of the outside option, :math:`f_m(v) = v_2 / v_3` would match an observed conditional mean income within those who choose the outside option.

A micro dataset :math:`d`, often a survey, is defined by survey weights :math:`w_{dijt}`. For example, :math:`w_{dijt} = 1\{j \neq 0, t \in T_d\}` defines a micro dataset that is a selected sample of inside purchasers in a few markets :math:`T_d \subset T`, giving each market an equal sampling weight. Different micro datasets are independent.

A micro dataset will often admit multiple micro moment parts. Each micro moment part :math:`p` is defined by its dataset :math:`d_p` and micro values :math:`v_{pijt}`. For example, a micro moment part :math:`p` with :math:`v_{pijt} = y_{it}x_{jt}` delivers the mean :math:`\bar{v}_p` or expectation :math:`v_p` of an interaction between some demographic :math:`y_{it}` and some product characteristic :math:`x_{jt}`.

A micro moment is a function of one or more micro moment parts. The simplest type is a function of only one micro moment part, and matches the simple average defined by the micro moment part. For example, :math:`f_m(v) = v_p` with :math:`v_{pijt} = y_{it} x_{jt}` matches the mean of an interaction between :math:`y_{it}` and :math:`x_{jt}`. Non-simple averages such as conditional means, covariances, correlations, or regression coefficients can be matched by choosing an appropriate function :math:`f_m`. For example, :math:`f_m(v) = v_1 / v_2` with :math:`v_{1ijt} = y_{it}x_{jt}1\{j \neq 0\}` and :math:`v_{2ijt} = 1\{j \neq 0\}` matches the conditional mean of an interaction between :math:`y_{it}` and :math:`x_{jt}` among those who do not choose the outside option :math:`j = 0`.

Technically, if not all micro moments :math:`m` are simple averages :math:`f_m(v) = v_m`, then the resulting estimator will no longer be a GMM estimator, but rather a more generic minimum distance estimator, since these "micro moments" are not technically sample moments. Regardless, the package uses GMM terminology for simplicity's sake, and the statistical expressions are all the same. Micro moments are computed for each :math:`\theta` and contribute to the GMM (or minimum distance) objective :math:`q(\theta)` in :eq:`objective`. Their derivatives with respect to :math:`\theta` are added as rows to :math:`\bar{G}` in :eq:`averaged_moments_jacobian`, and blocks are added to both :math:`W` and :math:`S` in :eq:`2sls_W` and :eq:`W`. The covariance between standard moments and micro moments is zero, so these matrices are block-diagonal. The delta method delivers the covariance matrix for the micro moments:

.. math:: S_M = \frac{\partial f(v)}{\partial v'} S_P \frac{\partial f(v)'}{\partial v}.
   :label: scaled_micro_moment_covariances

The scaled covariance between micro moment parts :math:`p` and :math:`q` in :math:`S_P` is zero if they are based on different micro datasets :math:`d_p` \neq d_q`; otherwise, if based on the same dataset :math:`d_p = d_q = d`,

.. math:: S_{P,pq} = \frac{N}{N_d} \text{Cov}(v_{pi_nj_nt_n}, v_{qi_nj_nt_n}),
   :label: scaled_micro_part_covariance

in which

.. math:: \text{Cov}(v_{pi_nj_nt_n}, v_{qi_nj_nt_n}) = \frac{\sum_{t \in T} \sum_{i \in I_t} \sum_{j \in J_t \cup \{0\}} w_{it} s_{ijt} w_{dijt} (v_{pijt} - v_p)(v_{qijt} - v_q)}{\sum_{t \in T} \sum_{i \in I_t} \sum_{j \in J_t \cup \{0\}} w_{it} s_{ijt} w_{dijt}}.
    :label: micro_part_covariance

Micro moment parts based on second choice are averages over values :math:`v_{pijkt}` where :math:`k` indexes second choices, and are based on datasets defined by survey weights :math:`w_{dijkt}`. A sample micro moment part is

.. math:: \bar{v}_p = \frac{1}{N_{d_p}} \sum_{n \in N_{d_p}} v_{pi_nj_nk_nt_n}.

Its simulated analogue is

.. math:: v_p = \frac{\sum_{t \in T} \sum_{i \in I_t} \sum_{j, k \in J_t \cup \{0\}} w_{it} s_{ijt} s_{ik(-j)t} w_{d_pijkt} v_{pijkt}}{\sum_{t \in T} \sum_{i \in I_t} \sum_{j, k \in J_t \cup \{0\}} w_{it} s_{ijt} s_{ik(-j)t} w_{d_pijkt}},

in which :math:`s_{ik(-j)t}` is the probability of choosing :math:`k` when :math:`j` is removed from the choice set. One can also define micro moment parts based on second choices where a group of products :math:`h(j)` containing the first choice :math:`j` is removed from the choice set. In this case, the above second choice probabilities become :math:`s_{ik(-h(j))t}`.

Covariances are defined analogously.


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

.. math:: \Gamma_{jk} \approx \sum_{i \in I_t} w_{it} s_{ijt}s_{ikt}\frac{\partial U_{ikt}}{\partial p_{kt}}.

Equilibrium prices are computed by iterating over the :math:`\zeta`-markup equation in :eq:`zeta`,

.. math:: p \leftarrow c + \zeta(p),
   :label: zeta_contraction

which, unlike :eq:`eta`, is a contraction. Iteration terminates when the norm of firms' first order conditions, :math:`||\Lambda(p)(p - c - \zeta(p))||`, is less than a small number.

If marginal costs depend on quantity, then they also depend on prices and need to be updated during each iteration: :math:`c_{jt} = c_{jt}(s_{jt}(p))`.
