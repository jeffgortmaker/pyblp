Background
==========

The following sections provide a brief overview of the BLP model and how it is estimated.


The Model
---------

At a high level, there are :math:`t = 1, 2, \dotsc, T` markets, each with :math:`j = 1, 2, \dotsc, J_t` products produced by :math:`f = 1, 2, \dotsc, F_t` firms. There are :math:`i = 1, 2, \dotsc, I_t` agents who choose among the :math:`J_t` products and an outside good, denoted by :math:`j = 0`. The set :math:`\mathscr{J}_{ft} \subset \{1, 2, \ldots, J_t\}` denotes the products produced by firm :math:`f` in market :math:`t`.


Demand-Side
~~~~~~~~~~~

Observed demand-side product characteristics are contained in the :math:`N \times K_1` matrix of linear characteristics, :math:`X_1`, and the :math:`N \times K_2` matrix of nonlinear characteristics, :math:`X_2`. Characteristic columns can overlap. For example, either matrix may contain a constant column and both matrices often contain a column of prices. Unobserved demand-side product characteristics, :math:`\xi`, are a :math:`N \times 1` column vector.

In market :math:`t`, observed agent characteristics are a :math:`I_t \times D` matrix called demographics, :math:`d`. Note that in :ref:`references:Nevo (2000)`, :math:`d` refers to the number of demographics and :math:`D` to the matrix; the opposite is employed here so that all numbers are capital letters. Unobserved characteristics are a :math:`I_t \times K_2` matrix, :math:`\nu`, which consists of rows that are usually assumed to be independent draws from a mean-zero multivariate normal distribution.

The indirect utility of agent :math:`i` from purchasing product :math:`j` in market :math:`t` is

.. math:: U_{jti} = \delta_{jt} + \mu_{jti} + \epsilon_{jti},
   :label: utilities

in which the mean utility for products is

.. math:: \delta = X_1\beta + \xi,

and the type-specific portion for all products and agents in a single market is

.. math:: \mu = X_2(\Sigma\nu' + \Pi d').

.. note::

   Unlike some of the existing literature, :math:`X_1` includes endogenous characteristics. That is, it includes characteristics that are functions of prices, :math:`p`, and :math:`\beta` includes parameters on these endogenous characteristics. The notation :math:`X_1^p` is used to denote these endogenous characteristics, and :math:`\alpha` is used to denote the parameters (which are a subset of those in :math:`\beta`) on these endogenous characteristics. Often, :math:`X_1^p` is simply a single column of prices.

The model incorporates both observable (demographic) and unobservable taste heterogeneity heterogeneity though random coefficients. For the unobserved heterogeneity, we let :math:`\nu_i` denote draws from a :math:`K_2` vector independent normals. These are scaled by a :math:`K_2 \times K_2` matrix :math:`\Sigma` which denotes Cholesky decomposition of the covariance matrix for the unobserved taste heterogeneity. The :math:`K_2 \times D` matrix :math:`\Pi` denotes how agent tastes vary with demographics.

Random idiosyncratic preferences :math:`\epsilon_{jti}` are assumed to be Type I Extreme Value, so that conditional on the heterogeneous coefficients, marketshares follow the well known logit form. Aggregate marketshares are obtained by integrating out over the distribution of individual heterogeneity.

.. math:: s_{jt} = \int s_{jti}(\mu_{ti}) f(\mu_{it}) \partial \mu_{it}  \approx \sum_{i=1}^{I_t} w_i s_{jti},
   :label: shares

Market shares can be approximated with Monte Carlo integration or quadrature rules as in :ref:`references:Heiss and Winschel (2008)` and :ref:`references:Judd and Skrainka (2011)` in which :math:`w` is a :math:`I_t \times 1` column vector of integration weights and the probability that agent :math:`i` chooses product :math:`j` in market :math:`t` is

.. math:: s_{jti} = \frac{\exp(\delta_{jt} + \mu_{jti})}{1 + \sum_{k=1}^{J_t} \exp(\delta_{kt} + \mu_{kti})}.
   :label: probabilities
   
Supply-Side
~~~~~~~~~~~

Observed supply-side product characteristics are the :math:`N \times K_3` matrix of cost characteristics, :math:`X_3`. Prices cannot be cost characteristics, but non-price product characteristics often overlap with the demand-side characteristics in :math:`X_1` and :math:`X_2`. Unobserved supply-side product characteristics, :math:`\omega`, are a :math:`N \times 1` column vector. In contrast to the notation employed by :ref:`references:Berry, Levinsohn, and Pakes (1995)`, the notation for observed cost characteristics here is similar to the notation for demand-side characteristics.

Firms play a differentiated Bertrand-Nash pricing game. Firm :math:`f` produces a subset :math:`\mathscr{J}_{ft} \subset \{1, 2, \ldots, J_t\}` of the products in market :math:`t` and chooses prices to maximize the sum of population-normalized gross expected profits, which for product :math:`j` in market :math:`t` are

.. math:: \pi_{jt} = (p_{jt} - c_{jt})s_{jt},

in which marginal costs for all products are defined according to either a linear or a log-linear specification:

.. math:: \tilde{c} = X_3\gamma + \omega \quad\text{where}\quad \tilde{c} = c \quad\text{or}\quad \tilde{c} = \log c.
   :label: costs

The :math:`K_3 \times 1` column vector :math:`\gamma` measures how marginal costs vary with cost characteristics. Regardless of how marginal costs are specified, the first-order conditions of firms in a market can be rewritten after suppressing market subscripts as

.. math:: p = c + \eta.
   :label: eta_markup

Called the BLP-markup equation in :ref:`references:Morrow and Skerlos (2011)`, the markup term is

.. math:: \eta = -(O \circ \frac{\partial s}{\partial p})^{-1}s,
   :label: eta

in which the market's owenership matrix, :math:`O`, is definited in terms of its corresponding cooperation matrix, :math:`\kappa`, by :math:`O_{jk} = \kappa_{fg}` where :math:`j \in \mathscr{J}_{ft}` and :math:`g \in \mathscr{J}_{gt}`. Usually, :math:`\kappa = I`, the identity matrix, so :math:`O_{jk}` is simply :math:`1` if the same firm produces products :math:`j` and :math:`k`, and is :math:`0` otherwise.

The Jacobian in the BLP-markup equation can be decomposed into

.. math:: \frac{\partial s}{\partial p} = \Lambda - \Gamma,

in which :math:`\Lambda` is a diagonal :math:`J_t \times J_t` matrix that can be approximated by

.. math:: \Lambda_{jj} = \sum_{i=1}^{I_t} w_i s_{jti}\frac{\partial U_{jti}}{\partial p_{jt}}
   :label: capital_lambda

and :math:`\Gamma` is a more dense :math:`J_t \times J_t` matrix that can be approximated by

.. math:: \Gamma_{jk} = \sum_{i=1}^{I_t} w_i s_{jti}s_{kti}\frac{\partial U_{jti}}{\partial p_{jt}}.
   :label: capital_gamma

Derivatives in these expressions are derived from the definition of :math:`U` in :eq:`utilities`. An alternative form of the first-order conditions is called the :math:`\zeta`-markup equation in :ref:`references:Morrow and Skerlos (2011)`:

.. math:: p = c + \zeta,
   :label: zeta_markup

in which the markup term is

.. math:: \zeta = \Lambda^{-1}(O \circ \Gamma)'(p - c) - \Lambda^{-1}.
   :label: zeta


Identification
~~~~~~~~~~~~~~

The GMM moments, :math:`g`, which are :math:`N \times (M_D + M_S)`, are defined by

.. math:: g_n = \begin{bmatrix} Z_{nD}\xi_n & Z_{nS}\omega_n \end{bmatrix},

in which :math:`Z_D` and :math:`Z_S` are :math:`N \times M_D` and :math:`N \times M_S` matrices of demand- and supply-side instruments.

The moment conditions are

.. math:: \mathrm{E}[g_n] = 0.
   :label: moments

The full set of demand-side instruments include excluded demand-side instruments along with include all exogenous product characteristics from :math:`X_1` (and hence :math:`X_2`), except for those including price, :math:`X_1^p`. Similarly, the full set of supply-side instruments include excluded supply-side instruments along with :math:`X_3`.


Estimation
----------

There are four sets of parameters to be estimated: :math:`\beta` (which may include :math:`\alpha`), :math:`\Sigma`, :math:`\Pi`, and :math:`\gamma`. If the supply side is not considered, only the first three sets of parameters are estimated. The linear parameters, :math:`\beta` and :math:`\gamma`, may be concentrated out of the problem. The exception is :math:`\alpha`, which cannot be concentrated out when there is a supply side because it is needed to compute marginal costs. Linear parameters that are not concentrated out along with unknown elements in the remaining matrices of nonlinear parameters, :math:`\Sigma` and :math:`\Pi`, are collectively referred to as :math:`\theta`, a :math:`P \times 1` column vector.

The GMM problem is

.. math:: \min_\theta \bar{g}'W\bar{g},
   :label: objective

in which :math:`\bar{g}` is the sample mean of the moment conditions and :math:`W` is a weighting matrix. The objective value is scaled by :math:`N^2` for comparability's sake.

Conventionally, the 2SLS weighting matrix is used in the first stage:

.. math:: W = \begin{bmatrix} (Z_D'Z_D)^{-1} & 0 \\ 0 & (Z_S'Z_S)^{-1} \end{bmatrix}.

With two-step or iterated GMM, the weighting matrix is updated before each subsequent stage according to :math:`W = S^{-1}`. For robust weighting matrices,

.. math:: S = \frac{1}{N}\sum_{n=1}^N g_ng_n'.

For clustered weighting matrices, which account for arbitrary correlation within :math:`c = 1, 2, \dotsc, C` clusters,

.. math:: S = \frac{1}{N}\sum_{c=1}^C q_cq_c',

where, letting the set :math:`\mathscr{J}_c \subset \{1, 2, \ldots, N\}` denote the products in cluster :math:`c`,

.. math:: q_c = \sum_{j\in\mathscr{J}_c} g_j.

Before being used to update the weighting matrix, the sample moments are often centered.

On the other hand, for unadjusted weighting matrix, the instruments are simply scaled by estimated error term covariances:

.. math:: S = \frac{1}{N} \begin{bmatrix} \sigma_{\xi}^2 Z_D'Z_D & \sigma_{\xi\omega} Z_D'Z_S \\ \sigma_{\xi\omega} Z_S'Z_D & \sigma_{\omega}^2 Z_S'Z_S \end{bmatrix}

where :math:`\sigma_{\xi}^2` and :math:`\sigma_{\omega}^2` are the sample variances of :math:`\xi` and :math:`\omega`, and :math:`\sigma_{\xi\omega}` is their sample covariance.

In each stage, a nonlinear optimizer is used to find values of :math:`\hat{\theta}` that minimize the GMM objective. The gradient of the objective is typically computed to speed up optimization.


The Objective
~~~~~~~~~~~~~

Given a :math:`\hat{\theta}`, the first step towards computing its associated objective value is computing :math:`\delta(\hat{\theta})` in each market with the following standard contraction:

.. math:: \delta \leftarrow \delta + \log s - \log s(\delta, \hat{\theta})

where :math:`s` are the market's observed shares and :math:`s(\hat{\theta}, \delta)` are shares evaluated at :math:`\hat{\theta}` and the current iteration's :math:`\delta`. As noted in the appendix of :ref:`references:Nevo (2000)`, exponentiating both sides of the contraction mapping and iterating over :math:`\exp(\delta)` gives an alternate formulation that can be faster. Conventional starting values are those that solve the Logit model.

If the supply side is considered, the BLP-markup equation from :eq:`eta_markup` is employed to compute marginal costs,

.. math:: c(\hat{\theta}) = p - \eta(\hat{\theta}).

Unlike when there is only a demand side, :math:`\theta` must contain :math:`\alpha` here because it is needed to compute :math:`\eta`.

The conditional independence assumption in :eq:`moments` is used to recover the concentrated out linear parameters with

.. math:: \begin{bmatrix} \hat{\beta} \\ \hat{\gamma} \end{bmatrix} = (X'ZWZ'X)^{-1}X'ZWZ'y(\hat{\theta}),
   :label: iv

where the linear parameters and instruments are stacked in a block diagonal fashion,

.. math:: X = \begin{bmatrix} X_1 & 0 \\ 0 & X_3 \end{bmatrix} \quad\text{and}\quad Z = \begin{bmatrix} Z_D & 0 \\ 0 & Z_S \end{bmatrix},

and the mean utility along with marginal costs according to their specification in :eq:`costs` are stacked as well,

.. math:: y(\hat{\theta}) = \begin{bmatrix} \delta(\hat{\theta}) \\ \tilde{c}(\hat{\theta}) \end{bmatrix}.

If any linear parameters were not concentrated out but rather included in :math:`\theta` (such as :math:`\alpha`, which cannot be concentrated out when there is a supply side), their contributions are subtracted from :math:`y(\hat{\theta})` before it is used to recover the concentrated out parameters.

The demand-side linear parameters are used to recover the unobserved demand-side product characteristics,

.. math:: \xi(\hat{\theta}) = \delta(\hat{\theta}) - X_1\hat{\beta},
   :label: xi

and the same is done for the supply side,

.. math:: \omega(\hat{\theta}) = \tilde{c}(\hat{\theta}) - X_3\hat{\gamma}.
   :label: omega

Finally, interacting the estimated unobserved product characteristics with the instruments gives the GMM objective value in :eq:`objective`.


The Gradient
~~~~~~~~~~~~

The gradient of the GMM objective in :eq:`objective` is :math:`2\bar{G}'W\bar{g}`, in which :math:`\bar{g}` is the mean of the sample moments and :math:`\bar{G}` is the mean of the Jacobian of the sample moments with respect to :math:`\theta`. This Jacobian, :math:`G`, which is :math:`N \times (M_D + M_S) + P`, is defined by

.. math:: G_n = \begin{bmatrix} Z_{nD}'\frac{\partial\xi}{\partial\theta} \\ Z_{nS}'\frac{\partial\omega}{\partial\theta} \end{bmatrix} = \begin{bmatrix} Z_{nD}'\frac{\partial\delta}{\partial\theta} \\ Z_{nS}'\frac{\partial\tilde{c}}{\partial\theta} \end{bmatrix}.

The demand-side Jacobian can be computed by writing :math:`\delta` as an implicit function of :math:`s`,

.. math:: \frac{\partial\delta}{\partial\theta} = -\left(\frac{\partial s}{\partial\delta}\right)^{-1}\frac{\partial s}{\partial\theta}.

Derivatives in this expression are derived directly from the definition of :math:`s` in :eq:`shares`.

The supply-side Jacobian can be derived from the BLP-markup equation in :eq:`eta_markup`,

.. math:: \frac{\partial\tilde{c}}{\partial\theta_p} = -\frac{\partial\tilde{c}}{\partial c}\frac{\partial\eta}{\partial\theta}.

The first term in this expression depends on whether marginal costs are defined according either to a linear or a log-linear specification, and the second term is derived from the definition of :math:`\eta` in :eq:`eta`. Specifically, letting :math:`A = O \circ (\Gamma - \Lambda)`,

.. math:: \frac{\partial\eta}{\partial\theta} = -A^{-1}\left(\frac{\partial A}{\partial\theta}\eta + \frac{\partial A}{\partial\xi}\eta\frac{\partial\xi}{\partial\theta}\right),

in which

.. math:: \frac{\partial A}{\partial\theta} = O \circ \left(\frac{\partial\Gamma}{\partial\theta} - \frac{\partial\Lambda}{\partial\theta}\right) \quad\text{and}\quad \frac{\partial A}{\partial\xi} = O \circ \left(\frac{\partial\Gamma}{\partial\xi} - \frac{\partial\Lambda}{\partial\xi}\right)

are derived from the definitions of :math:`\Gamma` and :math:`\Lambda` in :eq:`capital_gamma` and :eq:`capital_lambda`.


Standard Errors
~~~~~~~~~~~~~~~

Computing standard errors requires an estimate of the Jacobian of the moments with respect to all the parameters, which is the same as the above expression for :math:`G`, except that it includes terms for concentrated out parameters in :math:`\beta` and :math:`\gamma`, which are relatively simple because :math:`\partial\xi / \partial\beta = -X_1` and :math:`\partial\omega / \partial\gamma = -X_3`.

Before updating the weighting matrix, standard errors are extracted from

.. math:: \hat{\text{Var}}\begin{pmatrix} \hat{\theta} \\ \hat{\beta} \\ \hat{\gamma} \end{pmatrix} = (\bar{G}'W\bar{G})^{-1}\bar{G}'WSW\bar{G}(\bar{G}'W\bar{G})^{-1},

in which :math:`S` is defined in the same way as it is defined when computing the weighting matrix.

If the weighting matrix was chosen such that :math:`W = S^{-1}`, then

.. math:: \hat{\text{Var}}\begin{pmatrix} \hat{\theta} \\ \hat{\beta} \\ \hat{\gamma} \end{pmatrix} = (\bar{G}'W\bar{G})^{-1}.

Standard errors extracted from an estimate of this last expression are called unadjusted. One caveat is that after only one GMM step, the above expression for the unadjusted covariance matrix is missing the estimated variance of the error term. In this one case, :math:`W` is replaced with an updated unadjusted weighting matrix, which properly scales the expression.


Fixed Effect Absorption
~~~~~~~~~~~~~~~~~~~~~~~

One way to include demand-side fixed effects is to construct a large number of indicator variables and include them in :math:`X_1` (and hence in :math:`Z_D`). Similarly, indicator variables can be added to :math:`X_3` (and hence in :math:`Z_S`) to incorporate supply-side fixed effects. However, this approach becomes infeasible when there are a large amount of data or a large number of fixed effects because estimation with many indicator variables can be both memory- and processor-intensive. In particular, inversion of large matrices in :eq:`iv` can be problematic.

An alternative is to absorb or partial out fixed effects. If there is only one demand-side fixed effect, that is, if :math:`E_D = 1`, the procedure is simple and efficient: :math:`X_1`, :math:`Z_D`, and :math:`\delta(\hat{\theta})` are de-meaned within each level of the fixed effect. If there is only one supply-side effect, that is, if :math:`E_S = 1`, the same is done with :math:`X_3`, :math:`Z_S`, and :math:`\tilde{c}(\hat{\theta})`.

Estimates and structural error terms computed with the de-meaned or residualized data are guaranteed by the Frish-Waugh-Lovell (FWL) theorem of :ref:`references:Frisch and Waugh (1933)` and :ref:`references:Lovell (1963)` to be the same as results computed when fixed effects are included as indicator variables.

When :math:`E_D > 1` or :math:`E_S > 1`, the iterative de-meaning algorithm of :ref:`references:Rios-Avila (2015)` can be employed to absorb the multiple fixed effects. Iterative de-meaning can be processor-intensive, but for large amounts of data or for large numbers of fixed effects, it is often preferable to including indicator variables. When :math:`E_D = 2` or :math:`E_S = 2`, the more performant algorithm of :ref:`references:Somaini and Wolak (2016)` can be used instead.


Random Coefficients Nested Logit
--------------------------------

Incorporating parameters that measure within nesting group correlation gives the random coefficients nested logit (RCNL) model described, for example, by :ref:`references:Grigolon and Verboven (2014)`. In this model, there are :math:`h = 1, 2, \dotsc, H` nesting groups and each product :math:`j` is assigned to a group :math:`h(j)`. The set :math:`\mathscr{J}_{ht} \subset \{1, 2, \ldots, J_t\}` denotes the products in group :math:`h` and market :math:`t`.

In the RCNL model, the error term is decomposed into

.. math:: \epsilon_{jti} = \bar{\epsilon}_{h(j)ti} + (1 - \rho_{h(j)})\bar{\epsilon}_{jti},

in which :math:`\bar{\epsilon}_{jti}` is Type I Extreme Value and the group-specific term :math:`\bar{\epsilon}_{h(j)ti}` is distributed such that :math:`\epsilon_{jti}` is also Type I Extreme Value. 

The nesting parameter, :math:`\rho_{h(j)} \in [0, 1]`, measures within nesting group correlation. Collectively, :math:`\rho` can be either a scalar that corresponds to all groups or a :math:`H \times 1` column vector to give each group a different nesting parameter. The standard BLP model arises when :math:`\rho = 0`. On the other hand, setting any :math:`\rho_h = 1` creates division by zero errors during estimation. Values larger than one are inconsistent with utility maximization.

Under nesting, the expression for choice probabilities in :eq:`probabilities` is more complicated:

.. math:: s_{jti} = \frac{\exp[V_{jti} / (1 - \rho_{h(j)})]}{\exp[V_{h(j)ti} / (1 - \rho_{h(j)})]}\cdot\frac{\exp V_{h(j)ti}}{1 + \sum_{h=1}^H \exp V_{hti}}

where

.. math:: V_{jti} = \delta_{jt} + \mu_{jti}

and

.. math:: V_{hti} = (1 - \rho_h)\log\sum_{k\in\mathscr{J}_{ht}} \exp[V_{kti} / (1 - \rho_h)].

During estimation, unknown elements in :math:`\rho` are included in :math:`\theta`. Otherwise, estimation proceeds exactly as described in the above sections, except that expressions derived from definitions of :math:`U` in :eq:`utilities` and :math:`s` in :eq:`shares` are more complicated. In particular, Jacobians are much simpler when :math:`\rho = 0`.


Logit and Nested Logit Benchmarks
---------------------------------

Excluding :math:`X_2` and :math:`\Sigma` leaves the simple Logit model (or the nested Logit model), which serves as a simple benchmark for the full random coefficients BLP model (or the full RCNL model). Although it lacks the realism of the full model, estimation of the Logit or nested Logit model is much simpler. Specifically, a closed-form solution for the mean utility means that fixed point iteration is not required. In the simple Logit model, this solution is

.. math:: \delta_{jt} = \log s_{jt} - \log s_{0t},

and in the nested Logit model, it is

.. math:: \delta_{jt} = \log s_{jt} - \log s_{0t} - \rho_{h(j)}\log\frac{s_{jt}}{s_{h(j)t}}

where

.. math:: s_{h(j)t} = \sum_{k\in\mathscr{J}_{h(j)t}} s_{kt}.

In the simple Logit model, a lack of nonlinear parameters means that optimization is not required either. Importantly, a supply side can still be estimated jointly with demand. The only difference in the above sections, other than the absence of nonlinear characteristics and parameters, is that there is simply a single, representative agent in each market. That is, each :math:`I_t = 1` with :math:`w_1 = 1`.


Bertrand-Nash Prices and Shares
-------------------------------

Computing equilibrium prices and shares is necessary during post-estimation to evaluate counterfactuals such as mergers. Similarly, synthetic data can be simulated in a straightforward manner according to a demand-side specification, but if the data are to simultaneously conform to a supply-side specification as well, it is necessary to compute equilibrium prices and shares that are implied by the other synthetic data.

To efficiently compute equilibrium prices, the :math:`\zeta`-markup equation from :ref:`references:Morrow and Skerlos (2011)` in :eq:`zeta_markup` is employed in the following contraction:

.. math:: p \leftarrow c + \zeta(p).

When computing :math:`\zeta(p)`, shares :math:`s(p)` associated with the candidate equilibrium prices are computed according to their definition in :eq:`shares`.

Of course, marginal costs, :math:`c`, are required to iterate over the contraction. When evaluating counterfactuals, costs are usually computed first according to the BLP-markup equation in :eq:`eta_markup`. When simulating synthetic data, marginal costs are simulated according their specification in :eq:`costs`.
