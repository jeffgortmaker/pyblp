Background
==========

The following sections provide a brief overview of the BLP model and how it is estimated.


The Model
---------

At a high level, there are :math:`t = 1, 2, \dotsc, T` markets, each with :math:`j = 1, 2, \dotsc, J_t` products produced by :math:`f = 1, 2, \dotsc, F_t` firms. There are :math:`i = 1, 2, \dotsc, I_t` agents who choose among the :math:`J_t` products and an outside good, denoted by :math:`j = 0`. The set :math:`\mathscr{J}_{ft} \subset \{1, 2, \ldots, J_t\}` denotes the products produced by firm :math:`f` in market :math:`t`.


Demand-Side
~~~~~~~~~~~

Observed demand-side product characteristics are contained in the :math:`N \times K_1` matrix of linear characteristics, :math:`X_1`, and the :math:`N \times K_2` matrix of nonlinear characteristics, :math:`X_2`. Characteristic columns can overlap. For example, either matrix may contain a constant column and both matrices often contain a column of prices. Unobserved demand-side product characteristics, :math:`\xi`, are a :math:`N \times 1` column vector.

In market :math:`t`, observed agent characteristics are a :math:`I_t \times D` matrix called demographics, :math:`d`. Note that in :ref:`Nevo (2000) <n00>`, :math:`d` refers to the number of demographics and :math:`D` to the matrix; the opposite is employed here so that all numbers are capital letters. Unobserved characteristics are a :math:`I_t \times K_2` matrix, :math:`\nu`, which consists of rows that are usually assumed to be independent draws from a mean-zero multivariate normal distribution.

The indirect utility of agent :math:`i` from consuming product :math:`j` in market :math:`t` is

.. math:: U_{jti} = \delta_{jt} + \mu_{jti} + \epsilon_{jti},
   :label: utilities

in which the mean utility for products is

.. math:: \delta = X_1\beta + \xi,

and the type-specific portion for all products and agents in a single market is

.. math:: \mu = X_2(\Sigma\nu' + \Pi d').

The :math:`K_2 \times K_2` matrix :math:`\Sigma` is the Cholesky decomposition of the covariance matrix that defines the multivariate normal distribution from which each :math:`\nu_i` is assumed to be drawn. The :math:`K_2 \times D` matrix :math:`\Pi` measures how agent tastes vary with demographics.

Random idiosyncratic preferences :math:`\epsilon_{jti}` are assumed to be Type I Extreme Value so that market shares can be approximated with Monte Carlo integration or quadrature rules as in :ref:`Heiss and Winschel (2008) <hw08>` and :ref:`Skrainka and Judd (2011) <sj11>`:

.. math:: s_{jt} = \sum_{i=1}^{I_t} w_i s_{jti},
   :label: shares

in which :math:`w` is a :math:`I_t \times 1` column vector of integration weights and the probability that agent :math:`i` chooses product :math:`j` in market :math:`t` is

.. math:: s_{jti} = \frac{\exp(\delta_{jt} + \mu_{jti})}{1 + \sum_{k=1}^{J_t} \exp(\delta_{kt} + \mu_{kti})}.
   :label: probabilities


Supply-Side
~~~~~~~~~~~

Observed supply-side product characteristics are the :math:`N \times K_3` matrix of cost characteristics, :math:`X_3`. Prices cannot be cost characteristics, but non-price product characteristics often overlap with the demand-side characteristics in :math:`X_1` and :math:`X_2`. Unobserved supply-side product characteristics, :math:`\omega`, are a :math:`N \times 1` column vector. Note that in contrast to :ref:`Berry, Levinsohn, and Pakes (1995) <blp95>`, the notation for observed cost characteristics is similar to the notation for demand-side characteristics.

Firms play a differentiated Bertrand-Nash pricing game. Firm :math:`f` produces a subset :math:`\mathscr{J}_{ft} \subset \{1, 2, \ldots, J_t\}` of the products in market :math:`t` and chooses prices to maximize the sum of population-normalized gross expected profits, which for product :math:`j` in market :math:`t` are

.. math:: \pi_{jt} = (p_{jt} - c_{jt})s_{jt},

in which marginal costs for all products are defined according to either a linear or a log-linear specification:

.. math:: \tilde{c} = X_3\gamma + \omega \quad\text{where}\quad \tilde{c} = c \quad\text{or}\quad \tilde{c} = \log c.
   :label: costs

The :math:`K_3 \times 1` column vector :math:`\gamma` measures how marginal costs vary with cost characteristics. Regardless of how marginal costs are specified, the first-order conditions of firms in a market can be rewritten after suppressing market subscripts as

.. math:: p = c + \eta.
   :label: eta_markup

Called the BLP-markup equation in :ref:`Morrow and Skerlos (2011) <ms11>`, the markup term is

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

Derivatives in these expressions are derived from the definition of :math:`U` in :eq:`utilities`. An alternative form of the first-order conditions is called the :math:`\zeta`-markup equation in :ref:`Morrow and Skerlos (2011) <ms11>`:

.. math:: p = c + \zeta,
   :label: zeta_markup

in which the markup term is

.. math:: \zeta = \Lambda^{-1}(O \circ \Gamma)'(p - c) - \Lambda^{-1}.
   :label: zeta


Identification
~~~~~~~~~~~~~~

The unobserved product characteristics can be stacked to form a combined error term,

.. math:: u = \begin{bmatrix} \xi \\ \omega \end{bmatrix},

and similarly, :math:`Z_D` and :math:`Z_S`, which are :math:`N \times M_D` and :math:`N \times M_S` matrices of demand- and supply-side instruments, can be stacked to form a combined block-diagonal matrix of instruments,

.. math:: Z = \begin{bmatrix} Z_D & 0 \\ 0 & Z_S \end{bmatrix}.

The GMM moment conditions are

.. math:: \mathrm{E}[g_i] = 0 \quad\text{where}\quad g_i = u_iZ_i.
   :label: moments

Demand-side instruments include all non-price product characteristics from :math:`X_1` and :math:`X_2`, and supply-side instruments include :math:`X_3`. Since cost characteristics are often good demand-side instruments and vice versa, both :math:`Z_D` and :math:`Z_S` often include all characteristics.


Estimation
----------

There are four sets of parameters to be estimated: :math:`\beta`, :math:`\Sigma`, :math:`\Pi`, and :math:`\gamma`. If the supply side is not considered, only the first three sets of parameters are estimated. The linear parameters, :math:`\beta` and :math:`\gamma`, are concentrated out of the problem. Unknown elements in the remaining matrices of nonlinear parameters, :math:`\Sigma` and :math:`\Pi`, are collectively referred to as :math:`\theta`, a :math:`P \times 1` column vector. If demographics are not considered, :math:`\theta` will only consist of elements from :math:`\Sigma`.

The GMM problem is

.. math:: \min_\theta u'ZWZ'u,
   :label: objective

in which :math:`W` is a combined block-diagonal weighting matrix that consists of separate demand- and supply-side weighting matrices,

.. math:: W = \begin{bmatrix} W_D & 0 \\ 0 & W_S \end{bmatrix},

which is assumed to have an inverse that is a consistant estimate of :math:`\mathrm{E}[Z'uu'Z]`.

If only the demand side is considered, :math:`u = \xi`, :math:`Z = Z_D`, and :math:`W = W_D`.

Conventionally, the 2SLS weighting matrix, :math:`W = (Z'Z)^{-1}`, is used in the first stage. With two-step or iterated GMM, the weighting matrix is updated before each subsequent stage according to :math:`W = S^{-1}`. For robust weighting matrices, :math:`S = N^{-1}g'g`. For clustered weighting matrices, which account for arbitrary correlation within :math:`c = 1, 2, \dotsc, C` clusters,

.. math:: S = N^{-1}\sum_{c=1}^C q_cq_c'.

where, letting the set :math:`\mathscr{J}_c \subset \{1, 2, \ldots, N\}` denote the products in cluster :math:`c`,

.. math:: q_c = \sum_{j\in\mathscr{J}_c} g_j.

Before being used to update the weighting matrix, the sample moments are often centered. That is, :math:`g - \bar{g}` is often used instead.

On the other hand, for unadjusted weighting matrices, the instruments are simply scaled by the estimated variance of the error term:

.. math:: S = N^{-1} \hat{\sigma}_u^2 Z'Z \quad\text{where}\quad \hat{\sigma}_u^2 = N^{-1}(u - \bar{u})'(u - \bar{u})

In each stage, a nonlinear optimizer is used to find values of :math:`\hat{\theta}` that minimize the GMM objective. The gradient of the objective is typically computed to speed up optimization.


The Objective
~~~~~~~~~~~~~

Given a :math:`\hat{\theta}`, the first step towards computing its associated objective value is computing :math:`\delta(\hat{\theta})` in each market with the following standard contraction:

.. math:: \delta \leftarrow \delta + \log s - \log s(\delta, \hat{\theta})

where :math:`s` are the market's observed shares and :math:`s(\hat{\theta}, \delta)` are shares evaluated at :math:`\hat{\theta}` and the current iteration's :math:`\delta`. As noted in the appendix of :ref:`Nevo (2000) <n00>`, exponentiating both sides of the contraction mapping and iterating over :math:`\exp(\delta)` gives an alternate formulation that can be faster. Conventional starting values are those that solve the Logit model.

The mean utility in conjunction with the demand-side conditional independence assumption in :eq:`moments` is used to recover the demand-side linear parameters with

.. math:: \hat{\beta} = (X_1'Z_DW_DZ_D'X_1)^{-1}X_1'Z_DW_DZ_D'\delta(\hat{\theta}).
   :label: beta

The demand-side linear parameters are in turn are used to recover the unobserved demand-side product characteristics,

.. math:: \xi(\hat{\theta}) = \delta(\hat{\theta}) - X_1\hat{\beta}.
   :label: xi

If the supply side is considered, the BLP-markup equation from :eq:`eta_markup` is employed to compute marginal costs,

.. math:: c(\hat{\theta}) = p - \eta(\hat{\beta}, \hat{\theta}),

and in conjunction with the supply-side conditional independence assumption in :eq:`moments`, marginal costs are used to recover the supply-side linear parameters according to their specification in :eq:`costs` with

.. math:: \hat{\gamma} = (X_3'Z_SW_SZ_S'X_3)^{-1}X_3'Z_SW_SZ_S'\tilde{c}(\hat{\theta}).
   :label: gamma

The supply-side linear parameters are in turn are used to recover the unobserved supply-side product characteristics,

.. math:: \omega(\hat{\theta}) = \tilde{c}(\hat{\theta}) - X_3\hat{\gamma}.
   :label: omega

Finally, interacting the estimated unobserved product characteristics with the instruments gives the GMM objective value in :eq:`objective`.


The Gradient
~~~~~~~~~~~~

The gradient of the GMM objective in :eq:`objective` is

.. math:: 2\left(\frac{\partial u}{\partial\theta}\right)'ZWZ'u,

in which Jacobians of the unobserved product characteristics are stacked to form

.. math:: \frac{\partial u}{\partial\theta} = \begin{bmatrix} \frac{\partial\xi}{\partial\theta} \\ \frac{\partial\omega}{\partial\theta} \end{bmatrix} = \begin{bmatrix} \frac{\partial\delta}{\partial\theta} \\ \frac{\partial\tilde{c}}{\partial\theta} \end{bmatrix}.

The demand-side Jacobian can be computed by writing :math:`\delta` as an implicit function of :math:`s`:

.. math:: \frac{\partial\delta}{\partial\theta} = -\left(\frac{\partial s}{\partial\delta}\right)^{-1}\frac{\partial s}{\partial\theta}.

Derivatives in this expression are derived directly from the definition of :math:`s` in :eq:`shares`.

The supply-side Jacobian can be derived from the BLP-markup equation in :eq:`eta_markup`:

.. math:: \frac{\partial\tilde{c}}{\partial\theta_p} = -\frac{\partial\tilde{c}}{\partial c}\frac{\partial\eta}{\partial\theta}.

The first term in this expression depends on whether marginal costs are defined according either to a linear or a log-linear specification, and the second term is derived from the definition of :math:`\eta` in :eq:`eta`. Specifically, letting :math:`A = O \circ (\Gamma - \Lambda)`,

.. math:: \frac{\partial\eta}{\partial\theta} = -A^{-1}\left(\frac{\partial A}{\partial\theta}\eta + \frac{\partial A}{\partial\xi}\eta\frac{\partial\xi}{\partial\theta}\right),

in which

.. math:: \frac{\partial A}{\partial\theta} = O \circ \left(\frac{\partial\Gamma}{\partial\theta} - \frac{\partial\Lambda}{\partial\theta}\right) \quad\text{and}\quad \frac{\partial A}{\partial\xi} = O \circ \left(\frac{\partial\Gamma}{\partial\xi} - \frac{\partial\Lambda}{\partial\xi}\right)

are derived from the definitions of :math:`\Gamma` and :math:`\Lambda` in :eq:`capital_gamma` and :eq:`capital_lambda`.


Standard Errors
~~~~~~~~~~~~~~~

Computing standard errors requires an estimate of the Jacobian of the moments with respect to :math:`\theta`, :math:`\beta`, and :math:`\gamma`, which is

.. math:: G = N^{-1}Z'\begin{bmatrix} \frac{\partial\xi}{\partial\theta} & \frac{\partial\xi}{\partial\beta} & 0 \\ \frac{\partial\omega}{\partial\theta} & \frac{\partial\omega}{\partial\beta} & \frac{\partial\omega}{\partial\gamma} \end{bmatrix}.

Note that :math:`\partial\xi / \partial\beta = -X_1` and :math:`\partial\omega / \partial\gamma = -X_3`.

Before updating the weighting matrix, standard errors are extracted from

.. math:: \hat{\text{Var}}\begin{pmatrix} \theta \\ \beta \\ \gamma \end{pmatrix} = (G'WG)^{-1}G'WSWG(G'WG)^{-1},

For robust standard errors, :math:`S = N^{-1}g'g`. For clustered standard errors, which account for arbitrary correlation within :math:`c = 1, 2, \dotsc, C` clusters,

.. math:: S = \sum_{c=1}^C q_cq_c'

where, letting the set :math:`\mathscr{J}_c \subset \{1, 2, \ldots, N\}` denote the products in cluster :math:`c`,

.. math:: q_c = \sum_{j\in\mathscr{J}_c} g_j.

If the weighting matrix was chosen such that :math:`W = S^{-1}`, then

.. math:: \hat{\text{Var}}\begin{pmatrix} \hat{\theta} \\ \hat{\beta} \\ \hat{\gamma} \end{pmatrix} = (G'WG)^{-1}.

Standard errors extracted from an estimate of this last expression are called unadjusted. One caveat is that after only one GMM step, the above expression for the unadjusted covariance matrix is missing the estimated variance of the error term. In this one case, :math:`W` is replaced with an updated unadjusted weighting matrix. Doing so properly scales the expression.


Fixed Effect Absorption
~~~~~~~~~~~~~~~~~~~~~~~

One way to include demand-side fixed effects is to construct a large number of indicator variables and include them in :math:`X_1` and :math:`Z_D`. Similarly, indicator variables can be added to :math:`X_3` and :math:`Z_S` to incorporate supply-side fixed effects. However, this approach becomes infeasible when there are a large amount of data or a large number of fixed effects because estimation with many indicator variables can be both memory- and processor-intensive. In particular, inversion of large matrices in :eq:`beta` and :eq:`gamma` can be problematic.

An alternative is to absorb or partial out fixed effects. If there is only one demand-side fixed effect, that is, if :math:`E_D = 1`, the procedure is simple and efficient: :math:`X_1`, :math:`Z_D`, and :math:`\delta(\hat{\theta})` are de-meaned within each level of the fixed effect. If there is only one supply-side effect, that is, if :math:`E_S = 1`, the same is done with :math:`X_3`, :math:`Z_S`, and :math:`\tilde{c}(\hat{\theta})`.

Estimates computed with the de-meaned or residualized data are guaranteed by the Frish-Waugh-Lovell (FWL) theorem (:ref:`Frisch and Waugh, 1933 <fw33>`; :ref:`Lovell, 1963 <l63>`) to be the same as estimates computed when fixed effects are included as indicator variables.

When :math:`E_D > 1` or :math:`E_S > 1`, the iterative de-meaning algorithm of :ref:`Rios-Avila (2015) <r15>` can be applied to absorb the multiple fixed effects. Iterative de-meaning can be processor-intensive, but for large amounts of data or for large numbers of fixed effects, it is often preferable to including indicator variables. When :math:`E_D = 2` or :math:`E_S = 2`, the more performant algorithm of :ref:`Somaini and Wolak (2016) <sw16>` can be used instead.


Random Coefficients Nested Logit
--------------------------------

Incorporating parameters that measure within nesting group correlation gives rise to the random coefficients nested logit (RCNL) model described, for example, by :ref:`Grigolon and Verboven (2014) <gv14>`. In this model, there are :math:`h = 1, 2, \dotsc, H` nesting groups and each product :math:`j` is assigned to a group :math:`h(j)`. The set :math:`\mathscr{J}_{ht} \subset \{1, 2, \ldots, J_t\}` denotes the products in group :math:`h` and market :math:`t`.

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

To efficiently compute equilibrium prices, the :math:`\zeta`-markup equation from :ref:`Morrow and Skerlos (2011) <ms11>` in :eq:`zeta_markup` is employed in the following contraction:

.. math:: p \leftarrow c + \zeta(p).

When computing :math:`\zeta(p)`, shares :math:`s(p)` associated with the candidate equilibrium prices are computed according to their definition in :eq:`shares`.

Of course, marginal costs, :math:`c`, are required to iterate over the contraction. When evaluating counterfactuals, costs are usually computed first according to the BLP-markup equation in :eq:`eta_markup`. When simulating synthetic data, marginal costs are simulated according their specification in :eq:`costs`.
