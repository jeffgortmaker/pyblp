Notation
========

The notation in pyblp is a customized amalgamation of the notation employed by :ref:`references:Berry, Levinsohn, and Pakes (1995)`, :ref:`references:Nevo (2000)`, :ref:`references:Morrow and Skerlos (2011)`, :ref:`references:Grigolon and Verboven (2014)`, and others.


Indices
-------

=========  =====================
Index      Description
=========  =====================
:math:`j`  Products
:math:`t`  Markets
:math:`i`  Agents or individuals
:math:`f`  Firms
:math:`h`  Nests
:math:`c`  Clusters
:math:`m`  Micro moments
=========  =====================


Dimensions
----------

========================  ==================================================================================================
Dimension                 Description
========================  ==================================================================================================
:math:`T`                 Markets
:math:`N`                 Products across all markets
:math:`F`                 Firms across all markets
:math:`I`                 Agents across all markets
:math:`J_t`               Products in market :math:`t`
:math:`F_t`               Firms in market :math:`t`
:math:`I_t`               Agents in market :math:`t`
:math:`K_1`               Linear product characteristics
:math:`K_1^x`             Exogenous linear product characteristics
:math:`K_1^p`             Endogenous linear product characteristics
:math:`K_2`               Nonlinear product characteristics
:math:`K_3`               Cost product characteristics
:math:`D`                 Demographic variables
:math:`M_D`               Demand-side instruments, which is the number of exluded demand-side instruments plus :math:`K_1^x`
:math:`M_S`               Supply-side instruments, which is the number of exluded supply-side instruments plus :math:`K_3`
:math:`M_M`               Micro moments
:math:`M`                 Moments: the sum of :math:`M_D`, :math:`M_S`, and :math:`M_M`
:math:`E_D`               Absorbed dimensions of demand-side fixed effects
:math:`E_S`               Absorbed dimensions of supply-side fixed effects
:math:`H`                 Nesting groups
:math:`C`                 Clusters
:math:`P`                 Parameters
========================  ==================================================================================================


Sets
----

========================  ========================================================
Set                       Description
========================  ========================================================
:math:`\mathscr{J}_{ft}`  Products produced by firm :math:`f` in market :math:`t`
:math:`\mathscr{J}_{ht}`  Products in nesting group :math:`h` and market :math:`t`
:math:`\mathscr{J}_{ct}`  Products in cluster :math:`c` and market :math:`t`
:math:`\mathscr{T}_m`     Markets over which micro moment :math:`m` is averaged
========================  ========================================================


Matrices, Vectors, and Scalars
------------------------------

=====================================================  ============================  ====================================================================================
Symbol                                                 Dimensions                    Description
=====================================================  ============================  ====================================================================================
:math:`X_1`                                            :math:`N \times K_1`          Linear product characteristics
:math:`X_1^x`                                          :math:`N \times K_1^x`        Exogenous linear product characteristics
:math:`X_1^p`                                          :math:`N \times K_1^p`        Endogenous linear product characteristics
:math:`X_2`                                            :math:`N \times K_2`          Nonlinear product characteristics
:math:`X_3`                                            :math:`N \times K_3`          Cost product characteristics
:math:`\xi`                                            :math:`N \times 1`            Unobserved demand-side product characteristics
:math:`\omega`                                         :math:`N \times 1`            Unobserved supply-side product characteristics
:math:`p`                                              :math:`N \times 1`            Prices
:math:`s` (:math:`s_{jt}`)                             :math:`N \times 1`            Marketshares
:math:`s` (:math:`s_{ht}`)                             :math:`H \times 1`            Group shares in a market :math:`t`
:math:`s` (:math:`s_{jti}`)                            :math:`N \times I_t`          Choice probabiltiies in a market :math:`t`
:math:`c`                                              :math:`N \times 1`            Marginal costs
:math:`\tilde{c}`                                      :math:`N \times 1`            Linear or log-linear marginal costs, :math:`c` or :math:`\log c` 
:math:`\eta`                                           :math:`N \times 1`            Markup term from the BLP-markup equation
:math:`\zeta`                                          :math:`N \times 1`            Markup term from the :math:`\zeta`-markup equation
:math:`O`                                              :math:`J_t \times J_t`        Ownership matrix in market :math:`t`
:math:`\kappa`                                         :math:`F_t \times F_t`        Cooperation matrix in market :math:`t`
:math:`\Delta`                                         :math:`J_t \times J_t`        Intra-firm matrix of (negative) demand derivatives in market :math:`t`
:math:`\Lambda`                                        :math:`J_t \times J_t`        Diagonal matrix used to decompose :math:`\eta` and :math:`\zeta` in market :math:`t`
:math:`\Gamma`                                         :math:`J_t \times J_t`        Another matrix used to decompose :math:`\eta` and :math:`\zeta` in market :math:`t`
:math:`d`                                              :math:`I_t \times D`          Observed agent characteristics called demographics in market :math:`t`
:math:`\nu`                                            :math:`I_t \times K_2`        Unobserved agent characteristics called integration nodes in market :math:`t`
:math:`w`                                              :math:`I_t \times 1`          Integration weights in market :math:`t`
:math:`\delta`                                         :math:`N \times 1`            Mean utility
:math:`\mu`                                            :math:`J_t \times I_t`        Agent-specific portion of utility in market :math:`t`
:math:`\epsilon`                                       :math:`N \times 1`            Type I Extreme Value idiosyncratic preferences
:math:`\bar{\epsilon}` (:math:`\bar{\epsilon}_{jti}`)  :math:`N \times 1`            Type I Extreme Value term used to decompose :math:`\epsilon`
:math:`\bar{\epsilon}` (:math:`\bar{\epsilon}_{hti}`)  :math:`N \times 1`            Group-specific term used to decompose :math:`\epsilon`
:math:`U`                                              :math:`J_t \times I_t`        Indirect utilities
:math:`V` (:math:`V_{jti}`)                            :math:`J_t \times I_t`        Indirect utilities minus :math:`\epsilon`
:math:`V` (:math:`V_{hti}`)                            :math:`J_t \times I_t`        Inclusive value of a nesting group
:math:`\pi` (:math:`\pi_{jt}`)                         :math:`N \times 1`            Population-normalized gross expected profits
:math:`\pi` (:math:`\pi_{ft}`)                         :math:`F_t \times 1`          Population-normalized gross expected profits of a firm in market :math:`t`
:math:`\beta`                                          :math:`K_1 \times 1`          Demand-side linear parameters
:math:`\beta^x`                                        :math:`K_1^x \times 1`        Parameters in :math:`\beta` on exogenous product characteristics
:math:`\alpha`                                         :math:`K_1^p \times 1`        Parameters in :math:`\beta` on endogenous product characteristics
:math:`\Sigma`                                         :math:`K_2 \times K_2`        Cholesky root of the covariance matrix for unobserved taste heterogeneity
:math:`\Pi`                                            :math:`K_2 \times D`          Parameters that measures how agent tastes vary with demographics
:math:`\rho`                                           :math:`H \times 1`            Parameters that measures within nesting group correlation
:math:`\gamma`                                         :math:`K_3 \times 1`          Supply-side linear parameters
:math:`\theta`                                         :math:`P \times 1`            Parameters
:math:`Z_D`                                            :math:`N \times M_D`          Excluded demand-side instruments and :math:`X_1`, except for :math:`X_1^p`
:math:`Z_S`                                            :math:`N \times M_S`          Excluded supply-side instruments and :math:`X_3`
:math:`W`                                              :math:`M \times M`            Weighting matrix
:math:`S`                                              :math:`M \times M`            Moment covariances
:math:`q`                                              :math:`1 \times 1`            Objective value
:math:`g_D`                                            :math:`N \times M_D`          Demand-side moments
:math:`g_S`                                            :math:`N \times M_S`          Supply-side moments
:math:`g_M`                                            :math:`T \times M_M`          Micro moments
:math:`g` (:math:`g_{jt}`)                             :math:`N \times (M_D + M_S)`  Demand- and supply-side moments
:math:`g` (:math:`g_c`)                                :math:`C \times (M_D + M_S)`  Clustered demand- and supply-side moments
:math:`\bar{g}_D`                                      :math:`M_D \times 1`          Averaged demand-side moments
:math:`\bar{g}_S`                                      :math:`M_S \times 1`          Averaged supply-side moments
:math:`\bar{g}_M`                                      :math:`M_M \times 1`          Averaged micro moments
:math:`\bar{g}`                                        :math:`M \times 1`            Averaged moments
:math:`\bar{G}`                                        :math:`M \times P`            Jacobian of the averaged moments with respect to :math:`\theta`
:math:`\varepsilon`                                    :math:`J_t \times J_t`        Elasticities of demand in market :math:`t`
:math:`\mathscr{D}`                                    :math:`J_t \times J_t`        Diversion ratios in market :math:`t`
:math:`\bar{\mathscr{D}}`                              :math:`J_t \times J_t`        Long-run diversion ratios in market :math:`t`
:math:`\mathscr{M}`                                    :math:`N \times 1`            Markups
:math:`\mathscr{E}`                                    :math:`1 \times 1`            Aggregate elasticity of demand of a market
:math:`\text{CS}`                                      :math:`1 \times 1`            Population-normalized consumer surplus of a market
:math:`\text{HHI}`                                     :math:`1 \times 1`            Herfindahl-Hirschman Index of a market
=====================================================  ============================  ====================================================================================
