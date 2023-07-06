Notation
========

The notation in PyBLP is a customized amalgamation of the notation employed by :ref:`references:Berry, Levinsohn, and Pakes (1995)`, :ref:`references:Nevo (2000a)`, :ref:`references:Morrow and Skerlos (2011)`, :ref:`references:Grigolon and Verboven (2014)`, and others.


Indices
-------

=========  ==================
Index      Description
=========  ==================
:math:`j`  Products
:math:`t`  Markets
:math:`i`  Agents/individuals
:math:`f`  Firms
:math:`h`  Nests
:math:`c`  Clusters
:math:`m`  Micro moments
=========  ==================


Dimensions/Sets
---------------

=====================  ==========================================================================
Dimension/Set          Description
=====================  ==========================================================================
:math:`T`              Markets
:math:`N`              Products across all markets
:math:`F`              Firms across all markets
:math:`I`              Agents across all markets
:math:`J_t`            Products in market :math:`t`
:math:`F_t`            Firms in market :math:`t`
:math:`J_{ft}`         Products produced by firm :math:`f` in market :math:`t`
:math:`I_t`            Agents in market :math:`t`
:math:`K_1`            Demand-side linear product characteristics
:math:`K_1^\text{ex}`  Exogenous demand-side linear product characteristics
:math:`K_1^\text{en}`  Endogenous demand-side linear product characteristics
:math:`K_2`            Demand-side nonlinear product characteristics
:math:`K_3`            Supply-side product characteristics
:math:`K_3^\text{ex}`  Exogenous supply-side product characteristics
:math:`K_3^\text{en}`  Endogenous supply-side product characteristics
:math:`D`              Demographic variables
:math:`M_D`            Demand-side instruments
:math:`M_S`            Supply-side instruments
:math:`M_C`            Covariance instruments
:math:`M_M`            Micro moments
:math:`T_m`            Markets over which micro moment :math:`m` is averaged
:math:`T_{mn}`         Markets over which micro moments :math:`m` and :math:`n` are both averaged
:math:`N_m`            Observations underlying observed micro moment value :math:`m`.
:math:`M`              All moments
:math:`E_D`            Absorbed dimensions of demand-side fixed effects
:math:`E_S`            Absorbed dimensions of supply-side fixed effects
:math:`H`              Nesting groups
:math:`J_{ht}`         Products in nesting group :math:`h` and market :math:`t`
:math:`C`              Clusters
:math:`J_{ct}`         Products in cluster :math:`c` and market :math:`t`
=====================  ==========================================================================


Matrices, Vectors, and Scalars
------------------------------

=====================================================  ==================================  ====================================================================================
Symbol                                                 Dimensions                          Description
=====================================================  ==================================  ====================================================================================
:math:`X_1`                                            :math:`N \times K_1`                Demand-side linear product characteristics
:math:`X_1^\text{ex}`                                  :math:`N \times K_1^\text{ex}`      Exogenous demand-side linear product characteristics
:math:`X_1^\text{en}`                                  :math:`N \times K_1^\text{en}`      Endogenous demand-side linear product characteristics
:math:`X_2`                                            :math:`N \times K_2`                Demand-side Nonlinear product characteristics
:math:`X_3`                                            :math:`N \times K_3`                Supply-side product characteristics
:math:`X_3^\text{ex}`                                  :math:`N \times K_3^\text{ex}`      Exogenous supply-side product characteristics
:math:`X_3^\text{en}`                                  :math:`N \times K_3^\text{en}`      Endogenous supply-side product characteristics
:math:`\xi`                                            :math:`N \times 1`                  Unobserved demand-side product characteristics
:math:`\omega`                                         :math:`N \times 1`                  Unobserved supply-side product characteristics
:math:`p`                                              :math:`N \times 1`                  Prices
:math:`s` (:math:`s_{jt}`)                             :math:`N \times 1`                  Market shares
:math:`s` (:math:`s_{ht}`)                             :math:`H \times 1`                  Group shares in a market :math:`t`
:math:`s` (:math:`s_{ijt}`)                            :math:`N \times I_t`                Choice probabilities in a market :math:`t`
:math:`c`                                              :math:`N \times 1`                  Marginal costs
:math:`\tilde{c}`                                      :math:`N \times 1`                  Linear or log-linear marginal costs, :math:`c` or :math:`\log c`
:math:`\eta`                                           :math:`N \times 1`                  Markup term from the BLP-markup equation
:math:`\zeta`                                          :math:`N \times 1`                  Markup term from the :math:`\zeta`-markup equation
:math:`\mathscr{H}`                                    :math:`J_t \times J_t`              Ownership or product holdings matrix in market :math:`t`
:math:`\kappa`                                         :math:`F_t \times F_t`              Cooperation matrix in market :math:`t`
:math:`\Delta`                                         :math:`J_t \times J_t`              Intra-firm matrix of (negative, transposed) demand derivatives in market :math:`t`
:math:`\Lambda`                                        :math:`J_t \times J_t`              Diagonal matrix used to decompose :math:`\eta` and :math:`\zeta` in market :math:`t`
:math:`\Gamma`                                         :math:`J_t \times J_t`              Another matrix used to decompose :math:`\eta` and :math:`\zeta` in market :math:`t`
:math:`d`                                              :math:`I_t \times D`                Observed agent characteristics called demographics in market :math:`t`
:math:`\nu`                                            :math:`I_t \times K_2`              Unobserved agent characteristics called integration nodes in market :math:`t`
:math:`a`                                              :math:`I_t \times J_t`              Agent-specific product availability in market :math:`t`
:math:`w`                                              :math:`I_t \times 1`                Integration weights in market :math:`t`
:math:`\delta`                                         :math:`N \times 1`                  Mean utility
:math:`\mu`                                            :math:`J_t \times I_t`              Agent-specific portion of utility in market :math:`t`
:math:`\epsilon`                                       :math:`N \times 1`                  Type I Extreme Value idiosyncratic preferences
:math:`\bar{\epsilon}` (:math:`\bar{\epsilon}_{ijt}`)  :math:`N \times 1`                  Type I Extreme Value term used to decompose :math:`\epsilon`
:math:`\bar{\epsilon}` (:math:`\bar{\epsilon}_{iht}`)  :math:`N \times 1`                  Group-specific term used to decompose :math:`\epsilon`
:math:`U`                                              :math:`J_t \times I_t`              Indirect utilities
:math:`V` (:math:`V_{ijt}`)                            :math:`J_t \times I_t`              Indirect utilities minus :math:`\epsilon`
:math:`V` (:math:`V_{iht}`)                            :math:`J_t \times I_t`              Inclusive value of a nesting group
:math:`\pi` (:math:`\pi_{jt}`)                         :math:`N \times 1`                  Population-normalized gross expected profits
:math:`\pi` (:math:`\pi_{ft}`)                         :math:`F_t \times 1`                Population-normalized gross expected profits of a firm in market :math:`t`
:math:`\beta`                                          :math:`K_1 \times 1`                Demand-side linear parameters
:math:`\beta^\text{ex}`                                :math:`K_1^\text{ex} \times 1`      Parameters in :math:`\beta` on exogenous product characteristics
:math:`\alpha`                                         :math:`K_1^\text{en} \times 1`      Parameters in :math:`\beta` on endogenous product characteristics
:math:`\Sigma`                                         :math:`K_2 \times K_2`              Cholesky root of the covariance matrix for unobserved taste heterogeneity
:math:`\Pi`                                            :math:`K_2 \times D`                Parameters that measures how agent tastes vary with demographics
:math:`\rho`                                           :math:`H \times 1`                  Parameters that measures within nesting group correlation
:math:`\gamma`                                         :math:`K_3 \times 1`                Supply-side linear parameters
:math:`\gamma^\text{ex}`                               :math:`K_3^\text{ex} \times 1`      Parameters in :math:`\gamma` on exogenous product characteristics
:math:`\gamma^\text{en}`                               :math:`K_3^\text{en} \times 1`      Parameters in :math:`\gamma` on endogenous product characteristics
:math:`\theta`                                         :math:`P \times 1`                  Parameters
:math:`Z_D`                                            :math:`N \times M_D`                Demand-side instruments
:math:`Z_S`                                            :math:`N \times M_S`                Supply-side instruments
:math:`Z_C`                                            :math:`N \times M_C`                Covariance instruments
:math:`W`                                              :math:`M \times M`                  Weighting matrix
:math:`S`                                              :math:`M \times M`                  Moment covariances
:math:`q`                                              :math:`1 \times 1`                  Objective value
:math:`g_D`                                            :math:`N \times M_D`                Demand-side moments
:math:`g_S`                                            :math:`N \times M_S`                Supply-side moments
:math:`g_C`                                            :math:`N \times M_C`                Covariance moments
:math:`g_M`                                            :math:`I \times M_M`                Micro moments
:math:`g` (:math:`g_{jt}`)                             :math:`N \times (M_D + M_S + M_C)`  Demand-side, supply-side, and covariance moments
:math:`g` (:math:`g_c`)                                :math:`C \times (M_D + M_S + M_C)`  Clustered demand-side, supply-side, and covariance moments
:math:`\bar{g}_D`                                      :math:`M_D \times 1`                Averaged demand-side moments
:math:`\bar{g}_S`                                      :math:`M_S \times 1`                Averaged supply-side moments
:math:`\bar{g}_C`                                      :math:`M_C \times 1`                Averaged covariance moments
:math:`\bar{g}_M`                                      :math:`M_M \times 1`                Averaged micro moments
:math:`\bar{g}`                                        :math:`M \times 1`                  Averaged moments
:math:`\bar{G}`                                        :math:`M \times P`                  Jacobian of the averaged moments with respect to :math:`\theta`
:math:`\varepsilon`                                    :math:`J_t \times J_t`              Elasticities of demand in market :math:`t`
:math:`\mathscr{D}`                                    :math:`J_t \times J_t`              Diversion ratios in market :math:`t`
:math:`\bar{\mathscr{D}}`                              :math:`J_t \times J_t`              Long-run diversion ratios in market :math:`t`
:math:`\mathscr{M}`                                    :math:`N \times 1`                  Markups
:math:`\mathscr{E}`                                    :math:`1 \times 1`                  Aggregate elasticity of demand of a market
:math:`\text{CS}`                                      :math:`1 \times 1`                  Population-normalized consumer surplus of a market
:math:`\text{HHI}`                                     :math:`1 \times 1`                  Herfindahl-Hirschman Index of a market
=====================================================  ==================================  ====================================================================================
