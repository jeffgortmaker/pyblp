Notation
========

The notation in pyblp is a customized amalgamation of the notation employed by :ref:`references:Berry, Levinsohn, and Pakes (1995)`, :ref:`references:Nevo (2000)`, :ref:`references:Morrow and Skerlos (2011)`, :ref:`references:Grigolon and Verboven (2014)`, and others.


Dimensions and Sets
-------------------

========================  =========================================================================================================
Symbol                    Description
========================  =========================================================================================================
:math:`N`                 Products across all markets.
:math:`T`                 Markets.
:math:`J_t`               Products in market :math:`t`.
:math:`F_t`               Firms in market :math:`t`.
:math:`I_t`               Agents in market :math:`t`.
:math:`K_1`               Linear product characteristics.
:math:`K_1^p`             Endogenous linear product characteristics.
:math:`K_2`               Nonlinear product characteristics.
:math:`K_3`               Cost product characteristics.
:math:`D`                 Demographic variables.
:math:`M_D`               Demand-side instruments, which is the number of exluded demand-side instruments plus :math:`K_1 - K_1^p`.
:math:`M_S`               Supply-side instruments, which is the number of exluded supply-side instruments plus :math:`K_3`.
:math:`E_D`               Absorbed demand-side fixed effects.
:math:`E_S`               Absorbed supply-side fixed effects.
:math:`H`                 Nesting groups.
:math:`C`                 Clustering groups.
:math:`P`                 Parameters.
:math:`\mathscr{J}_c`     Products in cluster :math:`c`.
:math:`\mathscr{J}_{ft}`  Products produced by firm :math:`f` in market :math:`t`.
:math:`\mathscr{J}_{ht}`  Products in group :math:`h` and market :math:`t`.
========================  =========================================================================================================


Matrices, Vectors, and Scalars
------------------------------

Dimensions that can differ across markets are reported for a single market :math:`t`. Some notation differs depending on how it is indexed.

=====================================================  ===================================================  =====================================================================================
Symbol                                                 Dimensions                                           Description
=====================================================  ===================================================  =====================================================================================
:math:`X_1`                                            :math:`N \times K_1`                                 Linear product characteristics.
:math:`X_1^p`                                          :math:`N \times K_1^p`                               Endogenous linear product characteristics.
:math:`X_2`                                            :math:`N \times K_2`                                 Nonlinear product characteristics.
:math:`X_3`                                            :math:`N \times K_3`                                 Cost product characteristics.
:math:`\xi`                                            :math:`N \times 1`                                   Unobserved demand-side product characteristics or structural error term.
:math:`\omega`                                         :math:`N \times 1`                                   Unobserved supply-side product characteristics or structural error term.
:math:`p`                                              :math:`N \times 1`                                   Prices.
:math:`p^*`                                            :math:`N \times 1`                                   Bertrand-Nash prices after firm ID changes.
:math:`p^a`                                            :math:`N \times 1`                                   The same, assuming shares and their price derivatives are unaffected.
:math:`s` (:math:`s_{jt}`)                             :math:`N \times 1`                                   Shares.
:math:`s` (:math:`s_{ht}`)                             :math:`N \times 1`                                   Group shares.
:math:`s` (:math:`s_{jti}`)                            :math:`N \times I_t`                                 Choice probabiltiies.
:math:`c`                                              :math:`N \times 1`                                   Marginal costs.
:math:`\tilde{c}`                                      :math:`N \times 1`                                   Linear or log-linear marginal costs, :math:`c` or :math:`\log c` .
:math:`\eta`                                           :math:`N \times 1`                                   Markup term from the BLP-markup equation.
:math:`\eta^a`                                         :math:`N \times 1`                                   The same, assuming shares and their price derivatives are unaffected.
:math:`\zeta`                                          :math:`N \times 1`                                   Markup term from the :math:`\zeta`-markup equation.
:math:`\zeta^*`                                        :math:`N \times 1`                                   Markup term after firm ID changes from the :math:`\zeta`-markup equation.
:math:`O`                                              :math:`J_t \times J_t`                               Ownership matrix in market :math:`t`.
:math:`O^*`                                            :math:`J_t \times J_t`                               Ownership matrix in market :math:`t` after firm ID changes.
:math:`\kappa`                                         :math:`F_t \times F_t`                               Cooperation matrix in market :math:`t`.
:math:`\kappa^*`                                       :math:`F_t \times F_t`                               Cooperation matrix in market :math:`t` after firm ID changes.
:math:`\Delta`                                         :math:`J_t \times J_t`                               Intra-firm matrix of (negative) demand derivatives in market :math:`t`.
:math:`\Lambda`                                        :math:`J_t \times J_t`                               Diagonal matrix used to decompose :math:`\eta` and :math:`\zeta` in market :math:`t`.
:math:`\Gamma`                                         :math:`J_t \times J_t`                               Another matrix used to decompose :math:`\eta` and :math:`\zeta` in market :math:`t`.
:math:`d`                                              :math:`I_t \times D`                                 Observed agent characteristics called demographics in market :math:`t`.
:math:`\nu`                                            :math:`I_t \times K_2`                               Unobserved agent characteristics called integration nodes in market :math:`t`.
:math:`w`                                              :math:`I_t \times 1`                                 Integration weights in market :math:`t`.
:math:`\delta`                                         :math:`N \times 1`                                   Mean utility.
:math:`\mu`                                            :math:`J_t \times I_t`                               Type-specific portion of utility in market :math:`t`.
:math:`\epsilon`                                       :math:`N \times 1`                                   Type I Extreme Value idiosyncratic preferences.
:math:`\bar{\epsilon}` (:math:`\bar{\epsilon}_{jti}`)  :math:`N \times 1`                                   Type I Extreme Value term used to decompose :math:`\epsilon`.
:math:`\bar{\epsilon}` (:math:`\bar{\epsilon}_{hti}`)  :math:`N \times 1`                                   Group-specific term in the decomposition.
:math:`U`                                              :math:`J_t \times I_t`                               Indirect utilities.
:math:`V` (:math:`V_{jti}`)                            :math:`J_t \times I_t`                               Indirect utilities minus :math:`\epsilon`.
:math:`V` (:math:`V_{hti}`)                            :math:`J_t \times I_t`                               Inclusive value of a nesting group.
:math:`\beta`                                          :math:`K_1 \times 1`                                 Demand-side linear parameters.
:math:`\alpha`                                         :math:`K_1^p \times 1`                               Demand-side linear parameters on endogenous product characteristics.
:math:`\Sigma`                                         :math:`K_2 \times K_2`                               Cholesky root of the covariance matrix that measures agents' random tastes.
:math:`\Pi`                                            :math:`K_2 \times D`                                 Parameters that measures how agent tastes vary with demographics.
:math:`\rho`                                           :math:`H \times 1`                                   Parameters that measures within nesting group correlation.
:math:`\gamma`                                         :math:`K_3 \times 1`                                 Supply-side linear parameters.
:math:`\theta`                                         :math:`P \times 1`                                   Unknown elements in :math:`\Sigma`, :math:`\Pi`, and :math:`\rho`.
:math:`Z_D`                                            :math:`N \times M_D`                                 Excluded demand-side instruments and :math:`X_1`, except for :math:`X_1^p`.
:math:`Z_S`                                            :math:`N \times M_S`                                 Excluded supply-side instruments and :math:`X_3`.
:math:`W`                                              :math:`(M_D + M_S) \times (M_D + M_S)`               Weighting matrix.
:math:`S`                                              :math:`(M_D + M_S) \times (M_D + M_S)`               Sample moment covariances or inverse of the weighting matrix.
:math:`g`                                              :math:`N \times (M_D + M_S)`                         Sample moments.
:math:`q`                                              :math:`C \times (M_D + M_S)`                         Clustered sample moments.
:math:`G`                                              :math:`N \times (M_D + M_S) \times P`                Jacobian of the sample moments with respect to parameters.
:math:`\bar{g}`                                        :math:`(M_D + M_S) \times 1`                         Mean of the sample moments.
:math:`\bar{q}`                                        :math:`(M_D + M_S) \times 1`                         Clustered sample moments sum divided by :math:`N`.
:math:`\bar{G}`                                        :math:`(M_D + M_S) \times P`                         Mean of the Jacobian of the sample moments with respect to parameters.
:math:`\mathscr{Z}_D`                                  :math:`N \times P`                                   Optimal or efficient demand-side instruments for :math:`\theta`.
:math:`\mathscr{Z}_S`                                  :math:`N \times P`                                   Optimal or efficient supply-side instruments for :math:`\theta`.
:math:`E`                                              :math:`1 \times 1`                                   Aggregate elasticity of demand of a market.
:math:`\varepsilon`                                    :math:`J_t \times J_t`                               Elasticities of demand in market :math:`t`.
:math:`\mathscr{D}`                                    :math:`J_t \times J_t`                               Diversion ratios in market :math:`t`.
:math:`\bar{\mathscr{D}}`                              :math:`J_t \times J_t`                               Long-run diversion ratios in market :math:`t`.
:math:`\text{HHI}`                                     :math:`1 \times 1`                                   Herfindahl-Hirschman Index of a market.
:math:`\mathscr{M}`                                    :math:`N \times 1`                                   Markups.
:math:`\pi`                                            :math:`N \times 1`                                   Population-normalized gross expected profits.
:math:`\text{CS}`                                      :math:`1 \times 1`                                   Population-normalized consumer surplus of a market.
=====================================================  ===================================================  =====================================================================================
