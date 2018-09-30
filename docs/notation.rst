Notation
========

The notation in pyblp is a customized amalgamation of the notation employed by :ref:`Berry, Levinsohn, and Pakes (1995) <blp95>`, :ref:`Nevo (2000) <n00>`, :ref:`Morrow and Skerlos (2011) <ms11>`, :ref:`Grigolon and Verboven (2014) <gv14>`, and others.


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
:math:`P`                 Unknown nonlinear parameters.
:math:`\mathscr{J}_c`     Products in cluster :math:`c`.
:math:`\mathscr{J}_{ft}`  Products produced by firm :math:`f` in market :math:`t`.
:math:`\mathscr{J}_{ht}`  Products in group :math:`h` and market :math:`t`.
========================  =========================================================================================================


Matrices, Vectors, and Scalars
------------------------------

Dimensions that can differ across markets are reported for a single market :math:`t`.

=========================  ==========================================  ======================================================================================================================================================================
Symbol                     Dimensions                                  Description
=========================  ==========================================  ======================================================================================================================================================================
:math:`X_1`                :math:`N \times K_1`                        Linear product characteristics.
:math:`X_1^p`              :math:`N \times K_1^p`                      Endogenous linear product characteristics.
:math:`X_2`                :math:`N \times K_2`                        Nonlinear product characteristics.
:math:`X_3`                :math:`N \times K_3`                        Cost product characteristics.
:math:`\xi`                :math:`N \times 1`                          Unobserved demand-side product characteristics, or equivalently, the demand-side structural error term.
:math:`\omega`             :math:`N \times 1`                          Unobserved supply-side product characteristics, or equivalently, the supply-side structural error term.
:math:`p`                  :math:`N \times 1`                          Prices.
:math:`p^*`                :math:`N \times 1`                          Bertrand-Nash prices after firm ID changes.
:math:`p^a`                :math:`N \times 1`                          Bertrand-Nash prices after firm ID changes approximated with the assumption that shares and their price derivatives are unaffected by the changes.
:math:`s`                  :math:`N \times 1`                          Shares when indexed with :math:`j`, group shares when indexed with :math:`h`, and choice probabiltiies with :math:`I_t` columns when indexed with :math:`i`.
:math:`c`                  :math:`N \times 1`                          Marginal costs.
:math:`\tilde{c}`          :math:`N \times 1`                          Marginal costs, :math:`c`, under a linear specification, and :math:`\log c` under a log-linear specification.
:math:`\eta`               :math:`N \times 1`                          Markup term from the BLP-markup equation.
:math:`\eta^a`             :math:`N \times 1`                          Markup term after firm ID changes from the BLP-markup equation approximated with the assumption that shares and their price derivatives are unaffected by the changes.
:math:`\zeta`              :math:`N \times 1`                          Markup term from the :math:`\zeta`-markup equation.
:math:`\zeta^*`            :math:`N \times 1`                          Markup term after firm ID changes from the :math:`\zeta`-markup equation.
:math:`O`                  :math:`J_t \times J_t`                      Ownership matrix in market :math:`t`.
:math:`O^*`                :math:`J_t \times J_t`                      Ownership matrix in market :math:`t` after firm ID changes.
:math:`\kappa`             :math:`F_t \times F_t`                      Cooperation matrix in market :math:`t`.
:math:`\kappa^*`           :math:`F_t \times F_t`                      Cooperation matrix in market :math:`t` after firm ID changes.
:math:`\Lambda`            :math:`J_t \times J_t`                      Diagonal matrix used to decompose :math:`\eta` and :math:`\zeta` in market :math:`t`.
:math:`\Gamma`             :math:`J_t \times J_t`                      Another matrix used to decompose :math:`\eta` and :math:`\zeta` in market :math:`t`.
:math:`d`                  :math:`I_t \times D`                        Observed agent characteristics called demographics in market :math:`t`.
:math:`\nu`                :math:`I_t \times K_2`                      Unobserved agent characteristics called integration nodes in market :math:`t`.
:math:`w`                  :math:`I_t \times 1`                        Integration weights in market :math:`t`.
:math:`\delta`             :math:`N \times 1`                          Mean utility.
:math:`\mu`                :math:`J_t \times I_t`                      Type-specific portion of utility in market :math:`t`.
:math:`\epsilon`           :math:`N \times 1`                          Type I Extreme Value idiosyncratic preferences.
:math:`\bar{\epsilon}`     :math:`N \times 1`                          Type I Extreme Value term used to decompose :math:`\epsilon` when indexed with :math:`j` and the group-specific term in the decomposition when indexed with :math:`h`.
:math:`U`                  :math:`J_t \times I_t`                      Indirect utilities in market :math:`t`.
:math:`V`                  :math:`J_t \times I_t`                      Indirect utilities in market :math:`t` minus :math:`\epsilon` when indexed with :math:`j` and the inclusive value of a nesting group when indexed with :math:`h`.
:math:`\beta`              :math:`K_1 \times 1`                        Demand-side linear parameters.
:math:`\alpha`             :math:`K_1^p \times 1`                      Demand-side linear parameters on endogenous product characteristics.
:math:`\Sigma`             :math:`K_2 \times K_2`                      Cholesky decomposition of the covariance matrix that measures agents' random taste distribution.
:math:`\Pi`                :math:`K_2 \times D`                        Parameters that measures how agent tastes vary with demographics.
:math:`\rho`               :math:`H \times 1`                          Parameters that measures within nesting group correlation.
:math:`\gamma`             :math:`K_3 \times 1`                        Supply-side linear parameters.
:math:`\theta`             :math:`P \times 1`                          Unknown elements in :math:`\Sigma`, :math:`\Pi`, and :math:`\rho`.
:math:`Z_D`                :math:`N \times M_D`                        Full set of demand-side instruments: excluded demand-side instruments and :math:`X_1`, except for :math:`X_1^p`.
:math:`Z_S`                :math:`N \times M_S`                        Full set of supply-side instruments: excluded supply-side instruments and :math:`X_3`.
:math:`W_D`                :math:`M_D \times M_D`                      Demand-side weighting matrix.
:math:`W_S`                :math:`M_S \times M_S`                      Supply-side weighting matrix.
:math:`Z`                  :math:`2N \times (M_D + M_S)`               Block-diagonal instruments.
:math:`W`                  :math:`(M_D + M_S) \times (M_D + M_S)`      Block-diagonal weighting matrices.
:math:`u`                  :math:`2N \times 1`                         Stacked unobserved product characteristics, or equivalently, stacked structural error terms.
:math:`g`                  :math:`2N \times (M_D + M_S)`               Sample moments.
:math:`q`                  :math:`C \times (M_D + M_S)`                Within clustering group sample moment sums.
:math:`G`                  :math:`(M_D + M_S) \times (P + K_1 + K_2)`  Sample mean of the Jacobian of the moments with respect to all parameters.
:math:`\mathscr{Z}_D`      :math:`N \times (P + K_1^p)`                Optimal or efficient excluded demand-side instruments.
:math:`\mathscr{Z}_S`      :math:`N \times (P + K_1^p)`                Optimal or efficient excluded supply-side instruments.
:math:`E`                  :math:`1 \times 1`                          Aggregate elasticity of demand of a market.
:math:`\varepsilon`        :math:`J_t \times J_t`                      Elasticities of demand in market :math:`t`.
:math:`\mathscr{D}`        :math:`J_t \times J_t`                      Diversion ratios in market :math:`t`.
:math:`\bar{\mathscr{D}}`  :math:`J_t \times J_t`                      Long-run diversion ratios in market :math:`t`.
:math:`\text{HHI}`         :math:`1 \times 1`                          Herfindahl-Hirschman Index of a market.
:math:`\mathscr{M}`        :math:`N \times 1`                          Markups.
:math:`\pi`                :math:`N \times 1`                          Population-normalized gross expected profits.
:math:`\text{CS}`          :math:`1 \times 1`                          Population-normalized consumer surplus of a market.
=========================  ==========================================  ======================================================================================================================================================================
