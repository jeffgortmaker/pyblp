Version Notes
=============

These notes will only include major changes.


1.1
---
- Covariance restrictions
- Demographic-specific product availability


1.0
---

- Support matching smooth functions of micro means
- Optimal micro moments
- Support elimination of groups of products for second choices
- Micro data simulation
- Micro moment tutorials


0.13
----

- Overhauled micro moment API
- Product-specific demographics
- Passthrough calculations
- Added problem results methods to simulation results
- Profit Hessian computation
- Checks of pricing second order conditions
- Newton-based methods for computing equilibrium prices
- Large speedups for supply-side and micro moment derivatives
- Universal display for fixed point iteration progress
- Support adjusting for simulation error in moment covariances


0.12
----

- Refactored micro moment API
- Custom micro moments
- Properly scale micro moment covariances
- Pickling support


0.11
----

- Elasticities and diversion ratios with respect to mean utility
- Willingness to pay calculations


0.10
----

- Simplify micro moment API
- Second choice or diversion micro moments
- Add share clipping to make fixed point more robust
- Report covariance matrix estimates in addition to Cholesky root
- Approximation to the pure characteristics model
- Add option to always use finite differences


0.9
---

- More control over matrices of instruments
- Split off fixed effect absorption into companion package PyHDFE
- Scrambled Halton and Modified Latin Hypercube Sampling (MLHS) integration
- Importance sampling
- Quantity dependent marginal costs
- Speed up various matrix construction routines
- Option to do initial GMM update at starting values
- Update BLP example data to better replicate original paper
- Lognormal random coefficients
- Removed outdated default parameter bounds
- Change default objective scaling for more comparable objective values across problem sizes
- Add post-estimation routines to simplify integration error comparison


0.8
---

- Micro moments that match product and agent characteristic covariances
- Extended use of pseudo-inverses
- Added more information to error messages
- More flexible simulation interface
- Alternative way to simulate data with specified prices and shares
- Tests of overidentifying and model restrictions
- Report projected gradients and reduced Hessians
- Change objective gradient scaling
- Switch to a lower-triangular covariance matrix to fix a bug with off-diagonal parameters


0.7
---

- Support more fixed point and optimization solvers
- Hessian computation with finite differences
- Simplified interface for firm changes
- Construction of differentiation instruments
- Add collinearity checks
- Update notation and explanations


0.6
---

- Optimal instrument estimation
- Structured all results as classes
- Additional information in progress reports
- Parametric bootstrapping of post-estimation outputs
- Replaced all examples in the documentation with Jupyter notebooks
- Updated the instruments for the BLP example problem
- Improved support for multiple equation GMM
- Made concentrating out linear parameters optional
- Better support for larger nesting parameters
- Improved robustness to overflow


0.5
---

- Estimation of nesting parameters
- Performance improvements for matrix algebra and matrix construction
- Support for Python 3.7
- Computation of reasonable default bounds on nonlinear parameters
- Additional information in progress updates
- Improved error handling and documentation
- Simplified multiprocessing interface
- Cancelled out delta in the nonlinear contraction to improve performance
- Additional example data and improvements to the example problems
- Cleaned up covariance estimation
- Added type annotations and overhauled the testing suite


0.4
---

- Estimation of a Logit benchmark model
- Support for fixing of all nonlinear parameters
- More efficient two-way fixed effect absorption
- Clustered standard errors


0.3
---

- Patsy- and SymPy-backed R-style formula API
- More informative errors and displays of information
- Absorption of arbitrary fixed effects
- Reduction of memory footprint


0.2
---

- Improved support for longdouble precision
- Custom ownership matrices
- New benchmarking statistics
- Supply-side gradient computation
- Improved configuration for the automobile example problem


0.1
---

- Initial release
