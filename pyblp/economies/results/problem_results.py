"""Economy-level structuring of BLP problem results."""

import time
from typing import Any, Callable, Dict, Hashable, List, Optional, Sequence, TYPE_CHECKING, Tuple

import numpy as np
import scipy.linalg

from .abstract_problem_results import AbstractProblemResults
from ... import exceptions, options
from ...configurations.iteration import Iteration
from ...markets.results_market import ResultsMarket
from ...parameters import LinearParameters, NonlinearParameters
from ...utilities.algebra import multiply_matrix_and_tensor
from ...utilities.basics import (
    Array, Bounds, Error, TableFormatter, format_number, format_seconds, generate_items, output, output_progress
)
from ...utilities.statistics import IV, compute_gmm_parameter_covariances, compute_gmm_weights


# only import objects that create import cycles when checking types
if TYPE_CHECKING:
    from .bootstrapped_problem_results import BootstrappedProblemResults  # noqa
    from .optimal_instrument_results import OptimalInstrumentResults  # noqa
    from ..problem import Progress  # noqa


class ProblemResults(AbstractProblemResults):
    r"""Results of a solved BLP problem.

    Many results are class attributes. Other post-estimation outputs be computed by calling class methods.

    .. note::

       All methods in this class support :func:`parallel` processing. If multiprocessing is used, market-by-market
       computation of each post-estimation output will be distributed among the processes.

    Attributes
    ----------
    problem : `Problem`
        :class:`Problem` that created these results.
    last_results : `ProblemResults`
        :class:`ProblemResults` from the last GMM step.
    step : `int`
        GMM step that created these results.
    optimization_time : `float`
        Number of seconds it took the optimization routine to finish.
    cumulative_optimization_time : `float`
        Sum of :attr:`ProblemResults.optimization_time` for this step and all prior steps.
    total_time : `float`
        Sum of :attr:`ProblemResults.optimization_time` and the number of seconds it took to set up the GMM step and
        compute results after optimization had finished.
    cumulative_total_time : `float`
        Sum of :attr:`ProblemResults.total_time` for this step and all prior steps.
    optimization_iterations : `int`
        Number of major iterations completed by the optimization routine.
    cumulative_optimization_iterations : `int`
        Sum of :attr:`ProblemResults.optimization_iterations` for this step and all prior steps.
    objective_evaluations : `int`
        Number of GMM objective evaluations.
    cumulative_objective_evaluations : `int`
        Sum of :attr:`ProblemResults.objective_evaluations` for this step and all prior steps.
    fp_iterations : `ndarray`
        Number of major iterations completed by the iteration routine used to compute :math:`\delta(\hat{\theta})` in
        each market during each objective evaluation. Rows are in the same order as
        :attr:`Problem.unique_market_ids` and column indices correspond to objective evaluations.
    cumulative_fp_iterations : `ndarray`
        Concatenation of :attr:`ProblemResults.fp_iterations` for this step and all prior steps.
    contraction_evaluations : `ndarray`
        Number of times the contraction used to compute :math:`\delta(\hat{\theta})` was evaluated in each market during
        each objective evaluation. Rows are in the same order as :attr:`Problem.unique_market_ids` and column
        indices correspond to objective evaluations.
    cumulative_contraction_evaluations : `ndarray`
        Concatenation of :attr:`ProblemResults.contraction_evaluations` for this step and all prior steps.
    parameters : `ndarray`
        Stacked parameters: :math:`\hat{\theta}`, :math:`\hat{\beta}`, and :math:`\hat{\gamma}`, in that order.
    parameter_covariances : `ndarray`
        Estimated covariance matrix of the stacked parameters, from which standard errors are extracted.
    theta : `ndarray`
        Estimated unfixed nonlinear parameters, :math:`\hat{\theta}`.
    sigma : `ndarray`
        Estimated Cholesky decomposition of the covariance matrix that measures agents' random taste distribution,
        :math:`\hat{\Sigma}`.
    pi : `ndarray`
        Estimated parameters that measures how agent tastes vary with demographics, :math:`\hat{\Pi}`.
    rho : `ndarray`
        Estimated parameters that measure within nesting group correlations, :math:`\hat{\rho}`.
    beta : `ndarray`
        Estimated demand-side linear parameters, :math:`\hat{\beta}`.
    gamma : `ndarray`
        Estimated supply-side linear parameters, :math:`\hat{\gamma}`.
    sigma_se : `ndarray`
        Estimated standard errors for unknown :math:`\hat{\Sigma}` elements in :math:`\hat{\theta}`.
    pi_se : `ndarray`
        Estimated standard errors for unknown :math:`\hat{\Pi}` elements in :math:`\hat{\theta}`.
    rho_se : `ndarray`
        Estimated standard errors for unknown :math:`\hat{\rho}` elements in :math:`\hat{\theta}`.
    beta_se : `ndarray`
        Estimated standard errors for :math:`\hat{\beta}`.
    gamma_se : `ndarray`
        Estimated standard errors for :math:`\hat{\gamma}`.
    sigma_bounds : `tuple`
        Bounds for :math:`\Sigma` that were used during optimization, which are of the form ``(lb, ub)``.
    pi_bounds : `tuple`
        Bounds for :math:`\Pi` that were used during optimization, which are of the form ``(lb, ub)``.
    rho_bounds : `tuple`
        Bounds for :math:`\rho` that were used during optimization, which are of the form ``(lb, ub)``.
    delta : `ndarray`
        Estimated mean utility, :math:`\delta(\hat{\theta})`, which may have been residualized to absorb any demand-side
        fixed effects.
    true_delta : `ndarray`
        Estimated mean utility, :math:`\delta(\hat{\theta})`.
    tilde_costs : `ndarray`
        Estimated transformed marginal costs, :math:`\tilde{c}(\hat{\theta})`, which may have been residualized to
        absorb any demand-side fixed effects. Transformed marginal costs are simply :math:`\tilde{c} = c`, marginal
        costs, under a linear cost specification, and are :math:`\tilde{c} = \log c` under a log-linear specification.
        If `costs_bounds` were specified in :meth:`Problem.solve`, :math:`c` may have been clipped.
    true_tilde_costs : `ndarray`
        Estimated transformed marginal costs, :math:`\tilde{c}(\hat{\theta})`. Transformed marginal costs are simply
        :math:`\tilde{c} = c`, marginal costs, under a linear cost specification, and are :math:`\tilde{c} = \log c`
        under a log-linear specification. If `costs_bounds` were specified in :meth:`Problem.solve`, :math:`c` may have
        been clipped.
    clipped_costs : `ndarray`
        Vector of booleans indicating whether the associated marginal costs were clipped. All elements will be ``False``
        if `costs_bounds` in :meth:`Problem.solve` was not specified.
    xi : `ndarray`
        Estimated unobserved demand-side product characteristics, :math:`\xi(\hat{\theta})`, or equivalently, the
        demand-side structural error term, which includes the contribution of any absorbed demand-side fixed effects.
    true_xi : `ndarray
        Estimated unobserved demand-side product characteristics, :math:`\xi(\hat{\theta})`.
    omega : `ndarray`
        Estimated unobserved supply-side product characteristics, :math:`\omega(\hat{\theta})`, or equivalently, the
        supply-side structural error term, which includes the contribution of any absorbed supply-side fixed effects.
    true_omega : `ndarray`
        Estimated unobserved supply-side product characteristics, :math:`\omega(\hat{\theta})`, or equivalently, the
        supply-side structural error term.
    objective : `float`
        GMM objective value.
    xi_by_theta_jacobian : `ndarray`
        Estimated :math:`\partial\xi / \partial\theta = \partial\delta / \partial\theta`.
    omega_by_theta_jacobian : `ndarray`
        Estimated :math:`\partial\omega / \partial\theta = \partial\tilde{c} / \partial\theta`.
    omega_by_beta_jacobian : `ndarray`
        Estimated :math:`\partial\omega / \partial\beta = \partial\tilde{c} / \partial\beta`.
    gradient : `ndarray`
        Estimated gradient of the GMM objective with respect to :math:`\theta`. This is still computed once at the end
        of an optimization routine that was configured to not use analytic gradients.
    gradient_norm : `ndarray`
        Infinity norm of :attr:`ProblemResults.gradient`.
    sigma_gradient : `ndarray`
        Estimated gradient of the GMM objective with respect to unknown :math:`\Sigma` elements in :math:`\theta`.
    pi_gradient : `ndarray`
        Estimated gradient of the GMM objective with respect to unknown :math:`\Pi` elements in :math:`\theta`.
    rho_gradient : `ndarray`
        Estimated gradient of the GMM objective with respect to unknown :math:`\rho` elements in :math:`\theta`.
    WD : `ndarray`
        Demand-side weighting matrix, :math:`W_D`, used to compute these results.
    WS : `ndarray`
        Supply-side weighting matrix, :math:`W_S`, used to compute these results.
    updated_WD : `ndarray`
        Updated demand-side weighting matrix.
    updated_WS : `ndarray`
        Updated supply-side weighting matrix.

    Examples
    --------
    For examples of how to use class methods, refer to the :doc:`Examples </examples>` section.

    """

    last_results: Optional['ProblemResults']
    step: int
    optimization_time: float
    cumulative_optimization_time: float
    total_time: float
    cumulative_total_time: float
    optimization_iterations: int
    cumulative_optimization_iterations: int
    objective_evaluations: int
    cumulative_objective_evaluations: int
    fp_iterations: Array
    cumulative_fp_iterations: Array
    contraction_evaluations: Array
    cumulative_contraction_evaluations: Array
    parameters: Array
    parameter_covariances: Array
    theta: Array
    sigma: Array
    pi: Array
    rho: Array
    beta: Array
    gamma: Array
    sigma_se: Array
    pi_se: Array
    rho_se: Array
    beta_se: Array
    gamma_se: Array
    sigma_bounds: tuple
    pi_bounds: tuple
    rho_bounds: tuple
    delta: Array
    true_delta: Array
    tilde_costs: Array
    true_tilde_costs: Array
    clipped_costs: Array
    xi: Array
    true_xi: Array
    omega: Array
    true_omega: Array
    objective: Array
    xi_by_theta_jacobian: Array
    omega_by_theta_jacobian: Array
    omega_by_beta_jacobian: Array
    gradient: Array
    gradient_norm: Array
    sigma_gradient: Array
    pi_gradient: Array
    rho_gradient: Array
    WD: Array
    WS: Array
    updated_WD: Array
    updated_WS: Array
    _costs_type: str
    _se_type: str
    _errors: List[Error]
    _linear_parameters: LinearParameters
    _nonlinear_parameters: NonlinearParameters

    def __init__(
            self, progress: 'Progress', last_results: Optional['ProblemResults'], step_start_time: float,
            optimization_start_time: float, optimization_end_time: float, iterations: int, evaluations: int,
            iteration_mappings: Sequence[Dict[Hashable, int]], evaluation_mappings: Sequence[Dict[Hashable, int]],
            costs_type: str, costs_bounds: Bounds, center_moments: bool, W_type: str, se_type: str) -> None:
        """Compute cumulative progress statistics, update weighting matrices, and estimate standard errors."""

        # initialize values from the progress structure
        super().__init__(progress.problem)
        self._errors = progress.errors
        self.problem = progress.problem
        self.WD = progress.WD
        self.WS = progress.WS
        self.theta = progress.theta
        self.true_delta = progress.true_delta
        self.true_tilde_costs = progress.true_tilde_costs
        self.xi_by_theta_jacobian = progress.xi_jacobian
        self.omega_by_theta_jacobian = progress.omega_jacobian
        self.delta = progress.delta
        self.tilde_costs = progress.tilde_costs
        self.true_xi = progress.true_xi
        self.true_omega = progress.true_omega
        self.beta = progress.beta
        self.gamma = progress.gamma
        self.objective = progress.objective
        self.gradient = progress.gradient
        self.gradient_norm = progress.gradient_norm

        # store information about cost bounds
        self._costs_bounds = costs_bounds
        self.clipped_costs = progress.clipped_costs
        assert self.clipped_costs is not None

        # initialize counts and times
        self.step = 1
        self.total_time = self.cumulative_total_time = time.time() - step_start_time
        self.optimization_time = self.cumulative_optimization_time = optimization_end_time - optimization_start_time
        self.optimization_iterations = self.cumulative_optimization_iterations = iterations
        self.objective_evaluations = self.cumulative_objective_evaluations = evaluations
        self.fp_iterations = self.cumulative_fp_iterations = np.array(
            [[m[t] if m else 0 for m in iteration_mappings] for t in self.problem.unique_market_ids]
        )
        self.contraction_evaluations = self.cumulative_contraction_evaluations = np.array(
            [[m[t] if m else 0 for m in evaluation_mappings] for t in self.problem.unique_market_ids]
        )

        # initialize last results and add to cumulative values
        self.last_results = last_results
        if last_results is not None:
            self.step += last_results.step
            self.cumulative_total_time += last_results.cumulative_total_time
            self.cumulative_optimization_time += last_results.cumulative_optimization_time
            self.cumulative_optimization_iterations += last_results.cumulative_optimization_iterations
            self.cumulative_objective_evaluations += last_results.cumulative_objective_evaluations
            self.cumulative_fp_iterations = np.c_[
                last_results.cumulative_fp_iterations, self.cumulative_fp_iterations
            ]
            self.cumulative_contraction_evaluations = np.c_[
                last_results.cumulative_contraction_evaluations, self.cumulative_contraction_evaluations
            ]

        # store parameter information
        self.parameters = np.r_[np.c_[self.theta], self.beta, self.gamma]
        self._linear_parameters = LinearParameters(self.problem, self.beta, self.gamma)
        self._nonlinear_parameters = progress.nonlinear_parameters
        self.sigma_bounds = self._nonlinear_parameters.sigma_bounds
        self.pi_bounds = self._nonlinear_parameters.pi_bounds
        self.rho_bounds = self._nonlinear_parameters.rho_bounds

        # expand the nonlinear parameters and their gradient
        self.sigma, self.pi, self.rho = self._nonlinear_parameters.expand(self.theta)
        self.sigma_gradient, self.pi_gradient, self.rho_gradient = self._nonlinear_parameters.expand(
            self.gradient, nullify=True
        )

        # compute a version of xi that includes the contribution of any demand-side fixed effects
        self.xi = self.true_xi
        if self.problem.ED > 0:
            self.xi = self.true_delta - self.problem._compute_true_X1() @ self.beta

        # compute a version of omega that includes the contribution of any supply-side fixed effects
        self.omega = self.true_omega
        if self.problem.ES > 0:
            self.omega = self.true_tilde_costs - self.problem._compute_true_X3() @ self.gamma

        # update the weighting matrices
        self.updated_WD, WD_errors = compute_gmm_weights(
            self.true_xi, self.problem.products.ZD, W_type, center_moments, self.problem.products.clustering_ids
        )
        self.updated_WS, WS_errors = compute_gmm_weights(
            self.true_omega, self.problem.products.ZS, W_type, center_moments, self.problem.products.clustering_ids
        )
        self._errors.extend(WD_errors + WS_errors)

        # compute the Jacobian of omega with respect to beta, which will be used for computing standard errors
        self.omega_by_beta_jacobian = np.full((self.problem.N, self.problem.K1), np.nan, options.dtype)
        if self.problem.K3 > 0:
            # define a factory for computing the Jacobian of omega with respect to beta in markets
            def market_factory(s: Hashable) -> Tuple[ResultsMarket, Array, str]:
                """Build a market along with arguments used to compute the Jacobian."""
                market_s = ResultsMarket(self.problem, s, self.sigma, self.pi, self.rho, self.beta, self.true_delta)
                true_tilde_costs_s = self.true_tilde_costs[self.problem._product_market_indices[s]]
                return market_s, true_tilde_costs_s, costs_type

            # compute the Jacobian market-by-market
            generator = generate_items(
                self.problem.unique_market_ids, market_factory, ResultsMarket.compute_omega_by_beta_jacobian
            )
            for t, (omega_by_beta_jacobian_t, errors_t) in generator:
                self.omega_by_beta_jacobian[self.problem._product_market_indices[t]] = omega_by_beta_jacobian_t
                self._errors.extend(errors_t)

            # the Jacobian should be zero for any clipped marginal costs
            self.omega_by_beta_jacobian[self.clipped_costs.flat] = 0

        # stack errors, weights, instruments, Jacobian of the errors with respect to parameters, and clustering IDs
        if self.problem.K3 == 0:
            u = self.true_xi
            Z = self.problem.products.ZD
            W = self.WD
            jacobian = np.c_[self.xi_by_theta_jacobian, -self.problem.products.X1]
            stacked_clustering_ids = self.problem.products.clustering_ids
        else:
            u = np.r_[self.true_xi, self.true_omega]
            Z = scipy.linalg.block_diag(self.problem.products.ZD, self.problem.products.ZS)
            W = scipy.linalg.block_diag(self.WD, self.WS)
            jacobian = np.r_[
                np.c_[self.xi_by_theta_jacobian, -self.problem.products.X1, np.zeros_like(self.problem.products.X3)],
                np.c_[self.omega_by_theta_jacobian, self.omega_by_beta_jacobian, -self.problem.products.X3]
            ]
            stacked_clustering_ids = np.r_[self.problem.products.clustering_ids, self.problem.products.clustering_ids]

        # compute parameter covariances
        self.parameter_covariances, covariance_errors = compute_gmm_parameter_covariances(
            u, Z, W, jacobian, se_type, self.step, stacked_clustering_ids
        )
        self._errors.extend(covariance_errors)

        # extract standard errors
        with np.errstate(invalid='ignore'):
            se = np.sqrt(np.c_[self.parameter_covariances.diagonal()] / self.problem.N)
        self.sigma_se, self.pi_se, self.rho_se = self._nonlinear_parameters.expand(
            se[:self._nonlinear_parameters.P], nullify=True
        )
        self.beta_se = se[self._nonlinear_parameters.P:self._nonlinear_parameters.P + self.problem.K1]
        self.gamma_se = se[self._nonlinear_parameters.P + self.problem.K1:]
        if np.isnan(se).any():
            self._errors.append(exceptions.InvalidParameterCovariancesError())

        # store types that are used in other methods
        self._costs_type = costs_type
        self._se_type = se_type

    def __str__(self) -> str:
        """Format problem results as a string."""

        # construct a section containing summary information
        header = [
            ("Cumulative", "Total Time"), ("GMM", "Step"), ("Optimization", "Iterations"),
            ("Objective", "Evaluations"), ("Total Fixed Point", "Iterations"), ("Total Contraction", "Evaluations"),
            ("Objective", "Value"), ("Gradient", "Infinity Norm"),
        ]
        values = [
            format_seconds(self.cumulative_total_time),
            self.step,
            self.optimization_iterations,
            self.objective_evaluations,
            self.fp_iterations.sum(),
            self.contraction_evaluations.sum(),
            format_number(float(self.objective)),
            format_number(float(self.gradient_norm))
        ]
        if np.isfinite(self._costs_bounds).any():
            header.append(("Clipped", "Marginal Costs"))
            values.append(self.clipped_costs.sum())
        widths = [max(len(k1), len(k2), options.digits + 6 if i > 5 else 0) for i, (k1, k2) in enumerate(header)]
        formatter = TableFormatter(widths)
        sections = [[
            "Problem Results Summary:",
            formatter.line(),
            formatter([k[0] for k in header]),
            formatter([k[1] for k in header], underline=True),
            formatter(values),
            formatter.line()
        ]]

        # construct a standard error description
        if self._se_type == 'unadjusted':
            se_description = "Unadjusted SEs"
        elif self._se_type == 'robust':
            se_description = "Robust SEs"
        else:
            assert self._se_type == 'clustered'
            se_description = f'Robust SEs Adjusted for {np.unique(self.problem.products.clustering_ids).size} Clusters'

        # construct a section containing linear estimates
        sections.append([
            f"Linear Estimates ({se_description} in Parentheses):",
            self._linear_parameters.format_estimates(self.beta, self.gamma, self.beta_se, self.gamma_se)
        ])

        # construct a section containing nonlinear estimates
        if self.problem.K2 > 0 or self.problem.H > 0:
            sections.append([
                f"Nonlinear Estimates ({se_description} in Parentheses):",
                self._nonlinear_parameters.format_estimates(
                    self.sigma, self.pi, self.rho, self.sigma_se, self.pi_se, self.rho_se
                )
            ])

        # combine the sections into one string
        return "\n\n".join("\n".join(s) for s in sections)

    def bootstrap(
            self, draws: int = 1000, seed: Optional[int] = None, iteration: Optional[Iteration] = None) -> (
            'BootstrappedProblemResults'):
        """Use a parametric bootstrap to create an empirical distribution of results.

        The constructed :class:`BootstrappedProblemResults` can be used just like :class:`ProblemResults` to compute
        various post-estimation outputs. The only difference is that :class:`BootstrappedProblemResults` methods return
        arrays with an extra first dimension, along which bootstrapped results are stacked. These stacked results can
        be used to construct, for example, confidence intervals for post-estimation outputs.

        For each bootstrap draw, parameters are drawn from the estimated multivariate normal distribution of all
        parameters defined by :attr:`ProblemResults.parameters` and :attr:`ProblemResults.parameter_covariances`. Note
        that any bounds configured during the optimization routine will be used to bound parameter draws. These
        parameters are used to compute the implied mean utility, :math:`\delta`, and shares, :math:`s`. If a supply side
        was estimated, the implied marginal costs, :math:`c`, and prices, :math:`p`, are computed as well. Specifically,
        if a supply side was estimated, equilibrium prices and shares are computed by iterating over the
        :math:`\zeta`-markup equation from :ref:`Morrow and Skerlos (2011) <ms11>`.

        .. note::

           By default, the bootstrapping procedure can use a lot of memory. This is because it stores in memory all
           bootstrapped results (for all `draws`) at the same time. To reduce the memory footprint of the procedure,
           call this method in a loop with `draws` set to ``1``. In each iteration of the loop, compute the desired
           post-estimation output with the proper method of the returned :class:`BootstrappedProblemResults` class and
           store these outputs.

        Parameters
        ----------
        draws : `int, optional`
            The number of draws that will be taken from the joint distribution of the parameters. The default is
            ``1000``.
        seed : `int, optional`
            Passed to :class:`numpy.random.RandomState` to seed the random number generator before any draws are taken.
            By default, a seed is not passed to the random number generator.
        iteration : `Iteration, optional`
            :class:`Iteration` configuration used to compute bootstrapped prices by iterating over the
            :math:`\zeta`-markup equation from :ref:`Morrow and Skerlos (2011) <ms11>`. By default, if a supply side was
            estimated, this is ``Iteration('simple', {'tol': 1e-12})``. It is not used if a supply side was not
            estimated.

        Returns
        -------
        `BootstrappedProblemResults`
            Computed :class:`BootstrappedProblemResults`.

        """
        errors: List[Error] = []

        # keep track of long it takes to bootstrap results
        output("Bootstrapping results ...")
        start_time = time.time()

        # validate the number of draws
        if not isinstance(draws, int) or draws < 1:
            raise ValueError("draws must be a positive int.")

        # validate the iteration configuration
        if self.problem.K3 == 0:
            iteration = None
        elif iteration is None:
            iteration = Iteration('simple', {'tol': 1e-12})
        elif not isinstance(iteration, Iteration):
            raise TypeError("iteration must be None or an iteration instance.")

        # draw from the asymptotic distribution implied by the estimated parameters
        state = np.random.RandomState(seed)
        bootstrapped_parameters = np.atleast_3d(state.multivariate_normal(
            self.parameters.flatten(), self.parameter_covariances, draws
        ))

        # extract the parameters
        split_indices = [self._nonlinear_parameters.P, self._nonlinear_parameters.P + self.problem.K1]
        bootstrapped_theta, bootstrapped_beta, bootstrapped_gamma = np.split(
            bootstrapped_parameters, split_indices, axis=1
        )

        # extract bootstrapped nonlinear parameters
        bootstrapped_sigma = np.repeat(np.zeros_like(self.sigma[None]), draws, axis=0)
        bootstrapped_pi = np.repeat(np.zeros_like(self.pi[None]), draws, axis=0)
        bootstrapped_rho = np.repeat(np.zeros_like(self.rho[None]), draws, axis=0)
        for d in range(draws):
            bootstrapped_sigma[d], bootstrapped_pi[d], bootstrapped_rho[d] = self._nonlinear_parameters.expand(
                bootstrapped_theta[d]
            )
            bootstrapped_sigma[d] = np.clip(bootstrapped_sigma[d], *self.sigma_bounds)
            bootstrapped_pi[d] = np.clip(bootstrapped_pi[d], *self.pi_bounds)
            bootstrapped_rho[d] = np.clip(bootstrapped_rho[d], *self.rho_bounds)

        # compute bootstrapped prices, shares, delta and marginal costs
        iteration_mappings: List[Dict[Hashable, int]] = []
        evaluation_mappings: List[Dict[Hashable, int]] = []
        bootstrapped_prices = np.zeros((draws, self.problem.N, 1), options.dtype)
        bootstrapped_shares = np.zeros((draws, self.problem.N, 1), options.dtype)
        bootstrapped_delta = np.zeros((draws, self.problem.N, 1), options.dtype)
        bootstrapped_costs = np.zeros((draws, self.problem.N, int(self.problem.K3 > 0)), options.dtype)
        for d in output_progress(range(draws), draws, start_time):
            prices_d, shares_d, delta_d, costs_d, iterations_d, evaluations_d, errors_d = self._compute_bootstrap(
                iteration, bootstrapped_sigma[d], bootstrapped_pi[d], bootstrapped_rho[d], bootstrapped_beta[d],
                bootstrapped_gamma[d]
            )
            bootstrapped_prices[d] = prices_d
            bootstrapped_shares[d] = shares_d
            bootstrapped_delta[d] = delta_d
            bootstrapped_costs[d] = costs_d
            iteration_mappings.append(iterations_d)
            evaluation_mappings.append(evaluations_d)
            errors.extend(errors_d)

        # structure the results
        from .bootstrapped_problem_results import BootstrappedProblemResults  # noqa
        results = BootstrappedProblemResults(
            self, bootstrapped_sigma, bootstrapped_pi, bootstrapped_rho, bootstrapped_beta, bootstrapped_gamma,
            bootstrapped_prices, bootstrapped_shares, bootstrapped_delta, bootstrapped_costs, start_time, time.time(),
            draws, iteration_mappings, evaluation_mappings
        )
        output(f"Bootstrapped results after {format_seconds(results.computation_time)}.")
        output("")
        output(results)
        return results

    def _compute_bootstrap(
            self, iteration: Optional[Iteration], sigma: Array, pi: Array, rho: Array, beta: Array, gamma: Array) -> (
            Tuple[Array, Array, Array, Array, Dict[Hashable, int], Dict[Hashable, int], List[Error]]):
        """Compute the equilibrium prices, shares, marginal costs, and delta associated with bootstrapped parameters
        market-by-market
        """
        errors: List[Error] = []

        # compute delta (which will change under equilibrium prices) and marginal costs (which won't change)
        delta = self.true_delta + self.problem._compute_true_X1() @ (beta - self.beta)
        costs = self.true_tilde_costs + self.problem._compute_true_X3() @ (gamma - self.gamma)
        if self._costs_type == 'log':
            costs = np.exp(costs)

        # prices will only change if there is an iteration configuration
        prices = self.problem.products.prices if iteration is None else None

        # define a factory for computing bootstrapped prices, shares, and delta in markets
        def market_factory(s: Hashable) -> Tuple[ResultsMarket, Array, Optional[Array], Optional[Iteration]]:
            """Build a market along with arguments used to compute equilibrium prices and shares along with delta."""
            market_s = ResultsMarket(self.problem, s, sigma, pi, rho, beta, delta)
            costs_s = costs[self.problem._product_market_indices[s]]
            prices_s = prices[self.problem._product_market_indices[s]] if prices is not None else None
            return market_s, costs_s, prices_s, iteration

        # compute bootstrapped prices, shares, and delta market-by-market
        iteration_mapping: Dict[Hashable, int] = {}
        evaluation_mapping: Dict[Hashable, int] = {}
        equilibrium_prices = np.zeros_like(self.problem.products.prices)
        equilibrium_shares = np.zeros_like(self.problem.products.shares)
        generator = generate_items(self.problem.unique_market_ids, market_factory, ResultsMarket.solve_equilibrium)
        for t, (prices_t, shares_t, delta_t, errors_t, iterations_t, evaluations_t) in generator:
            equilibrium_prices[self.problem._product_market_indices[t]] = prices_t
            equilibrium_shares[self.problem._product_market_indices[t]] = shares_t
            delta[self.problem._product_market_indices[t]] = delta_t
            errors.extend(errors_t)
            iteration_mapping[t] = iterations_t
            evaluation_mapping[t] = evaluations_t

        # return all of the information associated with this bootstrap draw
        return equilibrium_prices, equilibrium_shares, delta, costs, iteration_mapping, evaluation_mapping, errors

    def compute_optimal_instruments(
            self, method: str = 'normal', draws: int = 100, seed: Optional[int] = None,
            expected_prices: Optional[Any] = None, iteration: Optional[Iteration] = None) -> 'OptimalInstrumentResults':
        r"""Estimate the set of optimal or efficient excluded instruments, :math:`\mathscr{Z}_D` and
        :math:`\mathscr{Z}_S`.

        Optimal instruments have been shown, for example, by :ref:`Reynaert and Verboven (2014) <rv14>`, to not only
        reduce bias in the BLP problem, but also to improve efficiency and stability.

        :ref:`Chamberlain's (1987) <c87>` optimal excluded instruments are

        .. math::

           \begin{bmatrix}
               \mathscr{Z}_D \\
               \mathscr{Z}_S
           \end{bmatrix}_{jt}
           = \text{Var}(\xi, \omega)^{-1}\operatorname{\mathbb{E}}\left[
           \begin{matrix}
               \frac{\partial\xi_{jt}}{\partial\theta} &
               \frac{\partial\xi_{jt}}{\partial\alpha} \\
               \frac{\partial\omega_{jt}}{\partial\theta} &
               \frac{\partial\omega_{jt}}{\partial\alpha}
           \end{matrix}
           \mathrel{\Bigg|} Z \right],

        The expectation is taken by integrating over the joint density of :math:`\xi` and :math:`\omega`. For each error
        term realization, if not already estimated, equilibrium prices are computed via iteration over the
        :math:`\zeta`-markup equation from :ref:`Morrow and Skerlos (2011) <ms11>`. Associated shares and :math:`\delta`
        are then computed before each Jacobian is evaluated.

        The expected Jacobians are estimated with the average over all computed Jacobian realizations. The normalizing
        matrix :math:`\text{Var}(\xi, \omega)^{-1}` is estimated with the sample covariance matrix of the error terms.

        Parameters
        ----------
        method : `str, optional`
            The method by which the integral over the joint density of :math:`\xi` and :math:`\omega` is computed. The
            following methods are supported:

                - ``'normal'`` (default) - Draw from the normal approximation to the joint distribution of the error
                  terms and take the average over the computed Jacobians (`draws` determines the number of draws).

                - ``'empirical'`` - Draw with replacement from the empirical joint distribution of the error terms and
                  take the average over the computed Jacobians (`draws` determines the number of draws).

                - ``'approximate'`` - Evaluate the Jacobians at the expected value of the error terms: zero (`draws`
                  will be ignored).

        draws : `int, optional`
            The number of draws that will be taken from the joint distribution of the error terms. This is ignored if
            `method` is ``'approximate'``. The default is ``100``.
        seed : `int, optional`
            Passed to :class:`numpy.random.RandomState` to seed the random number generator before any draws are taken.
            By default, a seed is not passed to the random number generator.
        expected_prices : `array-like, optional`
            Vector of expected prices conditional on all exogenous variables,
            :math:`\operatorname{\mathbb{E}}[p \mid Z]`, which is required if a supply side was not estimated. A common
            way to estimate this vector is with the fitted values from a reduced form regression of endogenous prices
            onto all exogenous variables, including instruments. An example is given in the documentation for the
            convenience function :func:`compute_fitted_values`.
        iteration : `Iteration, optional`
            :class:`Iteration` configuration used to estimate expected prices by iterating over the :math:`\zeta`-markup
            equation from :ref:`Morrow and Skerlos (2011) <ms11>`. By default, if a supply side was estimated, this is
            ``Iteration('simple', {'tol': 1e-12})``. It is not used if `expected_prices` is specified.

        Returns
        -------
        `OptimalInstrumentResults`
           Computed :class:`OptimalInstrumentResults`.

        """
        errors: List[Error] = []

        # keep track of long it takes to compute optimal excluded instruments
        output("Computing optimal excluded instruments ...")
        start_time = time.time()

        # validate the method and create a function that samples from the error distribution
        if method == 'approximate':
            sample = lambda: (np.zeros_like(self.xi), np.zeros_like(self.omega))
        else:
            state = np.random.RandomState(seed)
            if method == 'normal':
                if self.problem.K3 == 0:
                    variance = np.var(self.true_xi)
                    sample = lambda: (np.c_[state.normal(0, variance, self.problem.N)], self.omega)
                else:
                    covariances = np.cov(self.true_xi, self.true_omega, rowvar=False)
                    sample = lambda: np.hsplit(state.multivariate_normal([0, 0], covariances, self.problem.N), 2)
            elif method == 'empirical':
                if self.problem.K3 == 0:
                    sample = lambda: (self.true_xi[state.choice(self.problem.N, self.problem.N)], self.omega)
                else:
                    joint = np.c_[self.true_xi, self.true_omega]
                    sample = lambda: np.hsplit(joint[state.choice(self.problem.N, self.problem.N)], 2)
            else:
                raise ValueError("method must be 'approximate', 'normal', or 'empirical'.")

        # validate the number of draws (there will be only one for the approximate method)
        if method == 'approximate':
            draws = 1
        if not isinstance(draws, int) or draws < 1:
            raise ValueError("draws must be a positive int.")

        # validate expected prices or their iteration configuration
        if expected_prices is None:
            if self.problem.K3 == 0:
                raise TypeError("A supply side was not estimated, so expected_prices must be specified.")
            if iteration is None:
                iteration = Iteration('simple', {'tol': 1e-12})
            elif not isinstance(iteration, Iteration):
                raise TypeError("iteration must be None or an Iteration instance.")
        else:
            iteration = None
            expected_prices = np.c_[np.asarray(expected_prices, options.dtype)]
            if expected_prices.shape != (self.problem.N, 1):
                raise ValueError(f"expected_prices must be a {self.problem.N}-vector.")

        # average over Jacobian realizations
        iteration_mappings: List[Dict[Hashable, int]] = []
        evaluation_mappings: List[Dict[Hashable, int]] = []
        expected_xi_by_theta = np.zeros_like(self.xi_by_theta_jacobian)
        expected_xi_by_beta = np.zeros_like(self.problem.products.X1)
        expected_omega_by_theta = np.zeros_like(self.omega_by_theta_jacobian)
        expected_omega_by_beta = np.zeros_like(self.omega_by_beta_jacobian)
        for _ in output_progress(range(draws), draws, start_time):
            xi_by_theta_i, xi_by_beta_i, omega_by_theta_i, omega_by_beta_i, iterations_i, evaluations_i, errors_i = (
                self._compute_realizations(expected_prices, iteration, *sample())
            )
            expected_xi_by_theta += xi_by_theta_i / draws
            expected_xi_by_beta += xi_by_beta_i / draws
            expected_omega_by_theta += omega_by_theta_i / draws
            expected_omega_by_beta += omega_by_beta_i / draws
            iteration_mappings.append(iterations_i)
            evaluation_mappings.append(evaluations_i)
            errors.extend(errors_i)

        # output a warning about any errors
        if errors:
            output("")
            output(exceptions.MultipleErrors(errors))
            output("")

        # select columns in the expected Jacobians with respect to beta associated with endogenous characteristics
        endogenous_column_indices = [i for i, f in enumerate(self.problem._X1_formulations) if 'prices' in f.names]
        expected_xi_by_alpha = np.c_[expected_xi_by_beta[:, endogenous_column_indices]]
        expected_omega_by_alpha = np.c_[expected_omega_by_beta[:, endogenous_column_indices]]

        # compute the optimal instruments
        if self.problem.K3 == 0:
            inverse_covariance_matrix = np.c_[1 / np.var(self.true_xi)]
            demand_instruments = inverse_covariance_matrix * np.c_[expected_xi_by_theta, expected_xi_by_alpha]
            supply_instruments = np.full((self.problem.N, 0), np.nan, options.dtype)
        else:
            inverse_covariance_matrix = np.c_[scipy.linalg.inv(np.cov(self.true_xi, self.true_omega, rowvar=False))]
            jacobian = np.r_[
                np.c_[expected_xi_by_theta, expected_xi_by_alpha],
                np.c_[expected_omega_by_theta, expected_omega_by_alpha]
            ]
            tensor = multiply_matrix_and_tensor(inverse_covariance_matrix, np.stack(np.split(jacobian, 2), axis=1))
            demand_instruments, supply_instruments = np.split(tensor.reshape((self.problem.N, -1)), 2, axis=1)

        # structure the results
        from .optimal_instrument_results import OptimalInstrumentResults  # noqa
        results = OptimalInstrumentResults(
            self, demand_instruments, supply_instruments, inverse_covariance_matrix, expected_xi_by_theta,
            expected_xi_by_alpha, expected_omega_by_theta, expected_omega_by_alpha, start_time, time.time(), draws,
            iteration_mappings, evaluation_mappings
        )
        output(f"Computed optimal instruments after {format_seconds(results.computation_time)}.")
        output("")
        output(results)
        return results

    def _compute_realizations(
            self, expected_prices: Optional[Array], iteration: Optional[Iteration], xi: Array, omega: Array) -> (
            Tuple[Array, Array, Array, Array, Dict[Hashable, int], Dict[Hashable, int], List[Error]]):
        """If they have not already been estimated, compute the equilibrium prices, shares, and delta associated with a
        realization of xi and omega market-by-market. Then, compute realizations of Jacobians of xi and omega with
        respect to theta and beta.
        """
        errors: List[Error] = []

        # compute delta (which will change under equilibrium prices) and marginal costs (which won't change)
        delta = self.true_delta - self.true_xi + xi
        costs = tilde_costs = self.true_tilde_costs - self.true_omega + omega
        if self._costs_type == 'log':
            costs = np.exp(costs)

        # define a factory for computing realizations of prices, shares, and delta in markets
        def market_factory(s: Hashable) -> Tuple[ResultsMarket, Array, Optional[Array], Optional[Iteration]]:
            """Build a market along with arguments used to compute equilibrium prices and shares along with delta."""
            market_s = ResultsMarket(self.problem, s, self.sigma, self.pi, self.rho, self.beta, delta)
            costs_s = costs[self.problem._product_market_indices[s]]
            prices_s = expected_prices[self.problem._product_market_indices[s]] if expected_prices is not None else None
            return market_s, costs_s, prices_s, iteration

        # compute realizations of prices, shares, and delta market-by-market
        iteration_mapping: Dict[Hashable, int] = {}
        evaluation_mapping: Dict[Hashable, int] = {}
        equilibrium_prices = np.zeros_like(self.problem.products.prices)
        equilibrium_shares = np.zeros_like(self.problem.products.shares)
        generator = generate_items(self.problem.unique_market_ids, market_factory, ResultsMarket.solve_equilibrium)
        for t, (prices_t, shares_t, delta_t, errors_t, iterations_t, evaluations_t) in generator:
            equilibrium_prices[self.problem._product_market_indices[t]] = prices_t
            equilibrium_shares[self.problem._product_market_indices[t]] = shares_t
            delta[self.problem._product_market_indices[t]] = delta_t
            errors.extend(errors_t)
            iteration_mapping[t] = iterations_t
            evaluation_mapping[t] = evaluations_t

        # compute the Jacobian of xi with respect to theta
        xi_by_theta_jacobian, demand_errors = self._compute_demand_realizations(
            equilibrium_prices, equilibrium_shares, delta
        )
        errors.extend(demand_errors)

        # compute the Jacobian of xi with respect to beta (prices just need to be replaced in X1)
        xi_by_beta_jacobian = -self.problem._compute_true_X1({'prices': equilibrium_prices})

        # compute the Jacobians of omega with respect to theta and beta
        omega_by_theta_jacobian = np.full((self.problem.N, self._nonlinear_parameters.P), np.nan, options.dtype)
        omega_by_beta_jacobian = np.full((self.problem.N, self.problem.K1), np.nan, options.dtype)
        if self.problem.K3 > 0:
            omega_by_theta_jacobian, omega_by_beta_jacobian, supply_errors = self._compute_supply_realizations(
                equilibrium_prices, equilibrium_shares, delta, tilde_costs, xi_by_theta_jacobian
            )
            errors.extend(supply_errors)

        # return all of the information associated with this realization
        return (
            xi_by_theta_jacobian, xi_by_beta_jacobian, omega_by_theta_jacobian, omega_by_beta_jacobian,
            iteration_mapping, evaluation_mapping, errors
        )

    def _compute_demand_realizations(
            self, equilibrium_prices: Array, equilibrium_shares: Array, delta: Array) -> Tuple[Array, List[Error]]:
        """Compute a realization of the Jacobian of xi with respect to theta market-by-market. If necessary, revert
        problematic elements to their estimated values.
        """
        errors: List[Error] = []

        # check if the Jacobian does not need to be computed
        xi_by_theta_jacobian = np.full((self.problem.N, self._nonlinear_parameters.P), np.nan, options.dtype)
        if self._nonlinear_parameters.P == 0:
            return xi_by_theta_jacobian, errors

        # define a factory for computing the Jacobian of xi with respect to theta in markets
        def market_factory(s: Hashable) -> Tuple[ResultsMarket, NonlinearParameters]:
            """Build a market with the data realization along with arguments used to compute the Jacobian."""
            data_override_s = {
                'prices': equilibrium_prices[self.problem._product_market_indices[s]],
                'shares': equilibrium_shares[self.problem._product_market_indices[s]]
            }
            market_s = ResultsMarket(self.problem, s, self.sigma, self.pi, self.rho, self.beta, delta, data_override_s)
            return market_s, self._nonlinear_parameters

        # compute the Jacobian market-by-market
        generator = generate_items(
            self.problem.unique_market_ids, market_factory, ResultsMarket.compute_xi_by_theta_jacobian
        )
        for t, (xi_by_theta_jacobian_t, errors_t) in generator:
            xi_by_theta_jacobian[self.problem._product_market_indices[t]] = xi_by_theta_jacobian_t
            errors.extend(errors_t)

        # replace invalid elements
        bad_indices = ~np.isfinite(xi_by_theta_jacobian)
        if np.any(bad_indices):
            xi_by_theta_jacobian[bad_indices] = self.xi_by_theta_jacobian[bad_indices]
            errors.append(exceptions.XiByThetaJacobianReversionError(bad_indices))
        return xi_by_theta_jacobian, errors

    def _compute_supply_realizations(
            self, equilibrium_prices: Array, equilibrium_shares: Array, delta: Array, tilde_costs: Array,
            xi_jacobian: Array) -> Tuple[Array, Array, List[Error]]:
        """Compute realizations of the Jacobians of omega with respect to theta and beta market-by-market. If necessary,
        revert problematic elements to their estimated values.
        """
        errors: List[Error] = []

        # compute the Jacobian of beta with respect to theta, which is needed to compute the Jacobian of omega with
        #   respect to theta
        demand_iv = IV(self.problem.products.X1, self.problem.products.ZD, self.WD)
        beta_jacobian = demand_iv.estimate(xi_jacobian, residuals=False)
        errors.extend(demand_iv.errors)

        # define a factory for computing the Jacobians of omega with respect to theta and beta in markets
        def market_factory(s: Hashable) -> Tuple[ResultsMarket, Array, Array, Array, NonlinearParameters, str]:
            """Build a market with the data realization along with arguments used to compute the Jacobians."""
            data_override_s = {
                'prices': equilibrium_prices[self.problem._product_market_indices[s]],
                'shares': equilibrium_shares[self.problem._product_market_indices[s]]
            }
            market_s = ResultsMarket(self.problem, s, self.sigma, self.pi, self.rho, self.beta, delta, data_override_s)
            tilde_costs_s = tilde_costs[self.problem._product_market_indices[s]]
            xi_jacobian_s = xi_jacobian[self.problem._product_market_indices[s]]
            return market_s, tilde_costs_s, xi_jacobian_s, beta_jacobian, self._nonlinear_parameters, self._costs_type

        # compute the Jacobians market-by-market
        omega_by_theta_jacobian = np.full((self.problem.N, self._nonlinear_parameters.P), np.nan, options.dtype)
        omega_by_beta_jacobian = np.full((self.problem.N, self.problem.K1), np.nan, options.dtype)
        generator = generate_items(
            self.problem.unique_market_ids, market_factory, ResultsMarket.compute_omega_jacobians
        )
        for t, (omega_by_theta_jacobian_t, omega_by_beta_jacobian_t, errors_t) in generator:
            omega_by_theta_jacobian[self.problem._product_market_indices[t]] = omega_by_theta_jacobian_t
            omega_by_beta_jacobian[self.problem._product_market_indices[t]] = omega_by_beta_jacobian_t
            errors.extend(errors_t)

        # the Jacobians should be zero for any clipped marginal costs
        omega_by_theta_jacobian[self.clipped_costs.flat] = 0
        omega_by_beta_jacobian[self.clipped_costs.flat] = 0

        # replace invalid elements in the Jacobian of omega with respect to theta
        bad_indices = ~np.isfinite(omega_by_theta_jacobian)
        if np.any(bad_indices):
            omega_by_theta_jacobian[bad_indices] = self.omega_by_theta_jacobian[bad_indices]
            errors.append(exceptions.OmegaByThetaJacobianReversionError(bad_indices))

        # replace invalid elements in the Jacobian of omega with respect to beta
        bad_indices = ~np.isfinite(omega_by_theta_jacobian)
        if np.any(bad_indices):
            omega_by_beta_jacobian[bad_indices] = self.omega_by_beta_jacobian[bad_indices]
            errors.append(exceptions.OmegaByBetaJacobianReversionError(bad_indices))
        return omega_by_theta_jacobian, omega_by_beta_jacobian, errors

    def _coerce_matrices(self, matrices: Any) -> Array:
        """Coerce array-like stacked matrices into a stacked matrix and validate it."""
        matrices = np.c_[np.asarray(matrices, options.dtype)]
        if matrices.shape != (self.problem.N, self.problem._max_J):
            raise ValueError(f"matrices must be {self.problem.N} by {self.problem._max_J}.")
        return matrices

    def _coerce_optional_costs(self, costs: Optional[Any]) -> Array:
        """Coerce optional array-like costs into a column vector and validate it."""
        if costs is not None:
            costs = np.c_[np.asarray(costs, options.dtype)]
            if costs.shape != (self.problem.N, 1):
                raise ValueError(f"costs must be None or a {self.problem.N}-vector.")
        return costs

    def _coerce_optional_prices(self, prices: Optional[Any]) -> Array:
        """Coerce optional array-like prices into a column vector and validate it."""
        if prices is not None:
            prices = np.c_[np.asarray(prices, options.dtype)]
            if prices.shape != (self.problem.N, 1):
                raise ValueError(f"prices must be None or a {self.problem.N}-vector.")
        return prices

    def _coerce_optional_shares(self, shares: Optional[Any]) -> Array:
        """Coerce optional array-like shares into a column vector and validate it."""
        if shares is not None:
            shares = np.c_[np.asarray(shares, options.dtype)]
            if shares.shape != (self.problem.N, 1):
                raise ValueError(f"shares must be None or a {self.problem.N}-vector.")
        return shares

    def _combine_arrays(self, compute_market_results: Callable, fixed_args: Sequence, market_args: Sequence) -> Array:
        """Compute an array for each market and stack them into a single matrix. An array for a single market is
        computed by passing fixed_args (identical for all markets) and market_args (matrices with as many rows as there
        are products that are restricted to the market) to compute_market_results, a ResultsMarket method that returns
        the output for the market and a set of any errors encountered during computation.
        """
        errors: List[Error] = []

        # keep track of how long it takes to compute the arrays
        start_time = time.time()

        # define a factory for computing arrays in markets
        def market_factory(s: Hashable) -> tuple:
            """Build a market along with arguments used to compute arrays."""
            market_s = ResultsMarket(self.problem, s, self.sigma, self.pi, self.rho, self.beta, self.true_delta)
            args_s = [None if a is None else a[self.problem._product_market_indices[s]] for a in market_args]
            return (market_s, *fixed_args, *args_s)

        # construct a mapping from market IDs to market-specific arrays
        matrix_mapping: Dict[Hashable, Array] = {}
        generator = output_progress(
            generate_items(self.problem.unique_market_ids, market_factory, compute_market_results), self.problem.T,
            start_time
        )
        for t, (array_t, errors_t) in generator:
            matrix_mapping[t] = np.c_[array_t]
            errors.extend(errors_t)

        # output a warning about any errors
        if errors:
            output("")
            output(exceptions.MultipleErrors(errors))
            output("")

        # determine the number of rows and columns
        row_count = sum(matrix_mapping[t].shape[0] for t in self.problem.unique_market_ids)
        column_count = max(matrix_mapping[t].shape[1] for t in self.problem.unique_market_ids)

        # preserve the original product order or the sorted market order when stacking the arrays
        combined = np.full((row_count, column_count), np.nan, options.dtype)
        for t, matrix_t in matrix_mapping.items():
            if row_count == self.problem.N:
                combined[self.problem._product_market_indices[t], :matrix_t.shape[1]] = matrix_t
            else:
                combined[self.problem.unique_market_ids == t, :matrix_t.shape[1]] = matrix_t

        # output how long it took to compute the arrays
        end_time = time.time()
        output(f"Finished after {format_seconds(end_time - start_time)}.")
        output("")
        return combined
