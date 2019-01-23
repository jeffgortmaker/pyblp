"""Economy-level structuring of BLP problem results."""

import time
from typing import Any, Callable, Dict, Hashable, List, Optional, Sequence, TYPE_CHECKING, Tuple

import numpy as np
import scipy.linalg

from .results import Results
from .. import exceptions, options
from ..configurations.iteration import Iteration
from ..markets.results_market import ResultsMarket
from ..parameters import Parameters
from ..utilities.algebra import approximately_solve, multiply_matrix_and_tensor, precisely_compute_eigenvalues
from ..utilities.basics import (
    Array, Bounds, Error, TableFormatter, format_number, format_seconds, generate_items, output, output_progress
)
from ..utilities.statistics import compute_gmm_parameter_covariances, compute_gmm_weights


# only import objects that create import cycles when checking types
if TYPE_CHECKING:
    from .bootstrapped_results import BootstrappedResults  # noqa
    from .optimal_instrument_results import OptimalInstrumentResults  # noqa
    from ..economies.problem import Progress  # noqa


class ProblemResults(Results):
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
    fp_converged : `ndarray`
        Flags for convergence of the iteration routine used to compute :math:`\delta(\hat{\theta})` in each market
        during each objective evaluation. Rows are in the same order as :attr:`Problem.unique_market_ids` and column
        indices correspond to objective evaluations.
    cumulative_fp_converged : `ndarray`
        Concatenation of :attr:`ProblemResults.fp_converged` for this step and all prior steps.
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
    converged : `bool`
        Whether the optimization routine converged.
    cumulative_converged : `bool`
        Whether the optimization routine converged for this step and all prior steps.
    parameters : `ndarray`
        Stacked parameters in the following order: :math:`\hat{\theta}`, concentrated out elements of
        :math:`\hat{\beta}`, and concentrated out elements of :math:`\hat{\gamma}`.
    parameter_covariances : `ndarray`
        Estimated covariance matrix of the stacked parameters, from which standard errors are extracted.
    theta : `ndarray`
        Estimated unfixed parameters, :math:`\hat{\theta}` in the following order: :math:`\hat{\Sigma}`,
        :math:`\hat{\Pi}`, :math:`\hat{\rho}`, non-concentrated out elements from :math:`\hat{\beta}`, and
        non-concentrated out elements from :math:`\hat{\gamma}`.
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
        Estimated standard errors for :math:`\hat{\Sigma}`.
    pi_se : `ndarray`
        Estimated standard errors for :math:`\hat{\Pi}`.
    rho_se : `ndarray`
        Estimated standard errors for :math:`\hat{\rho}`.
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
    beta_bounds : `tuple`
        Bounds for :math:`\beta` that were used during optimization, which are of the form ``(lb, ub)``.
    gamma_bounds : `tuple`
        Bounds for :math:`\gamma` that were used during optimization, which are of the form ``(lb, ub)``.
    delta : `ndarray`
        Estimated mean utility, :math:`\delta(\hat{\theta})`.
    tilde_costs : `ndarray`
        Estimated transformed marginal costs, :math:`\tilde{c}(\hat{\theta})`. Transformed marginal costs are simply
        :math:`\tilde{c} = c`, marginal costs, under a linear cost specification, and are :math:`\tilde{c} = \log c`
        under a log-linear specification. If ``costs_bounds`` were specified in :meth:`Problem.solve`, :math:`c` may
        have been clipped.
    clipped_costs : `ndarray`
        Vector of booleans indicating whether the associated marginal costs were clipped. All elements will be ``False``
        if ``costs_bounds`` in :meth:`Problem.solve` was not specified.
    xi : `ndarray`
        Estimated unobserved demand-side product characteristics, :math:`\xi(\hat{\theta})`, or equivalently, the
        demand-side structural error term.
    omega : `ndarray`
        Estimated unobserved supply-side product characteristics, :math:`\omega(\hat{\theta})`, or equivalently, the
        supply-side structural error term.
    objective : `float`
        GMM objective value.
    xi_by_theta_jacobian : `ndarray`
        Estimated :math:`\partial\xi / \partial\theta = \partial\delta / \partial\theta`.
    omega_by_theta_jacobian : `ndarray`
        Estimated :math:`\partial\omega / \partial\theta = \partial\tilde{c} / \partial\theta`.
    gradient : `ndarray`
        Estimated gradient of the GMM objective with respect to :math:`\theta`, which is computed after the optimization
        routine finishes even if the routine was configured to not use analytic gradients.
    sigma_gradient : `ndarray`
        Estimated gradient of the GMM objective with respect to :math:`\Sigma` elements in :math:`\theta`.
    pi_gradient : `ndarray`
        Estimated gradient of the GMM objective with respect to :math:`\Pi` elements in :math:`\theta`.
    rho_gradient : `ndarray`
        Estimated gradient of the GMM objective with respect to :math:`\rho` elements in :math:`\theta`.
    beta_gradient : `ndarray`
        Estimated gradient of the GMM objective with respect to :math:`\beta` elements in :math:`\theta`.
    gamma_gradient : `ndarray`
        Estimated gradient of the GMM objective with respect to :math:`\gamma` elements in :math:`\theta`.
    gradient_norm : `ndarray`
        Infinity norm of :attr:`ProblemResults.gradient`.
    hessian : `ndarray`
        Estimated Hessian of the GMM objective with respect to :math:`\theta`. By default, this is computed with finite
        central differences after the optimization routine finishes.
    hessian_eigenvalues : `ndarray`
        Eigenvalues of :attr:`ProblemResults.hessian`.
    W : `ndarray`
        Weighting matrix, :math:`W`, used to compute these results.
    updated_W : `ndarray`
        Updated weighting matrix.

    Examples
    --------
        - :doc:`Tutorial </tutorial>`

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
    fp_converged: Array
    cumulative_fp_converged: Array
    fp_iterations: Array
    cumulative_fp_iterations: Array
    contraction_evaluations: Array
    cumulative_contraction_evaluations: Array
    converged: bool
    cumulative_converged: bool
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
    sigma_bounds: Bounds
    pi_bounds: Bounds
    rho_bounds: Bounds
    beta_bounds: Bounds
    gamma_bounds: Bounds
    delta: Array
    tilde_costs: Array
    clipped_costs: Array
    xi: Array
    omega: Array
    objective: Array
    xi_by_theta_jacobian: Array
    omega_by_theta_jacobian: Array
    gradient: Array
    gradient_norm: Array
    hessian: Array
    hessian_eigenvalues: Array
    sigma_gradient: Array
    pi_gradient: Array
    rho_gradient: Array
    beta_gradient: Array
    gamma_gradient: Array
    W: Array
    updated_W: Array
    _costs_type: str
    _se_type: str
    _errors: List[Error]
    _parameters: Parameters

    def __init__(
            self, progress: 'Progress', last_results: Optional['ProblemResults'], step_start_time: float,
            optimization_start_time: float, optimization_end_time: float, iterations: int, evaluations: int,
            converged_mappings: Sequence[Dict[Hashable, bool]], iteration_mappings: Sequence[Dict[Hashable, int]],
            evaluation_mappings: Sequence[Dict[Hashable, int]], converged: bool, costs_type: str, costs_bounds: Bounds,
            center_moments: bool, W_type: str, se_type: str) -> None:
        """Compute cumulative progress statistics, update weighting matrices, and estimate standard errors."""

        # initialize values from the progress structure
        super().__init__(progress.problem)
        self._errors = progress.errors
        self.problem = progress.problem
        self.W = progress.W
        self.theta = progress.theta
        self.delta = progress.delta
        self.tilde_costs = progress.tilde_costs
        self.xi_by_theta_jacobian = progress.xi_jacobian
        self.omega_by_theta_jacobian = progress.omega_jacobian
        self.xi = progress.xi
        self.omega = progress.omega
        self.beta = progress.beta
        self.gamma = progress.gamma
        self.objective = progress.objective
        self.gradient = progress.gradient
        self.gradient_norm = progress.gradient_norm
        self.hessian = progress.hessian

        # if the Hessian was computed, compute its eigenvalues and the ratio of the smallest to largest ones
        self.hessian_eigenvalues = np.full(progress.parameters.P, np.nan, options.dtype)
        if progress.parameters.P > 0 and np.isfinite(self.hessian).all():
            self.hessian_eigenvalues, successful = precisely_compute_eigenvalues(self.hessian)
            if not successful:
                self._errors.append(exceptions.HessianEigenvaluesError(self.hessian))

        # store information about cost bounds
        self._costs_bounds = costs_bounds
        self.clipped_costs = progress.clipped_costs

        # initialize counts, times, and convergence
        self.step = 1
        self.total_time = self.cumulative_total_time = time.time() - step_start_time
        self.optimization_time = self.cumulative_optimization_time = optimization_end_time - optimization_start_time
        self.optimization_iterations = self.cumulative_optimization_iterations = iterations
        self.objective_evaluations = self.cumulative_objective_evaluations = evaluations
        self.fp_converged = self.cumulative_fp_converged = np.array(
            [[m[t] if m else True for m in converged_mappings] for t in self.problem.unique_market_ids],
            dtype=np.int
        )
        self.fp_iterations = self.cumulative_fp_iterations = np.array(
            [[m[t] if m else 0 for m in iteration_mappings] for t in self.problem.unique_market_ids],
            dtype=np.int
        )
        self.contraction_evaluations = self.cumulative_contraction_evaluations = np.array(
            [[m[t] if m else 0 for m in evaluation_mappings] for t in self.problem.unique_market_ids],
            dtype=np.int
        )
        self.converged = self.cumulative_converged = converged

        # initialize last results and add to cumulative values
        self.last_results = last_results
        if last_results is not None:
            self.step += last_results.step
            self.cumulative_total_time += last_results.cumulative_total_time
            self.cumulative_optimization_time += last_results.cumulative_optimization_time
            self.cumulative_optimization_iterations += last_results.cumulative_optimization_iterations
            self.cumulative_objective_evaluations += last_results.cumulative_objective_evaluations
            self.cumulative_fp_converged = np.c_[
                last_results.cumulative_fp_converged, self.cumulative_fp_converged
            ]
            self.cumulative_fp_iterations = np.c_[
                last_results.cumulative_fp_iterations, self.cumulative_fp_iterations
            ]
            self.cumulative_contraction_evaluations = np.c_[
                last_results.cumulative_contraction_evaluations, self.cumulative_contraction_evaluations
            ]
            self.cumulative_converged = last_results.converged and converged

        # store estimated parameters and information about them (beta and gamma have already been stored above)
        self._parameters = progress.parameters
        self.sigma, self.pi, self.rho, _, _ = self._parameters.expand(self.theta)
        self.parameters = np.c_[np.r_[
            self.theta,
            self.beta[self._parameters.eliminated_beta_index],
            self.gamma[self._parameters.eliminated_gamma_index]
        ]]
        self.sigma_bounds = self._parameters.sigma_bounds
        self.pi_bounds = self._parameters.pi_bounds
        self.rho_bounds = self._parameters.rho_bounds
        self.beta_bounds = self._parameters.beta_bounds
        self.gamma_bounds = self._parameters.gamma_bounds

        # collect inputs to weighting matrix and standard error computation
        u_list = [self.xi]
        Z_list = [self.problem.products.ZD]
        jacobian_list = [np.c_[
            self.xi_by_theta_jacobian,
            -self.problem.products.X1[:, self._parameters.eliminated_beta_index.flat],
            np.zeros_like(self.problem.products.X3[:, self._parameters.eliminated_gamma_index.flat])
        ]]
        if self.problem.K3 > 0:
            u_list.append(self.omega)
            Z_list.append(self.problem.products.ZS)
            jacobian_list.append(np.c_[
                self.omega_by_theta_jacobian,
                np.zeros_like(self.problem.products.X1[:, self._parameters.eliminated_beta_index.flat]),
                -self.problem.products.X3[:, self._parameters.eliminated_gamma_index.flat]
            ])

        # update the weighting matrix
        with np.errstate(invalid='ignore'):
            self.updated_W, W_errors = compute_gmm_weights(
                u_list, Z_list, W_type, self.problem.products.clustering_ids, center_moments
            )
        self._errors.extend(W_errors)

        # compute parameter covariances (if this is the first step, an unadjusted weighting matrix needs to be used so
        #   that unadjusted covariances are scaled properly)
        update_W = se_type == 'unadjusted' and self.step == 1
        with np.errstate(all='ignore'):
            self.parameter_covariances, covariance_errors = compute_gmm_parameter_covariances(
                jacobian_list, u_list, Z_list, self.W, se_type, self.problem.products.clustering_ids, update_W
            )
        self._errors.extend(covariance_errors)

        # compute standard errors
        with np.errstate(invalid='ignore'):
            se = np.sqrt(np.c_[self.parameter_covariances.diagonal()] / self.problem.N)
        if np.isnan(se).any():
            self._errors.append(exceptions.InvalidParameterCovariancesError())

        # expand standard errors
        theta_se, eliminated_beta_se, eliminated_gamma_se = np.split(se, [
            self._parameters.P,
            self._parameters.P + self._parameters.eliminated_beta_index.sum()
        ])
        self.sigma_se, self.pi_se, self.rho_se, self.beta_se, self.gamma_se = (
            self._parameters.expand(theta_se, nullify=True)
        )
        self.beta_se[self._parameters.eliminated_beta_index] = eliminated_beta_se.flatten()
        self.gamma_se[self._parameters.eliminated_gamma_index] = eliminated_gamma_se.flatten()

        # expand gradients
        self.sigma_gradient, self.pi_gradient, self.rho_gradient, self.beta_gradient, self.gamma_gradient = (
            self._parameters.expand(self.gradient, nullify=True)
        )

        # store types that are used in other methods
        self._costs_type = costs_type
        self._se_type = se_type

    def __str__(self) -> str:
        """Format problem results (including parameters estimates) as a string."""

        # construct a standard error description
        if self._se_type == 'unadjusted':
            se_description = "Unadjusted SEs"
        elif self._se_type == 'robust':
            se_description = "Robust SEs"
        else:
            assert self._se_type == 'clustered'
            se_description = f'Robust SEs Adjusted for {np.unique(self.problem.products.clustering_ids).size} Clusters'

        # combine a summary table section and another with formatted estimates into one string
        return "\n\n".join([
            self._format_summary(),
            self._parameters.format_estimates(
                f"Estimates ({se_description} in Parentheses)", self.sigma, self.pi, self.rho, self.beta, self.gamma,
                self.sigma_se, self.pi_se, self.rho_se, self.beta_se, self.gamma_se
            )
        ])

    def _format_summary(self) -> str:
        """Format a summary table of problem results."""

        # at a minimum include time and the GMM step
        floats_index = 3
        header = [("", "Computation", "Time"), ("", "GMM", "Step")]
        values = [format_seconds(self.cumulative_total_time), self.step]

        # include any optimization information
        if self._parameters.P > 0:
            floats_index += 1
            header.append(("", "Optimization", "Iterations"))
            values.append(self.cumulative_optimization_iterations)

        # include objective evaluation information
        header.append(("", "Objective", "Evaluations"))
        values.append(self.cumulative_objective_evaluations)

        # include any fixed point information
        if np.any(self.cumulative_contraction_evaluations > 0):
            floats_index += 2
            header.extend([("", "Fixed Point", "Iterations"), ("", "Contraction", "Evaluations")])
            values.extend([self.cumulative_fp_iterations.sum(), self.cumulative_contraction_evaluations.sum()])

        # include any information about the final objective value
        header.append(("", "Objective", "Value"))
        values.append(format_number(float(self.objective)))
        if np.isfinite(self.gradient_norm):
            header.append(("", "Gradient", "Infinity Norm"))
            values.append(format_number(float(self.gradient_norm)))
        if np.isfinite(self.hessian_eigenvalues).any():
            if self.hessian_eigenvalues.size == 1:
                header.append(("", "Hessian", "Eigenvalue"))
                values.append(format_number(float(self.hessian_eigenvalues)))
            else:
                header.extend([
                    ("Smallest", "Hessian", "Eigenvalue"),
                    ("Largest", "Hessian", "Eigenvalue")
                ])
                values.extend([
                    format_number(float(np.min(self.hessian_eigenvalues))),
                    format_number(float(np.max(self.hessian_eigenvalues)))
                ])

        # include any information about clipped marginal costs
        if np.isfinite(self._costs_bounds).any():
            header.append(("Clipped", "Marginal", "Costs"))
            values.append(self.clipped_costs.sum())

        # format the table
        widths = [max(options.digits + 6 if i >= floats_index else 0, *map(len, k)) for i, k in enumerate(header)]
        formatter = TableFormatter(widths)
        lines = [
            "Problem Results Summary:",
            formatter.line()
        ]
        if any(k[0] for k in header):
            lines.append(formatter([k[0] for k in header]))
        lines.extend([
            formatter([k[1] for k in header]),
            formatter([k[2] for k in header], underline=True),
            formatter(values),
            formatter.line()
        ])
        return "\n".join(lines)

    def bootstrap(
            self, draws: int = 1000, seed: Optional[int] = None, iteration: Optional[Iteration] = None) -> (
            'BootstrappedResults'):
        r"""Use a parametric bootstrap to create an empirical distribution of results.

        The constructed :class:`BootstrappedResults` can be used just like :class:`ProblemResults` to compute various
        post-estimation outputs. The only difference is that :class:`BootstrappedResults` methods return arrays with an
        extra first dimension, along which bootstrapped results are stacked. These stacked results can be used to
        construct, for example, confidence intervals for post-estimation outputs.

        For each bootstrap draw, parameters are drawn from the estimated multivariate normal distribution of all
        parameters defined by :attr:`ProblemResults.parameters` and :attr:`ProblemResults.parameter_covariances`. Note
        that any bounds configured during the optimization routine will be used to bound parameter draws. These
        parameters are used to compute the implied mean utility, :math:`\delta`, and shares, :math:`s`. If a supply side
        was estimated, the implied marginal costs, :math:`c`, and prices, :math:`p`, are computed as well. Specifically,
        if a supply side was estimated, equilibrium prices and shares are computed by iterating over the
        :math:`\zeta`-markup equation from :ref:`references:Morrow and Skerlos (2011)`.

        .. note::

           By default, the bootstrapping procedure may use a lot of memory. This is because it stores in memory all
           bootstrapped results (for all ``draws``) at the same time. To reduce the memory footprint of the procedure,
           call this method in a loop with ``draws`` set to ``1``. In each iteration of the loop, compute the desired
           post-estimation output with the proper method of the returned :class:`BootstrappedResults` class and store
           these outputs.

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
            :math:`\zeta`-markup equation from :ref:`references:Morrow and Skerlos (2011)`. By default, if a supply side
            was estimated, this is ``Iteration('simple', {'atol': 1e-12})``. Analytic Jacobians are not supported for
            this contraction mapping and this configuration is not used if a supply side was not estimated.

        Returns
        -------
        `BootstrappedResults`
            Computed :class:`BootstrappedResults`.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

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
            iteration = Iteration('simple', {'atol': 1e-12})
        elif not isinstance(iteration, Iteration):
            raise TypeError("iteration must be None or an iteration instance.")
        elif iteration._compute_jacobian:
            raise ValueError("Analytic Jacobians are not supported for this contraction mapping.")

        # draw from the asymptotic distribution implied by the estimated parameters
        state = np.random.RandomState(seed)
        bootstrapped_parameters = np.atleast_3d(state.multivariate_normal(
            self.parameters.flatten(), self.parameter_covariances, draws
        ))

        # extract the parameters
        bootstrapped_sigma = np.zeros((draws, self.sigma.shape[0], self.sigma.shape[1]), options.dtype)
        bootstrapped_pi = np.zeros((draws, self.pi.shape[0], self.pi.shape[1]), options.dtype)
        bootstrapped_rho = np.zeros((draws, self.rho.shape[0], self.rho.shape[1]), options.dtype)
        bootstrapped_beta = np.zeros((draws, self.beta.shape[0], self.beta.shape[1]), options.dtype)
        bootstrapped_gamma = np.zeros((draws, self.gamma.shape[0], self.gamma.shape[1]), options.dtype)
        bootstrapped_theta, bootstrapped_eliminated_beta, bootstrapped_eliminated_gamma = np.split(
            bootstrapped_parameters,
            [self._parameters.P, self._parameters.P + self._parameters.eliminated_beta_index.sum()],
            axis=1
        )
        bootstrapped_beta[:, self._parameters.eliminated_beta_index.flat] = bootstrapped_eliminated_beta
        bootstrapped_gamma[:, self._parameters.eliminated_gamma_index.flat] = bootstrapped_eliminated_gamma
        for d in range(draws):
            bootstrapped_sigma[d], bootstrapped_pi[d], bootstrapped_rho[d], beta_d, gamma_d = self._parameters.expand(
                bootstrapped_theta[d]
            )
            bootstrapped_beta[d] = np.where(self._parameters.eliminated_beta_index, bootstrapped_beta[d], beta_d)
            bootstrapped_gamma[d] = np.where(self._parameters.eliminated_gamma_index, bootstrapped_gamma[d], gamma_d)
            bootstrapped_sigma[d] = np.clip(bootstrapped_sigma[d], *self.sigma_bounds)
            bootstrapped_pi[d] = np.clip(bootstrapped_pi[d], *self.pi_bounds)
            bootstrapped_rho[d] = np.clip(bootstrapped_rho[d], *self.rho_bounds)
            bootstrapped_beta[d] = np.clip(bootstrapped_beta[d], *self.beta_bounds)
            bootstrapped_gamma[d] = np.clip(bootstrapped_gamma[d], *self.gamma_bounds)

        # compute bootstrapped prices, shares, delta and marginal costs
        converged_mappings: List[Dict[Hashable, bool]] = []
        iteration_mappings: List[Dict[Hashable, int]] = []
        evaluation_mappings: List[Dict[Hashable, int]] = []
        bootstrapped_prices = np.zeros((draws, self.problem.N, 1), options.dtype)
        bootstrapped_shares = np.zeros((draws, self.problem.N, 1), options.dtype)
        bootstrapped_delta = np.zeros((draws, self.problem.N, 1), options.dtype)
        bootstrapped_costs = np.zeros((draws, self.problem.N, int(self.problem.K3 > 0)), options.dtype)
        for d in output_progress(range(draws), draws, start_time):
            prices_d, shares_d, delta_d, costs_d, converged_d, iterations_d, evaluations_d, errors_d = (
                self._compute_bootstrap(
                    iteration, bootstrapped_sigma[d], bootstrapped_pi[d], bootstrapped_rho[d], bootstrapped_beta[d],
                    bootstrapped_gamma[d]
                )
            )
            bootstrapped_prices[d] = prices_d
            bootstrapped_shares[d] = shares_d
            bootstrapped_delta[d] = delta_d
            bootstrapped_costs[d] = costs_d
            converged_mappings.append(converged_d)
            iteration_mappings.append(iterations_d)
            evaluation_mappings.append(evaluations_d)
            errors.extend(errors_d)

        # structure the results
        from .bootstrapped_results import BootstrappedResults  # noqa
        results = BootstrappedResults(
            self, bootstrapped_sigma, bootstrapped_pi, bootstrapped_rho, bootstrapped_beta, bootstrapped_gamma,
            bootstrapped_prices, bootstrapped_shares, bootstrapped_delta, bootstrapped_costs, start_time, time.time(),
            draws, converged_mappings, iteration_mappings, evaluation_mappings
        )
        output(f"Bootstrapped results after {format_seconds(results.computation_time)}.")
        output("")
        output(results)
        return results

    def _compute_bootstrap(
            self, iteration: Optional[Iteration], sigma: Array, pi: Array, rho: Array, beta: Array, gamma: Array) -> (
            Tuple[
                Array, Array, Array, Array, Dict[Hashable, bool], Dict[Hashable, int], Dict[Hashable, int], List[Error]
            ]):
        """Compute the equilibrium prices, shares, marginal costs, and delta associated with bootstrapped parameters
        market-by-market
        """
        errors: List[Error] = []

        # compute delta (which will change under equilibrium prices) and marginal costs (which won't change)
        delta = self.delta + self.problem._compute_true_X1() @ (beta - self.beta)
        costs = self.tilde_costs + self.problem._compute_true_X3() @ (gamma - self.gamma)
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
        converged_mapping: Dict[Hashable, bool] = {}
        iteration_mapping: Dict[Hashable, int] = {}
        evaluation_mapping: Dict[Hashable, int] = {}
        equilibrium_prices = np.zeros_like(self.problem.products.prices)
        equilibrium_shares = np.zeros_like(self.problem.products.shares)
        generator = generate_items(self.problem.unique_market_ids, market_factory, ResultsMarket.solve_equilibrium)
        for t, (prices_t, shares_t, delta_t, errors_t, converged_t, iterations_t, evaluations_t) in generator:
            equilibrium_prices[self.problem._product_market_indices[t]] = prices_t
            equilibrium_shares[self.problem._product_market_indices[t]] = shares_t
            delta[self.problem._product_market_indices[t]] = delta_t
            errors.extend(errors_t)
            converged_mapping[t] = converged_t
            iteration_mapping[t] = iterations_t
            evaluation_mapping[t] = evaluations_t

        # return all of the information associated with this bootstrap draw
        return (
            equilibrium_prices, equilibrium_shares, delta, costs, converged_mapping, iteration_mapping,
            evaluation_mapping, errors
        )

    def compute_optimal_instruments(
            self, method: str = 'normal', draws: int = 100, seed: Optional[int] = None,
            expected_prices: Optional[Any] = None, iteration: Optional[Iteration] = None) -> 'OptimalInstrumentResults':
        r"""Estimate optimal or efficient instruments, :math:`\mathscr{Z}_D` and :math:`\mathscr{Z}_S`.

        Optimal instruments have been shown, for example, by :ref:`references:Reynaert and Verboven (2014)`, to reduce
        bias, improve efficiency, and enhance stability of BLP estimates.

        In the spirit of :ref:`references:Chamberlain (1987)`, optimal instruments for :math:`\theta` are

        .. math::

           \begin{bmatrix}
               \mathscr{Z}_D \\
               \mathscr{Z}_S
           \end{bmatrix}_{jt}
           = \text{Var}(\xi, \omega)^{-1}\operatorname{\mathbb{E}}\left[
           \begin{matrix}
               \frac{\partial\xi_{jt}}{\partial\theta} \\
               \frac{\partial\omega_{jt}}{\partial\theta}
           \end{matrix}
           \mathrel{\Bigg|} Z \right],

        The expectation is taken by integrating over the joint density of :math:`\xi` and :math:`\omega`. For each error
        term realization, if not already estimated, equilibrium prices are computed via iteration over the
        :math:`\zeta`-markup equation from :ref:`references:Morrow and Skerlos (2011)`. Associated shares and
        :math:`\delta` are then computed before each Jacobian is evaluated.

        The expected Jacobians are estimated with the average over all computed Jacobian realizations. The normalizing
        matrix :math:`\text{Var}(\xi, \omega)^{-1}` is estimated with the sample covariance matrix of the error terms.

        Optimal instruments for linear parameters not included in :math:`\theta` are simple product characteristics, so
        they are not computed here but are rather included in the set of instruments by
        :meth:`OptimalInstrumentResults.to_problem`.

        Parameters
        ----------
        method : `str, optional`
            The method by which the integral over the joint density of :math:`\xi` and :math:`\omega` is computed. The
            following methods are supported:

                - ``'normal'`` (default) - Draw from the normal approximation to the joint distribution of the error
                  terms and take the average over the computed Jacobians (``draws`` determines the number of draws).

                - ``'empirical'`` - Draw with replacement from the empirical joint distribution of the error terms and
                  take the average over the computed Jacobians (``draws`` determines the number of draws).

                - ``'approximate'`` - Evaluate the Jacobians at the expected value of the error terms: zero (``draws``
                  will be ignored).

        draws : `int, optional`
            The number of draws that will be taken from the joint distribution of the error terms. This is ignored if
            ``method`` is ``'approximate'``. The default is ``100``.
        seed : `int, optional`
            Passed to :class:`numpy.random.RandomState` to seed the random number generator before any draws are taken.
            By default, a seed is not passed to the random number generator.
        expected_prices : `array-like, optional`
            Vector of expected prices conditional on all exogenous variables,
            :math:`\operatorname{\mathbb{E}}[p \mid Z]`. By default, if a supply side was estimated, ``iteration`` is
            used. If only a demand side was estimated, this is by default estimated with the fitted values from a
            reduced form regression of endogenous prices onto :math:`Z_D`: all exogenous variables including excluded
            instruments.
        iteration : `Iteration, optional`
            :class:`Iteration` configuration used to estimate expected prices by iterating over the :math:`\zeta`-markup
            equation from :ref:`references:Morrow and Skerlos (2011)`. By default, if a supply side was estimated, this
            is ``Iteration('simple', {'atol': 1e-12})``. Analytic Jacobians are not supported for this contraction
            mapping and this configuration is not used if ``expected_prices`` is specified.

        Returns
        -------
        `OptimalInstrumentResults`
           Computed :class:`OptimalInstrumentResults`.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        errors: List[Error] = []

        # keep track of long it takes to compute optimal instruments for theta
        output("Computing optimal instruments for theta ...")
        start_time = time.time()

        # validate the method and create a function that samples from the error distribution
        if method == 'approximate':
            sample = lambda: (np.zeros_like(self.xi), np.zeros_like(self.omega))
        else:
            state = np.random.RandomState(seed)
            if method == 'normal':
                if self.problem.K3 == 0:
                    variance = np.var(self.xi)
                    sample = lambda: (np.c_[state.normal(0, variance, self.problem.N)], self.omega)
                else:
                    covariances = np.cov(self.xi, self.omega, rowvar=False)
                    sample = lambda: np.hsplit(state.multivariate_normal([0, 0], covariances, self.problem.N), 2)
            elif method == 'empirical':
                if self.problem.K3 == 0:
                    sample = lambda: (self.xi[state.choice(self.problem.N, self.problem.N)], self.omega)
                else:
                    joint = np.c_[self.xi, self.omega]
                    sample = lambda: np.hsplit(joint[state.choice(self.problem.N, self.problem.N)], 2)
            else:
                raise ValueError("method must be 'approximate', 'normal', or 'empirical'.")

        # validate the number of draws (there will be only one for the approximate method)
        if method == 'approximate':
            draws = 1
        if not isinstance(draws, int) or draws < 1:
            raise ValueError("draws must be a positive int.")

        # validate expected prices or their integration configuration (or compute expected prices with a reduced form
        #   regression if unspecified and only a demand side)
        if expected_prices is not None:
            iteration = None
            expected_prices = np.c_[np.asarray(expected_prices, options.dtype)]
            if expected_prices.shape != (self.problem.N, 1):
                raise ValueError(f"expected_prices must be a {self.problem.N}-vector.")
        elif self.problem.K3 > 0:
            if iteration is None:
                iteration = Iteration('simple', {'atol': 1e-12})
            elif not isinstance(iteration, Iteration):
                raise TypeError("iteration must be None or an Iteration instance.")
        else:
            prices = self.problem.products.prices
            if self.problem._absorb_demand_ids is not None:
                prices, absorption_errors = self.problem._absorb_demand_ids(prices)
                errors.extend(absorption_errors)
            covariances = self.problem.products.ZD.T @ self.problem.products.ZD
            parameters, replacement = approximately_solve(covariances, self.problem.products.ZD.T @ prices)
            if replacement:
                errors.append(exceptions.FittedValuesInversionError(covariances, replacement))
            expected_prices = self.problem.products.ZD @ parameters + self.problem.products.prices - prices

        # average over Jacobian realizations
        converged_mappings: List[Dict[Hashable, bool]] = []
        iteration_mappings: List[Dict[Hashable, int]] = []
        evaluation_mappings: List[Dict[Hashable, int]] = []
        expected_xi_jacobian = np.zeros_like(self.xi_by_theta_jacobian)
        expected_omega_jacobian = np.zeros_like(self.omega_by_theta_jacobian)
        for _ in output_progress(range(draws), draws, start_time):
            xi_jacobian_i, omega_jacobian_i, converged_i, iterations_i, evaluations_i, errors_i = (
                self._compute_realizations(expected_prices, iteration, *sample())
            )
            expected_xi_jacobian += xi_jacobian_i / draws
            expected_omega_jacobian += omega_jacobian_i / draws
            converged_mappings.append(converged_i)
            iteration_mappings.append(iterations_i)
            evaluation_mappings.append(evaluations_i)
            errors.extend(errors_i)

        # output a warning about any errors
        if errors:
            output("")
            output(exceptions.MultipleErrors(errors))
            output("")

        # compute the optimal instruments
        if self.problem.K3 == 0:
            inverse_covariance_matrix = np.c_[1 / np.var(self.xi)]
            demand_instruments = inverse_covariance_matrix * expected_xi_jacobian
            supply_instruments = np.full((self.problem.N, 0), np.nan, options.dtype)
        else:
            inverse_covariance_matrix = np.c_[scipy.linalg.inv(np.cov(self.xi, self.omega, rowvar=False))]
            instruments = multiply_matrix_and_tensor(
                inverse_covariance_matrix,
                np.stack([expected_xi_jacobian, expected_omega_jacobian], axis=1)
            )
            demand_instruments, supply_instruments = np.split(instruments.reshape((self.problem.N, -1)), 2, axis=1)

        # structure the results
        from .optimal_instrument_results import OptimalInstrumentResults  # noqa
        results = OptimalInstrumentResults(
            self, demand_instruments, supply_instruments, inverse_covariance_matrix, expected_xi_jacobian,
            expected_omega_jacobian, expected_prices, start_time, time.time(), draws, converged_mappings,
            iteration_mappings, evaluation_mappings
        )
        output(f"Computed optimal instruments after {format_seconds(results.computation_time)}.")
        output("")
        output(results)
        return results

    def _compute_realizations(
            self, expected_prices: Optional[Array], iteration: Optional[Iteration], xi: Array, omega: Array) -> (
            Tuple[Array, Array, Dict[Hashable, bool], Dict[Hashable, int], Dict[Hashable, int], List[Error]]):
        """If they have not already been estimated, compute the equilibrium prices, shares, and delta associated with a
        realization of xi and omega market-by-market. Then, compute realizations of Jacobians of xi and omega with
        respect to theta.
        """
        errors: List[Error] = []

        # compute delta (which will change under equilibrium prices) and marginal costs (which won't change)
        delta = self.delta - self.xi + xi
        costs = tilde_costs = self.tilde_costs - self.omega + omega
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
        converged_mapping: Dict[Hashable, bool] = {}
        iteration_mapping: Dict[Hashable, int] = {}
        evaluation_mapping: Dict[Hashable, int] = {}
        equilibrium_prices = np.zeros_like(self.problem.products.prices)
        equilibrium_shares = np.zeros_like(self.problem.products.shares)
        generator = generate_items(self.problem.unique_market_ids, market_factory, ResultsMarket.solve_equilibrium)
        for t, (prices_t, shares_t, delta_t, errors_t, converged_t, iterations_t, evaluations_t) in generator:
            equilibrium_prices[self.problem._product_market_indices[t]] = prices_t
            equilibrium_shares[self.problem._product_market_indices[t]] = shares_t
            delta[self.problem._product_market_indices[t]] = delta_t
            errors.extend(errors_t)
            converged_mapping[t] = converged_t
            iteration_mapping[t] = iterations_t
            evaluation_mapping[t] = evaluations_t

        # compute the Jacobian of xi with respect to theta
        xi_jacobian, demand_errors = self._compute_demand_realization(equilibrium_prices, equilibrium_shares, delta)
        errors.extend(demand_errors)

        # compute the Jacobian of omega with respect to theta
        omega_jacobian = np.full((self.problem.N, self._parameters.P), np.nan, options.dtype)
        if self.problem.K3 > 0:
            omega_jacobian, supply_errors = self._compute_supply_realization(
                equilibrium_prices, equilibrium_shares, delta, tilde_costs, xi_jacobian
            )
            errors.extend(supply_errors)

        # return all of the information associated with this realization
        return xi_jacobian, omega_jacobian, converged_mapping, iteration_mapping, evaluation_mapping, errors

    def _compute_demand_realization(
            self, equilibrium_prices: Array, equilibrium_shares: Array, delta: Array) -> Tuple[Array, List[Error]]:
        """Compute a realization of the Jacobian of xi with respect to theta market-by-market. If necessary, revert
        problematic elements to their estimated values.
        """
        errors: List[Error] = []

        # check if the Jacobian does not need to be computed
        xi_jacobian = np.full((self.problem.N, self._parameters.P), np.nan, options.dtype)
        if self._parameters.P == 0:
            return xi_jacobian, errors

        # define a factory for computing the Jacobian of xi with respect to theta in markets
        def market_factory(s: Hashable) -> Tuple[ResultsMarket, Parameters]:
            """Build a market with the data realization along with arguments used to compute the Jacobian."""
            data_override_s = {
                'prices': equilibrium_prices[self.problem._product_market_indices[s]],
                'shares': equilibrium_shares[self.problem._product_market_indices[s]]
            }
            market_s = ResultsMarket(self.problem, s, self.sigma, self.pi, self.rho, self.beta, delta, data_override_s)
            return market_s, self._parameters

        # compute the Jacobian market-by-market
        generator = generate_items(
            self.problem.unique_market_ids, market_factory, ResultsMarket.compute_xi_by_theta_jacobian
        )
        for t, (xi_jacobian_t, errors_t) in generator:
            xi_jacobian[self.problem._product_market_indices[t]] = xi_jacobian_t
            errors.extend(errors_t)

        # replace invalid elements
        bad_jacobian_index = ~np.isfinite(xi_jacobian)
        if np.any(bad_jacobian_index):
            xi_jacobian[bad_jacobian_index] = self.xi_by_theta_jacobian[bad_jacobian_index]
            errors.append(exceptions.XiByThetaJacobianReversionError(bad_jacobian_index))
        return xi_jacobian, errors

    def _compute_supply_realization(
            self, equilibrium_prices: Array, equilibrium_shares: Array, delta: Array, tilde_costs: Array,
            xi_jacobian: Array) -> Tuple[Array, List[Error]]:
        """Compute a realization of the Jacobian of omega with respect to theta market-by-market. If necessary, revert
        problematic elements to their estimated values.
        """
        errors: List[Error] = []

        # define a factory for computing the Jacobian of omega with respect to theta in markets
        def market_factory(s: Hashable) -> Tuple[ResultsMarket, Array, Array, Parameters, str]:
            """Build a market with the data realization along with arguments used to compute the Jacobians."""
            data_override_s = {
                'prices': equilibrium_prices[self.problem._product_market_indices[s]],
                'shares': equilibrium_shares[self.problem._product_market_indices[s]]
            }
            market_s = ResultsMarket(self.problem, s, self.sigma, self.pi, self.rho, self.beta, delta, data_override_s)
            tilde_costs_s = tilde_costs[self.problem._product_market_indices[s]]
            xi_jacobian_s = xi_jacobian[self.problem._product_market_indices[s]]
            return market_s, tilde_costs_s, xi_jacobian_s, self._parameters, self._costs_type

        # compute the Jacobian market-by-market
        omega_jacobian = np.full((self.problem.N, self._parameters.P), np.nan, options.dtype)
        generator = generate_items(
            self.problem.unique_market_ids, market_factory, ResultsMarket.compute_omega_by_theta_jacobian
        )
        for t, (omega_jacobian_t, errors_t) in generator:
            omega_jacobian[self.problem._product_market_indices[t]] = omega_jacobian_t
            errors.extend(errors_t)

        # the Jacobian should be zero for any clipped marginal costs
        omega_jacobian[self.clipped_costs.flat] = 0

        # replace invalid elements
        bad_jacobian_index = ~np.isfinite(omega_jacobian)
        if np.any(bad_jacobian_index):
            omega_jacobian[bad_jacobian_index] = self.omega_by_theta_jacobian[bad_jacobian_index]
            errors.append(exceptions.OmegaByThetaJacobianReversionError(bad_jacobian_index))
        return omega_jacobian, errors

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

    def _combine_arrays(
            self, compute_market_results: Callable, fixed_args: Sequence = (), market_args: Sequence = ()) -> Array:
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
            indices_s = self.problem._product_market_indices[s]
            market_s = ResultsMarket(self.problem, s, self.sigma, self.pi, self.rho, self.beta, self.delta)
            args_s = [None if a is None else a[indices_s] for a in market_args]
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
