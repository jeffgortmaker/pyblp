"""Economy-level structuring of BLP problem results."""

import itertools
from pathlib import Path
import pickle
import time
from typing import Any, Dict, Hashable, List, Optional, Sequence, TYPE_CHECKING, Tuple, Union

import numpy as np
import scipy.linalg

from .economy_results import EconomyResults
from .. import exceptions, options
from ..configurations.integration import Integration
from ..configurations.iteration import Iteration
from ..markets.economy_results_market import EconomyResultsMarket
from ..micro import Moments
from ..primitives import Agents
from ..utilities.algebra import (
    approximately_invert, approximately_solve, compute_condition_number, precisely_compute_eigenvalues, vech_to_full
)
from ..utilities.basics import (
    Array, Bounds, Error, Mapping, RecArray, SolverStats, format_number, format_seconds, format_table, generate_items,
    get_indices, output, output_progress, update_matrices
)
from ..utilities.statistics import (
    compute_gmm_moment_covariances, compute_gmm_parameter_covariances, compute_gmm_moments_jacobian_mean,
    compute_gmm_weights, compute_sigma_squared_vector_covariances
)


# only import objects that create import cycles when checking types
if TYPE_CHECKING:
    from .bootstrapped_results import BootstrappedResults  # noqa
    from .importance_sampling_results import ImportanceSamplingResults  # noqa
    from .optimal_instrument_results import OptimalInstrumentResults  # noqa
    from ..economies.problem import ProblemEconomy, Progress  # noqa


class ProblemResults(EconomyResults):
    r"""Results of a solved BLP problem.

    Many results are class attributes. Other post-estimation outputs be computed by calling class methods.

    .. note::

       Methods in this class that compute one or more post-estimation output per market support :func:`parallel`
       processing. If multiprocessing is used, market-by-market computation of each post-estimation output will be
       distributed among the processes.

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
    converged : `bool`
        Whether the optimization routine converged.
    cumulative_converged : `bool`
        Whether the optimization routine converged for this step and all prior steps.
    optimization_iterations : `int`
        Number of major iterations completed by the optimization routine.
    cumulative_optimization_iterations : `int`
        Sum of :attr:`ProblemResults.optimization_iterations` for this step and all prior steps.
    objective_evaluations : `int`
        Number of GMM objective evaluations.
    cumulative_objective_evaluations : `int`
        Sum of :attr:`ProblemResults.objective_evaluations` for this step and all prior steps.
    fp_converged : `ndarray`
        Flags for convergence of the iteration routine used to compute :math:`\delta(\theta)` in each market during each
        objective evaluation. Rows are in the same order as :attr:`Problem.unique_market_ids` and column indices
        correspond to objective evaluations.
    cumulative_fp_converged : `ndarray`
        Concatenation of :attr:`ProblemResults.fp_converged` for this step and all prior steps.
    fp_iterations : `ndarray`
        Number of major iterations completed by the iteration routine used to compute :math:`\delta(\theta)` in each
        market during each objective evaluation. Rows are in the same order as :attr:`Problem.unique_market_ids` and
        column indices correspond to objective evaluations.
    cumulative_fp_iterations : `ndarray`
        Concatenation of :attr:`ProblemResults.fp_iterations` for this step and all prior steps.
    contraction_evaluations : `ndarray`
        Number of times the contraction used to compute :math:`\delta(\theta)` was evaluated in each market during each
        objective evaluation. Rows are in the same order as :attr:`Problem.unique_market_ids` and column indices
        correspond to objective evaluations.
    cumulative_contraction_evaluations : `ndarray`
        Concatenation of :attr:`ProblemResults.contraction_evaluations` for this step and all prior steps.
    parameters : `ndarray`
        Stacked parameters in the following order: :math:`\hat{\theta}`, concentrated out elements of
        :math:`\hat{\beta}`, and concentrated out elements of :math:`\hat{\gamma}`.
    parameter_covariances : `ndarray`
        Estimated asymptotic covariance matrix for :math:`\sqrt{N}(\hat{\theta} - \theta_0)`, in which :math:`\theta`
        are the stacked parameters. Standard errors are the square root of the diagonal of this matrix divided by
        :math:`N`. Parameter covariances are not estimated during the first step of two-step GMM.
    theta : `ndarray`
        Estimated unfixed parameters, :math:`\hat{\theta}`, in the following order: :math:`\hat{\Sigma}`,
        :math:`\hat{\Pi}`, :math:`\hat{\rho}`, non-concentrated out elements from :math:`\hat{\beta}`, and
        non-concentrated out elements from :math:`\hat{\gamma}`.
    sigma : `ndarray`
        Estimated Cholesky root of the covariance matrix for unobserved taste heterogeneity, :math:`\hat{\Sigma}`.
    sigma_squared : `ndarray`
        Estimated covariance matrix for unobserved taste heterogeneity, :math:`\hat{\Sigma}\hat{\Sigma}'`.
    pi : `ndarray`
        Estimated parameters that measures how agent tastes vary with demographics, :math:`\hat{\Pi}`.
    rho : `ndarray`
        Estimated parameters that measure within nesting group correlations, :math:`\hat{\rho}`.
    beta : `ndarray`
        Estimated demand-side linear parameters, :math:`\hat{\beta}`.
    gamma : `ndarray`
        Estimated supply-side linear parameters, :math:`\hat{\gamma}`.
    sigma_se : `ndarray`
        Estimated standard errors for :math:`\hat{\Sigma}`, which are not estimated in the first step of two-step GMM.
    sigma_squared_se : `ndarray`
        Estimated standard errors for :math:`\hat{\Sigma}\hat{\Sigma}'`, which are computed with the delta method, and
        are not estimated in the first step of two-step GMM.
    pi_se : `ndarray`
        Estimated standard errors for :math:`\hat{\Pi}`, which are not estimated in the first step of two-step GMM.
    rho_se : `ndarray`
        Estimated standard errors for :math:`\hat{\rho}`, which are not estimated in the first step of two-step GMM.
    beta_se : `ndarray`
        Estimated standard errors for :math:`\hat{\beta}`, which are not estimated in the first step of two-step GMM.
    gamma_se : `ndarray`
        Estimated standard errors for :math:`\hat{\gamma}`, which are not estimated in the first step of two-step GMM.
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
    sigma_labels : `list of str`
        Variable labels for rows and columns of :math:`\Sigma`, which are derived from the formulation for :math:`X_2`.
    pi_labels : `list of str`
        Variable labels for columns of :math:`\Pi`, which are derived from the formulation for demographics.
    rho_labels : `list of str`
        Variable labels for :math:`\rho`. If :math:`\rho` is not a scalar, this is :attr:`Problem.unique_nesting_ids`.
    beta_labels : `list of str`
        Variable labels for :math:`\beta`, which are derived from the formulation for :math:`X_1`.
    gamma_labels : `list of str`
        Variable labels for :math:`\gamma`, which are derived from the formulation for :math:`X_3`.
    theta_labels : `list of str`
        Variable labels for :math:`\theta`, which are derived from the above labels.
    delta : `ndarray`
        Estimated mean utility, :math:`\delta(\hat{\theta})`.
    clipped_shares : `ndarray`
        Vector of booleans indicating whether the associated simulated shares were clipped during the last fixed point
        iteration to compute :math:`\delta(\hat{\theta})`. All elements will be ``False`` if ``shares_bounds`` in
        :meth:`Problem.solve` is disabled (by default shares are bounded from below by a small number to alleviate
        issues with underflow and negative shares).
    tilde_costs : `ndarray`
        Estimated transformed marginal costs, :math:`\tilde{c}(\hat{\theta})` from :eq:`costs`. If ``costs_bounds`` were
        specified in :meth:`Problem.solve`, :math:`c` may have been clipped.
    clipped_costs : `ndarray`
        Vector of booleans indicating whether the associated marginal costs were clipped. All elements will be ``False``
        if ``costs_bounds`` in :meth:`Problem.solve` was not specified.
    xi : `ndarray`
        Estimated unobserved demand-side product characteristics, :math:`\xi(\hat{\theta})`, or equivalently, the
        demand-side structural error term. When there are demand-side fixed effects, this is
        :math:`\Delta\xi(\hat{\theta})` in :eq:`fe`. That is, fixed effects are not included.
    omega : `ndarray`
        Estimated unobserved supply-side product characteristics, :math:`\omega(\hat{\theta})`, or equivalently, the
        supply-side structural error term. When there are supply-side fixed effects, this is
        :math:`\Delta\omega(\hat{\theta})` in :eq:`fe`. That is, fixed effects are not included.
    xi_fe : `ndarray`
        Estimated demand-side fixed effects :math:`\xi_{k_1} + \cdots \xi_{k_{E_D}}` in :eq:`fe`, which are only
        computed when there are demand-side fixed effects.
    omega_fe : `ndarray`
        Estimated supply-side fixed effects :math:`\omega_{k_1} + \cdots \omega_{k_{E_D}}` in :eq:`fe`, which are only
        computed when there are supply-side fixed effects.
    micro : `ndarray`
        Micro moments, :math:`\bar{g}_M`, in :eq:`micro_moment`.
    micro_values : `ndarray`
        Estimated micro moment values, :math:`f_m(v)`. Rows are in the same order as :attr:`ProblemResults.micro`.
    micro_covariances : `ndarray`
        Estimated micro moment covariance matrix :math:`S_M` in :eq:`scaled_micro_moment_covariances` divided by
        :math:`N`. Equal to ``micro_sample_covariances`` if overridden in :meth:`Problem.solve`.
    moments : `ndarray`
        Moments, :math:`\bar{g}`, in :eq:`averaged_moments`.
    moments_jacobian : `ndarray`
        Jacobian :math:`\bar{G}` of moments with respect to :math:`\theta`, in :eq:`averaged_moments_jacobian`.
    simulation_covariances : `ndarray`
        Adjustment in :eq:`simulation_S` to moment covariances to account for simulation error. This will be all zeros
        unless ``resample_agent_data`` was specified in :meth:`Problem.solve`.
    objective : `float`
        GMM objective value, :math:`q(\hat{\theta})`, defined in :eq:`objective`. If ``scale_objective`` was ``True`` in
        :meth:`Problem.solve` (which is the default), this value was scaled by :math:`N` so that objective values are
        more comparable across different problem sizes. Note that in some of the BLP literature (and earlier versions of
        this package), this expression was previously scaled by :math:`N^2`.
    xi_by_theta_jacobian : `ndarray`
        Estimated :math:`\frac{\partial\xi}{\partial\theta} = \frac{\partial\delta}{\partial\theta}`, which is used to
        compute the gradient and standard errors.
    omega_by_theta_jacobian : `ndarray`
        Estimated :math:`\frac{\partial\omega}{\partial\theta} = \frac{\partial\tilde{c}}{\partial\theta}`, which is
        used to compute the gradient and standard errors.
    micro_by_theta_jacobian : `ndarray`
        Estimated :math:`\frac{\partial\bar{g}_M}{\partial\theta}`, which is used to compute the gradient and standard
        errors.
    gradient : `ndarray`
        Gradient of the GMM objective, :math:`\nabla q(\hat{\theta})`, defined in :eq:`gradient`. This is computed after
        the optimization routine finishes even if the routine was configured to not use analytic gradients.
    projected_gradient : `ndarray`
        Projected gradient of the GMM objective. When there are no parameter bounds, this will always be equal to
        :attr:`ProblemResults.gradient`. Otherwise, if an element in :math:`\hat{\theta}` is equal to its lower (upper)
        bound, the corresponding projected gradient value will be truncated at a maximum (minimum) of zero.
    projected_gradient_norm : `ndarray`
        Infinity norm of :attr:`ProblemResults.projected_gradient`.
    hessian : `ndarray`
        Estimated Hessian of the GMM objective. By default, this is computed with finite central differences after the
        optimization routine finishes.
    reduced_hessian : `ndarray`
        Reduced Hessian of the GMM objective. When there are no parameter bounds, this will always be equal to
        :attr:`ProblemResults.hessian`. Otherwise, if an element in :math:`\hat{\theta}` is equal to either its lower
        or upper bound, the corresponding row and column in the reduced Hessian will be all zeros.
    reduced_hessian_eigenvalues : `ndarray`
        Eigenvalues of :attr:`ProblemResults.reduced_hessian`.
    W : `ndarray`
        Weighting matrix, :math:`W`, used to compute these results.
    updated_W : `ndarray`
        Weighting matrix updated according to :eq:`W`.

    Examples
    --------
        - :doc:`Tutorial </tutorial>`

    """

    problem: 'ProblemEconomy'
    last_results: Optional['ProblemResults']
    step: int
    optimization_time: float
    cumulative_optimization_time: float
    total_time: float
    cumulative_total_time: float
    converged: bool
    cumulative_converged: bool
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
    parameters: Array
    parameter_covariances: Array
    theta: Array
    sigma: Array
    sigma_squared: Array
    pi: Array
    rho: Array
    beta: Array
    gamma: Array
    sigma_se: Array
    sigma_squared_se: Array
    pi_se: Array
    rho_se: Array
    beta_se: Array
    gamma_se: Array
    sigma_bounds: Bounds
    pi_bounds: Bounds
    rho_bounds: Bounds
    beta_bounds: Bounds
    gamma_bounds: Bounds
    sigma_labels: List[str]
    pi_labels: List[str]
    rho_labels: List[str]
    beta_labels: List[str]
    gamma_labels: List[str]
    theta_labels: List[str]
    delta: Array
    clipped_shares: Array
    tilde_costs: Array
    clipped_costs: Array
    xi: Array
    omega: Array
    xi_fe: Array
    omega_fe: Array
    micro: Array
    micro_values: Array
    micro_covariances: Array
    moments: Array
    moments_jacobian: Array
    simulation_covariances: Array
    objective: Array
    xi_by_theta_jacobian: Array
    omega_by_theta_jacobian: Array
    micro_by_theta_jacobian: Array
    gradient: Array
    projected_gradient: Array
    projected_gradient_norm: Array
    hessian: Array
    reduced_hessian: Array
    reduced_hessian_eigenvalues: Array
    W: Array
    updated_W: Array
    _scaled_objective: bool
    _shares_bounds: Bounds
    _costs_bounds: Bounds
    _se_type: str
    _formatted_moments: Optional[str]
    _errors: List[Error]

    def __init__(
            self, progress: 'Progress', last_results: Optional['ProblemResults'], step: int, last_step: bool,
            step_start_time: float, optimization_start_time: float, optimization_end_time: float,
            optimization_stats: SolverStats, iteration_stats: Sequence[Dict[Hashable, SolverStats]],
            scaled_objective: bool, shares_bounds: Bounds, costs_bounds: Bounds,
            micro_moment_covariances: Optional[Array], center_moments: bool, W_type: str, se_type: str,
            covariance_moments_mean: float) -> None:
        """Compute cumulative progress statistics, update weighting matrices, and estimate standard errors."""
        self.sigma, self.pi, self.rho, _, _ = progress.parameters.expand(progress.theta)
        self._errors = progress.errors
        self.problem = progress.problem
        self.W = progress.W
        self.theta = progress.theta
        self.delta = progress.delta
        self.tilde_costs = progress.tilde_costs
        self.micro = progress.micro
        self.micro_values = progress.micro_values
        self.moments = progress.mean_g
        self.simulation_covariances = progress.simulation_covariances
        self.xi_by_theta_jacobian = progress.xi_jacobian
        self.omega_by_theta_jacobian = progress.omega_jacobian
        self.micro_by_theta_jacobian = progress.micro_jacobian
        self.xi = progress.xi
        self.omega = progress.omega
        self.beta = progress.beta
        self.gamma = progress.gamma
        self.objective = progress.objective
        self.gradient = progress.gradient
        self.projected_gradient = progress.projected_gradient
        self.projected_gradient_norm = progress.projected_gradient_norm
        self.hessian = progress.hessian
        self.reduced_hessian = progress.reduced_hessian
        self.clipped_shares = progress.clipped_shares
        self.clipped_costs = progress.clipped_costs
        self._scaled_objective = scaled_objective
        self._shares_bounds = shares_bounds
        self._costs_bounds = costs_bounds
        self._se_type = se_type
        super().__init__(
            self.problem, progress.parameters, self.sigma, self.pi, self.rho, self.beta, self.gamma, self.delta
        )

        # if there are any fixed effects, compute their contributions to xi and omega
        self.xi_fe = np.zeros_like(self.xi)
        self.omega_fe = np.zeros_like(self.omega)
        if self.problem.ED > 0:
            self.xi_fe = self.delta - self.problem._compute_true_X1() @ self.beta - self.xi
        if self.problem.ES > 0:
            self.omega_fe = self.tilde_costs - self.problem._compute_true_X3() @ self.gamma - self.omega

        # either use the manually-specified micro moment covariances or estimated ones
        self.micro_covariances = micro_moment_covariances
        if micro_moment_covariances is None:
            self.micro_covariances = progress.micro_covariances

        # format micro moments for displaying information (micro moments themselves often contain lambda functions and
        #   are hence often not serializable)
        self._formatted_moments = None
        if progress.moments.MM > 0:
            self._formatted_moments = progress.moments.format("Estimated Micro Moments", self.micro_values)

        # if the reduced Hessian was computed, compute its eigenvalues and the ratio of the smallest to largest ones
        self.reduced_hessian_eigenvalues = np.full(self._parameters.P, np.nan, options.dtype)
        if self._parameters.P > 0 and np.isfinite(self.reduced_hessian).all():
            self.reduced_hessian_eigenvalues, successful = precisely_compute_eigenvalues(self.reduced_hessian)
            if not successful:
                self._errors.append(exceptions.HessianEigenvaluesError(self.reduced_hessian))

        # initialize counts, times, and convergence
        self.step = step
        self.total_time = self.cumulative_total_time = time.time() - step_start_time
        self.optimization_time = self.cumulative_optimization_time = optimization_end_time - optimization_start_time
        self.converged = self.cumulative_converged = optimization_stats.converged
        self.optimization_iterations = self.cumulative_optimization_iterations = optimization_stats.iterations
        self.objective_evaluations = self.cumulative_objective_evaluations = optimization_stats.evaluations
        self.fp_converged = self.cumulative_fp_converged = np.array(
            [[m[t].converged if m else True for m in iteration_stats] for t in self.problem.unique_market_ids],
            dtype=np.int64
        )
        self.fp_iterations = self.cumulative_fp_iterations = np.array(
            [[m[t].iterations if m else 0 for m in iteration_stats] for t in self.problem.unique_market_ids],
            dtype=np.int64
        )
        self.contraction_evaluations = self.cumulative_contraction_evaluations = np.array(
            [[m[t].evaluations if m else 0 for m in iteration_stats] for t in self.problem.unique_market_ids],
            dtype=np.int64
        )

        # initialize last results and add to cumulative values
        self.last_results = last_results
        if last_results is not None:
            self.cumulative_total_time += last_results.cumulative_total_time
            self.cumulative_optimization_time += last_results.cumulative_optimization_time
            self.cumulative_converged = last_results.converged and optimization_stats.converged
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

        # store estimated parameters and information about them (beta and gamma have already been stored above)
        self.sigma_squared = self.sigma @ self.sigma.T
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
        self.sigma_labels = self._parameters.sigma_labels
        self.pi_labels = self._parameters.pi_labels
        self.rho_labels = self._parameters.rho_labels
        self.beta_labels = self._parameters.beta_labels
        self.gamma_labels = self._parameters.gamma_labels
        self.theta_labels = self._parameters.theta_labels

        # ignore computational errors when updating the weighting matrix and computing covariances
        with np.errstate(all='ignore'):
            # update the weighting matrix
            S_for_weights = self._compute_S(progress.moments, W_type, covariance_moments_mean, center_moments)
            self.updated_W, W_errors = compute_gmm_weights(S_for_weights)
            self._errors.extend(W_errors)

            # only compute parameter covariances and standard errors if this is the last step
            self.moments_jacobian = np.full((self.moments.size, self.parameters.size), np.nan, options.dtype)
            self.parameter_covariances = np.full((self.parameters.size, self.parameters.size), np.nan, options.dtype)
            se = np.full((self.parameters.size, 1), np.nan, options.dtype)
            sigma_squared_vector_se = np.full((self.problem.K2 * (self.problem.K2 + 1) // 2, 1), np.nan, options.dtype)
            if last_step:
                S_for_covariances = S_for_weights
                if se_type != W_type or center_moments:
                    S_for_covariances = self._compute_S(progress.moments, se_type, covariance_moments_mean)

                # if this is the first step, an unadjusted weighting matrix needs to be used when computing unadjusted
                #   covariances so that they are scaled properly
                W_for_covariances = self.W
                if se_type == 'unadjusted' and self.step == 1:
                    W_for_covariances, W_for_covariances_errors = compute_gmm_weights(S_for_covariances)
                    self._errors.extend(W_for_covariances_errors)

                # compute parameter covariances
                self.moments_jacobian = self._compute_mean_G(progress.moments)
                self.parameter_covariances, se_errors = compute_gmm_parameter_covariances(
                    W_for_covariances, S_for_covariances, self.moments_jacobian, se_type
                )
                self._errors.extend(se_errors)

                # use the delta method to compute covariances for the parameters in sigma squared
                theta_covariances = self.parameter_covariances[:self._parameters.P, :self._parameters.P]
                sigma_vector_covariances = self._parameters.extract_sigma_vector_covariances(theta_covariances)
                sigma_squared_vector_covariances = compute_sigma_squared_vector_covariances(
                    self.sigma, sigma_vector_covariances
                )

                # compute standard errors
                se = np.sqrt(np.c_[self.parameter_covariances.diagonal()] / self.problem.N)
                sigma_squared_vector_se = np.sqrt(np.c_[sigma_squared_vector_covariances.diagonal()] / self.problem.N)
                if np.isnan(se).any() or np.isnan(sigma_squared_vector_se).any():
                    self._errors.append(exceptions.InvalidParameterCovariancesError())

        # expand standard errors
        theta_se, eliminated_beta_se, eliminated_gamma_se = np.split(se, [
            self._parameters.P,
            self._parameters.P + self._parameters.eliminated_beta_index.sum()
        ])
        self.sigma_se, self.pi_se, self.rho_se, self.beta_se, self.gamma_se = (
            self._parameters.expand(theta_se, nullify=True)
        )
        self.sigma_squared_se = vech_to_full(sigma_squared_vector_se, self.problem.K2)
        self.beta_se[self._parameters.eliminated_beta_index] = eliminated_beta_se.flatten()
        self.gamma_se[self._parameters.eliminated_gamma_index] = eliminated_gamma_se.flatten()

    def __str__(self) -> str:
        """Format problem results as a string."""
        sections = [self._format_summary(), self._format_cumulative_statistics()]

        # construct a standard error description
        if self._se_type == 'unadjusted':
            se_description = "Unadjusted SEs"
        elif self._se_type == 'robust':
            se_description = "Robust SEs"
        else:
            assert self._se_type == 'clustered'
            se_description = f'Robust SEs Adjusted for {np.unique(self.problem.products.clustering_ids).size} Clusters'

        # add sections formatting estimates and micro moments values
        sections.append(self._parameters.format_estimates(
            f"Estimates ({se_description} in Parentheses)", self.sigma, self.pi, self.rho, self.beta, self.gamma,
            self.sigma_squared, self.sigma_se, self.pi_se, self.rho_se, self.beta_se, self.gamma_se,
            self.sigma_squared_se
        ))
        if self._formatted_moments is not None:
            sections.append(self._formatted_moments)

        # join the sections into a single string
        return "\n\n".join(sections)

    def _compute_mean_G(self, moments: Moments) -> Array:
        """Compute the Jacobian of moments with respect to parameters."""
        Z_list = [self.problem.products.ZD]
        jacobian_list = [np.c_[
            self.xi_by_theta_jacobian,
            -self.problem.products.X1[:, self._parameters.eliminated_beta_index.flat],
            np.zeros_like(self.problem.products.X3[:, self._parameters.eliminated_gamma_index.flat])
        ]]
        if self.problem.K3 > 0:
            Z_list.append(self.problem.products.ZS)
            jacobian_list.append(np.c_[
                self.omega_by_theta_jacobian,
                np.zeros_like(self.problem.products.X1[:, self._parameters.eliminated_beta_index.flat]),
                -self.problem.products.X3[:, self._parameters.eliminated_gamma_index.flat]
            ])
            if self.problem.MC > 0:
                Z_list.append(self.problem.products.ZC)
                jacobian_list.append(np.c_[
                    self.xi_by_theta_jacobian * self.omega + self.xi * self.omega_by_theta_jacobian,
                    -self.problem.products.X1[:, self._parameters.eliminated_beta_index.flat] * self.omega,
                    self.xi * -self.problem.products.X3[:, self._parameters.eliminated_gamma_index.flat]
                ])
        mean_G = np.r_[
            compute_gmm_moments_jacobian_mean(jacobian_list, Z_list),
            np.c_[
                self.micro_by_theta_jacobian,
                np.zeros((moments.MM, self._parameters.eliminated_beta_index.sum()), options.dtype),
                np.zeros((moments.MM, self._parameters.eliminated_gamma_index.sum()), options.dtype)
            ]
        ]
        return mean_G

    def _compute_S(
            self, moments: Moments, S_type: str, covariance_moments_mean: float, center_moments: bool = False) -> Array:
        """Compute moment covariances."""
        u_list = [self.xi]
        Z_list = [self.problem.products.ZD]
        if self.problem.K3 > 0:
            u_list.append(self.omega)
            Z_list.append(self.problem.products.ZS)
            if self.problem.MC > 0:
                u_list.append(self.xi * self.omega - covariance_moments_mean)
                Z_list.append(self.problem.products.ZC)

        S = compute_gmm_moment_covariances(u_list, Z_list, S_type, self.problem.products.clustering_ids, center_moments)
        self.problem._detect_singularity(S, "the estimated covariance matrix of aggregate GMM moments")
        if moments.MM > 0:
            S = scipy.linalg.block_diag(S, self.problem.N * self.micro_covariances)

        S += self.simulation_covariances
        return S

    def _format_summary(self) -> str:
        """Format a summary table of problem results."""

        # construct the leftmost part of the table that always shows up
        header = [("GMM", "Step"), ("Objective", "Value")]
        values = [self.step, format_number(self.objective)]

        # add information about first order conditions
        if np.isfinite(self.projected_gradient_norm):
            if self._parameters.any_bounds:
                header.append(("Projected", "Gradient Norm"))
            else:
                header.append(("Gradient", "Norm"))
            values.append(format_number(self.projected_gradient_norm))

        # add information about second order conditions
        if np.isfinite(self.reduced_hessian_eigenvalues).any():
            hessian_type = "Reduced" if self._parameters.any_bounds else ""
            if self.reduced_hessian_eigenvalues.size == 1:
                header.append((hessian_type, "Hessian"))
                values.append(format_number(self.reduced_hessian))
            else:
                header.extend([
                    (f"{hessian_type} Hessian", "Min Eigenvalue"),
                    (f"{hessian_type} Hessian", "Max Eigenvalue")
                ])
                values.extend([
                    format_number(self.reduced_hessian_eigenvalues.min()),
                    format_number(self.reduced_hessian_eigenvalues.max())
                ])

        # add counts of any clipped shares or marginal costs
        if np.isfinite(self._shares_bounds).any():
            header.append(("Clipped", "Shares"))
            values.append(self.clipped_shares.sum())
        if np.isfinite(self._costs_bounds).any():
            header.append(("Clipped", "Costs"))
            values.append(self.clipped_costs.sum())

        # add information about the weighting matrix if this isn't an initial update
        if self.step > 0:
            header.append(("Weighting Matrix", "Condition Number"))
            values.append(format_number(compute_condition_number(self.W)))

        # add information about the covariance matrix
        if np.isfinite(self.parameter_covariances).any() and self.parameter_covariances.size > 1:
            header.append(("Covariance Matrix", "Condition Number"))
            values.append(format_number(compute_condition_number(self.parameter_covariances)))

        return format_table(header, values, title="Problem Results Summary")

    def _format_cumulative_statistics(self) -> str:
        """Format a table of cumulative statistics."""

        # construct the leftmost part of the top table that always shows up
        header = [("Computation", "Time")]
        values = [format_seconds(self.cumulative_total_time)]

        # add optimization convergence and iterations
        if self._parameters.P > 0:
            header.extend([("Optimizer", "Converged"), ("Optimization", "Iterations")])
            values.extend(["Yes" if self.cumulative_converged else "No", str(self.cumulative_optimization_iterations)])

        # add evaluations and iterations
        header.append(("Objective", "Evaluations"))
        values.append(str(self.cumulative_objective_evaluations))
        if np.any(self.cumulative_contraction_evaluations > 0):
            header.extend([("Fixed Point", "Iterations"), ("Contraction", "Evaluations")])
            values.extend([
                str(self.cumulative_fp_iterations.sum()),
                str(self.cumulative_contraction_evaluations.sum())]
            )

        return format_table(header, values, title="Cumulative Statistics")

    def to_pickle(self, path: Union[str, Path]) -> None:
        """Save these results as a pickle file.

        Parameters
        ----------
        path: `str or Path`
            File path to which these results will be saved.

        """
        with open(path, 'wb') as handle:
            pickle.dump(self, handle)

    def to_dict(
            self, attributes: Sequence[str] = (
                'step', 'optimization_time', 'cumulative_optimization_time', 'total_time', 'cumulative_total_time',
                'converged', 'cumulative_converged', 'optimization_iterations', 'cumulative_optimization_iterations',
                'objective_evaluations', 'cumulative_objective_evaluations', 'fp_converged', 'cumulative_fp_converged',
                'fp_iterations', 'cumulative_fp_iterations', 'contraction_evaluations',
                'cumulative_contraction_evaluations', 'parameters', 'parameter_covariances', 'theta', 'sigma',
                'sigma_squared', 'pi', 'rho', 'beta', 'gamma', 'sigma_se', 'sigma_squared_se', 'pi_se', 'rho_se',
                'beta_se', 'gamma_se', 'sigma_bounds', 'pi_bounds', 'rho_bounds', 'beta_bounds', 'gamma_bounds',
                'sigma_labels', 'pi_labels', 'rho_labels', 'beta_labels', 'gamma_labels', 'theta_labels', 'delta',
                'tilde_costs', 'clipped_shares', 'clipped_costs', 'xi', 'omega', 'xi_fe', 'omega_fe', 'micro',
                'micro_values', 'micro_covariances', 'moments', 'moments_jacobian', 'simulation_covariances',
                'objective', 'xi_by_theta_jacobian', 'omega_by_theta_jacobian', 'micro_by_theta_jacobian', 'gradient',
                'projected_gradient', 'projected_gradient_norm', 'hessian', 'reduced_hessian',
                'reduced_hessian_eigenvalues', 'W', 'updated_W'
            )) -> dict:
        """Convert these results into a dictionary that maps attribute names to values.

        Parameters
        ----------
        attributes : `sequence of str, optional`
            Name of attributes that will be added to the dictionary. By default, all :class:`ProblemResults` attributes
            are added except for :attr:`ProblemResults.problem` and :attr:`ProblemResults.last_results`.

        Returns
        -------
        `dict`
            Mapping from attribute names to values.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        return {k: getattr(self, k) for k in attributes}

    def run_hansen_test(self) -> float:
        r"""Test the validity of overidentifying restrictions with the Hansen :math:`J` test.

        Following :ref:`references:Hansen (1982)`, the :math:`J` statistic is

        .. math:: J = N\bar{g}(\hat{\theta})'W\bar{g}(\hat{\theta})
           :label: J

        where :math:`\bar{g}(\hat{\theta})` is defined in :eq:`averaged_moments` and :math:`W` is the optimal weighting
        matrix in :eq:`W`.

        .. note::

           The statistic can equivalently be written as :math:`J = Nq(\hat{\theta})` where the GMM objective value is
           defined in :eq:`objective`, or the same but without the :math:`N` if the GMM objective value was scaled by
           :math:`N`, which is the default behavior.

        When the overidentifying restrictions in this model are valid, the :math:`J` statistic is asymptotically
        :math:`\chi^2` with degrees of freedom equal to the number of overidentifying restrictions. This requires that
        there are more moments than parameters.

        .. warning::

           This test requires :attr:`ProblemResults.W` to be an optimal weighting matrix, so it should typically be run
           only after two-step GMM or after one-step GMM with a pre-specified optimal weighting matrix.

        Returns
        -------
        `float`
            The :math:`J` statistic.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        return (1 if self._scaled_objective else self.problem.N) * float(self.objective)

    def run_distance_test(self, unrestricted: 'ProblemResults') -> float:
        r"""Test the validity of model restrictions with the distance test.

        Following :ref:`references:Newey and West (1987)`, the distance or likelihood ratio-like statistic is

        .. math:: \text{LR} = J(\hat{\theta}^r) - J(\hat{\theta}^u)

        where :math:`J(\hat{\theta}^r)` is the :math:`J` statistic defined in :eq:`J` for this restricted model and
        :math:`J(\hat{\theta}^u)` is the :math:`J` statistic for the unrestricted model.

        .. note::

           The statistic can equivalently be written as
           :math:`\text{LR} = N[q(\hat{\theta}^r) - q(\hat{\theta}^u)]` where the GMM objective value is defined in
           :eq:`objective`, or the same but without the :math:`N` if the GMM objective value was scaled by :math:`N`,
           which is the default behavior.

        If the restrictions in this model are valid, the distance statistic is asymptotically :math:`\chi^2` with
        degrees of freedom equal to the number of restrictions.

        .. warning::

           This test requires each model's :attr:`ProblemResults.W` to be the optimal weighting matrix, so it should
           typically be run only after two-step GMM or after one-step GMM with pre-specified optimal weighting matrices.

        Parameters
        ----------
        unrestricted : `ProblemResults`
            :class:`ProblemResults` for the unrestricted model.

        Returns
        -------
        `float`
            The distance statistic.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        if not isinstance(unrestricted, ProblemResults):
            raise TypeError("unrestricted must be another ProblemResults.")
        if unrestricted.problem.N != self.problem.N:
            raise ValueError("unrestricted must have the same number of observations as these results.")
        restricted_J = (1 if self._scaled_objective else self.problem.N) * self.objective
        unrestricted_J = (1 if unrestricted._scaled_objective else unrestricted.problem.N) * unrestricted.objective
        return restricted_J - unrestricted_J

    def run_lm_test(self) -> float:
        r"""Test the validity of model restrictions with the Lagrange multiplier test.

        Following :ref:`references:Newey and West (1987)`, the Lagrange multiplier or score statistic is

        .. math::

           \text{LM} = N\bar{g}(\hat{\theta})'W\bar{G}(\hat{\theta})V\bar{G}(\hat{\theta})'W\bar{g}(\hat{\theta})

        where :math:`\bar{g}(\hat{\theta})` is defined in :eq:`averaged_moments`, :math:`\bar{G}(\hat{\theta})` is
        defined in :eq:`averaged_moments_jacobian`, :math:`W` is the optimal weighting matrix in :eq:`W`, and :math:`V`
        is the covariance matrix of :math:`\sqrt{N}(\hat{\theta} - \theta)` in :eq:`covariances`.

        If the restrictions in this model are valid, the Lagrange multiplier statistic is asymptotically :math:`\chi^2`
        with degrees of freedom equal to the number of restrictions.

        .. warning::

           This test requires :attr:`ProblemResults.W` to be an optimal weighting matrix, so it should typically be run
           only after two-step GMM or after one-step GMM with a pre-specified optimal weighting matrix.

        Returns
        -------
        `float`
            The Lagrange multiplier statistic.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        gradient = self.moments_jacobian.T @ self.W @ self.moments
        return self.problem.N * float(gradient.T @ self.parameter_covariances @ gradient)

    def run_wald_test(self, restrictions: Any, restrictions_jacobian: Any) -> float:
        r"""Test the validity of model restrictions with the Wald test.

        Following :ref:`references:Newey and West (1987)`, the Wald statistic is

        .. math:: \text{Wald} = Nr(\hat{\theta})'[R(\hat{\theta})VR(\hat{\theta})']^{-1}r(\hat{\theta})

        where the restrictions are :math:`r(\theta) = 0` under the test's null hypothesis, their Jacobian is
        :math:`R(\theta) = \frac{\partial r(\theta)}{\partial\theta}`, and :math:`V` is the covariance matrix of
        :math:`\sqrt{N}(\hat{\theta} - \theta)` in :eq:`covariances`.

        If the restrictions are valid, the Wald statistic is asymptotically :math:`\chi^2` with degrees of freedom equal
        to the number of restrictions.

        Parameters
        ----------
        restrictions : `array-like`
            Column vector of the model restrictions evaluated at the estimated parameters, :math:`r(\hat{\theta})`.
        restrictions_jacobian : `array-like`
            Estimated Jacobian of the restrictions with respect to all parameters, :math:`R(\hat{\theta})`. This matrix
            should have as many rows as ``restrictions`` and as many columns as
            :attr:`ProblemResults.parameter_covariances`.

        Returns
        -------
        `float`
            The Wald statistic.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """

        # validate the restrictions and their Jacobian
        restrictions = np.c_[np.asarray(restrictions, options.dtype)]
        restrictions_jacobian = np.c_[np.asarray(restrictions_jacobian, options.dtype)]
        if restrictions.shape != (restrictions.shape[0], 1):
            raise ValueError("restrictions must be a column vector.")
        if restrictions_jacobian.shape != (restrictions.shape[0], self.parameter_covariances.shape[0]):
            raise ValueError(
                f"restrictions_jacobian must be a {restrictions.shape[0]} by {self.parameter_covariances.shape[0]} "
                f"matrix."
            )

        # compute the statistic
        matrix = restrictions_jacobian @ self.parameter_covariances @ restrictions_jacobian.T
        inverted, replacement = approximately_invert(matrix)
        if replacement:
            output(exceptions.WaldInversionError(matrix, replacement))
        return self.problem.N * float(restrictions.T @ inverted @ restrictions)

    def bootstrap(
            self, draws: int = 1000, seed: Optional[int] = None, iteration: Optional[Iteration] = None,
            constant_costs: bool = True) -> 'BootstrappedResults':
        r"""Use a parametric bootstrap to create an empirical distribution of results.

        The constructed :class:`BootstrappedResults` can be used just like :class:`ProblemResults` to compute various
        post-estimation outputs for different markets. The only difference is that :class:`BootstrappedResults` methods
        return arrays with an extra first dimension, along which bootstrapped results are stacked. These stacked results
        can be used to construct, for example, confidence intervals for post-estimation outputs.

        For each bootstrap draw, parameters are drawn from the estimated multivariate normal distribution of all
        parameters defined by :attr:`ProblemResults.parameters` and :attr:`ProblemResults.parameter_covariances`
        (where the second covariance matrix is divided by :math:`N`). Any bounds configured in :meth:`Problem.solve`
        will also bound parameter draws. Each parameter draw is used to compute the implied mean utility,
        :math:`\delta`, and shares, :math:`s`. If a supply side was estimated, the implied marginal costs, :math:`c`,
        and prices, :math:`p`, are computed as well by iterating over the :math:`\zeta`-markup contraction in
        :eq:`zeta_contraction`. If marginal costs depend on prices through market shares, they will be updated to
        reflect different prices during each iteration of the routine.

        .. note::

           By default, parametric bootstrapping may use a lot of memory. This is because all bootstrapped results (for
           all ``draws``) are stored in memory at the same time. Memory usage can be reduced by calling this method in a
           loop with ``draws = 1``. In each iteration of the loop, compute the desired post-estimation output with the
           proper method of the returned :class:`BootstrappedResults` class and store these outputs.

        Parameters
        ----------
        draws : `int, optional`
            The number of draws that will be taken from the joint distribution of the parameters. The default value is
            ``1000``.
        seed : `int, optional`
            Passed to :class:`numpy.random.RandomState` to seed the random number generator before any draws are
            taken. By default, a seed is not passed to the random number generator.
        iteration : `Iteration, optional`
            :class:`Iteration` configuration used to compute bootstrapped prices by iterating over the
            :math:`\zeta`-markup equation in :eq:`zeta_contraction`. By default, if a supply side was estimated, this
            is ``Iteration('simple', {'atol': 1e-12})``. Analytic Jacobians are not supported for solving this system.
            This configuration is not used if a supply side was not estimated.
        constant_costs : `bool, optional`
            Whether to assume that marginal costs, :math:`c`, remain constant as equilibrium prices and shares change.
            By default this is ``True``, which means that firms treat marginal costs as constant when setting prices.
            If set to ``False``, marginal costs will be allowed to adjust if ``shares`` was included in
            the formulation for :math:`X_3` in :class:`Problem`. This is not relevant if a supply side was not
            estimated.

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
            raise ValueError("Analytic Jacobians are not supported for solving this system.")

        # draw from the asymptotic distribution implied by the estimated parameters
        state = np.random.RandomState(seed)
        bootstrapped_parameters = np.atleast_3d(state.multivariate_normal(
            self.parameters.flatten(), self.parameter_covariances / self.problem.N, draws
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

        # pre-compute X1 and X3 without any absorbed fixed effects
        true_X1 = self.problem._compute_true_X1()
        true_X3 = self.problem._compute_true_X3()

        def market_factory(
                pair: Tuple[int, Hashable]) -> (
                Tuple[EconomyResultsMarket, Array, Optional[Array], Optional[Iteration], bool]):
            """Build a market along with arguments used to compute equilibrium prices and shares along with delta."""
            c, s = pair
            indices_s = self.problem._product_market_indices[s]
            market_cs = EconomyResultsMarket(
                self.problem, s, self._parameters, bootstrapped_sigma[c], bootstrapped_pi[c], bootstrapped_rho[c],
                bootstrapped_beta[c], bootstrapped_gamma[c], self.delta + true_X1 @ (bootstrapped_beta[c] - self.beta)
            )
            costs_cs = self.tilde_costs[indices_s] + true_X3[indices_s] @ (bootstrapped_gamma[c] - self.gamma)
            if self.problem.costs_type == 'log':
                costs_cs = np.exp(costs_cs)
            prices_s = self.problem.products.prices[indices_s] if iteration is None else None
            return market_cs, costs_cs, prices_s, iteration, constant_costs

        # compute bootstrapped prices, shares, and deltas
        bootstrapped_prices = np.zeros((draws, self.problem.N, 1), options.dtype)
        bootstrapped_shares = np.zeros((draws, self.problem.N, 1), options.dtype)
        bootstrapped_delta = np.zeros((draws, self.problem.N, 1), options.dtype)
        iteration_stats: Dict[Hashable, SolverStats] = {}
        pairs = itertools.product(range(draws), self.problem.unique_market_ids)
        generator = generate_items(pairs, market_factory, EconomyResultsMarket.safely_solve_equilibrium_realization)
        for (d, t), (prices_dt, shares_dt, delta_dt, iteration_stats_dt, errors_dt) in generator:
            bootstrapped_prices[d, self.problem._product_market_indices[t]] = prices_dt
            bootstrapped_shares[d, self.problem._product_market_indices[t]] = shares_dt
            bootstrapped_delta[d, self.problem._product_market_indices[t]] = delta_dt
            iteration_stats[(d, t)] = iteration_stats_dt
            errors.extend(errors_dt)

        # output a warning about any errors
        if errors:
            output("")
            output(exceptions.MultipleErrors(errors))
            output("")

        # structure the results
        from .bootstrapped_results import BootstrappedResults  # noqa
        results = BootstrappedResults(
            self, bootstrapped_sigma, bootstrapped_pi, bootstrapped_rho, bootstrapped_beta, bootstrapped_gamma,
            bootstrapped_prices, bootstrapped_shares, bootstrapped_delta, start_time, time.time(), draws,
            iteration_stats
        )
        output(f"Bootstrapped results after {format_seconds(results.computation_time)}.")
        output("")
        output(results)
        return results

    def compute_optimal_instruments(
            self, method: str = 'approximate', draws: int = 1, seed: Optional[int] = None,
            expected_prices: Optional[Any] = None, iteration: Optional[Iteration] = None,
            constant_costs: bool = True) -> 'OptimalInstrumentResults':
        r"""Estimate feasible optimal or efficient instruments, :math:`Z_D^\text{opt}` and :math:`Z_S^\text{opt}`.

        Optimal instruments have been shown, for example, by :ref:`references:Reynaert and Verboven (2014)` and
        :ref:`references:Conlon and Gortmaker (2020)`, to reduce bias, improve efficiency, and enhance stability of BLP
        estimates.

        Optimal instruments in the spirit of :ref:`references:Amemiya (1977)` or :ref:`references:Chamberlain (1987)`
        are defined by

        .. math::
           :label: optimal_instruments

           \begin{bmatrix}
               Z_{D,jt}^\text{opt} \\
               Z_{S,jt}^\text{opt}
           \end{bmatrix}
           = \Sigma_{\xi\omega}^{-1}E\left[
           \begin{matrix}
               \frac{\partial\xi_{jt}}{\partial\theta} \\
               \frac{\partial\omega_{jt}}{\partial\theta}
           \end{matrix}
           \mathrel{\Bigg|} Z \right],

        in which :math:`Z` are all exogenous variables.

        Feasible optimal instruments are estimated by evaluating this expression at an estimated :math:`\hat{\theta}`.
        The expectation is taken by approximating an integral over the joint density of :math:`\xi` and :math:`\omega`.
        For each error term realization, if not already estimated, equilibrium prices and shares are computed by
        iterating over the :math:`\zeta`-markup contraction in :eq:`zeta_contraction`. If marginal costs depend on
        prices through market shares, they will be updated to reflect different prices during each iteration of the
        routine.

        The expected Jacobians are estimated with the average over all computed Jacobian realizations. The
        :math:`2 \times 2` normalizing matrix :math:`\Sigma_{\xi\omega}` is estimated with the sample covariance matrix
        of the error terms.

        Optimal instruments for linear parameters not included in :math:`\theta` are simple product characteristics, so
        they are not computed here but are rather included in the final set of instruments by
        :meth:`OptimalInstrumentResults.to_problem`.

        .. note::

           When both a supply and demand side are estimated, there are usually collinear rows in
           :eq:`optimal_instruments` because of overlapping product characteristics in :math:`X_1` and :math:`X_3`. The
           expression can be corrected by multiplying it with a conformable matrix of ones and zeros that remove the
           collinearity problem. The question of which rows to exclude is addressed in
           :meth:`OptimalInstrumentResults.to_problem`.

        .. warning::

           Currently, only optimal instruments for the standard demand- and supply-side moments are supported. If
           ``covariance_instruments`` were specified in ``product_data``, the computed optimal instruments will only be
           optimal with respect to the demand- and supply-side moments, not with respect to the addition of any
           covariance moments as well.

        Parameters
        ----------
        method : `str, optional`
            The method by which the integral over the joint density of :math:`\xi` and :math:`\omega` is approximated.
            The following methods are supported:

                - ``'approximate'`` (default) - Evaluate the Jacobians at the expected value of the error terms: zero
                  (``draws`` will be ignored).

                - ``'normal'`` - Draw from the normal approximation to the joint distribution of the error terms and
                  take the average over the computed Jacobians (``draws`` determines the number of draws).

                - ``'empirical'`` - Draw with replacement from the empirical joint distribution of the error terms and
                  take the average over the computed Jacobians (``draws`` determines the number of draws).

        draws : `int, optional`
            The number of draws that will be taken from the joint distribution of the error terms. This is ignored if
            ``method`` is ``'approximate'``. Because the default ``method`` is ``'approximate'``, the default number of
            draws is ``1``, even though it will be ignored. For ``'normal'`` or empirical, larger numbers such as
            ``100`` or ``1000`` are recommended.
        seed : `int, optional`
            Passed to :class:`numpy.random.RandomState` to seed the random number generator before any draws are
            taken. By default, a seed is not passed to the random number generator.
        expected_prices : `array-like, optional`
            Vector of expected prices conditional on all exogenous variables, :math:`E[p \mid Z]`. By default, if a
            supply side was estimated and ``shares`` did not enter into the formulation for :math:`X_3` in
            :class:`Problem`, ``iteration`` is used. Otherwise, this is by default estimated with the fitted values from
            a reduced form regression of endogenous prices onto :math:`Z_D`.
        iteration : `Iteration, optional`
            :class:`Iteration` configuration used to estimate expected prices by iterating over the :math:`\zeta`-markup
            contraction in :eq:`zeta_contraction`. By default, if a supply side was estimated, this is
            ``Iteration('simple', {'atol': 1e-12})``. Analytic Jacobians are not supported for solving this system.
            This configuration is not used if ``expected_prices`` is specified.
        constant_costs : `bool, optional`
            Whether to assume that marginal costs, :math:`c`, remain constant as equilibrium prices and shares change.
            By default this is ``True``, which means that firms treat marginal costs as constant when setting prices.
            If set to ``False``, marginal costs will be allowed to adjust if ``shares`` was included in
            the formulation for :math:`X_3` in :class:`Problem`. This is not relevant if a supply side was not
            estimated.

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
            sample = lambda: [np.zeros_like(self.xi), np.zeros_like(self.omega)]
        else:
            state = np.random.RandomState(seed)
            if method == 'normal':
                if self.problem.K3 == 0:
                    variance = np.var(self.xi)
                    sample = lambda: [np.c_[state.normal(0, variance, self.problem.N)], self.omega]
                else:
                    covariance_matrix = np.cov(self.xi, self.omega, rowvar=False)
                    sample = lambda: np.hsplit(state.multivariate_normal([0, 0], covariance_matrix, self.problem.N), 2)
            elif method == 'empirical':
                if self.problem.K3 == 0:
                    sample = lambda: [self.xi[state.choice(self.problem.N, self.problem.N)], self.omega]
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
        elif self.problem.K3 > 0 and 'shares' not in {n for f in self.problem._X3_formulations for n in f.names}:
            if iteration is None:
                iteration = Iteration('simple', {'atol': 1e-12})
            elif not isinstance(iteration, Iteration):
                raise TypeError("iteration must be None or an Iteration instance.")
            elif iteration._compute_jacobian:
                raise ValueError("Analytic Jacobians are not supported for solving this system.")
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

        # average over realizations
        computed_expected_prices = np.zeros_like(self.problem.products.prices)
        expected_shares = np.zeros_like(self.problem.products.shares)
        expected_xi_jacobian = np.zeros_like(self.xi_by_theta_jacobian)
        expected_omega_jacobian = np.zeros_like(self.omega_by_theta_jacobian)
        iteration_stats: List[Dict[Hashable, SolverStats]] = []
        for _ in output_progress(range(draws), draws, start_time):
            prices_i, shares_i, xi_jacobian_i, omega_jacobian_i, iteration_stats_i, errors_i = (
                self._compute_realizations(expected_prices, iteration, constant_costs, *sample())
            )
            computed_expected_prices += prices_i / draws
            expected_shares += shares_i / draws
            expected_xi_jacobian += xi_jacobian_i / draws
            expected_omega_jacobian += omega_jacobian_i / draws
            iteration_stats.append(iteration_stats_i)
            errors.extend(errors_i)

        # output a warning about any errors
        if errors:
            output("")
            output(exceptions.MultipleErrors(errors))
            output("")

        # compute the optimal instruments
        with np.errstate(all='ignore'):
            if self.problem.K3 == 0:
                inverse_covariance_matrix = np.c_[1 / np.var(self.xi)]
                demand_instruments = inverse_covariance_matrix * expected_xi_jacobian
                supply_instruments = np.full((self.problem.N, 0), np.nan, options.dtype)
            else:
                inverse_covariance_matrix = np.c_[scipy.linalg.inv(np.cov(self.xi, self.omega, rowvar=False))]
                expected_jacobian = np.stack([expected_xi_jacobian, expected_omega_jacobian], axis=1)
                instruments = inverse_covariance_matrix @ expected_jacobian
                demand_instruments, supply_instruments = np.split(instruments.reshape((self.problem.N, -1)), 2, axis=1)

        # structure the results
        from .optimal_instrument_results import OptimalInstrumentResults  # noqa
        results = OptimalInstrumentResults(
            self, demand_instruments, supply_instruments, inverse_covariance_matrix, expected_xi_jacobian,
            expected_omega_jacobian, computed_expected_prices, expected_shares, start_time, time.time(), draws,
            iteration_stats
        )
        output(f"Computed optimal instruments after {format_seconds(results.computation_time)}.")
        output("")
        output(results)
        return results

    def _compute_realizations(
            self, expected_prices: Optional[Array], iteration: Optional[Iteration], constant_costs: bool, xi: Array,
            omega: Array) -> Tuple[Array, Array, Array, Array, Dict[Hashable, SolverStats], List[Error]]:
        """If they have not already been estimated, compute the equilibrium prices, shares, and delta associated with a
        realization of xi and omega market-by-market. Then, compute realizations of Jacobians of xi and omega with
        respect to theta.
        """
        errors: List[Error] = []

        # compute delta (which will change under equilibrium prices) and marginal costs (which won't change)
        delta = self.delta - self.xi + xi
        costs = tilde_costs = self.tilde_costs - self.omega + omega
        if self.problem.costs_type == 'log':
            costs = np.exp(costs)

        def equilibrium_market_factory(
                s: Hashable) -> Tuple[EconomyResultsMarket, Array, Optional[Array], Optional[Iteration], bool]:
            """Build a market along with arguments used to compute equilibrium prices and shares along with delta."""
            market_s = EconomyResultsMarket(
                self.problem, s, self._parameters, self.sigma, self.pi, self.rho, self.beta, self.gamma, delta
            )
            costs_s = costs[self.problem._product_market_indices[s]]
            prices_s = expected_prices[self.problem._product_market_indices[s]] if expected_prices is not None else None
            return market_s, costs_s, prices_s, iteration, constant_costs

        # compute realizations of prices, shares, and delta market-by-market
        data_override = {
            'prices': np.zeros_like(self.problem.products.prices),
            'shares': np.zeros_like(self.problem.products.shares)
        }
        iteration_stats: Dict[Hashable, SolverStats] = {}
        generator = generate_items(
            self.problem.unique_market_ids, equilibrium_market_factory,
            EconomyResultsMarket.safely_solve_equilibrium_realization
        )
        for t, (prices_t, shares_t, delta_t, iteration_stats_t, errors_t) in generator:
            data_override['prices'][self.problem._product_market_indices[t]] = prices_t
            data_override['shares'][self.problem._product_market_indices[t]] = shares_t
            delta[self.problem._product_market_indices[t]] = delta_t
            iteration_stats[t] = iteration_stats_t
            errors.extend(errors_t)

        # only compute Jacobians if necessary
        xi_jacobian = np.full((self.problem.N, self._parameters.P), np.nan, options.dtype)
        omega_jacobian = np.full((self.problem.N, self._parameters.P), np.nan, options.dtype)
        if self._parameters.P > 0:
            def jacobian_market_factory(s: Hashable) -> Tuple[EconomyResultsMarket, Array]:
                """Build a market with the data realization along with arguments used to compute the Jacobians."""
                market_s = EconomyResultsMarket(
                    self.problem, s, self._parameters, self.sigma, self.pi, self.rho, self.beta, delta=delta,
                    data_override=data_override
                )
                tilde_costs_s = tilde_costs[self.problem._product_market_indices[s]]
                return market_s, tilde_costs_s

            # compute the Jacobians market-by-market
            generator = generate_items(
                self.problem.unique_market_ids, jacobian_market_factory,
                EconomyResultsMarket.safely_compute_jacobian_realizations
            )
            for t, (xi_jacobian_t, omega_jacobian_t, errors_t) in generator:
                xi_jacobian[self.problem._product_market_indices[t]] = xi_jacobian_t

                if self.problem.K3 > 0:
                    omega_jacobian[self.problem._product_market_indices[t]] = omega_jacobian_t

                errors.extend(errors_t)

            # replace invalid elements
            bad_xi_jacobian_index = ~np.isfinite(xi_jacobian)
            if np.any(bad_xi_jacobian_index):
                xi_jacobian[bad_xi_jacobian_index] = self.xi_by_theta_jacobian[bad_xi_jacobian_index]
                errors.append(exceptions.XiByThetaJacobianReversionError(bad_xi_jacobian_index))

            if self.problem.K3 > 0:
                bad_omega_jacobian_index = ~np.isfinite(omega_jacobian)
                if np.any(bad_omega_jacobian_index):
                    omega_jacobian[bad_omega_jacobian_index] = self.omega_by_theta_jacobian[bad_omega_jacobian_index]
                    errors.append(exceptions.OmegaByThetaJacobianReversionError(bad_omega_jacobian_index))

        return data_override['prices'], data_override['shares'], xi_jacobian, omega_jacobian, iteration_stats, errors

    def importance_sampling(
            self, draws: int, ar_constant: float = 1.0, seed: Optional[int] = None,
            agent_data: Optional[Mapping] = None, integration: Optional[Integration] = None,
            delta: Optional[Any] = None) -> 'ImportanceSamplingResults':
        r"""Use importance sampling to construct nodes and weights for integration.

        Importance sampling is done with the accept/reject procedure of
        :ref:`references:Berry, Levinsohn, and Pakes (1995)`. First, ``agent_data`` and/or ``integration`` are used to
        provide a large number of candidate sampling nodes :math:`\nu` and any demographics :math:`d`.

        Out of these candidate agent data, each candidate agent :math:`i` in market :math:`t` is accepted with
        probability :math:`\frac{1 - s_{i0t}}{M}` where :math:`M \geq 1` is some accept-reject constant. The probability
        of choosing an inside good :math:`1 - s_{i0t}`, is evaluated at the estimated :math:`\hat{\theta}` and
        :math:`\delta(\hat{\theta})`.

        Optionally, :meth:`ProblemResults.compute_delta` can be used to provide a more precise
        :math:`\delta(\hat{\theta})` than the estimated :attr:`ProblemResults.delta`. The idea is that more precise
        agent data (i.e., more integration nodes) would be infeasible to use during estimation, but is feasible here
        because :math:`\delta(\hat{\theta})` only needs to be computed once given a :math:`\hat{\theta}`.

        Out of the remaining accepted agents, :math:`I_t` equal to ``draws`` are randomly selected within each market
        :math:`t` and assigned integration weights :math:`w_{it} = \frac{1}{I_t} \cdot \frac{1 - s_{0t}}{1 - s_{i0t}}`.

        If this procedure accepts fewer than ``draws`` agents in a market, an exception will be raised. A good rule of
        thumb is to provide more candidate draws in each market than :math:`\frac{M \times I_t}{1 - s_{0t}}`.

        Parameters
        ----------
        draws : `int, optional`
            Number of draws to take from ``sampling_agent_data`` in each market.
        ar_constant : `float, optional`
            Accept/reject constant :math:`M \geq 1`, which is by default, ``1.0``.
        seed : `int, optional`
            Passed to :class:`numpy.random.RandomState` to seed the random number generator before importance sampling
            is done. By default, a seed is not passed to the random number generator.
        agent_data : `structured array-like, optional`
            Agent data from which draws will be sampled, which should have the same structure as ``agent_data`` in
            :class:`Problem`. The ``weights`` field does not need to be specified, and if it is specified it will be
            ignored. By default, the same agent data used to solve the problem will be used.
        integration : `Integration, optional`
            :class:`Integration` configuration for how to build nodes from which draws will be sampled, which will
            replace any ``nodes`` field in ``sampling_agent_data``. This configuration is required if
            ``sampling_agent_data`` is specified without a ``nodes`` field.
        delta : `array-like, optional`
            More precise :math:`\delta(\hat{\theta})` than the estimated :attr:`ProblemResults.delta`, which can be
            computed by passing a more precise integration rule to :meth:`ProblemResults.compute_delta`. By default,
            :attr:`ProblemResults.delta` is used.

        Returns
        -------
        `ImportanceSamplingResults`
           Computed :class:`ImportanceSamplingResults`.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        errors: List[Error] = []

        # keep track of long it takes to do importance sampling
        output("Importance sampling ...")
        start_time = time.time()

        # validate the configuration
        if self.problem.K2 == 0:
            raise ValueError("Importance sampling is only relevant when there are agent data.")
        if not isinstance(draws, int) or draws < 1:
            raise ValueError("draws must be a positive int.")
        if not isinstance(ar_constant, (int, float)) or ar_constant < 1:
            raise ValueError("ar_constant must be a float that is no less than 1.")

        # validate any more precise delta
        if delta is None:
            delta = self.delta
        else:
            delta = np.c_[np.asarray(delta, options.dtype)]
            if delta.shape != (self.problem.N, 1):
                raise ValueError(f"delta must be a vector with {self.problem.N} elements.")

        # construct agents that will be sampled from
        agents = self.problem.agents
        if agent_data is not None or integration is not None:
            agents = Agents(
                self.problem.products, self.problem.agent_formulation, agent_data, integration, check_weights=False
            )

        # compute importance sampling weights
        weights, weights_errors = self._compute_importance_weights(agents, delta, draws, ar_constant, seed)
        errors.extend(weights_errors)

        # output a warning about any errors
        if errors:
            output("")
            output(exceptions.MultipleErrors(errors))
            output("")

        # update the agent data
        with np.errstate(all='ignore'):
            sampled_agents = update_matrices(agents, {'weights': (weights, options.dtype)})
            sampled_agents = sampled_agents[weights.flat > 0]

        # structure the results
        from .importance_sampling_results import ImportanceSamplingResults  # noqa
        results = ImportanceSamplingResults(self, sampled_agents, start_time, time.time(), draws)
        output(f"Finished importance sampling after {format_seconds(results.computation_time)}.")
        output("")
        output(results)
        return results

    def _compute_importance_weights(
            self, agents: RecArray, delta: Array, draws: int, ar_constant: float, seed: Optional[int]) -> (
            Tuple[Array, List[Error]]):
        """Compute the importance sampling weights associated with a set of agents."""
        errors: List[Error] = []
        market_indices = get_indices(agents.market_ids)

        def market_factory(s: Hashable) -> Tuple[EconomyResultsMarket]:
            """Build a market use to compute probabilities."""
            market_s = EconomyResultsMarket(
                self.problem, s, self._parameters, self.sigma, self.pi, self.rho, delta=delta,
                agents_override=agents[market_indices[s]]
            )
            return market_s,

        # compute weights market-by-market
        state = np.random.RandomState(seed)
        weights = np.zeros_like(agents.weights)
        generator = generate_items(
            self.problem.unique_market_ids, market_factory, EconomyResultsMarket.safely_compute_probabilities
        )
        for t, (probabilities_t, errors_t) in generator:
            errors.extend(errors_t)
            with np.errstate(all='ignore'):
                inside_share_t = self.problem.products.shares[self.problem._product_market_indices[t]].sum()
                inside_probabilities = probabilities_t.sum(axis=0)
                probability_cutoffs_t = state.uniform(size=inside_probabilities.size)
                accept_indices_t = np.where(probability_cutoffs_t < inside_probabilities / ar_constant)[0]
                try:
                    sampled_indices_t = state.choice(accept_indices_t, size=draws, replace=False)
                except ValueError:
                    raise RuntimeError(
                        f"The number of accepted draws in market '{t}' was {accept_indices_t.size}, which is less then "
                        f"{draws}. Either decrease the number of desired draws in each market or increase the size of "
                        f"sampling_agent_data and/or sampling_integration."
                    )
                weights_t = np.zeros_like(inside_probabilities)
                weights_t[sampled_indices_t] = inside_share_t / inside_probabilities[sampled_indices_t] / draws
                weights[market_indices[t]] = weights_t[:, None]

        return weights, errors
