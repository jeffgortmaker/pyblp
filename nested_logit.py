"""
Pure nested logit estimation with two-step efficient GMM and GMM standard errors.

- Analytic delta inversion (pure nested logit).
- Initial GMM with identity weighting, then compute optimal weight and re-run GMM.
- Compute asymptotic covariance of parameters (rho and beta) using numerical derivatives.

Requires: numpy, scipy
"""

import numpy as np
from scipy.optimize import minimize_scalar

# -----------------------------
# Reused helper functions
# -----------------------------

def compute_nest_shares(product_shares, nesting_ids):
    product_shares = np.asarray(product_shares)
    single_market = product_shares.ndim == 1
    if single_market:
        product_shares = product_shares[None, :]
    T, J = product_shares.shape
    H = int(np.max(nesting_ids) + 1)
    shares_by_nest = np.zeros_like(product_shares)
    for h in range(H):
        mask = (np.array(nesting_ids) == h)
        if not mask.any():
            continue
        nest_sum = product_shares[:, mask].sum(axis=1)
        shares_by_nest[:, mask] = nest_sum[:, None]
    if single_market:
        return shares_by_nest[0]
    return shares_by_nest

def analytic_delta_nested(product_shares, nesting_ids, rho, s0=None):
    arr = np.asarray(product_shares)
    single_market = (arr.ndim == 1)
    if single_market:
        arr = arr[None, :]
    T, J = arr.shape
    nesting_ids = np.asarray(nesting_ids)
    if np.isscalar(rho):
        rho_arr = np.full(int(np.max(nesting_ids) + 1), rho)
    else:
        rho_arr = np.asarray(rho)
    if s0 is None:
        s0 = 1.0 - arr.sum(axis=1)
    s0 = np.asarray(s0)
    if s0.ndim == 0:
        s0 = np.full(T, float(s0))
    shares_by_nest = compute_nest_shares(arr, nesting_ids)
    eps = 1e-12
    delta = np.log(arr + eps) - np.log(s0[:, None] + eps) - rho_arr[nesting_ids] * (np.log(arr + eps) - np.log(shares_by_nest + eps))
    if single_market:
        return delta[0]
    return delta

def two_stage_least_squares(Z, X, y):
    Z = np.asarray(Z)
    X = np.asarray(X)
    y = np.asarray(y)
    ZZ = Z.T @ Z
    if np.linalg.matrix_rank(ZZ) < ZZ.shape[0]:
        ZZ = ZZ + 1e-8 * np.eye(ZZ.shape[0])
    Pz = Z @ np.linalg.inv(ZZ) @ Z.T
    Xp = Pz @ X
    XX = X.T @ Xp
    if np.linalg.matrix_rank(XX) < XX.shape[0]:
        XX = XX + 1e-8 * np.eye(XX.shape[0])
    beta = np.linalg.solve(XX, X.T @ (Pz @ y))
    resid = y - X @ beta
    return beta, resid

# -----------------------------
# Utilities for stacked data
# -----------------------------

def build_delta_stacked_for_rho(rho_scalar, shares_by_market, nesting_ids_by_market):
    """
    Given scalar rho, returns stacked delta vector (length N), and also returns list-of-arrays of per-market deltas.
    """
    delta_list = []
    markets = sorted(list(shares_by_market.keys()))
    for t in markets:
        s_t = np.array(shares_by_market[t], dtype=float)
        s0t = 1.0 - s_t.sum()
        nesting_ids_t = np.array(nesting_ids_by_market[t], dtype=int)
        delta_t = analytic_delta_nested(s_t, nesting_ids_t, rho_scalar, s0=s0t)
        delta_list.append(delta_t)
    delta_stacked = np.concatenate(delta_list)
    return delta_stacked, delta_list

# -----------------------------
# GMM machinery, scaled by N (sample size)
# -----------------------------

def compute_moments_and_S(delta_stacked, X, Z):
    """
    Compute sample moments g = (1/N) Z' xi and estimate S = (1/N) sum_i (z_i z_i' xi_i^2)
    Inputs:
      delta_stacked: (N,) delta per observation
      X: (N,K)
      Z: (N,L)
    Returns:
      g: (L,) moments (scaled by 1/N)
      xi: (N,)
      S: (L,L) covariance of moments (scaled by 1/N)
    """
    N = X.shape[0]
    # estimate beta by 2SLS using delta as dependent var
    beta_hat, resid_beta = two_stage_least_squares(Z, X, delta_stacked)
    xi = delta_stacked - X @ beta_hat  # (N,)
    # compute g = (1/N) Z' xi
    g = (Z.T @ xi) / N
    # compute S = (1/N) sum_i (z_i z_i' xi_i^2) = (Z * xi).T @ (Z * xi) / N
    Z_xi = Z * xi[:, None]  # (N, L)
    S = (Z_xi.T @ Z_xi) / N
    return g, xi, S, beta_hat

def gmm_objective_scaled(rho_scalar, shares_by_market, nesting_ids_by_market, X, Z, W):
    """
    GMM objective based on scaled moments (g = 1/N Z' xi): obj = g' W g
    """
    N = X.shape[0]
    delta_stacked, _ = build_delta_stacked_for_rho(rho_scalar, shares_by_market, nesting_ids_by_market)
    g, xi, S, beta_hat = compute_moments_and_S(delta_stacked, X, Z)
    obj = float(g.T @ W @ g)
    return obj

# -----------------------------
# Numerical derivative for d delta / d rho (stacked)
# -----------------------------

def numeric_derivative_delta_rho(rho, shares_by_market, nesting_ids_by_market, h=1e-6):
    """
    Compute stacked derivative d delta_stacked / d rho at scalar rho via central differences.
    Returns vector ddelta_drho of length N.
    """
    rho_plus = rho + h
    rho_minus = rho - h
    delta_plus, _ = build_delta_stacked_for_rho(rho_plus, shares_by_market, nesting_ids_by_market)
    delta_minus, _ = build_delta_stacked_for_rho(rho_minus, shares_by_market, nesting_ids_by_market)
    ddelta = (delta_plus - delta_minus) / (2.0 * h)
    return ddelta

# -----------------------------
# Two-step GMM estimator + standard errors
# -----------------------------

def two_step_gmm_with_se(shares_by_market, nesting_ids_by_market, X, Z, rho_bounds=(1e-4, 0.99)):
    """
    Two-step efficient GMM for rho, and compute standard errors for (rho, beta).
    Returns dictionary with estimates and standard errors.
    """
    N = X.shape[0]
    # Initial W = identity (LxL)
    L = Z.shape[1]
    W0 = np.eye(L)

    # 1) First step: minimize objective with W0
    obj1 = lambda r: gmm_objective_scaled(r, shares_by_market, nesting_ids_by_market, X, Z, W0)
    sol1 = minimize_scalar(obj1, bounds=rho_bounds, method='bounded', options={'xatol':1e-4})
    rho1 = float(sol1.x)

    # Compute moments and S at rho1
    delta1, _ = build_delta_stacked_for_rho(rho1, shares_by_market, nesting_ids_by_market)
    g1, xi1, S1, beta1 = compute_moments_and_S(delta1, X, Z)

    # 2) Compute optimal weight W = S^{-1}
    # Regularize S if needed
    S_mat = S1
    try:
        W_opt = np.linalg.inv(S_mat)
    except np.linalg.LinAlgError:
        S_reg = S_mat + 1e-8 * np.eye(S_mat.shape[0])
        W_opt = np.linalg.inv(S_reg)

    # 3) Second step: minimize objective with W_opt, starting from rho1
    obj2 = lambda r: gmm_objective_scaled(r, shares_by_market, nesting_ids_by_market, X, Z, W_opt)
    sol2 = minimize_scalar(obj2, bounds=rho_bounds, method='bounded', options={'xatol':1e-6})
    rho_hat = float(sol2.x)
    success = sol2.success

    # recompute final delta, g, xi, S, beta at rho_hat
    delta_hat, _ = build_delta_stacked_for_rho(rho_hat, shares_by_market, nesting_ids_by_market)
    g_hat, xi_hat, S_hat, beta_hat = compute_moments_and_S(delta_hat, X, Z)

    # Use S_hat to form final W (robust)
    try:
        W_final = np.linalg.inv(S_hat)
    except np.linalg.LinAlgError:
        S_hat_reg = S_hat + 1e-8 * np.eye(S_hat.shape[0])
        W_final = np.linalg.inv(S_hat_reg)

    # -------------------------
    # Compute derivative matrix D = d g / d theta
    # theta = [rho, beta (K x 1)]  -> dimension p = 1 + K
    # g = (1/N) Z' xi = (1/N) Z'( delta(rho) - X beta )
    # So:
    #   dg/dbeta = -(1/N) Z' X   (L x K)
    #   dg/drho  = (1/N) Z' ( d delta / d rho )   (L x 1)
    # -------------------------
    K = X.shape[1]
    p = 1 + K
    # compute dg/dbeta
    dg_dbeta = - (Z.T @ X) / N  # shape (L, K)
    # compute dg/drho numerically
    ddelta_drho = numeric_derivative_delta_rho(rho_hat, shares_by_market, nesting_ids_by_market, h=1e-6)  # (N,)
    dg_drho = (Z.T @ ddelta_drho) / N  # (L,)
    # assemble D (L x p)
    D = np.zeros((Z.shape[1], p))
    D[:, 0] = dg_drho
    D[:, 1:] = dg_dbeta

    # Compute GMM covariance:
    # Var(theta) = (D' W D)^{-1} (D' W S W D) (D' W D)^{-1} / N
    # Note: our g, S were already scaled by 1/N; the 1/N factor for Var appears explicitly
    DW = D.T @ W_final
    try:
        A = np.linalg.inv(DW @ D)
    except np.linalg.LinAlgError:
        A = np.linalg.inv(DW @ D + 1e-8 * np.eye(D.shape[1]))
    middle = DW @ S_hat @ (W_final @ D)
    Var_theta = (A @ middle @ A) / N

    # extract standard errors
    se_theta = np.sqrt(np.diag(Var_theta))
    se_rho = se_theta[0]
    se_beta = se_theta[1:]

    # Also compute conventional 2SLS se for beta as a sanity check (not used for reporting here).
    # 2SLS variance (white) = (X'PzX)^{-1} X' Pz diag(xi^2) Pz X (X'PzX)^{-1}
    ZZ = Z.T @ Z
    if np.linalg.matrix_rank(ZZ) < ZZ.shape[0]:
        ZZ = ZZ + 1e-8 * np.eye(ZZ.shape[0])
    Pz = Z @ np.linalg.inv(ZZ) @ Z.T
    XPzX = X.T @ Pz @ X
    try:
        XPzX_inv = np.linalg.inv(XPzX)
    except np.linalg.LinAlgError:
        XPzX_inv = np.linalg.inv(XPzX + 1e-8 * np.eye(XPzX.shape[0]))
    # compute sandwich
    Xi_diag = xi_hat  # (N,)
    ZX = Z * Xi_diag[:, None]
    S_beta = (X.T @ (Pz @ np.diag(Xi_diag**0) ) @ X)  # not correct shape; skip elaborate 2SLS robust here
    # skip detailed 2SLS se derivation — rely on GMM var above which covers rho and beta jointly.

    results = {
        'rho_hat': rho_hat,
        'se_rho': se_rho,
        'beta_hat': beta_hat,
        'se_beta': se_beta,
        'delta_hat': delta_hat,
        'xi_hat': xi_hat,
        'g_hat': g_hat,
        'S_hat': S_hat,
        'W_final': W_final,
        'success': success,
        'sol': sol2,
        'Var_theta': Var_theta
    }
    return results

# -----------------------------
# Synthetic data example (same as before)
# -----------------------------

def synthetic_data_example():
    rng = np.random.default_rng(12345)
    T = 40
    J_per_market = 4
    H = 2
    beta_true = np.array([-1.5, 0.8])
    rho_true = 0.35
    shares_by_market = {}
    nesting_ids_by_market = {}
    X_rows = []
    Z_rows = []
    market_index = []
    for t in range(T):
        quality = rng.normal(loc=0.0, scale=1.0, size=J_per_market)
        cost = rng.normal(loc=1.0, scale=0.5, size=J_per_market)
        price = cost + 0.3 * quality + rng.normal(scale=0.2, size=J_per_market)
        X_t = np.column_stack([price, quality])
        other_inst = rng.normal(size=J_per_market)
        Z_t = np.column_stack([np.ones(J_per_market), cost, other_inst])
        nesting_ids = rng.integers(0, H, size=J_per_market)
        xi = rng.normal(scale=0.5, size=J_per_market)
        delta_true = X_t @ beta_true + xi
        V_j = delta_true.copy()
        V_h = np.zeros(H)
        for h in range(H):
            mask = (nesting_ids == h)
            if not mask.any():
                V_h[h] = -1e9
            else:
                V_h[h] = (1 - rho_true) * np.log(np.sum(np.exp(V_j[mask] / (1 - rho_true))))
        exp_V_h = np.exp(V_h)
        denom = 1.0 + exp_V_h.sum()
        s_j = np.empty(J_per_market)
        for j in range(J_per_market):
            h = nesting_ids[j]
            num = np.exp(V_j[j] / (1 - rho_true)) / np.exp(V_h[h] / (1 - rho_true))
            s_j[j] = num * (np.exp(V_h[h]) / denom)
        s_j = np.clip(s_j, 1e-8, 1 - 1e-8)
        shares_by_market[t] = s_j
        nesting_ids_by_market[t] = nesting_ids
        X_rows.append(X_t)
        Z_rows.append(Z_t)
        market_index.extend([t] * J_per_market)
    X_stacked = np.vstack(X_rows)
    Z_stacked = np.vstack(Z_rows)
    market_indices = np.array(market_index)
    return shares_by_market, nesting_ids_by_market, X_stacked, Z_stacked, market_indices

# -----------------------------
# Run the example
# -----------------------------
if __name__ == "__main__":
    shares_by_market, nesting_ids_by_market, X, Z, market_indices = synthetic_data_example()
    results = two_step_gmm_with_se(shares_by_market, nesting_ids_by_market, X, Z)
    print("rho_hat:", results['rho_hat'], "se:", results['se_rho'])
    print("beta_hat:", results['beta_hat'])
    print("se_beta:", results['se_beta'])
    print("GMM success:", results['success'])