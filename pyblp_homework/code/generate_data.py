import numpy as np
import pandas as pd
from scipy.optimize import root

# This script manually generates data for a BLP model, following the logic
# of the pyblp library but without using its high-level Simulation class.
# This approach makes the data generation process transparent and customizable.

# --- Section 1: Model and Simulation Parameters ---

# Set a seed for reproducibility
np.random.seed(0)

# Set up dimensions of the simulation
T = 600  # Number of markets
J = 4    # Number of products per market
N = 1000 # Number of simulated consumers per market

# True parameter values for the utility and cost functions
alpha = -2      # Price coefficient
beta1 = 1       # Quality coefficient (referred to as x in the code)
gamma0 = 0.5    # Intercept for marginal costs
gamma1 = 0.25   # Coefficient on the cost shifter (w)

# Distribution of the two random coefficients (on satellite and wired)
beta_mean = np.array([4, 4])
beta_cov = np.array([[1, 0], [0, 1]])

# Distribution of unobserved demand (xi) and cost (omega) shocks
shocks_mean = np.array([0, 0])
shocks_cov = np.array([[1, 0.25], [0.25, 1]]) # 0.25 correlation

# --- Section 2: Draw Exogenous Variables and Unobservables ---

print("Generating exogenous variables and unobservables...")

# Exogenous product characteristics
quality = np.abs(np.random.randn(T, J)) # Corresponds to 'x' in original code
cost_shifter = np.abs(np.random.randn(T, J)) # Corresponds to 'w' in original code

# Unobserved demand-side (xi) and supply-side (omega) shocks
shocks = np.random.multivariate_normal(shocks_mean, shocks_cov, size=(T, J))
xi = shocks[:, :, 0]
omega = shocks[:, :, 1]

# Consumer-specific random coefficients for satellite and wired
beta_i = np.random.multivariate_normal(beta_mean, beta_cov, size=(T, N))

# Create indicator variables for the two types of products
satellite = np.zeros((T, J)); satellite[:, 0:2] = 1
wired = np.zeros((T, J)); wired[:, 2:4] = 1

# True marginal costs, as a function of the cost shifter and shock
mc = np.exp(gamma0 + gamma1 * cost_shifter + omega)

# --- Section 3: Solve for Equilibrium Prices ---

# This section manually solves for the Bertrand-Nash equilibrium prices market by market.
# This is the core logic that pyblp's `simulation.replace_endogenous` handles internally.

def calculate_shares_and_derivs(p_t, t):
    """Calculates market shares and their price derivatives for a single market t.
    This is the manual equivalent of the demand side of a pyblp.Market object.
    """
    # Mean utility (delta) for all products in market t
    delta_t = alpha * p_t + beta1 * quality[t, :] + xi[t, :]

    # Consumer-specific utility component (mu)
    mu_it = (beta_i[t, :, 0][:, np.newaxis] * satellite[t, :] +
             beta_i[t, :, 1][:, np.newaxis] * wired[t, :])

    # Total utility for each consumer-product pair
    u_it = delta_t + mu_it

    # Compute choice probabilities (logit formula)
    exp_u = np.exp(u_it)
    denom = 1 + np.sum(exp_u, axis=1, keepdims=True)
    s_it = exp_u / denom

    # Aggregate to find market shares
    s_t = np.mean(s_it, axis=0)

    # Calculate the matrix of price derivatives (ds_j / dp_k)
    dsdp = np.zeros((J, J))
    for j in range(J):
        for k in range(J):
            if j == k:
                deriv_integrand = alpha * s_it[:, j] * (1 - s_it[:, k])
            else:
                deriv_integrand = -alpha * s_it[:, j] * s_it[:, k]
            dsdp[j, k] = np.mean(deriv_integrand)

    return s_t, dsdp

def foc_system(p_t, t):
    """Defines the system of J first-order conditions for a market t for scipy.optimize.root."""
    s_t, dsdp = calculate_shares_and_derivs(p_t, t)
    # For single-product firms, the FOC is: p_j - mc_j = -s_j / (ds_j/dp_j)
    # The function should return a vector of zeros at the solution.
    return p_t - mc[t, :] + s_t / np.diag(dsdp)

def solve_prices_ms_iteration(t, initial_prices, tol=1e-12, max_iter=5000):
    """Solves for equilibrium prices using a fixed-point iteration.
    This is a simplified version of the Morrow-Skerlos (2011) zeta-markup contraction
    used by pyblp, valid for the single-product firm case.
    """
    p_old = initial_prices
    for _ in range(max_iter):
        s_t, dsdp = calculate_shares_and_derivs(p_old, t)
        diag_dsdp = np.diag(dsdp)
        if np.any(np.abs(diag_dsdp) < 1e-14):
            return None, False # Avoid division by zero
        
        # Markup formula for single-product firms
        p_new = mc[t, :] - s_t / diag_dsdp
        
        if np.max(np.abs(p_new - p_old)) < tol:
            return p_new, True # Convergence
        
        p_old = p_new
        
    return None, False # Failed to converge

print("\nSolving for equilibrium prices in each market...")

equilibrium_prices = np.zeros((T, J))
equilibrium_shares = np.zeros((T, J))

for t in range(T):
    initial_prices = mc[t, :] # Use marginal costs as a starting guess
    
    # Method 1: Morrow-Skerlos (2011) style fixed-point iteration
    p_star, success = solve_prices_ms_iteration(t, initial_prices)
    
    # Method 2: Fallback to a generic nonlinear solver if iteration fails
    if not success:
        solution_root = root(foc_system, initial_prices, args=(t,))
        if solution_root.success:
            p_star = solution_root.x
            success = True

    if success:
        s_star, _ = calculate_shares_and_derivs(p_star, t)
        equilibrium_prices[t, :] = p_star
        equilibrium_shares[t, :] = s_star
    else:
        print(f"Warning: Price solution failed for market {t}. Using NaNs.")
        equilibrium_prices[t, :] = np.nan
        equilibrium_shares[t, :] = np.nan

    if (t + 1) % 100 == 0:
        print(f"  ...solved for {t + 1} / {T} markets.")

print("Finished solving for equilibrium prices.")

# --- Section 4: Assemble and Save the Dataset ---

# Reshape data into a long format (one row per product-market)
data = pd.DataFrame({
    'market_ids': np.repeat(np.arange(T), J),
    'firm_ids': np.tile(np.arange(J), T), # Single-product firms
    'shares': equilibrium_shares.flatten(),
    'prices': equilibrium_prices.flatten(),
    'quality': quality.flatten(),
    'cost_shifter': cost_shifter.flatten(),
    'demand_shock': xi.flatten(),
    'cost_shock': omega.flatten(),
    'satellite': np.tile(satellite[0,:], T),
    'wired': np.tile(wired[0,:], T)
})

# Drop markets where the price solution failed
data.dropna(inplace=True)

# Save the dataset
output_path = '/Users/marek/Library/CloudStorage/Dropbox/github/pyblp/pyblp_homework/data/blp_data.csv'
data.to_csv(output_path, index=False)

print(f"\nDataset successfully generated and saved to {output_path}")
print(f"Number of observations: {len(data)}")
print("\nFirst 5 rows of the dataset:")
print(data.head())
