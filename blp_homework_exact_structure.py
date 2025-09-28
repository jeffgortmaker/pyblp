# %% [markdown]
# # PhD Empirical IO
# ## Fall 2024
# ## Prof. Conlon
# ## Homework Assignment
# 
# *Thanks to Phil Haile and Jaewon Lee for coding tips and other highly useful feedback on this problem set.*
# 
# **Due Oct 18**
# 
# ---
# 
# ## Overview
# 
# You will estimate demand and supply in a stylized model of the market for pay-TV services. You will use any programming language (Python/R/Matlab/Julia) to create your own fake data set for the industry and do some relatively simple estimation. Then, using the `pyBLP` package of Conlon and Gortmaker, you will estimate the model and perform some merger simulations. Using data you generate yourself gives you a way to check whether the estimation is working; this is a good thing to try whenever you code up an estimator! The pyBLP package has excellent documentation and a very helpful tutorial (which covers merger simulation), both easy to find (https://pyblp.readthedocs.io/en/stable/). You may want to work through the tutorial notebooks available with the documentation (or on the Github page).
# 
# To install `pyBLP` you need to have Python 3 installed, I recommend Anaconda https://www.anaconda.com/distribution/. If you have python installed you simply need to type:
# ```
# pip install pyblp
# ```
# or 
# ```
# pip install git+https://github.com/jeffgortmaker/pyblp
# ```
# 
# Please submit a single printed document presenting your answers to the questions below, requested results, and code. Write this up cleanly with nice tables where appropriate. You may work in groups of up to 3 on the coding, but your write-up must be your own work and must indicate who your partners are.
# 
# You can do parts (2) and (3) in R, Matlab, Julia, or Python. Parts (4) and (5) use `pyblp` which you can run in R using `reticulate` if you really want.

# %%
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import fsolve, root
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
import warnings
warnings.filterwarnings('ignore')

# For pyBLP estimation
import pyblp

# Set random seed for reproducibility
np.random.seed(42)

print("Libraries imported successfully")
print(f"NumPy version: {np.__version__}")
print(f"PyBLP version: {pyblp.__version__}")

# %% [markdown]
# ## 1. Model
# 
# There are $T$ markets, each with four inside goods $j \in \{1,2,3,4\}$ and an outside option. Goods 1 and 2 are satellite television services (e.g., DirecTV and Dish); goods 3 and 4 are wired television services (e.g., Frontier and Comcast in New Haven).
# 
# The conditional indirect utility of consumer $i$ for good $j$ in market $t$ is given by
# 
# $$u_{ijt} = \beta^{(1)} x_{jt} + \beta_i^{(2)} satellite_{jt} + \beta_i^{(3)} wired_{jt} + \alpha p_{jt} + \xi_{jt} + \epsilon_{ijt} \quad j > 0$$
# 
# $$u_{i0t} = \epsilon_{i0t}$$
# 
# where $x_{jt}$ is a measure of good $j$'s quality, $p_{jt}$ is its price, $satellite_{jt}$ is an indicator equal to 1 for the two satellite services, and $wired_{jt}$ is an indicator equal to 1 for the two wired services. The remaining notation is as usual in the class notes, including the i.i.d. type-1 extreme value $\epsilon_{ijt}$. Each consumer purchases the good giving them the highest conditional indirect utility.
# 
# Goods are produced by single-product firms. Firm $j$'s (log) marginal cost in market $t$ is
# 
# $$\ln mc_{jt} = \gamma^0 + w_{jt}\gamma^1 + \omega_{jt}/8$$
# 
# where $w_{jt}$ is an observed cost shifter. Firms compete by simultaneously choosing prices in each market under complete information. Firm $j$ has profit
# 
# $$\pi_{jt} = \max_{p_{jt}} M_t(p_{jt} - mc_{jt})s_{jt}(p_t)$$

# %%
# Define model parameters (true values)
class ModelParameters:
    def __init__(self):
        # Demand parameters
        self.beta1 = 1.0          # Quality coefficient
        self.alpha = -2.0         # Price coefficient
        self.beta2_mean = 4.0     # Mean satellite preference
        self.beta3_mean = 4.0     # Mean wired preference  
        self.beta_std = 1.0       # Standard deviation of random coefficients
        
        # Supply parameters
        self.gamma0 = 0.5         # Cost intercept
        self.gamma1 = 0.25        # Cost shifter coefficient
        
        # Market structure
        self.T = 600              # Number of markets
        self.J = 4                # Number of inside products
        
        # Simulation parameters
        self.n_draws = 1000       # Number of simulation draws

# Initialize parameters
params = ModelParameters()

print("Model Parameters:")
print(f"β¹ (quality): {params.beta1}")
print(f"α (price): {params.alpha}")
print(f"β²ᵢ (satellite) ~ N({params.beta2_mean}, {params.beta_std}²)")
print(f"β³ᵢ (wired) ~ N({params.beta3_mean}, {params.beta_std}²)")
print(f"γ⁰ (cost intercept): {params.gamma0}")
print(f"γ¹ (cost shifter): {params.gamma1}")

# %% [markdown]
# ## 2. Generate Fake Data
# 
# Generate a data set from the model above. Let
# 
# $$\beta^{(1)} = 1, \quad \beta_i^{(k)} \sim \text{iid } N(4,1) \text{ for } k=2,3$$
# $$\alpha = -2$$
# $$\gamma^{(0)} = 1/2, \quad \gamma^{(1)} = 1/4$$

# %% [markdown]
# ### 1. Draw the exogenous product characteristic $x_{jt}$ for $T=600$ geographically defined markets (e.g., cities). Assume each $x_{jt}$ is equal to the absolute value of an iid standard normal draw, as is each $w_{jt}$. Simulate demand and cost unobservables as well, specifying
# 
# $$\begin{pmatrix} \xi_{jt} \\ \omega_{jt} \end{pmatrix} \sim N\left(\begin{pmatrix} 0 \\ 0 \end{pmatrix}, \begin{pmatrix} 1 & 0.25 \\ 0.25 & 1 \end{pmatrix}\right) \text{ iid across } j,t$$

# %%
def generate_exogenous_data(params):
    """Generate exogenous variables and unobservables."""
    
    # Product characteristics (absolute value of standard normal)
    x_jt = np.abs(np.random.normal(0, 1, (params.T, params.J)))
    w_jt = np.abs(np.random.normal(0, 1, (params.T, params.J)))
    
    # Unobservables with specified correlation structure
    sigma_matrix = np.array([[1.0, 0.25], [0.25, 1.0]])
    
    # Generate correlated unobservables
    unobservables = np.random.multivariate_normal([0, 0], sigma_matrix, size=(params.T, params.J))
    xi_jt = unobservables[:, :, 0]    # Demand unobservables
    omega_jt = unobservables[:, :, 1] # Supply unobservables
    
    # Create product indicators
    satellite_jt = np.zeros((params.T, params.J))
    satellite_jt[:, :2] = 1  # Products 1,2 are satellite
    
    wired_jt = np.zeros((params.T, params.J))
    wired_jt[:, 2:] = 1      # Products 3,4 are wired
    
    # Marginal costs
    ln_mc_jt = params.gamma0 + params.gamma1 * w_jt + omega_jt / 8
    mc_jt = np.exp(ln_mc_jt)
    
    return {
        'x_jt': x_jt,
        'w_jt': w_jt,
        'xi_jt': xi_jt,
        'omega_jt': omega_jt,
        'satellite_jt': satellite_jt,
        'wired_jt': wired_jt,
        'mc_jt': mc_jt,
        'ln_mc_jt': ln_mc_jt
    }

# Generate exogenous data
exog_data = generate_exogenous_data(params)

print("Exogenous Data Generated:")
print(f"Quality x_jt: mean={exog_data['x_jt'].mean():.3f}, std={exog_data['x_jt'].std():.3f}")
print(f"Cost shifter w_jt: mean={exog_data['w_jt'].mean():.3f}, std={exog_data['w_jt'].std():.3f}")
print(f"Marginal costs mc_jt: mean={exog_data['mc_jt'].mean():.3f}, range=[{exog_data['mc_jt'].min():.3f}, {exog_data['mc_jt'].max():.3f}]")

# Check correlation between demand and supply unobservables
xi_flat = exog_data['xi_jt'].flatten()
omega_flat = exog_data['omega_jt'].flatten()
correlation = np.corrcoef(xi_flat, omega_flat)[0, 1]
print(f"Correlation between ξ and ω: {correlation:.3f} (should be ≈ 0.25)")

# %% [markdown]
# ### 2. Solve for the equilibrium prices for each good in each market.

# %% [markdown]
# #### (a) Start by writing a procedure to approximate the derivatives of market shares with respect to prices (taking prices, shares, $x$, and demand parameters as inputs). The key steps are:

# %% [markdown]
# ##### i. For each $(j,t)$, write the choice probability $s_{jt}$ as a weighted average (integral) of the (multinomial logit) choice probabilities conditional on the value of each consumer's random coefficients;

# %%
def generate_consumer_draws(params):
    """Generate draws for consumer heterogeneity (done once)."""
    
    # Draw random coefficients for satellite and wired preferences
    beta_satellite_draws = np.random.normal(params.beta2_mean, params.beta_std, params.n_draws)
    beta_wired_draws = np.random.normal(params.beta3_mean, params.beta_std, params.n_draws)
    
    return beta_satellite_draws, beta_wired_draws

# Generate consumer draws once (to avoid jittering)
beta_sat_draws, beta_wire_draws = generate_consumer_draws(params)

print(f"Generated {params.n_draws} consumer draws:")
print(f"Satellite preference draws: mean={beta_sat_draws.mean():.3f}, std={beta_sat_draws.std():.3f}")
print(f"Wired preference draws: mean={beta_wire_draws.mean():.3f}, std={beta_wire_draws.std():.3f}")

# %% [markdown]
# ##### ii. Anticipating differentiation under the integral sign, derive the analytical expression for the derivative of the integrand with respect to each $p_{kt}$;

# %%
def compute_individual_utilities(prices, market_t, exog_data, params, beta_sat_draws, beta_wire_draws):
    """Compute individual utilities for all consumers and products in a market."""
    
    n_draws = len(beta_sat_draws)
    J = params.J
    
    # Initialize utility matrix: [n_draws x (J+1)] where last column is outside option
    utilities = np.zeros((n_draws, J + 1))
    
    # Compute utilities for inside goods
    for j in range(J):
        # Base utility (common across consumers)
        base_utility = (params.beta1 * exog_data['x_jt'][market_t, j] + 
                       params.alpha * prices[j] + 
                       exog_data['xi_jt'][market_t, j])
        
        # Add individual-specific preferences
        individual_preferences = (beta_sat_draws * exog_data['satellite_jt'][market_t, j] + 
                                beta_wire_draws * exog_data['wired_jt'][market_t, j])
        
        utilities[:, j] = base_utility + individual_preferences
    
    # Outside option utility (normalized to 0)
    utilities[:, J] = 0
    
    return utilities

def compute_shares_and_derivatives(prices, market_t, exog_data, params, beta_sat_draws, beta_wire_draws):
    """Compute market shares and their derivatives with respect to prices."""
    
    # Get individual utilities
    utilities = compute_individual_utilities(prices, market_t, exog_data, params, beta_sat_draws, beta_wire_draws)
    
    # Compute choice probabilities using multinomial logit
    exp_utilities = np.exp(utilities)
    choice_probs = exp_utilities / exp_utilities.sum(axis=1, keepdims=True)
    
    # Market shares (average over consumers)
    shares = choice_probs[:, :params.J].mean(axis=0)  # Only inside goods
    
    # Compute derivatives
    derivatives = np.zeros((params.J, params.J))
    
    for j in range(params.J):
        for k in range(params.J):
            if j == k:
                # Own-price derivative: ∂s_j/∂p_j = α * E[s_{ij}(1 - s_{ij})]
                deriv_terms = params.alpha * choice_probs[:, j] * (1 - choice_probs[:, j])
            else:
                # Cross-price derivative: ∂s_j/∂p_k = -α * E[s_{ij} * s_{ik}]
                deriv_terms = -params.alpha * choice_probs[:, j] * choice_probs[:, k]
            
            derivatives[j, k] = deriv_terms.mean()
    
    return shares, derivatives

# Test the derivative computation
test_prices = exog_data['mc_jt'][0] * 1.2  # Initial guess for market 0
test_shares, test_derivatives = compute_shares_and_derivatives(
    test_prices, 0, exog_data, params, beta_sat_draws, beta_wire_draws
)

print("Test computation for market 0:")
print(f"Shares: {test_shares}")
print(f"Own-price derivatives (diagonal): {np.diag(test_derivatives)}")
print(f"Sum of shares: {test_shares.sum():.6f}")

# %% [markdown]
# ##### iii. Use the expression you obtained in (2) and simulation draws of the random coefficients to approximate the integral that corresponds to $\partial s_{jt}/\partial p_{kt}$ for each $j$ and $k$ (i.e., replace the integral with the mean over the values at each simulation draw).

# %%
def test_derivative_precision(market_t, exog_data, params, n_draws_list=[100, 500, 1000, 2000]):
    """Test how many simulation draws are needed for precise approximations."""
    
    print("Testing derivative precision with different numbers of draws:")
    print("-" * 60)
    
    # Fixed test prices
    test_prices = exog_data['mc_jt'][market_t] * 1.3
    
    results = []
    
    for n_draws in n_draws_list:
        # Generate draws for this test
        beta_sat = np.random.normal(params.beta2_mean, params.beta_std, n_draws)
        beta_wire = np.random.normal(params.beta3_mean, params.beta_std, n_draws)
        
        # Compute shares and derivatives
        shares, derivatives = compute_shares_and_derivatives(
            test_prices, market_t, exog_data, params, beta_sat, beta_wire
        )
        
        results.append({
            'n_draws': n_draws,
            'shares': shares.copy(),
            'own_derivatives': np.diag(derivatives).copy()
        })
        
        print(f"n_draws = {n_draws:4d}: shares = {shares}, own_derivs = {np.diag(derivatives)}")
    
    # Check convergence
    final_shares = results[-1]['shares']
    final_derivs = results[-1]['own_derivatives']
    
    print(f"\nConvergence check (vs n_draws={n_draws_list[-1]}):")
    for i, result in enumerate(results[:-1]):
        share_diff = np.max(np.abs(result['shares'] - final_shares))
        deriv_diff = np.max(np.abs(result['own_derivatives'] - final_derivs))
        print(f"n_draws = {result['n_draws']:4d}: max_share_diff = {share_diff:.6f}, max_deriv_diff = {deriv_diff:.6f}")
    
    return results

# Test precision with different numbers of draws
precision_results = test_derivative_precision(0, exog_data, params)

# %% [markdown]
# ##### iv. Experiment to see how many simulation draws you need to get precise approximations and check this again at the equilibrium shares and prices you obtain below.

# %%
print("Based on the precision tests, 1000 draws appear sufficient for stable results.")
print(f"Using {params.n_draws} draws for equilibrium computation.")

# %% [markdown]
# Note: you do not want to take new simulation draws of the random coefficients each time you call this procedure. This is because, if you did so, the attempt to solve for equilibrium prices (below) may never converge due to "jittering" across iterations. So take your simulation draws only once, outside the procedure you write here.

# %% [markdown]
# #### (b) The FOC for firm $j$'s profit maximization problem in market $t$ is
# 
# $$(p_{jt} - mc_{jt})\frac{\partial s_{jt}(p_t)}{\partial p_{jt}} + s_{jt} = 0$$
# 
# $$\implies p_{jt} - mc_{jt} = -\left(\frac{\partial s_{jt}(p_t)}{\partial p_{jt}}\right)^{-1} s_{jt} \tag{FOC}$$

# %%
def profit_foc(prices, market_t, exog_data, params, beta_sat_draws, beta_wire_draws):
    """Compute first-order conditions for profit maximization."""
    
    # Get shares and derivatives
    shares, derivatives = compute_shares_and_derivatives(
        prices, market_t, exog_data, params, beta_sat_draws, beta_wire_draws
    )
    
    # FOC residuals: (p_j - mc_j) * (∂s_j/∂p_j) + s_j = 0
    foc_residuals = np.zeros(params.J)
    
    for j in range(params.J):
        markup_term = derivatives[j, j]  # ∂s_j/∂p_j
        
        if abs(markup_term) > 1e-10:  # Avoid division by zero
            foc_residuals[j] = ((prices[j] - exog_data['mc_jt'][market_t, j]) * markup_term + 
                               shares[j])
        else:
            # If derivative is too small, just set price close to marginal cost
            foc_residuals[j] = prices[j] - exog_data['mc_jt'][market_t, j] - 0.1
    
    return foc_residuals

# Test FOC computation
test_foc = profit_foc(test_prices, 0, exog_data, params, beta_sat_draws, beta_wire_draws)
print(f"Test FOC residuals: {test_foc}")

# %% [markdown]
# #### (c) Substituting in your approximation of each $\left(\frac{\partial s_{jt}(p_t)}{\partial p_{jt}}\right)$, solve the system of equations (FOC) ($J$ equations per market) for the equilibrium prices in each market.

# %% [markdown]
# ##### i. To do this you will need to solve a system of $J \times J$ nonlinear equations. Make sure to check the exit flag for each market to make sure you have a solution.

# %%
def solve_equilibrium_fsolve(exog_data, params, beta_sat_draws, beta_wire_draws):
    """Solve for equilibrium prices using scipy's root."""
    
    print("Solving equilibrium prices using scipy.optimize.root...")
    print("-" * 60)
    
    equilibrium_prices = np.zeros((params.T, params.J))
    success_flags = np.zeros(params.T)
    
    for t in range(params.T):
        if t % 100 == 0:
            print(f"Market {t+1}/{params.T}")
        
        # Initial guess: marginal cost plus 20% markup
        initial_prices = exog_data['mc_jt'][t] * 1.2
        
        try:
            # Define FOC function for this market
            def foc_market(prices):
                return profit_foc(prices, t, exog_data, params, beta_sat_draws, beta_wire_draws)
            
            # Solve using scipy.optimize.root with the hybrid method (Powell's method)
            # This method is more robust for non-smooth problems
            result = root(
                fun=foc_market,
                x0=initial_prices,
                method='hybr',  # Modified Powell's hybrid method (same as fsolve but with better options)
                options={
                    'col_deriv': 0,
                    'xtol': 1e-6,  # Tolerance for solution
                    'maxfev': 1000,  # Maximum number of function evaluations
                    'band': None,
                    'eps': None,  # Use default step size for the numerical Jacobian
                    'factor': 100,  # Initial step bound
                    'diag': None
                }
            )
            
            # Check if solution is valid
            if result.success and np.all(result.fun < 1e-4):  # Check if FOC is close to zero
                solution = result.x
                # Additional economic sense checks
                if np.all(solution > 0) and np.all(solution > exog_data['mc_jt'][t]):
                    equilibrium_prices[t] = solution
                    success_flags[t] = 1
                else:
                    # Solution doesn't make economic sense, use fallback
                    equilibrium_prices[t] = exog_data['mc_jt'][t] * 1.3
                    success_flags[t] = 0
            else:
                # Convergence failed, use fallback
                equilibrium_prices[t] = exog_data['mc_jt'][t] * 1.3
                success_flags[t] = 0
                
        except Exception as e:
            # Handle any errors
            equilibrium_prices[t] = exog_data['mc_jt'][t] * 1.3
            success_flags[t] = 0
    
    success_rate = success_flags.mean()
    print(f"\nEquilibrium solved successfully for {success_rate:.1%} of markets")
    
    if success_rate < 0.8:
        print(f"Warning: Low success rate ({success_rate:.1%})")
    
    return equilibrium_prices, success_flags

# Solve equilibrium using fsolve
eq_prices_fsolve, success_fsolve = solve_equilibrium_fsolve(
    exog_data, params, beta_sat_draws, beta_wire_draws
)

print(f"Average equilibrium prices (fsolve): {eq_prices_fsolve.mean(axis=0)}")
markups = (eq_prices_fsolve.mean(axis=0) / exog_data['mc_jt'].mean(axis=0) - 1) * 100
print(f"Average markups: {', '.join(f'{m:.2f}%' for m in markups)}")

# %% [markdown]
# ##### ii. Do this again using the algorithm of Morrow and Skerlos (2011), discussed in section 3.6 of Conlon and Gortmaker (2019) (and in the `pyBLP` "problem simulation tutorial"). Use the numerical integration approach you used in step (a) to approximate the terms defined in equation (25) of Conlon and Gortmaker. If you get different results using this method, resolve this discrepancy either by correcting your code or explaining why your preferred method is the one to be trusted.

# %%
def solve_equilibrium_morrow_skerlos(exog_data, params, beta_sat_draws, beta_wire_draws, max_iter=100, tol=1e-6):
    """Solve equilibrium using Morrow-Skerlos (2011) algorithm."""
    
    print("Solving equilibrium using Morrow-Skerlos algorithm...")
    print("-" * 50)
    
    ms_prices = np.zeros((params.T, params.J))
    
    for t in range(params.T):
        if t % 100 == 0:
            print(f"Market {t+1}/{params.T}")
        
        # Initial prices
        prices = exog_data['mc_jt'][t] * 1.2
        
        for iteration in range(max_iter):
            # Compute shares and derivatives at current prices
            shares, derivatives = compute_shares_and_derivatives(
                prices, t, exog_data, params, beta_sat_draws, beta_wire_draws
            )
            
            # Morrow-Skerlos update: p_j^{new} = mc_j - s_j / (∂s_j/∂p_j)
            new_prices = np.zeros(params.J)
            
            for j in range(params.J):
                if abs(derivatives[j, j]) > 1e-10:
                    markup = -shares[j] / derivatives[j, j]
                    new_prices[j] = exog_data['mc_jt'][t, j] + markup
                else:
                    # If derivative is too small, keep current price
                    new_prices[j] = prices[j]
            
            # Check convergence
            price_change = np.max(np.abs(new_prices - prices))
            if price_change < tol:
                break
            
            # Update with damping for stability
            damping = 0.7
            prices = damping * new_prices + (1 - damping) * prices
        
        ms_prices[t] = prices
    
    # Compare with fsolve results
    max_diff = np.abs(eq_prices_fsolve - ms_prices).max()
    avg_diff = np.abs(eq_prices_fsolve - ms_prices).mean()
    
    print(f"\nComparison with fsolve results:")
    print(f"Maximum price difference: {max_diff:.6f}")
    print(f"Average price difference: {avg_diff:.6f}")
    
    if max_diff > 1e-3:
        print("Warning: Significant difference between methods detected")
    else:
        print("✓ Methods agree within tolerance")
    
    return ms_prices

# Solve using Morrow-Skerlos algorithm
eq_prices_ms = solve_equilibrium_morrow_skerlos(
    exog_data, params, beta_sat_draws, beta_wire_draws
)

print(f"Average equilibrium prices (M-S): {eq_prices_ms.mean(axis=0)}")

# Choose preferred method (fsolve tends to be more robust)
equilibrium_prices = eq_prices_fsolve.copy()
print(f"\nUsing fsolve results as preferred equilibrium prices.")

# %% [markdown]
# ### 3. Calculate "observed" shares for your fake data set using your parameters, your draws of $x$, $w$, $\beta_i$, $\omega$, $\xi$, and your equilibrium prices.

# %%
def compute_market_shares(equilibrium_prices, exog_data, params, beta_sat_draws, beta_wire_draws):
    """Compute market shares at equilibrium prices."""
    
    print("Computing market shares at equilibrium prices...")
    
    market_shares = np.zeros((params.T, params.J))
    
    for t in range(params.T):
        shares, _ = compute_shares_and_derivatives(
            equilibrium_prices[t], t, exog_data, params, beta_sat_draws, beta_wire_draws
        )
        market_shares[t] = shares
    
    return market_shares

# Compute equilibrium shares
market_shares = compute_market_shares(equilibrium_prices, exog_data, params, beta_sat_draws, beta_wire_draws)

print("Market shares computed:")
print(f"Average shares by product: {market_shares.mean(axis=0)}")
print(f"Share ranges: min={market_shares.min(axis=0)}, max={market_shares.max(axis=0)}")
print(f"Average total inside share: {market_shares.sum(axis=1).mean():.4f}")

# Create dataset for estimation
def create_estimation_dataset(equilibrium_prices, market_shares, exog_data, params):
    """Create dataset in long format for estimation."""
    
    data_list = []
    
    for t in range(params.T):
        for j in range(params.J):
            data_list.append({
                'market_ids': t,
                'product_ids': j,
                'firm_ids': j,  # Single-product firms
                'shares': market_shares[t, j],
                'prices': equilibrium_prices[t, j],
                'x': exog_data['x_jt'][t, j],
                'w': exog_data['w_jt'][t, j],
                'satellite': exog_data['satellite_jt'][t, j],
                'wired': exog_data['wired_jt'][t, j],
                'xi': exog_data['xi_jt'][t, j],
                'omega': exog_data['omega_jt'][t, j],
                'marginal_cost': exog_data['mc_jt'][t, j]
            })
    
    dataset = pd.DataFrame(data_list)
    
    # Add outside good shares
    dataset['outside_share'] = 1 - dataset.groupby('market_ids')['shares'].transform('sum')
    dataset['log_share_diff'] = np.log(dataset['shares']) - np.log(dataset['outside_share'])
    
    return dataset

# Create dataset
dataset = create_estimation_dataset(equilibrium_prices, market_shares, exog_data, params)

print(f"\nDataset created:")
print(f"Shape: {dataset.shape}")
print(f"Markets: {dataset['market_ids'].nunique()}")
print(f"Products per market: {dataset.groupby('market_ids').size().iloc[0]}")

# Display summary statistics
print(f"\nSummary Statistics:")
summary_vars = ['shares', 'prices', 'x', 'w', 'marginal_cost']
print(dataset[summary_vars].describe().round(4))

# %% [markdown]
# ## 3. Estimate Some Mis-specified Models

# %% [markdown]
# ### 4. Estimate the plain multinomial logit model of demand by OLS (ignoring the endogeneity of prices).

# %%
import statsmodels.api as sm

def estimate_logit_ols(dataset):
    """Estimate multinomial logit by OLS."""
    
    print("Estimating Multinomial Logit by OLS")
    print("-" * 40)
    
    # Dependent variable: log(s_j/s_0)
    y = dataset['log_share_diff'].values
    
    # Independent variables: x, prices, satellite, wired
    X = dataset[['x', 'prices', 'satellite', 'wired']].values
    X = sm.add_constant(X)  # Add constant term
    
    # OLS estimation
    model = sm.OLS(y, X).fit()
    
    # Extract results
    results = {
        'method': 'Logit OLS',
        'constant': model.params[0],
        'beta_x': model.params[1],
        'alpha_price': model.params[2],
        'beta_satellite': model.params[3],
        'beta_wired': model.params[4],
        'se_constant': model.bse[0],
        'se_x': model.bse[1],
        'se_price': model.bse[2],
        'se_satellite': model.bse[3],
        'se_wired': model.bse[4],
        'r_squared': model.rsquared,
        'n_obs': len(y),
        'model': model
    }
    
    return results

# Estimate OLS logit
ols_results = estimate_logit_ols(dataset)

print("OLS Logit Results:")
print(f"Constant:           {ols_results['constant']:8.4f} (SE: {ols_results['se_constant']:.4f})")
print(f"Quality (x):        {ols_results['beta_x']:8.4f} (SE: {ols_results['se_x']:.4f})")
print(f"Price:              {ols_results['alpha_price']:8.4f} (SE: {ols_results['se_price']:.4f})")
print(f"Satellite:          {ols_results['beta_satellite']:8.4f} (SE: {ols_results['se_satellite']:.4f})")
print(f"Wired:              {ols_results['beta_wired']:8.4f} (SE: {ols_results['se_wired']:.4f})")
print(f"R-squared:          {ols_results['r_squared']:.4f}")
print(f"Observations:       {ols_results['n_obs']}")

print(f"\nComparison with True Parameters:")
print(f"Parameter        True      OLS    ")
print(f"Quality          1.000   {ols_results['beta_x']:7.3f}")
print(f"Price           -2.000   {ols_results['alpha_price']:7.3f}")
print(f"Satellite        4.000   {ols_results['beta_satellite']:7.3f}")
print(f"Wired            4.000   {ols_results['beta_wired']:7.3f}")

# %% [markdown]
# ### 5. Re-estimate the multinomial logit model of demand by two-stage least squares, instrumenting for prices with the exogenous demand shifters $x$ and excluded cost shifters $w$. Discuss how the results differ from those obtained by OLS.

# %%
from statsmodels.sandbox.regression.gmm import IV2SLS

def create_instruments(dataset):
    """Create instruments including BLP-style instruments."""
    
    dataset_with_instruments = dataset.copy()
    
    # Create competitor characteristics
    for market in dataset['market_ids'].unique():
        market_data = dataset[dataset['market_ids'] == market].copy()
        
        for idx, row in market_data.iterrows():
            j = int(row['product_ids'])
            
            # Competing products' quality (sum of other products' x)
            competing_x = market_data[market_data['product_ids'] != j]['x'].sum()
            dataset_with_instruments.loc[
                (dataset_with_instruments['market_ids'] == market) & 
                (dataset_with_instruments['product_ids'] == j), 
                'competing_x'
            ] = competing_x
            
            # Same nest quality (quality of other product in same nest)
            if j < 2:  # Satellite products
                other_satellite = market_data[
                    (market_data['satellite'] == 1) & 
                    (market_data['product_ids'] != j)
                ]['x']
                same_nest_x = other_satellite.iloc[0] if len(other_satellite) > 0 else 0
            else:  # Wired products
                other_wired = market_data[
                    (market_data['wired'] == 1) & 
                    (market_data['product_ids'] != j)
                ]['x']
                same_nest_x = other_wired.iloc[0] if len(other_wired) > 0 else 0
            
            dataset_with_instruments.loc[
                (dataset_with_instruments['market_ids'] == market) & 
                (dataset_with_instruments['product_ids'] == j), 
                'same_nest_x'
            ] = same_nest_x
    
    return dataset_with_instruments

def estimate_logit_2sls(dataset_with_instruments):
    """Estimate multinomial logit by 2SLS."""
    
    print("Estimating Multinomial Logit by 2SLS")
    print("-" * 40)
    
    # Dependent variable
    y = dataset_with_instruments['log_share_diff'].values
    
    # Exogenous variables
    exog = dataset_with_instruments[['x', 'satellite', 'wired']].values
    exog = sm.add_constant(exog)
    
    # Endogenous variable (price)
    endog = dataset_with_instruments[['prices']].values
    
    # Instruments (including exogenous variables)
    instruments = ['x', 'w', 'satellite', 'wired', 'competing_x', 'same_nest_x']
    instr = dataset_with_instruments[instruments].values
    instr = sm.add_constant(instr)
    
    # First stage regression for diagnostics
    first_stage = sm.OLS(endog.flatten(), instr).fit()
    
    # 2SLS estimation
    iv_model = IV2SLS(y, exog, endog, instr).fit()
    
    results = {
        'method': 'Logit 2SLS',
        'constant': iv_model.params[0],
        'beta_x': iv_model.params[1],
        'beta_satellite': iv_model.params[2],
        'beta_wired': iv_model.params[3],
        'alpha_price': iv_model.params[4],  # Price is last (endogenous)
        'se_constant': iv_model.bse[0],
        'se_x': iv_model.bse[1],
        'se_satellite': iv_model.bse[2],
        'se_wired': iv_model.bse[3],
        'se_price': iv_model.bse[4],
        'r_squared': iv_model.rsquared,
        'n_obs': len(y),
        'first_stage_r2': first_stage.rsquared,
        'first_stage_f': first_stage.fvalue,
        'model': iv_model,
        'instruments': instruments
    }
    
    return results

# Create instruments and estimate 2SLS
dataset_with_instruments = create_instruments(dataset)
sls_results = estimate_logit_2sls(dataset_with_instruments)

print("2SLS Logit Results:")
print(f"Constant:           {sls_results['constant']:8.4f} (SE: {sls_results['se_constant']:.4f})")
print(f"Quality (x):        {sls_results['beta_x']:8.4f} (SE: {sls_results['se_x']:.4f})")
print(f"Satellite:          {sls_results['beta_satellite']:8.4f} (SE: {sls_results['se_satellite']:.4f})")
print(f"Wired:              {sls_results['beta_wired']:8.4f} (SE: {sls_results['se_wired']:.4f})")
print(f"Price:              {sls_results['alpha_price']:8.4f} (SE: {sls_results['se_price']:.4f})")
print(f"R-squared:          {sls_results['r_squared']:.4f}")

print(f"\nFirst-stage diagnostics:")
print(f"R-squared:          {sls_results['first_stage_r2']:.4f}")
print(f"F-statistic:        {sls_results['first_stage_f']:.2f}")

print(f"\nComparison of OLS vs 2SLS:")
print(f"Parameter        True      OLS      2SLS")
print(f"Quality          1.000   {ols_results['beta_x']:7.3f}   {sls_results['beta_x']:7.3f}")
print(f"Price           -2.000   {ols_results['alpha_price']:7.3f}   {sls_results['alpha_price']:7.3f}")
print(f"Satellite        4.000   {ols_results['beta_satellite']:7.3f}   {sls_results['beta_satellite']:7.3f}")
print(f"Wired            4.000   {ols_results['beta_wired']:7.3f}   {sls_results['beta_wired']:7.3f}")

print(f"\nDiscussion:")
print("- 2SLS corrects for price endogeneity, bringing price coefficient closer to true value")
print("- OLS price coefficient is upward biased (less negative) due to correlation with unobservables")
print(f"- First-stage F-statistic of {sls_results['first_stage_f']:.1f} indicates instruments have adequate strength")

# %% [markdown]
# ### 6. Now estimate a nested logit model by two-stage least squares, treating "satellite" and "wired" as the two nests for the inside goods. You will probably want to review the discussion of the nested logit in Berry (1994). Note that Berry focuses on the special case in which all the "nesting parameters" are the same; you should allow a different nesting parameter for each nest. In Berry's notation, this means letting the parameter $\sigma$ become $\sigma_{g(j)}$, where $g(j)$ indicates the group (satellite or wired) to which each inside good $j$ belongs. Without reference to the results, explain the way(s) that this model is misspecified. (Hint: students tend to get this question wrong; recall that I suggested you review Berry 94).

# %%
class NestedLogitEstimator:
    """Nested Logit estimation following Berry (1994) with nest-specific parameters."""
    
    def __init__(self, dataset):
        self.dataset = dataset.copy()
        self.prepare_nested_data()
    
    def prepare_nested_data(self):
        """Prepare data structures for nested logit estimation."""
        
        # Create nest identifier
        self.dataset['nest'] = 'wired'  # Default
        self.dataset.loc[self.dataset['satellite'] == 1, 'nest'] = 'satellite'
        
        # Compute nest shares (sum within each nest-market)
        nest_shares = self.dataset.groupby(['market_ids', 'nest'])['shares'].sum().reset_index()
        nest_shares.columns = ['market_ids', 'nest', 'nest_share']
        
        # Merge back to dataset
        self.dataset = self.dataset.merge(nest_shares, on=['market_ids', 'nest'])
        
        # Compute within-nest shares
        self.dataset['within_nest_share'] = self.dataset['shares'] / self.dataset['nest_share']
        
        # Handle numerical issues
        self.dataset['within_nest_share'] = np.clip(self.dataset['within_nest_share'], 1e-12, 1-1e-12)
        self.dataset['nest_share'] = np.clip(self.dataset['nest_share'], 1e-12, 1-1e-12)
        
        print(f"Nested logit data prepared:")
        print(f"Average nest shares - Satellite: {self.dataset[self.dataset['nest']=='satellite']['nest_share'].mean():.4f}")
        print(f"Average nest shares - Wired: {self.dataset[self.dataset['nest']=='wired']['nest_share'].mean():.4f}")
    
    def berry_inversion(self, sigma_satellite, sigma_wired):
        """Berry (1994) inversion with nest-specific nesting parameters."""
        
        delta = np.zeros(len(self.dataset))
        
        for i, (_, row) in enumerate(self.dataset.iterrows()):
            # Choose appropriate sigma parameter
            sigma = sigma_satellite if row['nest'] == 'satellite' else sigma_wired
            
            # Berry inversion: δ_j = log(s_j) - σ_{g(j)} * log(s_{j|g}) - log(s_g)
            delta[i] = (np.log(row['shares']) - 
                       sigma * np.log(row['within_nest_share']) - 
                       np.log(row['nest_share']))
        
        return delta
    
    def gmm_objective(self, sigma_params, X, Z):
        """GMM objective function for nest-specific parameters."""
        
        sigma_satellite, sigma_wired = sigma_params
        
        # Parameter bounds
        if sigma_satellite <= 0 or sigma_satellite >= 1 or sigma_wired <= 0 or sigma_wired >= 1:
            return 1e10
        
        try:
            # Berry inversion
            delta = self.berry_inversion(sigma_satellite, sigma_wired)
            
            # 2SLS regression
            ZZ_inv = np.linalg.inv(Z.T @ Z)
            ZX = Z.T @ X
            Pi_hat = ZZ_inv @ ZX
            X_hat = Z @ Pi_hat
            
            XhXh_inv = np.linalg.inv(X_hat.T @ X_hat)
            beta_hat = XhXh_inv @ X_hat.T @ delta
            
            # Residuals and GMM moments
            residuals = delta - X @ beta_hat
            moments = Z.T @ residuals / len(residuals)
            objective = moments.T @ moments
            
            return objective
        
        except:
            return 1e10
    
    def estimate(self, instruments):
        """Estimate nested logit via 2SLS with nest-specific parameters."""
        
        from scipy.optimize import minimize
        
        print("Estimating Nested Logit with nest-specific parameters...")
        
        # Design matrices
        X = self.dataset[['x', 'prices', 'satellite', 'wired']].values
        X = np.column_stack([np.ones(len(X)), X])  # Add constant
        
        # Instruments
        Z = self.dataset[instruments].values
        Z = np.column_stack([np.ones(len(Z)), Z])
        
        # Initial sigma values
        initial_sigma = [0.5, 0.5]
        bounds = [(0.01, 0.99), (0.01, 0.99)]
        
        # Optimize GMM objective
        result = minimize(
            fun=self.gmm_objective,
            x0=initial_sigma,
            args=(X, Z),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-8}
        )
        
        sigma_satellite, sigma_wired = result.x
        
        # Final parameter estimates
        delta = self.berry_inversion(sigma_satellite, sigma_wired)
        
        ZZ_inv = np.linalg.inv(Z.T @ Z)
        ZX = Z.T @ X
        Pi_hat = ZZ_inv @ ZX
        X_hat = Z @ Pi_hat
        
        XhXh_inv = np.linalg.inv(X_hat.T @ X_hat)
        beta_hat = XhXh_inv @ X_hat.T @ delta
        
        # Standard errors
        residuals = delta - X @ beta_hat
        n, k = len(residuals), len(beta_hat)
        sigma2 = np.sum(residuals**2) / (n - k)
        var_beta = sigma2 * XhXh_inv
        se_beta = np.sqrt(np.diag(var_beta))
        
        results = {
            'method': 'Nested Logit 2SLS',
            'sigma_satellite': sigma_satellite,
            'sigma_wired': sigma_wired,
            'constant': beta_hat[0],
            'beta_x': beta_hat[1],
            'alpha_price': beta_hat[2],
            'beta_satellite': beta_hat[3],
            'beta_wired': beta_hat[4],
            'se_constant': se_beta[0],
            'se_x': se_beta[1],
            'se_price': se_beta[2],
            'se_satellite': se_beta[3],
            'se_wired': se_beta[4],
            'gmm_objective': result.fun,
            'convergence': result.success,
            'n_obs': n
        }
        
        return results

# Estimate nested logit
nested_estimator = NestedLogitEstimator(dataset_with_instruments)
nested_instruments = ['x', 'w', 'satellite', 'wired', 'competing_x', 'same_nest_x']
nested_results = nested_estimator.estimate(nested_instruments)

print("\n" + "="*50)
print("NESTED LOGIT ESTIMATION RESULTS")
print("="*50)

print("Nesting Parameters:")
print(f"σ_satellite = {nested_results['sigma_satellite']:.4f}")
print(f"σ_wired     = {nested_results['sigma_wired']:.4f}")

print(f"\nDemand Parameters:")
print(f"Constant:    {nested_results['constant']:8.4f} (SE: {nested_results['se_constant']:.4f})")
print(f"Quality (x): {nested_results['beta_x']:8.4f} (SE: {nested_results['se_x']:.4f})")
print(f"Price:       {nested_results['alpha_price']:8.4f} (SE: {nested_results['se_price']:.4f})")
print(f"Satellite:   {nested_results['beta_satellite']:8.4f} (SE: {nested_results['se_satellite']:.4f})")
print(f"Wired:       {nested_results['beta_wired']:8.4f} (SE: {nested_results['se_wired']:.4f})")

print(f"\nModel diagnostics:")
print(f"GMM objective: {nested_results['gmm_objective']:.6f}")
print(f"Converged: {nested_results['convergence']}")

print(f"\nModel Misspecification Discussion:")
print("The nested logit model is misspecified relative to our true DGP in the following ways:")
print("1. FUNCTIONAL FORM: The true model has individual-specific random coefficients")
print("   β_i^(2) ~ N(4,1) and β_i^(3) ~ N(4,1), while nested logit assumes a specific")
print("   correlation structure within nests only.")
print("2. SUBSTITUTION PATTERNS: Nested logit restricts substitution to be identical within")
print("   nests and between nests, while the true model allows more flexible substitution")
print("   patterns through the full random coefficients structure.")
print("3. Berry (1994) shows nested logit can be derived as a special case of random")
print("   coefficients logit, but our DGP doesn't satisfy those restrictions.")

# %% [markdown]
# ### 7. Using the nested logit results, provide a table comparing the estimated own-price elasticities to the true own-price elasticities. Provide two additional tables showing the true matrix of diversion ratios and the diversion ratios implied by your estimates.

# %%
def compute_true_elasticities_and_diversion(market_id=0):
    """Compute true own-price elasticities and diversion ratios from the DGP."""
    
    # Get equilibrium prices and shares for the market
    prices = equilibrium_prices[market_id]
    shares, derivatives = compute_shares_and_derivatives(
        prices, market_id, exog_data, params, beta_sat_draws, beta_wire_draws
    )
    
    # Own-price elasticities: η_jj = (∂s_j/∂p_j) * (p_j/s_j)
    true_own_elasticities = np.zeros(params.J)
    true_cross_elasticities = np.zeros((params.J, params.J))
    
    for j in range(params.J):
        for k in range(params.J):
            if shares[j] > 0:
                elasticity = derivatives[j, k] * prices[k] / shares[j]
                if j == k:
                    true_own_elasticities[j] = elasticity
                true_cross_elasticities[j, k] = elasticity
    
    # Diversion ratios: D_jk = -η_jk / η_jj
    true_diversion_ratios = np.zeros((params.J, params.J))
    for j in range(params.J):
        if abs(true_own_elasticities[j]) > 1e-10:
            for k in range(params.J):
                if j != k:
                    true_diversion_ratios[j, k] = -true_cross_elasticities[j, k] / true_own_elasticities[j]
    
    return true_own_elasticities, true_cross_elasticities, true_diversion_ratios, shares

def compute_nested_logit_elasticities(nested_results, market_id=0):
    """Compute elasticities for nested logit model."""
    
    # Get market data
    market_data = nested_estimator.dataset[nested_estimator.dataset['market_ids'] == market_id].copy()
    n_products = len(market_data)
    
    # Parameters
    alpha = nested_results['alpha_price']
    sigma_satellite = nested_results['sigma_satellite']
    sigma_wired = nested_results['sigma_wired']
    
    # Elasticity matrix
    elasticities = np.zeros((n_products, n_products))
    
    for i, (_, prod_i) in enumerate(market_data.iterrows()):
        for j, (_, prod_j) in enumerate(market_data.iterrows()):
            
            price_j = prod_j['prices']
            share_i = prod_i['shares']
            nest_i = prod_i['nest']
            nest_j = prod_j['nest']
            
            if i == j:  # Own-price elasticity
                sigma_g = sigma_satellite if nest_i == 'satellite' else sigma_wired
                nest_share = prod_i['nest_share']
                within_share = prod_i['within_nest_share']
                
                # Own-price elasticity for nested logit
                elasticity = alpha * price_j * (
                    1 - sigma_g * (1 - within_share) - (1 - sigma_g) * nest_share
                )
                
            else:  # Cross-price elasticity
                if nest_i == nest_j:  # Same nest
                    sigma_g = sigma_satellite if nest_i == 'satellite' else sigma_wired
                    within_share_j = prod_j['within_nest_share']
                    nest_share_g = prod_i['nest_share']
                    
                    elasticity = alpha * price_j * (
                        sigma_g * within_share_j + (1 - sigma_g) * nest_share_g
                    )
                    
                else:  # Different nests
                    sigma_g_j = sigma_satellite if nest_j == 'satellite' else sigma_wired
                    nest_share_j = prod_j['nest_share']
                    
                    elasticity = alpha * price_j * (1 - sigma_g_j) * nest_share_j
            
            elasticities[i, j] = elasticity
    
    return elasticities

def compute_diversion_ratios(elasticity_matrix):
    """Compute diversion ratios from elasticity matrix."""
    n = elasticity_matrix.shape[0]
    diversion_ratios = np.zeros((n, n))
    
    for i in range(n):
        own_elasticity = elasticity_matrix[i, i]
        if abs(own_elasticity) > 1e-10:
            for j in range(n):
                if i != j:
                    diversion_ratios[i, j] = -elasticity_matrix[i, j] / own_elasticity
    
    return diversion_ratios

# Compute true elasticities and diversion ratios
true_own_elast, true_cross_elast, true_diversion, true_shares = compute_true_elasticities_and_diversion(0)

# Compute nested logit elasticities
nested_elasticities = compute_nested_logit_elasticities(nested_results, 0)
nested_own_elast = np.diag(nested_elasticities)
nested_diversion = compute_diversion_ratios(nested_elasticities)

# Create comparison tables
print("\n" + "="*60)
print("ELASTICITY AND DIVERSION RATIO ANALYSIS")
print("="*60)

# Own-price elasticities comparison
elasticity_comparison = pd.DataFrame({
    'Product': [f'Product {j+1}' for j in range(4)],
    'Type': ['Satellite', 'Satellite', 'Wired', 'Wired'],
    'True': true_own_elast,
    'Nested_Logit': nested_own_elast
})

print("Own-Price Elasticities Comparison (Market 0):")
print(elasticity_comparison.round(4))

# True diversion ratios
print(f"\nTrue Diversion Ratios (Market 0):")
true_diversion_df = pd.DataFrame(
    true_diversion,
    columns=[f'To_Prod_{j+1}' for j in range(4)],
    index=[f'From_Prod_{i+1}' for i in range(4)]
)
print(true_diversion_df.round(4))

# Nested logit diversion ratios
print(f"\nNested Logit Diversion Ratios (Market 0):")
nested_diversion_df = pd.DataFrame(
    nested_diversion,
    columns=[f'To_Prod_{j+1}' for j in range(4)],
    index=[f'From_Prod_{i+1}' for i in range(4)]
)
print(nested_diversion_df.round(4))

print(f"\nInterpretation:")
print("- Nested logit captures within-nest vs cross-nest substitution patterns")
print("- Higher diversion ratios within nests (satellite-to-satellite, wired-to-wired)")
print("- Lower diversion ratios across nests due to product differentiation")
print("- True model allows more flexible substitution through random coefficients")

# %% [markdown]
# ## 4. Estimate the Correctly Specified Model
# 
# Use the `pyBLP` package to estimate the correctly specified model. Allow `pyBLP` to construct approximations to the optimal instruments, using the exogenous demand shifters and exogenous cost shifters.

# %%
def prepare_pyblp_data(dataset):
    """Prepare data in PyBLP format."""
    
    pyblp_data = dataset.copy()
    
    # Required PyBLP columns
    pyblp_data['market_ids'] = pyblp_data['market_ids']
    pyblp_data['firm_ids'] = pyblp_data['firm_ids'] 
    pyblp_data['product_ids'] = pyblp_data['product_ids']
    
    # Demand instruments
    pyblp_data['demand_instruments0'] = pyblp_data['x']
    pyblp_data['demand_instruments1'] = pyblp_data['w']
    pyblp_data['demand_instruments2'] = pyblp_data['competing_x']
    pyblp_data['demand_instruments3'] = pyblp_data['same_nest_x']
    
    # Supply instruments  
    pyblp_data['supply_instruments0'] = pyblp_data['w']
    pyblp_data['supply_instruments1'] = pyblp_data['x']
    pyblp_data['supply_instruments2'] = pyblp_data['competing_x']
    
    return pyblp_data

def create_agent_data(dataset, n_agents=200):
    """Create agent data for PyBLP integration."""
    
    np.random.seed(42)  # For reproducibility
    
    markets = dataset['market_ids'].unique()
    agent_data = []
    
    for market_id in markets:
        for agent_id in range(n_agents):
            # Draw random coefficients matching our DGP
            beta_sat_draw = np.random.normal(4.0, 1.0)
            beta_wire_draw = np.random.normal(4.0, 1.0)
            
            agent_data.append({
                'market_ids': market_id,
                'weights': 1.0 / n_agents,
                'nodes0': beta_sat_draw,   # For satellite random coefficient
                'nodes1': beta_wire_draw   # For wired random coefficient
            })
    
    return pd.DataFrame(agent_data)

# Prepare PyBLP data
print("Preparing data for PyBLP estimation...")
pyblp_product_data = prepare_pyblp_data(dataset_with_instruments)
pyblp_agent_data = create_agent_data(dataset_with_instruments, n_agents=200)

print(f"Product data shape: {pyblp_product_data.shape}")
print(f"Agent data shape: {pyblp_agent_data.shape}")
print(f"Markets: {pyblp_product_data['market_ids'].nunique()}")
print(f"Agents per market: {pyblp_agent_data.groupby('market_ids').size().iloc[0]}")

# %% [markdown]
# ### 8. Report a table with the estimates of the demand parameters and standard errors. Do this three times: once when you estimate demand alone, then again when you estimate jointly with supply; and again with the "optimal IV".