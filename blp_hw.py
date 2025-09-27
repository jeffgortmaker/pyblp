# Economics 600a - BLP Homework Solution Code
# Prof. P. Haile, Fall 2025

# ==============================================================================
# SECTION 0: IMPORTS AND SETUP
# ==============================================================================
import numpy as np
import pandas as pd
import pyblp
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.optimize import root

# Set random seed for reproducibility
np.random.seed(0)
print(f"pyBLP Version: {pyblp.__version__}")


# ==============================================================================
# SECTION 3: GENERATE FAKE DATA
# ==============================================================================
print("\n--- Section 3: Generating Fake Data ---")

# --- 3.1: Model Parameters and Data Generation ---
T = 600  # Number of markets
J = 4    # Number of products per market
N = T * J # Total observations

# True Demand Parameters
BETA_1 = 1
ALPHA = -2
BETA_MEAN_2_3 = 4
BETA_STD_2_3 = 1

# True Supply Parameters
GAMMA_0 = 0.5
GAMMA_1 = 0.25

# Generate Exogenous Data
market_ids = np.repeat(np.arange(T), J)
firm_ids = np.tile(np.arange(J), T)
x = np.abs(np.random.randn(N))
satellite = np.tile([1, 1, 0, 0], T)
wired = np.tile([0, 0, 1, 1], T)
w = np.abs(np.random.randn(N))

# Demand and cost unobservables
cov_matrix = np.array([[1, 0.25], [0.25, 1]])
shocks = np.random.multivariate_normal([0, 0], cov_matrix, N)
xi = shocks[:, 0]
omega = shocks[:, 1]

# Generate Consumer-level random coefficients
NUM_CONSUMERS = 1000
beta_i_2 = np.random.normal(BETA_MEAN_2_3, BETA_STD_2_3, (T, NUM_CONSUMERS))
beta_i_3 = np.random.normal(BETA_MEAN_2_3, BETA_STD_2_3, (T, NUM_CONSUMERS))

# Calculate True Marginal Costs
mc = np.exp(GAMMA_0 + w * GAMMA_1 + omega / 8)

# Initial DataFrame
df = pd.DataFrame({
    'market_ids': market_ids,
    'firm_ids': firm_ids,
    'x': x,
    'satellite': satellite,
    'wired': wired,
    'w': w,
    'xi': xi,
    'mc': mc
})

# --- 3.2: Solve for Equilibrium Prices ---

# Function to compute shares and their derivatives for a single market
def compute_shares_and_derivs(prices, market_id_val):
    market_df = df[df['market_ids'] == market_id_val]

    # Consumer-specific utility for this market
    delta = BETA_1 * market_df['x'].values[:, np.newaxis] + \
            ALPHA * prices[:, np.newaxis] + \
            market_df['xi'].values[:, np.newaxis]

    mu = market_df['satellite'].values[:, np.newaxis] * beta_i_2[market_id_val, :] + \
         market_df['wired'].values[:, np.newaxis] * beta_i_3[market_id_val, :]

    utility = delta + mu

    # Compute choice probabilities (logit formula)
    exp_utility = np.exp(utility)
    denom = 1 + np.sum(exp_utility, axis=0)
    s_ij = exp_utility / denom

    # Compute market shares by averaging over consumers
    shares = np.mean(s_ij, axis=1)

    # Compute derivatives d(s_j)/d(p_k)
    derivs = np.zeros((J, J))
    for j in range(J):
        for k in range(J):
            if j == k:
                integrand = ALPHA * s_ij[j, :] * (1 - s_ij[j, :])
            else:
                integrand = -ALPHA * s_ij[j, :] * s_ij[k, :]
            derivs[j, k] = np.mean(integrand)

    return shares, derivs

# Morrow and Skerlos (2011) fixed-point algorithm to solve for prices
def solve_prices_for_market(market_id_val):
    market_df = df[df['market_ids'] == market_id_val]
    market_mc = market_df['mc'].values
    prices = market_mc + 1 # Initial price guess

    for _ in range(100):
        shares, derivs = compute_shares_and_derivs(prices, market_id_val)
        
        try:
            inv_derivs = np.linalg.inv(derivs)
        except np.linalg.LinAlgError:
            print(f"Warning: Singular derivative matrix for market {market_id_val}. Using pseudo-inverse.")
            inv_derivs = np.linalg.pinv(derivs)

        new_prices = market_mc - np.diag(inv_derivs) * shares

        if np.max(np.abs(new_prices - prices)) < 1e-8:
            return new_prices
        prices = new_prices

    print(f"Warning: Price solver did not converge for market {market_id_val}")
    return prices

# Solve for prices in all markets
equilibrium_prices = np.zeros(N)
for t in range(T):
    market_mask = (df['market_ids'] == t)
    equilibrium_prices[market_mask] = solve_prices_for_market(t)
df['prices'] = equilibrium_prices
print("Finished solving for equilibrium prices.")

# --- 3.3: Calculate "Observed" Market Shares ---
equilibrium_shares = np.zeros(N)
for t in range(T):
    market_mask = (df['market_ids'] == t)
    prices_t = df.loc[market_mask, 'prices'].values
    shares_t, _ = compute_shares_and_derivs(prices_t, t)
    equilibrium_shares[market_mask] = shares_t
df['shares'] = equilibrium_shares

market_shares_sum = df.groupby('market_ids')['shares'].sum()
df = df.merge(market_shares_sum.rename('inside_share_sum'), on='market_ids')
df['outside_shares'] = 1 - df['inside_share_sum']
print("Calculated final 'observed' market shares.")

# --- 3.4: Instrument Check ---
df['rival_x_sum'] = df.groupby('market_ids')['x'].transform(lambda x: x.sum() - x)
df['rival_w_sum'] = df.groupby('market_ids')['w'].transform(lambda w: w.sum() - w)

formula = 'prices ~ x + w + satellite + wired + rival_x_sum + rival_w_sum'
first_stage = smf.ols(formula, data=df).fit()

print("\n--- Instrument Relevance Check (First Stage) ---")
print(f"F-statistic: {first_stage.fvalue:.2f}")
print(f"R-squared: {first_stage.rsquared:.3f}")


# ==============================================================================
# SECTION 4: ESTIMATE MIS-SPECIFIED MODELS
# ==============================================================================
print("\n--- Section 4: Estimating Mis-specified Models ---")
df['log_share_diff'] = np.log(df['shares']) - np.log(df['outside_shares'])

# --- 4.5: Plain Multinomial Logit (OLS) ---
ols_model = smf.ols('log_share_diff ~ x + satellite + wired + prices', data=df).fit()
print("\n--- 5. OLS Results (Biased) ---")
print(ols_model.summary().tables[1])

# --- 4.6: Multinomial Logit (2SLS / IV) ---
iv_model = sm.IV2SLS.from_formula(
    'log_share_diff ~ 1 + x + satellite + wired + [prices ~ w + rival_x_sum + rival_w_sum]',
    data=df
).fit()
print("\n--- 6. 2SLS (IV) Results ---")
print(iv_model.summary().tables[1])

# --- 4.7: Nested Logit (IV) ---
df['group_share'] = df.groupby(['market_ids', 'satellite'])['shares'].transform('sum')
df['within_group_share'] = df['shares'] / df['group_share']
df['log_within_share'] = np.log(df['within_group_share'])
df['is_satellite'] = (df['satellite'] == 1).astype(int)
df['is_wired'] = (df['wired'] == 1).astype(int)
df['nesting_satellite'] = df['is_satellite'] * df['log_within_share']
df['nesting_wired'] = df['is_wired'] * df['log_within_share']

nested_iv_model = sm.IV2SLS.from_formula(
    'log_share_diff ~ 1 + x + [prices ~ w + rival_w_sum] + [nesting_satellite + nesting_wired ~ rival_x_sum]',
    data=df
).fit()

sigma_satellite = 1 - nested_iv_model.params['nesting_satellite']
sigma_wired = 1 - nested_iv_model.params['nesting_wired']
print("\n--- 7. Nested Logit (IV) Results ---")
print(nested_iv_model.summary().tables[1])
print(f"\nEstimated Nesting Parameter (Satellite): {sigma_satellite:.4f}")
print(f"Estimated Nesting Parameter (Wired): {sigma_wired:.4f}")


# ==============================================================================
# SECTION 5: ESTIMATE THE CORRECTLY SPECIFIED MODEL
# ==============================================================================
print("\n--- Section 5: Estimating Correctly Specified Model with pyBLP ---")

# --- 5.9: pyBLP Estimation ---
product_data = df.copy()
product_data['product_ids'] = df['firm_ids']
product_data['demand_ids'] = product_data['product_ids']

X1_formula = 'prices + x'
X2_formula = 'satellite + wired'
instrument_config = pyblp.Instrument('x + w', 'sum')

# Problem for Demand Estimation
problem = pyblp.Problem(
    product_formulations=(
        pyblp.Formulation(X1_formula),
        pyblp.Formulation(X2_formula)
    ),
    product_data=product_data,
    integration=pyblp.Integration('monte_carlo', size=NUM_CONSUMERS, seed=0)
)
problem.add_optimal_instruments(instrument_config)

# Estimate Demand Side Only
gmm_options = {'method': '1s'}
demand_only_results = problem.solve(sigma=np.eye(2), optim_options=gmm_options)
print("\n--- 9. pyBLP Demand-Only Estimation Results ---")
print(demand_only_results)

# Problem for Joint Estimation
supply_problem = pyblp.Problem(
    product_formulations=(
        pyblp.Formulation(X1_formula),
        pyblp.Formulation(X2_formula),
        pyblp.Formulation('1 + w') # Supply side
    ),
    product_data=product_data,
    integration=pyblp.Integration('monte_carlo', size=NUM_CONSUMERS, seed=0)
)
supply_problem.add_optimal_instruments(instrument_config)

# Estimate Jointly
joint_results = supply_problem.solve(sigma=np.eye(2), gamma=np.zeros(2), optim_options=gmm_options)
print("\n--- 9. pyBLP Joint Estimation Results ---")
print(joint_results)


# --- 5.10: Elasticities and Diversion Ratios ---
# (Comparison tables are best built from the printed results above)
# This code just shows how to calculate them
true_elasticities = problem.compute_elasticities(prices=df['prices'].values)
est_elasticities = joint_results.compute_elasticities()

avg_true_elasticity = np.mean(np.diag(true_elasticities.reshape(T, J, J)), axis=0)
avg_est_elasticity = np.mean(np.diag(est_elasticities.reshape(T, J, J)), axis=0)
print("\n--- 10. Average Own-Price Elasticities (True vs. Estimated) ---")
print(f"True: {avg_true_elasticity}")
print(f"Estimated: {avg_est_elasticity}")


# ==============================================================================
# SECTION 6: MERGER SIMULATION
# ==============================================================================
print("\n--- Section 6: Merger Simulation ---")

# --- 6.12 & 6.13: Mergers of (1,2) and (1,3) ---
merger1_ids = [0, 1]
merger2_ids = [0, 2]
merger1_prices = joint_results.compute_merger_prices(merger1_ids)
merger2_prices = joint_results.compute_merger_prices(merger2_ids)

base_prices_avg = df.groupby('firm_ids')['prices'].mean().values
merger1_prices_avg = merger1_prices.reshape(T, J).mean(axis=0)
merger2_prices_avg = merger2_prices.reshape(T, J).mean(axis=0)

price_change_df = pd.DataFrame({
    'Product': ['1 (Sat)', '2 (Sat)', '3 (Wired)', '4 (Wired)'],
    'Merger 1&2 Change (%)': (merger1_prices_avg / base_prices_avg - 1) * 100,
    'Merger 1&3 Change (%)': (merger2_prices_avg / base_prices_avg - 1) * 100
})
print("\n--- 12 & 13. Predicted Merger-Induced Price Changes (%) ---")
print(price_change_df.to_string(index=False))


# --- 6.15: Merger of (1,2) with Cost Savings ---
initial_mc = joint_results.compute_costs()
merger_mc = initial_mc.copy()
merger1_mask = np.isin(product_data['firm_ids'], merger1_ids)
merger_mc[merger1_mask] *= 0.85 # 15% reduction

merger1_eff_prices = joint_results.compute_merger_prices(merger1_ids, costs=merger_mc)
merger1_eff_prices_avg = merger1_eff_prices.reshape(T, J).mean(axis=0)

# Calculate welfare changes (assuming M_t = 1000)
M = 1000
base_welfare = joint_results.compute_consumer_surpluses() * M
merger1_eff_welfare = joint_results.compute_consumer_surpluses(prices=merger1_eff_prices) * M
total_welfare_change = joint_results.compute_total_welfare_change(merger1_ids, costs=merger_mc) * M

eff_price_change_df = pd.DataFrame({
    'Product': ['1 (Sat)', '2 (Sat)', '3 (Wired)', '4 (Wired)'],
    'No Efficiencies Change (%)': price_change_df['Merger 1&2 Change (%)'],
    '15% Cost Savings Change (%)': (merger1_eff_prices_avg / base_prices_avg - 1) * 100
})

print("\n--- 15. Merger Price Changes with 15% Cost Savings ---")
print(eff_price_change_df.to_string(index=False))
print(f"\nPredicted Change in Consumer Welfare: ${np.sum(merger1_eff_welfare - base_welfare):,.2f}")
print(f"Predicted Change in Total Welfare:    ${np.sum(total_welfare_change):,.2f}")