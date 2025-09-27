import pyblp
import pandas as pd
import numpy as np

# --- Load Data ---
data_path = '/Users/marek/Library/CloudStorage/Dropbox/github/pyblp/pyblp_homework/data/blp_data.csv'
product_data = pd.read_csv(data_path)

# --- Set up for pyBLP ---

# Add agent data (for simulation)
agent_data = pyblp.build_integration(pyblp.Integration('monte_carlo', 2), 1000)

# Define instruments
# Use exogenous demand shifters (own and rival quality) and cost shifters
x_vars = ['quality']
instrument_config = pyblp.build_blp_instruments(pyblp.Formulation('1 + quality'), x_vars)

# Define model formulation
# Demand: quality + prices ~ satellite + wired
# Supply: ~ 1 + cost_shifter (w)
demand_formulation = pyblp.Formulation('quality + prices')
random_coeffs_formulation = pyblp.Formulation('satellite + wired')
supply_formulation = pyblp.Formulation('~ 1 + cost_shifter')
product_formulations = (demand_formulation, random_coeffs_formulation, supply_formulation)

# --- Question 8: Estimate the Correctly Specified Model ---
print("--- Question 8: Estimating Correctly Specified Models with pyBLP ---")

# --- 1. Demand-side only estimation ---
problem_demand = pyblp.Problem(product_formulations[:2], product_data, agent_data=agent_data, integration=pyblp.Integration('monte_carlo', 2))
results_demand = problem_demand.solve(sigma=np.eye(2), instrument_config=instrument_config)

# --- 2. Joint estimation (Demand + Supply) ---
problem_joint = pyblp.Problem(product_formulations, product_data, agent_data=agent_data, integration=pyblp.Integration('monte_carlo', 2))
results_joint = problem_joint.solve(sigma=np.eye(2), instrument_config=instrument_config)

# --- 3. Joint estimation with Optimal Instruments ---
optimal_instrument_config = pyblp.build_optimal_instruments(demand_formulation, random_coeffs_formulation, results_joint.product_data)
results_optimal_iv = problem_joint.solve(sigma=np.eye(2), instrument_config=optimal_instrument_config)

# --- Display Results in a Table ---

true_params = {
    'quality': 1,
    'prices': -2,
    'Mean(satellite)': 4,
    'Mean(wired)': 4,
    'Sigma(satellite, satellite)': 1,
    'Sigma(wired, wired)': 1,
    'Intercept (gamma0)': 0.5,
    'cost_shifter (gamma1)': 0.25
}

# Create series for each estimation result
demand_series = pd.Series(np.concatenate([results_demand.beta, results_demand.pi.flatten(), np.diag(results_demand.sigma)]),
                          index=['quality', 'prices', 'Mean(satellite)', 'Mean(wired)', 'Sigma(satellite, satellite)', 'Sigma(wired, wired)'])

joint_series = pd.Series(np.concatenate([results_joint.beta, results_joint.pi.flatten(), np.diag(results_joint.sigma), results_joint.gamma]),
                         index=['quality', 'prices', 'Mean(satellite)', 'Mean(wired)', 'Sigma(satellite, satellite)', 'Sigma(wired, wired)', 'Intercept (gamma0)', 'cost_shifter (gamma1)'])

optimal_iv_series = pd.Series(np.concatenate([results_optimal_iv.beta, results_optimal_iv.pi.flatten(), np.diag(results_optimal_iv.sigma), results_optimal_iv.gamma]),
                                index=['quality', 'prices', 'Mean(satellite)', 'Mean(wired)', 'Sigma(satellite, satellite)', 'Sigma(wired, wired)', 'Intercept (gamma0)', 'cost_shifter (gamma1)'])


results_df = pd.DataFrame({
    'True Value': pd.Series(true_params),
    'Demand Only': demand_series,
    'Joint Estimation': joint_series,
    'Optimal IV': optimal_iv_series
})


print("\nParameter Estimation Results:")
print(results_df.round(3))
