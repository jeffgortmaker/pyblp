import numpy as np
import pandas as pd
import pyblp

# Set a seed for reproducibility
pyblp.options.seed = 0

# Section 2: Generate Fake Data using pyblp.Simulation

# Set up parameters
T = 600  # Number of markets
J = 4    # Number of products per market
N = 1000 # Number of simulated consumers per market

# True parameter values
beta1 = 1
alpha = -2
gamma0 = 0.5
gamma1 = 0.25

# Distribution of random coefficients
# Note: pyblp uses Cholesky of covariance matrix for sigma
beta_cov = np.array([[1, 0], [0, 1]])
sigma = np.linalg.cholesky(beta_cov)

# Define pyblp formulations for the model
product_formulations = (
    pyblp.Formulation('0 + prices + quality'),
    pyblp.Formulation('0 + satellite + wired'),
    pyblp.Formulation('1 + cost_shifter')
)

# Create the product data structure required by pyblp
# We will let pyblp simulate the exogenous characteristics for now
id_data = pyblp.build_id_data(T=T, J=J, F=J) # F=J for single-product firms

# Create an integration configuration for consumer tastes
integration = pyblp.Integration('monte_carlo', size=N)

# Initialize the Simulation
simulation = pyblp.Simulation(
    product_formulations=product_formulations,
    product_data=id_data,
    beta=np.array([alpha, beta1]), # beta for prices, quality
    sigma=sigma,
    gamma=np.array([gamma0, gamma1]), # gamma for intercept, cost_shifter
    integration=integration,
    correlation=0.25, # Correlation between xi and omega
    seed=0
)

print("Initialized pyblp.Simulation.")

# --- Solve for equilibrium prices and shares ---
# The replace_endogenous method solves the model using the Morrow-Skerlos algorithm
results = simulation.replace_endogenous(constant_costs=False)

print("Finished solving for equilibrium prices and shares.")

# --- 3. Assemble and save the dataset ---

# Extract the simulated data from the results object
data = results.product_data

# Convert to a pandas DataFrame for easier inspection and saving
data_df = pd.DataFrame({
    'market_ids': data.market_ids.flatten(),
    'firm_ids': data.firm_ids.flatten(),
    'product_ids': np.arange(T * J),
    'shares': data.shares.flatten(),
    'prices': data.prices.flatten(),
    'quality': data.quality.flatten(),
    'cost_shifter': data.cost_shifter.flatten(),
    'demand_shock': simulation.xi.flatten(),
    'cost_shock': simulation.omega.flatten(),
    'satellite': data.satellite.flatten(),
    'wired': data.wired.flatten()
})

# Save the dataset
output_path = '/Users/marek/Library/CloudStorage/Dropbox/github/pyblp/pyblp_homework/data/blp_data.csv'
data_df.to_csv(output_path, index=False)

print(f"\nDataset successfully generated and saved to {output_path}")
print(f"Number of observations: {len(data_df)}")
print("\nFirst 5 rows of the dataset:")
print(data_df.head())
