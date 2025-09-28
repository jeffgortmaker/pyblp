import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class ManualNestedLogit:
    def __init__(self, data, nest_structure):
        """
        Manual implementation of Nested Logit estimation
        
        Parameters:
        data: DataFrame with columns: ['market_id', 'product_id', 'price', 'x1', 'x2', ..., 'share', 'outside_share']
        nest_structure: Dictionary mapping product_id to nest_id
        """
        self.data = data.copy()
        self.nest_structure = nest_structure
        self.data['nest_id'] = self.data['product_id'].map(nest_structure)
        
    def compute_delta(self, params, market_data):
        """
        Compute mean utilities delta = X*beta - alpha*price
        """
        alpha = params[0]  # price coefficient
        beta = params[1:]  # characteristic coefficients
        
        prices = market_data['price'].values
        X = market_data[[col for col in market_data.columns if col.startswith('x')]].values
        
        delta = X.dot(beta) - alpha * prices
        return delta
    
    def compute_nest_shares(self, delta, market_data, rho):
        """
        Compute nest-level shares and within-nest shares
        """
        nest_ids = market_data['nest_id'].values
        unique_nests = np.unique(nest_ids)
        
        # Compute exp(delta/rho) for each product
        exp_delta_rho = np.exp(delta / rho)
        
        # Compute nest denominators D_g = sum(exp(delta/rho)) for each nest
        D_g = {}
        for nest in unique_nests:
            mask = (nest_ids == nest)
            D_g[nest] = np.sum(exp_delta_rho[mask])
        
        # Compute within-nest shares
        within_nest_shares = {}
        for nest in unique_nests:
            mask = (nest_ids == nest)
            within_nest_shares[nest] = exp_delta_rho[mask] / D_g[nest]
        
        return D_g, within_nest_shares
    
    def predicted_shares(self, params, market_data):
        """
        Compute predicted market shares using nested logit formula
        """
        alpha, beta, rho = params[0], params[1:-1], params[-1]
        
        # Add outside good (delta = 0, nest = 0)
        market_data_extended = market_data.copy()
        outside_good = {'product_id': 0, 'price': 0, 'nest_id': 0}
        for col in market_data.columns:
            if col.startswith('x'):
                outside_good[col] = 0
        market_data_extended = pd.concat([pd.DataFrame([outside_good]), market_data_extended], ignore_index=True)
        
        # Compute delta for all products (including outside good)
        prices = market_data_extended['price'].values
        X_cols = [col for col in market_data_extended.columns if col.startswith('x')]
        X = market_data_extended[X_cols].values
        
        delta = X.dot(beta) - alpha * prices
        delta[0] = 0  # Outside good utility
        
        # Compute nest shares
        nest_ids = market_data_extended['nest_id'].values
        unique_nests = np.unique(nest_ids)
        
        exp_delta_rho = np.exp(delta / rho)
        D_g = {}
        for nest in unique_nests:
            mask = (nest_ids == nest)
            D_g[nest] = np.sum(exp_delta_rho[mask])
        
        # Compute D = sum(D_g^rho) over all nests
        D = sum([D_g[nest]**rho for nest in unique_nests])
        
        # Compute predicted shares
        predicted_shares = np.zeros(len(market_data_extended))
        for i, (nest, d_i) in enumerate(zip(nest_ids, exp_delta_rho)):
            predicted_shares[i] = (d_i / D_g[nest]) * (D_g[nest]**rho / D)
        
        return predicted_shares[1:]  # Exclude outside good
    
    def contraction_mapping(self, params, market_data, tol=1e-10, max_iter=1000):
        """
        Berry contraction mapping to solve for delta
        """
        alpha, beta, rho = params[0], params[1:-1], params[-1]
        observed_shares = market_data['share'].values
        
        # Initial delta
        delta_old = np.log(observed_shares) - np.log(market_data['outside_share'].iloc[0])
        
        for iteration in range(max_iter):
            # Add outside good
            delta_full = np.concatenate([[0], delta_old])
            nest_ids_full = np.concatenate([[0], market_data['nest_id'].values])
            
            # Compute predicted shares
            exp_delta_rho = np.exp(delta_full / rho)
            unique_nests = np.unique(nest_ids_full)
            
            D_g = {}
            for nest in unique_nests:
                mask = (nest_ids_full == nest)
                D_g[nest] = np.sum(exp_delta_rho[mask])
            
            D = sum([D_g[nest]**rho for nest in unique_nests])
            
            predicted_shares = np.zeros(len(delta_full))
            for i, (nest, d_i) in enumerate(zip(nest_ids_full, exp_delta_rho)):
                predicted_shares[i] = (d_i / D_g[nest]) * (D_g[nest]**rho / D)
            
            # Update delta
            delta_new = delta_old + np.log(observed_shares) - np.log(predicted_shares[1:])
            
            # Check convergence
            if np.max(np.abs(delta_new - delta_old)) < tol:
                break
            
            delta_old = delta_new
        
        return delta_new
    
    def objective_function(self, params, market_data):
        """
        GMM objective function: minimize difference between predicted and observed shares
        """
        try:
            # Solve for delta using contraction mapping
            delta = self.contraction_mapping(params, market_data)
            
            # Compute instrument matrix (assuming instruments are the characteristics)
            Z = market_data[[col for col in market_data.columns if col.startswith('x')]].values
            prices = market_data['price'].values
            Z = np.column_stack([prices.reshape(-1, 1), Z])  # Include price as instrument
            
            # Compute errors: delta - X*beta + alpha*price
            alpha, beta = params[0], params[1:-1]
            X_cols = [col for col in market_data.columns if col.startswith('x')]
            X = market_data[X_cols].values
            
            errors = delta - (X.dot(beta) - alpha * prices)
            
            # GMM objective: errors' * Z * W * Z' * errors
            W = np.linalg.inv(Z.T @ Z)  # Weighting matrix
            gmm_obj = errors.T @ Z @ W @ Z.T @ errors
            
            return gmm_obj
            
        except:
            return 1e10  # Return large value if optimization fails
    
    def estimate(self, initial_params=None, method='BFGS'):
        """
        Estimate nested logit model parameters
        """
        if initial_params is None:
            # Reasonable starting values: negative price coefficient, small positive rho
            n_features = len([col for col in self.data.columns if col.startswith('x')])
            initial_params = np.concatenate([[-1.0], np.ones(n_features) * 0.1, [0.7]])
        
        results = {}
        
        for market_id in self.data['market_id'].unique():
            market_data = self.data[self.data['market_id'] == market_id].copy()
            
            # Add outside share (complement to 1)
            total_share = market_data['share'].sum()
            market_data['outside_share'] = 1 - total_share
            
            # Estimate parameters for this market
            res = minimize(self.objective_function, initial_params, 
                          args=(market_data,), method=method)
            
            results[market_id] = {
                'params': res.x,
                'success': res.success,
                'message': res.message,
                'fun': res.fun
            }
        
        self.results = results
        return results
    
    def summary(self):
        """Print estimation results summary"""
        if not hasattr(self, 'results'):
            print("No results available. Run estimate() first.")
            return
        
        param_names = ['alpha(price)'] + \
                     [f'beta_{i}' for i in range(len(self.results[list(self.results.keys())[0]]['params']) - 2)] + \
                     ['rho']
        
        print("Nested Logit Estimation Results")
        print("=" * 50)
        
        for market_id, result in self.results.items():
            print(f"\nMarket {market_id}:")
            for name, value in zip(param_names, result['params']):
                print(f"  {name}: {value:.4f}")
            print(f"  Convergence: {result['success']}")

# Example usage and test data generation
def generate_test_data(n_markets=10, n_products=5, n_nests=3):
    """Generate synthetic test data for nested logit estimation"""
    np.random.seed(42)
    
    data = []
    for market in range(n_markets):
        for product in range(n_products):
            nest = product % n_nests  # Simple nesting structure
            
            row = {
                'market_id': market,
                'product_id': product + 1,
                'price': np.random.lognormal(2, 0.5),
                'x1': np.random.normal(0, 1),
                'x2': np.random.normal(0, 1),
                'nest_id': nest
            }
            data.append(row)
    
    df = pd.DataFrame(data)
    
    # Generate realistic shares using nested logit formula
    for market in range(n_markets):
        market_data = df[df['market_id'] == market]
        alpha, beta1, beta2, rho = -1, 0.5, 0.3, 0.7
        
        # Compute utilities
        utilities = beta1 * market_data['x1'] + beta2 * market_data['x2'] - alpha * market_data['price']
        
        # Add outside good
        utilities_full = np.concatenate([[0], utilities.values])
        nest_ids_full = np.concatenate([[0], market_data['nest_id'].values])
        
        # Compute nested logit shares
        exp_u_rho = np.exp(utilities_full / rho)
        unique_nests = np.unique(nest_ids_full)
        
        D_g = {}
        for nest in unique_nests:
            mask = (nest_ids_full == nest)
            D_g[nest] = np.sum(exp_u_rho[mask])
        
        D = sum([D_g[nest]**rho for nest in unique_nests])
        
        shares = np.zeros(len(utilities_full))
        for i, (nest, u_i) in enumerate(zip(nest_ids_full, exp_u_rho)):
            shares[i] = (u_i / D_g[nest]) * (D_g[nest]**rho / D)
        
        # Assign shares to products (excluding outside good)
        df.loc[df['market_id'] == market, 'share'] = shares[1:]
    
    # Create nest structure dictionary
    nest_structure = dict(zip(df['product_id'], df['nest_id']))
    
    return df, nest_structure

# Demonstration
if __name__ == "__main__":
    # Generate test data
    print("Generating test data...")
    test_data, nest_structure = generate_test_data()
    
    # Initialize estimator
    print("Initializing nested logit estimator...")
    estimator = ManualNestedLogit(test_data, nest_structure)
    
    # Estimate parameters
    print("Estimating parameters...")
    results = estimator.estimate()
    
    # Print results
    estimator.summary()
    
    # True parameters for comparison
    true_params = [-1, 0.5, 0.3, 0.7]
    param_names = ['alpha(price)', 'beta_x1', 'beta_x2', 'rho']
    
    print("\nTrue Parameters:")
    for name, value in zip(param_names, true_params):
        print(f"  {name}: {value:.4f}")