import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
import warnings

class NestedLogit:
    """
    Manual implementation of Nested Logit model for discrete choice estimation.
    
    This implementation follows the standard nested logit specification where:
    - Products are grouped into nests
    - Within-nest correlation is captured by nesting parameter rho
    - Choice probabilities have two-stage structure
    """
    
    def __init__(self, data, nest_id_col='nest_id', market_id_col='market_id', 
                 share_col='shares', price_col='prices'):
        """
        Initialize the nested logit model.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Product-level data with shares, characteristics, and nest identifiers
        nest_id_col : str
            Column name for nest identifiers
        market_id_col : str  
            Column name for market identifiers
        share_col : str
            Column name for market shares
        price_col : str
            Column name for prices
        """
        self.data = data.copy()
        self.nest_id_col = nest_id_col
        self.market_id_col = market_id_col
        self.share_col = share_col
        self.price_col = price_col
        
        # Create market-product structure
        self.markets = sorted(data[market_id_col].unique())
        self.nests = sorted(data[nest_id_col].unique())
        self.n_markets = len(self.markets)
        self.n_nests = len(self.nests)
        
        # Pre-compute useful quantities
        self._setup_data_structures()
        
    def _setup_data_structures(self):
        """Pre-compute data structures for efficient estimation."""
        
        # Create nest-level shares
        self.nest_shares = self.data.groupby([self.market_id_col, self.nest_id_col])[self.share_col].sum().reset_index()
        self.nest_shares.columns = [self.market_id_col, self.nest_id_col, 'nest_share']
        
        # Merge nest shares back to product data
        self.data = self.data.merge(self.nest_shares, on=[self.market_id_col, self.nest_id_col])
        
        # Compute within-nest shares
        self.data['within_nest_share'] = self.data[self.share_col] / self.data['nest_share']
        
        # Handle numerical issues
        self.data['within_nest_share'] = np.clip(self.data['within_nest_share'], 1e-12, 1-1e-12)
        self.data['nest_share'] = np.clip(self.data['nest_share'], 1e-12, 1-1e-12)
        
        # Log transforms for convenience
        self.data['log_within_share'] = np.log(self.data['within_nest_share'])
        self.data['log_nest_share'] = np.log(self.data['nest_share'])
        
    def compute_inclusive_values(self, X, beta, rho):
        """
        Compute inclusive values (log-sum within each nest).
        
        Parameters:
        -----------
        X : np.array
            Design matrix of product characteristics
        beta : np.array
            Linear parameters
        rho : float
            Nesting parameter (0 < rho < 1)
            
        Returns:
        --------
        np.array : Inclusive values for each nest in each market
        """
        # Compute deterministic utilities
        utilities = X @ beta
        
        # Add utilities to data temporarily
        data_temp = self.data.copy()
        data_temp['utility'] = utilities
        
        # Compute inclusive values by nest and market
        def compute_iv(group):
            return logsumexp(group['utility'] / rho) * rho
            
        inclusive_values = data_temp.groupby([self.market_id_col, self.nest_id_col]).apply(compute_iv)
        
        return inclusive_values.values
    
    def predict_shares(self, X, beta, rho, lambda_param=None):
        """
        Predict market shares given parameters.
        
        Parameters:
        -----------
        X : np.array
            Design matrix
        beta : np.array
            Linear parameters  
        rho : float
            Nesting parameter
        lambda_param : float, optional
            Scale parameter for nest choice (default: 1)
            
        Returns:
        --------
        tuple : (product_shares, nest_shares)
        """
        if lambda_param is None:
            lambda_param = 1.0
            
        # Compute utilities
        utilities = X @ beta
        data_temp = self.data.copy()
        data_temp['utility'] = utilities
        
        # Compute inclusive values
        def compute_iv(group):
            return logsumexp(group['utility'] / rho) * rho
            
        iv_by_nest = data_temp.groupby([self.market_id_col, self.nest_id_col]).apply(compute_iv)
        
        # Merge inclusive values back
        iv_df = iv_by_nest.reset_index()
        iv_df.columns = [self.market_id_col, self.nest_id_col, 'inclusive_value']
        data_temp = data_temp.merge(iv_df, on=[self.market_id_col, self.nest_id_col])
        
        # Compute nest choice probabilities
        nest_choice_probs = []
        for market in self.markets:
            market_data = data_temp[data_temp[self.market_id_col] == market]
            nest_ivs = market_data.groupby(self.nest_id_col)['inclusive_value'].first().values
            
            # Add outside option (normalized to 0)
            all_ivs = np.concatenate([[0], nest_ivs / lambda_param])
            nest_probs = np.exp(all_ivs) / np.sum(np.exp(all_ivs))
            
            # Skip outside option probability
            nest_choice_probs.extend(nest_probs[1:])
        
        # Compute within-nest choice probabilities
        data_temp['exp_util_rho'] = np.exp(data_temp['utility'] / rho)
        nest_sums = data_temp.groupby([self.market_id_col, self.nest_id_col])['exp_util_rho'].sum()
        
        data_temp = data_temp.merge(nest_sums.rename('nest_sum'), on=[self.market_id_col, self.nest_id_col])
        data_temp['within_nest_prob'] = data_temp['exp_util_rho'] / data_temp['nest_sum']
        
        # Final product probabilities
        nest_prob_dict = {}
        idx = 0
        for market in self.markets:
            for nest in self.nests:
                nest_prob_dict[(market, nest)] = nest_choice_probs[idx]
                idx += 1
        
        data_temp['nest_choice_prob'] = data_temp.apply(
            lambda row: nest_prob_dict.get((row[self.market_id_col], row[self.nest_id_col]), 0), 
            axis=1
        )
        
        predicted_shares = data_temp['within_nest_prob'] * data_temp['nest_choice_prob']
        
        return predicted_shares, data_temp['nest_choice_prob'].values
    
    def log_likelihood(self, params, X, instruments=None):
        """
        Compute log-likelihood for nested logit model.
        
        Parameters:
        -----------
        params : np.array
            Parameters [beta, rho] where beta are linear coefficients
        X : np.array
            Design matrix
        instruments : np.array, optional
            Instrumental variables matrix
            
        Returns:
        --------
        float : Negative log-likelihood
        """
        beta = params[:-1]
        rho = params[-1]
        
        # Constrain rho to (0, 1)
        if rho <= 0 or rho >= 1:
            return 1e10
            
        try:
            # Compute utilities
            utilities = X @ beta
            data_temp = self.data.copy()
            data_temp['utility'] = utilities
            
            # Compute log-likelihood contributions
            ll_contributions = []
            
            for market in self.markets:
                market_data = data_temp[data_temp[self.market_id_col] == market]
                
                # Compute inclusive values for this market
                nest_ivs = {}
                for nest in self.nests:
                    nest_products = market_data[market_data[self.nest_id_col] == nest]
                    if len(nest_products) > 0:
                        nest_ivs[nest] = logsumexp(nest_products['utility'].values / rho) * rho
                
                # Nest choice denominator (including outside option)
                all_ivs = [0] + list(nest_ivs.values())  # 0 for outside option
                log_nest_denom = logsumexp(all_ivs)
                
                # Product contributions
                for _, product in market_data.iterrows():
                    nest = product[self.nest_id_col]
                    share = product[self.share_col]
                    
                    if share > 0:
                        # Within-nest probability
                        nest_products = market_data[market_data[self.nest_id_col] == nest]
                        log_nest_sum = logsumexp(nest_products['utility'].values / rho)
                        log_within_prob = product['utility'] / rho - log_nest_sum
                        
                        # Nest choice probability  
                        log_nest_prob = nest_ivs[nest] - log_nest_denom
                        
                        # Total log probability
                        log_prob = log_within_prob + log_nest_prob
                        ll_contributions.append(share * log_prob)
            
            return -np.sum(ll_contributions)
            
        except:
            return 1e10
    
    def estimate_mle(self, formula, instruments=None, initial_params=None, 
                     method='L-BFGS-B', options=None):
        """
        Estimate nested logit via maximum likelihood.
        
        Parameters:
        -----------
        formula : str
            Patsy-style formula for covariates (e.g., 'price + horsepower + constant')
        instruments : list, optional
            List of instrumental variable column names
        initial_params : np.array, optional
            Initial parameter values
        method : str
            Optimization method
        options : dict
            Optimizer options
            
        Returns:
        --------
        dict : Estimation results
        """
        # Parse formula and create design matrix
        X = self._create_design_matrix(formula)
        n_params = X.shape[1]
        
        # Initial parameters
        if initial_params is None:
            initial_params = np.concatenate([np.zeros(n_params), [0.5]])  # rho = 0.5
        
        # Parameter bounds (rho must be in (0, 1))
        bounds = [(None, None)] * n_params + [(0.01, 0.99)]
        
        # Optimization options
        if options is None:
            options = {'maxiter': 1000, 'ftol': 1e-8}
        
        # Optimize
        result = minimize(
            fun=self.log_likelihood,
            x0=initial_params,
            args=(X, instruments),
            method=method,
            bounds=bounds,
            options=options
        )
        
        # Compute standard errors (basic implementation)
        hessian = self._compute_hessian(result.x, X)
        try:
            se = np.sqrt(np.diag(np.linalg.inv(hessian)))
        except:
            se = np.full(len(result.x), np.nan)
            warnings.warn("Could not compute standard errors")
        
        # Package results
        beta_est = result.x[:-1]
        rho_est = result.x[-1]
        
        results = {
            'beta': beta_est,
            'rho': rho_est,
            'se_beta': se[:-1],
            'se_rho': se[-1],
            'log_likelihood': -result.fun,
            'convergence': result.success,
            'n_iter': result.nit,
            'params_all': result.x,
            'se_all': se,
            'X': X,
            'formula': formula
        }
        
        return results
    
    def estimate_2sls(self, formula, instruments, rho_initial=0.5):
        """
        Two-stage least squares estimation using Berry (1994) inversion.
        
        Parameters:
        -----------
        formula : str
            Formula for product characteristics
        instruments : list
            Instrumental variable column names
        rho_initial : float
            Initial guess for rho parameter
            
        Returns:
        --------
        dict : 2SLS estimation results
        """
        # Create design matrices
        X = self._create_design_matrix(formula)
        Z = self._create_design_matrix(' + '.join(instruments))
        
        def objective_2sls(rho):
            # Berry inversion for given rho
            delta = self._berry_inversion(rho)
            
            # 2SLS regression of delta on X using instruments Z
            # First stage: X = Z * gamma + u
            gamma = np.linalg.solve(Z.T @ Z, Z.T @ X)
            X_hat = Z @ gamma
            
            # Second stage: delta = X_hat * beta + e  
            beta = np.linalg.solve(X_hat.T @ X_hat, X_hat.T @ delta)
            residuals = delta - X @ beta
            
            # GMM objective (with identity weighting matrix)
            moments = Z.T @ residuals
            objective = moments.T @ moments / len(residuals)
            
            return objective
        
        # Optimize over rho
        rho_result = minimize(objective_2sls, x0=[rho_initial], 
                             bounds=[(0.01, 0.99)], method='L-BFGS-B')
        
        rho_est = rho_result.x[0]
        
        # Final parameter estimates at optimal rho
        delta = self._berry_inversion(rho_est)
        Z = self._create_design_matrix(' + '.join(instruments))
        X = self._create_design_matrix(formula)
        
        # 2SLS estimation
        gamma = np.linalg.solve(Z.T @ Z, Z.T @ X)
        X_hat = Z @ gamma
        beta_est = np.linalg.solve(X_hat.T @ X_hat, X_hat.T @ delta)
        
        # Standard errors (simplified)
        residuals = delta - X @ beta_est
        sigma2 = np.sum(residuals**2) / (len(residuals) - len(beta_est))
        var_beta = sigma2 * np.linalg.inv(X_hat.T @ X_hat)
        se_beta = np.sqrt(np.diag(var_beta))
        
        return {
            'beta': beta_est,
            'rho': rho_est,
            'se_beta': se_beta,
            'se_rho': np.nan,  # Would need bootstrap or other method
            'delta': delta,
            'objective': rho_result.fun,
            'convergence': rho_result.success
        }
    
    def _berry_inversion(self, rho):
        """Berry (1994) inversion to recover mean utilities."""
        delta = np.zeros(len(self.data))
        
        for i, (_, product) in enumerate(self.data.iterrows()):
            # Observed shares
            s_j = product[self.share_col]
            s_nest = product['nest_share'] 
            s_within = product['within_nest_share']
            
            # Berry inversion formula for nested logit
            # delta_j = log(s_j) - rho * log(s_j/s_nest) - log(s_nest)
            if s_j > 0 and s_within > 0 and s_nest > 0:
                delta[i] = np.log(s_j) - rho * np.log(s_within)
            else:
                delta[i] = -10  # Handle zero shares
                
        return delta
    
    def _create_design_matrix(self, formula):
        """Create design matrix from formula."""
        # Simple implementation - extend as needed
        if formula == 'constant':
            return np.ones((len(self.data), 1))
        
        terms = [term.strip() for term in formula.split('+')]
        matrices = []
        
        for term in terms:
            if term == 'constant' or term == '1':
                matrices.append(np.ones((len(self.data), 1)))
            elif term in self.data.columns:
                matrices.append(self.data[term].values.reshape(-1, 1))
            else:
                raise ValueError(f"Unknown term: {term}")
        
        return np.hstack(matrices)
    
    def _compute_hessian(self, params, X, epsilon=1e-5):
        """Compute numerical Hessian."""
        n = len(params)
        hessian = np.zeros((n, n))
        
        f0 = self.log_likelihood(params, X)
        
        for i in range(n):
            for j in range(i, n):
                params_pp = params.copy()
                params_pm = params.copy() 
                params_mp = params.copy()
                params_mm = params.copy()
                
                params_pp[i] += epsilon
                params_pp[j] += epsilon
                
                params_pm[i] += epsilon
                params_pm[j] -= epsilon
                
                params_mp[i] -= epsilon
                params_mp[j] += epsilon
                
                params_mm[i] -= epsilon
                params_mm[j] -= epsilon
                
                fpp = self.log_likelihood(params_pp, X)
                fpm = self.log_likelihood(params_pm, X) 
                fmp = self.log_likelihood(params_mp, X)
                fmm = self.log_likelihood(params_mm, X)
                
                hessian[i, j] = (fpp - fpm - fmp + fmm) / (4 * epsilon**2)
                hessian[j, i] = hessian[i, j]
        
        return hessian
    
    def compute_elasticities(self, results, market_id=None):
        """Compute own and cross-price elasticities."""
        if market_id is None:
            market_id = self.markets[0]
            
        # Get market data
        market_data = self.data[self.data[self.market_id_col] == market_id].copy()
        
        # Predict shares at estimated parameters
        X = self._create_design_matrix(results['formula'])
        market_X = X[self.data[self.market_id_col] == market_id]
        
        pred_shares, _ = self.predict_shares(market_X, results['beta'], results['rho'])
        
        # Find price coefficient (assume last coefficient is price)
        alpha = results['beta'][-1]  # Price coefficient
        
        # Compute elasticities
        n_products = len(market_data)
        elasticities = np.zeros((n_products, n_products))
        
        for i in range(n_products):
            for j in range(n_products):
                price_j = market_data.iloc[j][self.price_col]
                share_i = pred_shares[i]
                
                if i == j:  # Own price elasticity
                    nest_i = market_data.iloc[i][self.nest_id_col]
                    nest_j = market_data.iloc[j][self.nest_id_col]
                    
                    if nest_i == nest_j:  # Same nest
                        sigma_group = self.data[
                            (self.data[self.market_id_col] == market_id) & 
                            (self.data[self.nest_id_col] == nest_i)
                        ][self.share_col].sum()
                        
                        elasticity = (alpha * price_j / results['rho']) * (1 - share_i / sigma_group) + \
                                   alpha * price_j * (1 - sigma_group) - alpha * price_j
                    else:
                        elasticity = alpha * price_j * (1 - share_i)
                        
                    elasticities[i, j] = elasticity
                    
                else:  # Cross price elasticity
                    nest_i = market_data.iloc[i][self.nest_id_col] 
                    nest_j = market_data.iloc[j][self.nest_id_col]
                    share_j = pred_shares[j]
                    
                    if nest_i == nest_j:  # Same nest
                        elasticity = (alpha * price_j / results['rho']) * (share_j / 
                                    self.data[(self.data[self.market_id_col] == market_id) & 
                                            (self.data[self.nest_id_col] == nest_i)][self.share_col].sum()) + \
                                   alpha * price_j * share_j
                    else:  # Different nests
                        elasticity = alpha * price_j * share_j
                        
                    elasticities[i, j] = elasticity
        
        return elasticities


# Example usage and testing
def simulate_nested_logit_data(n_markets=50, n_nests=3, products_per_nest=4, seed=42):
    """Simulate nested logit data for testing."""
    np.random.seed(seed)
    
    data = []
    product_id = 0
    
    for market in range(n_markets):
        for nest in range(n_nests):
            for product in range(products_per_nest):
                # Product characteristics
                price = np.random.lognormal(1.5, 0.3)
                quality = np.random.normal(0, 1)
                
                # Utility components
                beta_price = -1.0
                beta_quality = 1.5
                beta_constant = 2.0
                rho_true = 0.7
                
                utility = beta_constant + beta_price * price + beta_quality * quality
                
                data.append({
                    'market_id': market,
                    'nest_id': nest,
                    'product_id': product_id,
                    'price': price,
                    'quality': quality,
                    'utility': utility
                })
                
                product_id += 1
    
    df = pd.DataFrame(data)
    
    # Compute shares using nested logit formula
    def compute_shares(group):
        # Within-nest shares
        exp_util_rho = np.exp(group['utility'] / 0.7)
        within_shares = exp_util_rho / exp_util_rho.sum()
        
        # Nest inclusive value
        iv = 0.7 * np.log(exp_util_rho.sum())
        
        group['within_share'] = within_shares
        group['inclusive_value'] = iv
        return group
    
    df = df.groupby(['market_id', 'nest_id']).apply(compute_shares).reset_index(drop=True)
    
    # Compute nest shares
    def compute_nest_shares(group):
        # All inclusive values in market (plus outside option = 0)
        ivs = group.groupby('nest_id')['inclusive_value'].first().values
        all_ivs = np.concatenate([[0], ivs])
        
        # Nest probabilities
        nest_probs = np.exp(all_ivs) / np.exp(all_ivs).sum()
        
        # Map back to products
        for nest_id in group['nest_id'].unique():
            nest_prob = nest_probs[nest_id + 1]  # +1 because outside option is first
            mask = group['nest_id'] == nest_id
            group.loc[mask, 'nest_share'] = nest_prob
            
        return group
    
    df = df.groupby('market_id').apply(compute_nest_shares).reset_index(drop=True)
    
    # Final product shares
    df['shares'] = df['within_share'] * df['nest_share']
    
    return df[['market_id', 'nest_id', 'product_id', 'price', 'quality', 'shares']]


# Testing the implementation
if __name__ == "__main__":
    # Generate test data
    test_data = simulate_nested_logit_data()
    print("Generated test data with shape:", test_data.shape)
    print("\nFirst few rows:")
    print(test_data.head(10))
    
    # Initialize model
    model = NestedLogit(test_data, nest_id_col='nest_id')
    
    # Estimate via MLE
    print("\n" + "="*50)
    print("Maximum Likelihood Estimation")
    print("="*50)
    
    results_mle = model.estimate_mle('price + quality + constant')
    
    print(f"True parameters: β_constant=2.0, β_price=-1.0, β_quality=1.5, ρ=0.7")
    print(f"Estimated parameters:")
    print(f"  β_constant = {results_mle['beta'][2]:.3f} (SE: {results_mle['se_beta'][2]:.3f})")
    print(f"  β_price    = {results_mle['beta'][0]:.3f} (SE: {results_mle['se_beta'][0]:.3f})")
    print(f"  β_quality  = {results_mle['beta'][1]:.3f} (SE: {results_mle['se_beta'][1]:.3f})")
    print(f"  ρ          = {results_mle['rho']:.3f} (SE: {results_mle['se_rho']:.3f})")
    print(f"Log-likelihood: {results_mle['log_likelihood']:.2f}")
    print(f"Converged: {results_mle['convergence']}")
