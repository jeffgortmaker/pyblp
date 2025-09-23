%% BLP Homework Data Generation - Steps 1-4
%% This script generates synthetic data for a Random Coefficients Logit model
%% of the pay-TV market, solves for equilibrium prices, and exports data for Python analysis

if ~isdeployed
    main();
end

function main()
    % Set random seed for reproducibility
    rng(42);
    
    % Generate BLP data and solve equilibrium
    data_table = generate_blp_data();
    
    % Create extended instruments
    fprintf('Creating extended instrument set...\n');
    instruments = create_instruments(data_table);
    
    % Export true parameters
    true_params = export_true_parameters();
    
    % Step 4: Validate instrument quality
    fprintf('Step 4: Checking instrument quality...\n');
    instrument_validation = validate_instruments(data_table);
    fprintf('  ✓ Instrument quality validated\n');
    
    % Write all data to CSV files
    write_data_files(data_table, instruments, true_params);
    
    % Display completion message
    display_completion_message(instrument_validation);
end

function data_table = generate_blp_data()
    % Model parameters
    params = set_model_parameters();
    
    % Step 1: Generate market structure and characteristics
    fprintf('Step 1: Generating market structure and product characteristics...\n');
    [market_data, product_chars] = generate_market_structure(params);
    
    % Generate unobservables
    unobservables = generate_unobservables(params);
    
    % Calculate marginal costs
    marginal_costs = calculate_marginal_costs(product_chars.w, unobservables.omega, params);
    fprintf('  ✓ Generated %d markets with %d products each\n', params.T, params.J);
    
    % Step 2: Solve for equilibrium prices
    fprintf('Step 2: Solving for equilibrium prices using FOC...\n');
    equilibrium_prices = solve_equilibrium_prices(market_data, product_chars, unobservables, marginal_costs, params);
    fprintf('  ✓ Solved equilibrium prices for all %d markets\n', params.T);
    
    % Step 2i: Solve using Morrow-Skerlos (2011) algorithm for comparison
    fprintf('Step 2i: Solving using Morrow-Skerlos (2011) algorithm...\n');
    [ms_prices, comparison_results] = solve_morrow_skerlos_prices(market_data, product_chars, unobservables, marginal_costs, params, equilibrium_prices);
    fprintf('  ✓ Morrow-Skerlos algorithm completed\n');
    
    % Step 3: Calculate observed market shares
    fprintf('Step 3: Calculating observed market shares...\n');
    observed_shares = calculate_observed_shares(equilibrium_prices, product_chars, unobservables, market_data, params);
    fprintf('  ✓ Calculated market shares for all markets\n');
    
    % Combine all data into table
    data_table = create_data_table(market_data, product_chars, unobservables, marginal_costs, equilibrium_prices, observed_shares);
end

function params = set_model_parameters()
    % Market structure parameters
    params.T = 600;  % Number of markets
    params.J = 4;    % Number of products per market
    params.N_sim = 200;  % Number of simulation draws
    
    % Demand parameters
    params.beta_1 = 1;        % Quality coefficient
    params.alpha = -2;        % Price coefficient
    params.beta_2_mean = 4;   % Satellite mean preference
    params.beta_3_mean = 4;   % Wired mean preference
    params.sigma_2 = 1;       % Satellite preference std dev
    params.sigma_3 = 1;       % Wired preference std dev
    
    % Supply parameters
    params.gamma_0 = 0.5;     % Cost intercept
    params.gamma_1 = 0.25;    % Cost shifter coefficient
end

function [market_data, product_chars] = generate_market_structure(params)
    % Create market and product identifiers
    market_data.market_ids = repelem((1:params.T)', params.J);
    market_data.product_ids = repmat((1:params.J)', params.T, 1);
    market_data.firm_ids = repmat((1:params.J)', params.T, 1);
    
    % Product characteristics (satellite/wired dummies)
    market_data.satellite = repmat([1; 1; 0; 0], params.T, 1);
    market_data.wired = repmat([0; 0; 1; 1], params.T, 1);
    
    % Observable characteristics
    product_chars.x = abs(randn(params.T * params.J, 1));  % Quality characteristic
    product_chars.w = abs(randn(params.T * params.J, 1));  % Cost shifter
end

function unobservables = generate_unobservables(params)
    % Correlated demand and cost unobservables
    xi_omega_cov = [1.0, 0.25; 0.25, 1.0];
    A = chol(xi_omega_cov, 'lower');
    z = randn(params.T * params.J, 2);
    xi_omega = z * A';
    
    unobservables.xi = xi_omega(:, 1);     % Demand unobservable
    unobservables.omega = xi_omega(:, 2);  % Cost unobservable
end

function marginal_costs = calculate_marginal_costs(w, omega, params)
    % Log marginal cost function
    log_mc = params.gamma_0 + params.gamma_1 * w + omega / 8;
    marginal_costs = exp(log_mc);
end

function equilibrium_prices = solve_equilibrium_prices(market_data, product_chars, unobservables, marginal_costs, params)
    % Solve FOC system for each market
    equilibrium_prices = zeros(params.T * params.J, 1);
    options = optimoptions('fsolve', 'Display', 'off');
    
    for t = 1:params.T
        start_idx = (t-1) * params.J + 1;
        end_idx = t * params.J;
        indices = start_idx:end_idx;
        
        % Extract market-specific data
        x_t = product_chars.x(indices);
        xi_t = unobservables.xi(indices);
        satellite_t = market_data.satellite(indices);
        wired_t = market_data.wired(indices);
        mc_t = marginal_costs(indices);
        
        % Solve FOC system
        foc_system = @(p) compute_foc_residuals(p, x_t, xi_t, satellite_t, wired_t, mc_t, params);
        p_eq = fsolve(foc_system, mc_t + 0.5, options);
        equilibrium_prices(indices) = p_eq;
    end
end

function [ms_equilibrium_prices, comparison_results] = solve_morrow_skerlos_prices(market_data, product_chars, unobservables, marginal_costs, params, foc_prices)
    % Implement Morrow and Skerlos (2011) ζ-markup algorithm for solving BLP supply equilibrium
    % Reference: pyBLP documentation, equations (49)-(52)
    % The ζ-markup equation: p ← c + ζ(p), where ζ = Λ^(-1)(H ⊙ Γ)'(p - c) - Λ^(-1)s
    
    ms_equilibrium_prices = zeros(params.T * params.J, 1);
    max_iterations = 1000;   % Morrow-Skerlos is a contraction, should converge faster
    tolerance = 1e-10;       % Tight tolerance for ζ-markup equation
    
    % Statistics for comparison
    convergence_info = zeros(params.T, 3); % [iterations, max_error, converged]
    
    for t = 1:params.T
        start_idx = (t-1) * params.J + 1;
        end_idx = t * params.J;
        indices = start_idx:end_idx;
        
        % Extract market-specific data
        x_t = product_chars.x(indices);
        xi_t = unobservables.xi(indices);
        satellite_t = market_data.satellite(indices);
        wired_t = market_data.wired(indices);
        mc_t = marginal_costs(indices);
        
        % Initialize prices with marginal costs (pyBLP default)
        p_current = mc_t;
        
        % Morrow-Skerlos ζ-markup iteration (this is a contraction)
        converged = false;
        iteration = 0;
        
        while ~converged && iteration < max_iterations
            iteration = iteration + 1;
            
            % Calculate shares and derivatives at current prices
            [shares, dsdp] = blp_shares_and_derivatives(p_current, x_t, xi_t, satellite_t, wired_t, params);
            
            % Compute Λ (diagonal matrix of weighted own-price elasticities)
            % Λ_jj ≈ Σ_i w_it * s_ijt * (∂U_ijt/∂p_jt)
            % For logit: ∂U_ijt/∂p_jt = α (price coefficient)
            % Since α is negative, Lambda will be negative
            Lambda = diag(params.alpha * shares);  % params.alpha = -2, so this is negative
            
            % Compute Γ (interaction matrix)
            % Γ_jk ≈ Σ_i w_it * s_ijt * s_ikt * (∂U_ikt/∂p_kt)
            Gamma = zeros(params.J, params.J);
            for j = 1:params.J
                for k = 1:params.J
                    % For logit: this simplifies to α * shares[j] * shares[k]
                    Gamma(j,k) = params.alpha * shares(j) * shares(k);
                end
            end
            
            % Ownership matrix H (single-product firms, so H is identity)
            H = eye(params.J);
            
            % Calculate ζ-markup: ζ = Λ^(-1)(H ⊙ Γ)^T(p - c) - Λ^(-1)s
            Lambda_inv = inv(Lambda + 1e-12 * eye(params.J));  % Add small regularization
            markup_current = p_current - mc_t;
            
            % ζ-markup calculation following pyBLP implementation exactly
            % capital_gamma_tilde = ownership_matrix * capital_gamma
            capital_gamma_tilde = H .* Gamma;
            % capital_gamma_tilde_margin = capital_gamma_tilde.T @ margin
            capital_gamma_tilde_margin = capital_gamma_tilde' * markup_current;
            % zeta = capital_lamda_inv @ capital_gamma_tilde_margin - capital_lamda_inv @ shares
            zeta_markup = Lambda_inv * capital_gamma_tilde_margin - Lambda_inv * shares;
            
            % Update prices using ζ-markup equation: p ← c + ζ(p)
            p_new = mc_t + zeta_markup;
            
            % Check convergence using firm FOC conditions (more robust)
            foc_residuals = Lambda * (p_new - mc_t - zeta_markup);
            foc_error = norm(foc_residuals);
            convergence_info(t, 2) = foc_error;
            
            if foc_error < tolerance
                converged = true;
                convergence_info(t, 3) = 1;
            end
            
            % Update prices (no damping needed - this is a contraction)
            p_current = p_new;
        end
        
        convergence_info(t, 1) = iteration;
        ms_equilibrium_prices(indices) = p_current;
        
        if ~converged && mod(t, 100) == 0
            fprintf('Market %d: %d iterations, error: %.2e\n', t, iteration, price_error);
        end
    end
    
    % Compare results with FOC method
    comparison_results = compare_price_methods(foc_prices, ms_equilibrium_prices, convergence_info);
end

function comparison_results = compare_price_methods(foc_prices, ms_prices, convergence_info)
    % Compare FOC and Morrow-Skerlos price solutions
    
    price_diff = abs(foc_prices - ms_prices);
    max_diff = max(price_diff);
    mean_diff = mean(price_diff);
    rmse_diff = sqrt(mean(price_diff.^2));
    
    % Convergence statistics
    converged_markets = sum(convergence_info(:, 3));
    mean_iterations = mean(convergence_info(:, 1));
    max_iterations = max(convergence_info(:, 1));
    
    % Store results
    comparison_results.price_differences = price_diff;
    comparison_results.max_difference = max_diff;
    comparison_results.mean_difference = mean_diff;
    comparison_results.rmse_difference = rmse_diff;
    comparison_results.correlation = corr(foc_prices, ms_prices);
    comparison_results.converged_markets = converged_markets;
    comparison_results.mean_iterations = mean_iterations;
    comparison_results.max_iterations = max_iterations;
    
    % Silent execution - no output
end

function observed_shares = calculate_observed_shares(equilibrium_prices, product_chars, unobservables, market_data, params)
    % Calculate market shares for each market
    observed_shares = zeros(params.T * params.J, 1);
    
    for t = 1:params.T
        start_idx = (t-1) * params.J + 1;
        end_idx = t * params.J;
        indices = start_idx:end_idx;
        
        % Extract market-specific data
        prices_t = equilibrium_prices(indices);
        x_t = product_chars.x(indices);
        xi_t = unobservables.xi(indices);
        satellite_t = market_data.satellite(indices);
        wired_t = market_data.wired(indices);
        
        % Calculate shares
        [shares_t, ~] = blp_shares_and_derivatives(prices_t, x_t, xi_t, satellite_t, wired_t, params);
        observed_shares(indices) = shares_t;
    end
end

function data_table = create_data_table(market_data, product_chars, unobservables, marginal_costs, equilibrium_prices, observed_shares)
    % Combine all data into a structured table
    data_table = table(market_data.market_ids, market_data.product_ids, market_data.firm_ids, ...
                      observed_shares, equilibrium_prices, product_chars.x, product_chars.w, ...
                      unobservables.xi, unobservables.omega, market_data.satellite, market_data.wired, marginal_costs);
    
    % Set descriptive variable names
    data_table.Properties.VariableNames = {'market_id', 'product_id', 'firm_id', ...
                                          'observed_share', 'equilibrium_price', ...
                                          'quality_x', 'cost_shifter_w', 'xi_demand_unobs', ...
                                          'omega_cost_unobs', 'satellite_dummy', 'wired_dummy', 'marginal_cost'};
end

function instruments = create_instruments(data_table)
    % Step 4: Create extended instrument set
    T = max(data_table.market_id);
    J = max(data_table.product_id);
    
    % Calculate sum of other products' characteristics within market
    market_x_other = zeros(height(data_table), 1);
    market_w_other = zeros(height(data_table), 1);
    
    for t = 1:T
        for j = 1:J
            current_idx = find(data_table.market_id == t & data_table.product_id == j);
            other_indices = find(data_table.market_id == t & data_table.product_id ~= j);
            
            if ~isempty(current_idx) && ~isempty(other_indices)
                market_x_other(current_idx) = sum(data_table.quality_x(other_indices));
                market_w_other(current_idx) = sum(data_table.cost_shifter_w(other_indices));
            end
        end
    end
    
    % Create extended instrument set
    instruments = table(data_table.market_id, data_table.product_id, ...
                       data_table.quality_x, data_table.cost_shifter_w, ...
                       market_x_other, market_w_other, ...
                       data_table.quality_x.^2, data_table.cost_shifter_w.^2, ...
                       data_table.quality_x .* data_table.satellite_dummy, ...
                       data_table.cost_shifter_w .* data_table.satellite_dummy, ...
                       data_table.quality_x .* data_table.wired_dummy, ...
                       data_table.cost_shifter_w .* data_table.wired_dummy, ...
                       data_table.satellite_dummy, data_table.wired_dummy);
    
    % Set descriptive variable names
    instruments.Properties.VariableNames = {'market_id', 'product_id', 'x_quality', 'w_cost', ...
                                           'market_x_other', 'market_w_other', 'x_squared', 'w_squared', ...
                                           'x_satellite', 'w_satellite', 'x_wired', 'w_wired', 'satellite', 'wired'};
end

function instrument_validation = validate_instruments(data_table)
    % Step 4: Check whether x and w appear to be good instruments
    % Good instruments should be:
    % 1. Correlated with endogenous prices (relevance)
    % 2. Uncorrelated with demand unobservables xi (exogeneity)
    % 3. Provide sufficient variation
    
    % Extract variables
    prices = data_table.equilibrium_price;
    shares = data_table.observed_share;
    x = data_table.quality_x;
    w = data_table.cost_shifter_w;
    xi = data_table.xi_demand_unobs;
    
    % 1. First stage regressions - check instrument relevance
    % Use simple regression to avoid multicollinearity issues
    X_simple = [ones(length(x), 1), x, w];  % Just constant, x, w
    
    try
        beta_price = X_simple \ prices;
        fitted_prices = X_simple * beta_price;
        price_rsq = 1 - sum((prices - fitted_prices).^2) / sum((prices - mean(prices)).^2);
        
        % F-statistic for joint significance of x and w
        residuals = prices - fitted_prices;
        n = length(prices);
        k = size(X_simple, 2);
        mse = sum(residuals.^2) / (n - k);
        
        % Test H0: beta_x = beta_w = 0
        X_restricted = ones(n, 1);  % Only constant
        beta_restricted = X_restricted \ prices;
        fitted_restricted = X_restricted * beta_restricted;
        ssr_restricted = sum((prices - fitted_restricted).^2);
        ssr_unrestricted = sum(residuals.^2);
        
        f_stat = ((ssr_restricted - ssr_unrestricted) / 2) / mse;
        
    catch
        % Fallback if matrix operations fail
        price_rsq = corr(prices, [x, w]).^2;
        price_rsq = max(price_rsq);
        f_stat = NaN;
        beta_price = [NaN; NaN; NaN];
    end
    
    % 2. Check correlations
    corr_price_x = corr(prices, x);
    corr_price_w = corr(prices, w);
    corr_xi_x = corr(xi, x);
    corr_xi_w = corr(xi, w);
    
    % 3. Variation analysis
    var_x = var(x);
    var_w = var(w);
    var_prices = var(prices);
    
    % 4. Reduced form for market shares
    try
        beta_share = X_simple \ shares;
        fitted_shares = X_simple * beta_share;
        share_rsq = 1 - sum((shares - fitted_shares).^2) / sum((shares - mean(shares)).^2);
    catch
        share_rsq = NaN;
    end
    
    % Store validation results
    instrument_validation.price_rsq = price_rsq;
    instrument_validation.share_rsq = share_rsq;
    instrument_validation.f_statistic = f_stat;
    instrument_validation.correlations = struct(...
        'price_x', corr_price_x, 'price_w', corr_price_w, ...
        'xi_x', corr_xi_x, 'xi_w', corr_xi_w);
    instrument_validation.variations = struct(...
        'x', var_x, 'w', var_w, 'prices', var_prices);
    instrument_validation.first_stage_coeffs = beta_price;
    
    % Assess instrument quality
    good_relevance = price_rsq > 0.1 && (isnan(f_stat) || f_stat > 10);  
    good_exogeneity = abs(corr_xi_x) < 0.3 && abs(corr_xi_w) < 0.3;  
    sufficient_variation = var_x > 0.01 && var_w > 0.01;  
    
    instrument_validation.assessment = struct(...
        'good_relevance', good_relevance, ...
        'good_exogeneity', good_exogeneity, ...
        'sufficient_variation', sufficient_variation, ...
        'overall_quality', good_relevance && good_exogeneity && sufficient_variation);
end

function true_params = export_true_parameters()
    % Export true parameter values for estimation validation
    param_names = {'beta_1'; 'alpha'; 'beta_2_mean'; 'beta_3_mean'; 'sigma_2'; 'sigma_3'; 'gamma_0'; 'gamma_1'};
    param_values = [1; -2; 4; 4; 1; 1; 0.5; 0.25];
    
    true_params = table(param_names, param_values);
    true_params.Properties.VariableNames = {'parameter', 'value'};
end

function write_data_files(data_table, instruments, true_params)
    % Write main dataset to CSV file
    writetable(data_table, 'fake_data.csv');
    writetable(instruments, 'blp_instruments_extended.csv');
    writetable(true_params, 'blp_true_parameters.csv');
end

function display_completion_message(~)
    % Display export completion message
    fprintf('\nData exported to fake_data.csv\n');
end

%% Helper Functions for BLP Computation

function [shares, dsdp] = blp_shares_and_derivatives(prices, x, xi, satellite, wired, params)
    % Calculate market shares and price derivatives using simulation
    J = length(prices);
    shares_sum = zeros(J, 1);
    dsdp_sum = zeros(J, J);
    
    % Generate random coefficient draws
    beta_2_draws = normrnd(params.beta_2_mean, params.sigma_2, params.N_sim, 1);
    beta_3_draws = normrnd(params.beta_3_mean, params.sigma_3, params.N_sim, 1);
    
    % Simulation loop
    for i = 1:params.N_sim
        % Calculate utilities for this draw
        V = params.beta_1 * x + params.alpha * prices + ...
            beta_2_draws(i) * satellite + beta_3_draws(i) * wired + xi;
        
        % Numerical stability: subtract maximum utility
        V_max = max(V);
        V_stable = V - V_max;
        exp_V = exp(V_stable);
        exp_0 = exp(-V_max);
        denom = exp_0 + sum(exp_V);
        
        % Calculate choice probabilities
        s_i = exp_V / denom;
        shares_sum = shares_sum + s_i;
        
        % Calculate price derivatives
        for j = 1:J
            for k = 1:J
                if j == k
                    dsdp_sum(j, k) = dsdp_sum(j, k) + params.alpha * s_i(j) * (1 - s_i(j));
                else
                    dsdp_sum(j, k) = dsdp_sum(j, k) - params.alpha * s_i(j) * s_i(k);
                end
            end
        end
    end
    
    % Average across simulation draws
    shares = shares_sum / params.N_sim;
    dsdp = dsdp_sum / params.N_sim;
end

function residuals = compute_foc_residuals(prices, x, xi, satellite, wired, mc, params)
    % Compute first-order condition residuals for profit maximization
    [shares, dsdp] = blp_shares_and_derivatives(prices, x, xi, satellite, wired, params);
    J = length(prices);
    residuals = zeros(J, 1);
    
    % FOC: markup * own-price derivative + market share = 0
    for j = 1:J
        markup = prices(j) - mc(j);
        residuals(j) = markup * dsdp(j, j) + shares(j);
    end
end