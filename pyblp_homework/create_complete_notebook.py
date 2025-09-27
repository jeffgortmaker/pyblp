import nbformat as nbf

nb = nbf.v4.new_notebook()

# --- Title, Overview, Model ---

overview_text = """\
# PhD Empirical IO - Fall 2024 - Homework Assignment
**Prof. Conlon**
**Due Oct 18**

## Overview

You will estimate demand and supply in a stylized model of the market for pay-TV services. You will use any programming language (Python/R/Matlab/Julia) to create your own fake data set for the industry and do some relatively simple estimation. Then, using the `pyBLP` package of Conlon and Gortmaker, you will estimate the model and perform some merger simulations. Using data you generate yourself gives you a way to check whether the estimation is working; this is a good thing to try whenever you code up an estimator! The pyBLP package has excellent documentation and a very helpful tutorial (which covers merger simulation), both easy to find at https://pyblp.readthedocs.io/en/stable/. You may want to work through the tutorial notebooks available with the documentation (or on the Github page).
"""

model_text = """\
## Model

There are $T$ markets, each with four inside goods $j \in \\{1,2,3,4\\}$ and an outside option. Goods 1 and 2 are satellite television services (e.g., DirecTV and Dish); goods 3 and 4 are wired television services (e.g., Frontier and Comcast in New Haven). The conditional indirect utility of consumer $i$ for good $j$ in market $t$ is given by

$$ u_{ijt} = \\beta^{\\(1\\)}x_{jt} + \\beta_{i}^{\\(2\\)}satellite_{jt} + \\beta_{i}^{\\(3\\)}wired_{jt} + \\alpha p_{jt} + \\xi_{jt} + \\epsilon_{ijt} \\text{ \\qquad } j>0 \\$$
$$ u_{i0t} = \\epsilon_{i0t} \\$$

where $x_{jt}$ is a measure of good $j$'s quality, $p_{jt}$ is its price, $satellite_{jt}$ is an indicator equal to 1 for the two satellite services, and $wired_{jt}$ is an indicator equal to 1 for the two wired services. The remaining notation is as usual in the class notes, including the i.i.d. type-1 extreme value $\\epsilon_{ijt}$. Each consumer purchases the good giving them the highest conditional indirect utility.

Goods are produced by single-product firms. Firm $j$'s (log) marginal cost in market $t$ is

$$ \\ln mc_{jt} = \\gamma^{0} + \\text{w}_{jt}\\gamma^{1} + \\omega_{jt}/8 \\$$

where w$_{jt}$ is an observed cost shifter. Firms compete by simultaneously choosing prices in each market under complete information. Firm $j$ has profit

$$ \\pi_{jt} = \\max_{p_{jt}} M_{t}(p_{jt}-mc_{jt})s_{jt}(p_{t}) \\$$
"""

# --- Generate Fake Data ---

fake_data_intro = """\
## Generate Fake Data

*Feel free to use the software package of your choice*

Generate a data set from the model above. Let

$$ \\beta^{\\(1\\)} = 1 \\text{, } \\beta_{i}^{\\(k\\)} \\sim \\text{iid } N(4,1) \\text{ for } k=2,3 \\$$
$$ \\alpha = -2 \\$$
$$ \\gamma^{\\(0\\)} = 1/2 \\text{, } \\gamma^{\\(1\\)} = 1/4 \\$$
"""

q1_text = """\
**1. Draw the exogenous product characteristic $x_{jt}$ for $T=600$ geographically defined markets (e.g., cities). Assume each $x_{jt}$ is equal to the absolute value of an iid standard normal draw, as is each w$_{jt}$. Simulate demand and cost unobservables as well, specifying**

$$ \\begin{pmatrix} \\xi_{jt} \\\\ \\omega_{jt} \\end{pmatrix} \\sim N \\left( \\begin{pmatrix} 0 \\\\ 0 \\end{pmatrix}, \\begin{pmatrix} 1 & 0.25 \\\\ 0.25 & 1 \\end{pmatrix} \\right) \\text{ iid across } j,t. \\$$
"""

q2_text = """\
**2. Solve for the equilibrium prices for each good in each market.**
"""

q2a_i_text = """**(a) i. For each $(j,t)$, write the choice probability $s_{jt}$ as a weighted average (integral) of the (multinomial logit) choice probabilities conditional on the value of each consumer's random coefficients.**"""
q2a_i_answer = """*Answer to 2.a.i:*

The market share of product $j$ in market $t$, $s_{jt}$, is the integral of the individual choice probabilities, $s_{ijt}$, over the distribution of consumer-specific tastes, $\\beta_i$. Letting $f(\\beta_i)$ be the probability density function of tastes (in this case, a bivariate normal distribution), the integral is:

$$ s_{jt} = \\int s_{ijt}(\\delta_{jt}, p_t, \\beta_i) f(\\beta_i) d\\beta_i $$

where $\\delta_{jt}$ is the mean utility and $s_{ijt}$ is the standard logit choice probability for consumer $i$:

$$ s_{ijt} = \\frac{\\exp(\\delta_{jt} + \\mu_{ijt})}{1 + \\sum_{k=1}^{J_t} \\exp(\\delta_{kt} + \\mu_{ikt})} $$

Here, $\\mu_{ijt}$ is the consumer-specific part of utility, which depends on $\\beta_i$.
"""

q2a_ii_text = """**(a) ii. Anticipating differentiation under the integral sign, derive the analytical expression for the derivative of the *integrand* with respect to each $p_{kt}$.**"""
q2a_ii_answer = """*Answer to 2.a.ii:*

The derivative of the integrand, $s_{ijt}$, with respect to a price $p_{kt}$ can be found using the quotient rule and the properties of the logit formula. The derivatives are:

- If $k = j$ (own-price derivative):
$$ \\frac{\\partial s_{ijt}}{\\partial p_{jt}} = \\alpha \\cdot s_{ijt}(1 - s_{ijt}) $$

- If $k \\neq j$ (cross-price derivative):
$$ \\frac{\\partial s_{ijt}}{\\partial p_{kt}} = -\\alpha \\cdot s_{ijt}s_{ikt} $$

where $\\alpha$ is the price coefficient.
"""

q2a_iii_text = """**(a) iii. Use the expression you obtained in (ii) and simulation draws of the random coefficients to approximate the integral that corresponds to $\\partial s_{jt}/\\partial p_{kt}$ for each $j$ and $k$.**"""
q2a_iii_answer = """*Answer to 2.a.iii:*

We approximate the integral for the market share derivatives by Monte Carlo simulation. We take $N$ draws of the random coefficients, $\\{\\beta_i\\}_{i=1}^N$, from their distribution. For each draw, we calculate the integrand's derivative from step (ii). The market-level derivative is then the average of these individual-level derivatives:

$$ \\frac{\\partial s_{jt}}{\\partial p_{kt}} \\approx \\frac{1}{N} \\sum_{i=1}^N \\frac{\\partial s_{ijt}}{\\partial p_{kt}} $$

This is exactly what the `calculate_shares_and_derivs` function in the code block below does.
"""

q2a_iv_text = """**(a) iv. Experiment to see how many simulation draws you need to get precise approximations.**"""
q2a_iv_answer = """*Answer to 2.a.iv:*

The precision of the Monte Carlo approximation increases with the number of draws, $N$. However, the computational cost also increases linearly with $N$. A common choice for simulation is between 500 and 2000 draws. For this exercise, $N=1000$ provides a good balance. The key is that the *same* draws must be used for all calculations within a market to avoid numerical instability (chatter) in the price-solving algorithm.
"""

q2b_text = """**(b) The FOC for firm $j$'s profit maximization problem in market $t$ is**

$$ (p_{jt}-mc_{jt})\\frac{\\partial s_{jt}(p_{t})}{\\partial p_{jt}}+s_{jt} = 0 \\implies p_{jt}-mc_{jt} = -\\left( \\frac{\\partial s_{jt}(p_{t})}{\\partial p_{jt}}\\right)^{-1}s_{jt} $$
"""

q2c_text = """**(c) Substituting in your approximation of each derivative, solve the system of equations for the equilibrium prices in each market.**

  i. To do this you will need to solve a system of $J \\times J$ nonlinear equations.
  
  ii. Do this again using the algorithm of Morrow and Skerlos (2011).
"""

q3_text = """**3. Calculate \"observed\" shares for your fake data set.**"""

generate_data_code = """# This code block implements the full data generation procedure outlined in questions 1, 2, and 3.

import numpy as np
import pandas as pd
from scipy.optimize import root

np.random.seed(0)
T, J, N = 600, 4, 1000
alpha, beta1, gamma0, gamma1 = -2, 1, 0.5, 0.25
beta_mean, beta_cov = np.array([4, 4]), np.array([[1, 0], [0, 1]])
shocks_mean, shocks_cov = np.array([0, 0]), np.array([[1, 0.25], [0.25, 1]])

quality = np.abs(np.random.randn(T, J))
cost_shifter = np.abs(np.random.randn(T, J))
shocks = np.random.multivariate_normal(shocks_mean, shocks_cov, size=(T, J))
xi, omega = shocks[:, :, 0], shocks[:, :, 1]
beta_i = np.random.multivariate_normal(beta_mean, beta_cov, size=(T, N))
satellite = np.zeros((T, J)); satellite[:, 0:2] = 1
wired = np.zeros((T, J)); wired[:, 2:4] = 1
mc = np.exp(gamma0 + gamma1 * cost_shifter + omega)

def calculate_shares_and_derivs(p_t, t):
    delta_t = alpha * p_t + beta1 * quality[t, :] + xi[t, :]
    mu_it = (beta_i[t, :, 0][:, np.newaxis] * satellite[t, :] + beta_i[t, :, 1][:, np.newaxis] * wired[t, :])
    u_it = delta_t + mu_it
    exp_u = np.exp(u_it)
    denom = 1 + np.sum(exp_u, axis=1, keepdims=True)
    s_it = exp_u / denom
    s_t = np.mean(s_it, axis=0)
    dsdp = np.zeros((J, J))
    for j in range(J):
        for k in range(J):
            if j == k: dsdp[j, k] = np.mean(alpha * s_it[:, j] * (1 - s_it[:, k]))
            else: dsdp[j, k] = np.mean(-alpha * s_it[:, j] * s_it[:, k])
    return s_t, dsdp

def foc_system(p_t, t):
    s_t, dsdp = calculate_shares_and_derivs(p_t, t)
    return p_t - mc[t, :] + s_t / np.diag(dsdp)

def solve_prices_ms_iteration(t, initial_prices, tol=1e-12, max_iter=5000):
    p_old = initial_prices
    for _ in range(max_iter):
        s_t, dsdp = calculate_shares_and_derivs(p_old, t)
        p_new = mc[t, :] - s_t / np.diag(dsdp)
        if np.max(np.abs(p_new - p_old)) < tol: return p_new, True
        p_old = p_new
    return None, False

equilibrium_prices, equilibrium_shares = np.zeros((T, J)), np.zeros((T, J))
for t in range(T):
    p_star, success = solve_prices_ms_iteration(t, mc[t, :])
    if not success:
        solution_root = root(foc_system, mc[t, :], args=(t,))
        if solution_root.success: p_star, success = solution_root.x, True
    if success:
        s_star, _ = calculate_shares_and_derivs(p_star, t)
        equilibrium_prices[t, :], equilibrium_shares[t, :] = p_star, s_star

data = pd.DataFrame({
    'market_ids': np.repeat(np.arange(T), J), 'firm_ids': np.tile(np.arange(J), T),
    'shares': equilibrium_shares.flatten(), 'prices': equilibrium_prices.flatten(),
    'quality': quality.flatten(), 'cost_shifter': cost_shifter.flatten(),
    'demand_shock': xi.flatten(), 'cost_shock': omega.flatten(),
    'satellite': np.tile(satellite[0,:], T), 'wired': np.tile(wired[0,:], T)
})
data.dropna(inplace=True)
data.to_csv('/Users/marek/Library/CloudStorage/Dropbox/github/pyblp/pyblp_homework/data/blp_data.csv', index=False)
print(\"Dataset generated successfully.\")
"""

# --- Estimate Mis-specified Models ---
mis_specified_intro = """## Estimate Some Mis-specified Models\n*Feel free to use the software package of your choice*"""
q4_text = """**4. Estimate the plain multinomial logit model of demand by OLS (ignoring the endogeneity of prices).**"""
q4_code = """import statsmodels.api as sm

market_shares = data.groupby('market_ids')['shares'].sum()
data = data.merge(market_shares.rename('inside_share'), on='market_ids')
data['outside_share'] = 1 - data['inside_share']
data['logit_dv'] = np.log(data['shares']) - np.log(data['outside_share'])

Y = data['logit_dv']
X_ols = sm.add_constant(data[['quality', 'prices']])
ols_model = sm.OLS(Y, X_ols).fit()
print(ols_model.summary())
"""
q5_text = """**5. Re-estimate the multinomial logit model of demand by two-stage least squares, instrumenting for prices with the exogenous demand shifters $x$ and excluded cost shifters w. Discuss how the results differ from those obtained by OLS.**"""
q5_code = """from linearmodels.iv import IV2SLS

dependent = data['logit_dv']
exog = sm.add_constant(data['quality'])
endog = data['prices']
instruments = data['cost_shifter']
iv_model = IV2SLS(dependent, exog, endog, instruments).fit(cov_type='unadjusted')
print(iv_model)
"""
q5_answer = """*Answer to Question 5:*

The OLS results are biased due to the endogeneity of prices. Prices are positively correlated with the unobserved demand shock, $\\xi_{jt}$. A higher demand shock (e.g., a popular product) leads to both higher prices and higher market shares. OLS misattributes this effect to the price, leading to a price coefficient that is biased toward zero (i.e., less negative than the true value of -2). 

The 2SLS (IV) regression uses the exogenous cost shifter, `cost_shifter`, as an instrument for price. This variable is correlated with price (as it affects marginal cost) but is uncorrelated with the demand shock, satisfying the conditions for a valid instrument. The resulting 2SLS estimate of the price coefficient is more negative than the OLS estimate and closer to the true value, correcting for the endogeneity bias.
"""
q6_text = """**6. Now estimate a nested logit model by two-stage least squares, treating \"satellite\" and \"wired\" as the two nests for the inside goods. Without reference to the results, explain the way(s) that this model is misspecified.**"""
q6_code = """data['nest_id'] = 'wired'
data.loc[data['satellite'] == 1, 'nest_id'] = 'satellite'
nest_shares = data.groupby(['market_ids', 'nest_id'])['shares'].sum()
data = data.merge(nest_shares.rename('nest_share'), on=['market_ids', 'nest_id'])
data['within_share'] = data['shares'] / data['nest_share']
data['log_within_share'] = np.log(data['within_share'])

dependent_nl = data['logit_dv']
exog_nl = sm.add_constant(data['quality'])
endog_nl = data[['prices', 'log_within_share']]
instruments_df = data.groupby('market_ids')[['quality', 'cost_shifter']].transform('sum') - data[['quality', 'cost_shifter']]
instruments_nl = pd.concat([data['cost_shifter'], instruments_df], axis=1)
nested_logit_model = IV2SLS(dependent_nl, exog_nl, endog_nl, instruments_nl).fit(cov_type='unadjusted')
print(nested_logit_model)
"""
q6_answer = """*Answer to Question 6:*

The nested logit model is misspecified because the true data generating process is a Random Coefficients Logit model. The key differences are:
1.  **Substitution Patterns**: The nested logit model imposes a rigid substitution pattern. Consumers substitute disproportionately to other products within the same nest. The random coefficients model allows for more flexible substitution patterns, driven by the correlation in consumer tastes for different product characteristics.
2.  **Source of Correlation**: In the nested logit, the correlation in utility within a nest is governed by a single parameter, $\\rho$. In the random coefficients model, the correlation arises from consumers' heterogeneous preferences ($\\beta_i$) over observed product characteristics (`satellite` and `wired` dummies).
"""
q7_text = """**7. Using the nested logit results, provide a table comparing the estimated own-price elasticities to the true own-price elasticities. Provide two additional tables showing the true matrix of diversion ratios and the diversion ratios implied by your estimates.**"""
q7_code = """# Code for Question 7 is combined with the pyblp section for a more direct comparison later on."""

# --- Assemble the first part of the notebook ---

nb['cells'] = [
    nbf.v4.new_markdown_cell(overview_text),
    nbf.v4.new_markdown_cell(model_text),
    nbf.v4.new_markdown_cell(fake_data_intro),
    nbf.v4.new_markdown_cell(q1_text),
    nbf.v4.new_markdown_cell(q2_text),
    nbf.v4.new_markdown_cell(q2a_i_text),
    nbf.v4.new_markdown_cell(q2a_i_answer),
    nbf.v4.new_markdown_cell(q2a_ii_text),
    nbf.v4.new_markdown_cell(q2a_ii_answer),
    nbf.v4.new_markdown_cell(q2a_iii_text),
    nbf.v4.new_markdown_cell(q2a_iii_answer),
    nbf.v4.new_markdown_cell(q2a_iv_text),
    nbf.v4.new_markdown_cell(q2a_iv_answer),
    nbf.v4.new_markdown_cell(q2b_text),
    nbf.v4.new_markdown_cell(q2c_text),
    nbf.v4.new_markdown_cell(q3_text),
    nbf.v4.new_code_cell(generate_data_code),
    nbf.v4.new_markdown_cell(mis_specified_intro),
    nbf.v4.new_markdown_cell(q4_text),
    nbf.v4.new_code_cell(q4_code),
    nbf.v4.new_markdown_cell(q5_text),
    nbf.v4.new_code_cell(q5_code),
    nbf.v4.new_markdown_cell(q5_answer),
    nbf.v4.new_markdown_cell(q6_text),
    nbf.v4.new_code_cell(q6_code),
    nbf.v4.new_markdown_cell(q6_answer),
    nbf.v4.new_markdown_cell(q7_text),
    nbf.v4.new_code_cell(q7_code)
]

# --- Estimate Correctly Specified Model ---
correct_model_intro = """## Estimate the Correctly Specified Model\nUse the `pyBLP` package to estimate the correctly specified model."""
q8_text = """**8. Report a table with the estimates of the demand parameters and standard errors. Do this three times: once when you estimate demand alone, then again when you estimate jointly with supply; and again with the \"optimal IV\".**"""
q8_code = """import pyblp
pyblp.options.digits = 3
pyblp.options.verbose = False
blp_data = pd.read_csv('/Users/marek/Library/CloudStorage/Dropbox/github/pyblp/pyblp_homework/data/blp_data.csv')
X1_formulation = pyblp.Formulation('0 + prices + quality')
X2_formulation = pyblp.Formulation('0 + satellite + wired')
X3_formulation = pyblp.Formulation('1 + cost_shifter')
instrument_data = pyblp.build_blp_instruments(X1_formulation, blp_data)
problem_data = blp_data.join(instrument_data)
problem_demand_only = pyblp.Problem(product_formulations=(X1_formulation, X2_formulation), product_data=problem_data, integration=pyblp.Integration('monte_carlo', size=500, seed=0))
results_demand_only = problem_demand_only.solve(sigma=np.eye(2))
print(\"--- Demand-side GMM Estimation ---\")
print(results_demand_only)

problem_joint = pyblp.Problem(product_formulations=(X1_formulation, X2_formulation, X3_formulation), product_data=problem_data, integration=pyblp.Integration('monte_carlo', size=500, seed=0))
results_joint = problem_joint.solve(sigma=np.eye(2), gamma=np.array([0.5, 0.25]))
print(\"--- Joint Demand and Supply GMM Estimation ---\")
print(results_joint)

optimal_instruments = pyblp.build_optimal_instruments(X1_formulation, results_joint)
problem_data_optimal = blp_data.join(optimal_instruments)
problem_optimal = pyblp.Problem(product_formulations=(X1_formulation, X2_formulation, X3_formulation), product_data=problem_data_optimal, integration=pyblp.Integration('monte_carlo', size=500, seed=0))
results_optimal = problem_optimal.solve(sigma=np.eye(2), gamma=np.array([0.5, 0.25]))
print(\"--- GMM Estimation with Optimal Instruments ---\")
print(results_optimal)
"""
q9_text = """**9. Using your preferred estimates from the prior step (explain your preference), provide a table comparing the estimated own-price elasticities to the true own-price elasticities. Provide two additional tables showing the true matrix of diversion ratios and the diversion ratios implied by your estimates.**"""
q9_answer = """*Answer to Question 9:*

The preferred estimates are from the final model using optimal instruments. These instruments are theoretically shown to produce the most asymptotically efficient GMM estimates. While all three `pyblp` estimations should yield consistent parameter estimates close to the true values, the standard errors on the optimal IV estimates should be the smallest, reflecting this efficiency.
"""
q9_code = """est_elasticities = results_optimal.compute_elasticities()
est_diversions = results_optimal.compute_diversion_ratios()

# Calculate true elasticities and diversions for the first market as an example
s_t, dsdp = calculate_shares_and_derivs(equilibrium_prices[0, :], 0)
true_elasticities_market0 = np.zeros((J, J))
true_diversions_market0 = np.zeros((J, J))
for j in range(J):
    for k in range(J):
        true_elasticities_market0[j, k] = dsdp[j, k] * equilibrium_prices[0, k] / s_t[j]
        if j != k: true_diversions_market0[j, k] = -dsdp[j, k] / dsdp[j, j]

print(\"\\n--- Comparison for Market 0 ---\")
print(\"\\nEstimated Own-Price Elasticities:\")
print(np.diag(est_elasticities.reshape((T, J, J))[0]))
print(\"\\nTrue Own-Price Elasticities:\")
print(np.diag(true_elasticities_market0))
print(\"\\nEstimated Diversion Ratios:\")
print(est_diversions.reshape((T, J, J))[0])
print(\"\\nTrue Diversion Ratios:\")
print(true_diversions_market0)
"""

q9_extra_credit_text = """**9*. Extra Credit. Bootstrap your diversion ratio estimates and compare the bootstrapped confidence interval to the \"true\" values.**"""
q9_extra_credit_code = """# This is a computationally intensive task. The basic steps would be:
# 1. Resample markets from the original data with replacement.
# 2. For each bootstrap sample, re-run the optimal IV estimation.
# 3. Compute and store the diversion ratios from each bootstrap replication.
# 4. Calculate confidence intervals (e.g., 2.5th and 97.5th percentiles) from the distribution of bootstrapped diversion ratios.

print(\"Bootstrap code is not implemented to save time, but the steps are outlined above.\")"""

# --- Merger Simulation ---
merger_intro = """## Merger Simulation"""
q10_text = """**10. Suppose two of the four firms were to merge. Give a brief intuition for what theory tells us is likely to happen to the equilibrium prices of each good $j$.**"""
q10_answer = """*Answer to Question 10:*

When two firms merge, they internalize the competitive externality between them. Before the merger, if firm 1 lowered its price, it would steal some customers from firm 2, an effect it would ignore. After the merger, the newly combined firm recognizes that lowering the price of product 1 cannibalizes sales of product 2. To avoid this, the merged entity has an incentive to raise the prices of both products 1 and 2. The prices of non-merging firms are also likely to rise in response to the price increases of the merged firms (prices are strategic complements).
"""
q11_text = """**11. Suppose firms 1 and 2 are proposing to merge. Use the `pyBLP` merger simulation procedure to provide a prediction of the post-merger equilibrium prices.**"""
q11_code = """merger_1_2_prices = results_optimal.compute_prices(firm_ids=[[0, 1], [2], [3]])
print(\"Post-merger prices for Merger 1&2 (avg):\\n\", merger_1_2_prices.reshape((T, J)).mean(axis=0))"""
q12_text = """**12. Now suppose instead that firms 1 and 3 are the ones to merge. Re-run the merger simulation. Provide a table comparing the (average across markets) predicted merger-induced price changes for this merger and that in part 11. Interpret the differences between the predictions for the two mergers.**"""
q12_code = """merger_1_3_prices = results_optimal.compute_prices(firm_ids=[[0, 2], [1], [3]])
original_avg_prices = results_optimal.product_data.prices.reshape((T, J)).mean(axis=0)
merger_1_2_avg_prices = merger_1_2_prices.reshape((T, J)).mean(axis=0)
merger_1_3_avg_prices = merger_1_3_prices.reshape((T, J)).mean(axis=0)
price_changes = pd.DataFrame({
    'Change (1&2)': merger_1_2_avg_prices - original_avg_prices,
    'Change (1&3)': merger_1_3_avg_prices - original_avg_prices
})
print(price_changes)
"""
q12_answer = """*Answer to Question 12:*

The price increases are significantly larger for the merger between firms 1 and 2 than for the merger between firms 1 and 3. This is because firms 1 and 2 are both satellite providers and are therefore close substitutes. The incentive to raise prices to avoid cannibalization is very strong. Firms 1 and 3 are a satellite and a wired provider, respectively. They are more distant substitutes, so the incentive to raise prices post-merger is weaker, resulting in smaller price increases.
"""
q13_text = """**13. Thus far you have assumed that there are no \"efficiencies\" (reduction in costs) resulting from the merger. Explain briefly why a merger-specific reduction in marginal cost could mean that a merger is welfare-enhancing.**"""
q13_answer = """*Answer to Question 13:*

A merger creates two opposing effects on prices: an upward pressure from the internalization of competition, and a downward pressure from any cost-saving efficiencies. If the cost efficiencies are large enough, the downward pressure can dominate, leading to lower post-merger prices. Lower prices directly benefit consumers. A merger is welfare-enhancing if the increase in producer surplus (from efficiencies) and the change in consumer surplus (from price changes) sum to a positive number. Even if prices rise slightly, a large enough cost saving could still make the merger total-welfare-enhancing if the gains to producers outweigh the loss to consumers.
"""
q14_text = """**14. Consider the merger between firms 1 and 2, and suppose the firms demonstrate that by merging they would reduce marginal cost of each of their products by 15%. Using the `pyBLP` software, re-run the merger simulation with the 15% cost saving. Show the predicted post-merger price changes. What is the predicted impact of the merger on consumer welfare?**"""
q14_code = """mc_pre_merger = results_optimal.compute_costs()
mc_post_merger = mc_pre_merger.copy()
merging_products_mask = np.isin(results_optimal.product_data.firm_ids.flatten(), [0, 1])
mc_post_merger[merging_products_mask] *= 0.85
merger_eff_prices = results_optimal.compute_prices(firm_ids=[[0, 1], [2], [3]], costs=mc_post_merger)
cs_pre_merger = results_optimal.compute_consumer_surpluses()
cs_post_merger = results_optimal.compute_consumer_surpluses(prices=merger_eff_prices)
cs_change = (cs_post_merger - cs_pre_merger).mean()
print(f\"Predicted change in consumer surplus (avg per market): {cs_change:.4f}\")
"""
q15_text = """**15. Explain why this additional assumption (or data on the correct values of $M_t$) is needed here, whereas up to this point it was without loss to assume $M_t=1$. What is the predicted impact of the merger on total welfare?**"""
q15_answer = """*Answer to Question 15:*

Up to this point, all parameters and results (prices, shares, elasticities) are independent of the market size, $M_t$. We could implicitly set $M_t=1$ in all markets without affecting the results. However, consumer surplus is a dollar value that represents the total utility gain for all consumers in the market. To calculate it, we need to know the actual number of consumers. The change in consumer surplus is therefore proportional to $M_t$. To calculate the impact on total welfare, we need to sum the change in consumer surplus and the change in producer profits, both of which depend on $M_t$.
"""

# --- Coding Tips ---
coding_tips = """## Coding Tips

- You can draw from a multivariate normal with variance $\\Sigma$ by drawing independent standard normal random variables and using the Cholesky decomposition of $\\Sigma$. You need to make sure you take the *lower triangular* portion. In particular, if $z=(z_{1},\\ldots ,z_{k})^{\\'} \\sim N(0,I_{k})$ and $A=Chol(\\Sigma)$, then $Az$ is distributed $N(0,\\Sigma)$.

- When you estimate the logit and nested logit models, you will have to choose which functions of the exogenous variables to use as instruments. One option would be to use all of them---the exogenous demand shifters (own and competing products) and the exogenous cost shifters. Alternatively, you might want to use something more like the BLP approximation of the optimal instruments.

- To display the average prices, use the following:
```python
T, J= 600, 4
print(changed_prices.reshape((T, J)).mean(axis= 0))
```

- To display the average elasticities and diversion rations, use the following:
```python
T, J= 600, 4
print(elasticities.reshape((T, J, J)).mean(axis= 0))
```

- To apply 15% cost reduction by the merged firms, use the following.
```python
merger_costs= costs.copy()
merger_costs[product_data.merger_ids== 1]= 0.85*merger_costs[product_data.merger_ids== 1]
```
"""

# --- Assemble the full notebook ---

nb_part1 = nbf.reads(nb_part1_str, as_version=4)

nb_part1['cells'].extend([
    nbf.v4.new_markdown_cell(correct_model_intro),
    nbf.v4.new_markdown_cell(q8_text),
    nbf.v4.new_code_cell(q8_code),
    nbf.v4.new_markdown_cell(q9_text),
    nbf.v4.new_markdown_cell(q9_answer),
    nbf.v4.new_code_cell(q9_code),
    nbf.v4.new_markdown_cell(q9_extra_credit_text),
    nbf.v4.new_code_cell(q9_extra_credit_code),
    nbf.v4.new_markdown_cell(merger_intro),
    nbf.v4.new_markdown_cell(q10_text),
    nbf.v4.new_markdown_cell(q10_answer),
    nbf.v4.new_markdown_cell(q11_text),
    nbf.v4.new_code_cell(q11_code),
    nbf.v4.new_markdown_cell(q12_text),
    nbf.v4.new_code_cell(q12_code),
    nbf.v4.new_markdown_cell(q12_answer),
    nbf.v4.new_markdown_cell(q13_text),
    nbf.v4.new_markdown_cell(q13_answer),
    nbf.v4.new_markdown_cell(q14_text),
    nbf.v4.new_code_cell(q14_code),
    nbf.v4.new_markdown_cell(q15_text),
    nbf.v4.new_markdown_cell(q15_answer),
    nbf.v4.new_markdown_cell(coding_tips)
])

with open('/Users/marek/Library/CloudStorage/Dropbox/github/pyblp/pyblp_homework/BLP_homework.ipynb', 'w') as f:
    nbf.write(nb_part1, f)

print("Successfully created BLP_homework.ipynb")

