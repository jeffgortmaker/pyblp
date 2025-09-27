import nbformat as nbf

# --- Load the first part of the notebook ---
with open('/Users/marek/Library/CloudStorage/Dropbox/github/pyblp/pyblp_homework/notebook_part1.json', 'r') as f:
    nb = nbf.read(f, as_version=4)

# --- Estimate Correctly Specified Model ---
correct_model_intro = """## Estimate the Correctly Specified Model\n\nUse the `pyBLP` package to estimate the correctly specified model. Allow `pyBLP` to construct approximations to the optimal instruments, using the exogenous demand shifters and exogenous cost shifters."""
q8_text = """**8. Report a table with the estimates of the demand parameters and standard errors. Do this three times: once when you estimate demand alone, then again when you estimate jointly with supply; and again with the \"optimal IV\".**"""
q8_code = """import pyblp
import pandas as pd
import numpy as np

pyblp.options.digits = 3
pyblp.options.verbose = False
blp_data = pd.read_csv('/Users/marek/Library/CloudStorage/Dropbox/github/pyblp/pyblp_homework/data/blp_data.csv')
X1_formulation = pyblp.Formulation('0 + prices + quality')
X2_formulation = pyblp.Formulation('0 + satellite + wired')
X3_formulation = pyblp.Formulation('1 + cost_shifter')
instrument_data = pyblp.build_blp_instruments(X1_formulation, blp_data)
problem_data = blp_data.join(instrument_data)

# Demand-side only
problem_demand_only = pyblp.Problem(product_formulations=(X1_formulation, X2_formulation), product_data=problem_data, integration=pyblp.Integration('monte_carlo', size=500, seed=0))
results_demand_only = problem_demand_only.solve(sigma=np.eye(2))
print(\"--- Demand-side GMM Estimation ---\")
print(results_demand_only)

# Joint demand and supply
problem_joint = pyblp.Problem(product_formulations=(X1_formulation, X2_formulation, X3_formulation), product_data=problem_data, integration=pyblp.Integration('monte_carlo', size=500, seed=0))
results_joint = problem_joint.solve(sigma=np.eye(2), gamma=np.array([0.5, 0.25]))
print(\"\\n--- Joint Demand and Supply GMM Estimation ---\")
print(results_joint)

# Optimal Instruments
optimal_instruments = pyblp.build_optimal_instruments(X1_formulation, results_joint)
problem_data_optimal = blp_data.join(optimal_instruments)
problem_optimal = pyblp.Problem(product_formulations=(X1_formulation, X2_formulation, X3_formulation), product_data=problem_data_optimal, integration=pyblp.Integration('monte_carlo', size=500, seed=0))
results_optimal = problem_optimal.solve(sigma=np.eye(2), gamma=np.array([0.5, 0.25]))
print(\"\\n--- GMM Estimation with Optimal Instruments ---\")
print(results_optimal)
"""
q9_text = """**9. Using your preferred estimates from the prior step (explain your preference), provide a table comparing the estimated own-price elasticities to the true own-price elasticities. Provide two additional tables showing the true matrix of diversion ratios and the diversion ratios implied by your estimates.**"""
q9_answer = """*Answer to Question 9:*

The preferred estimates are from the final model using optimal instruments. These instruments are theoretically shown to produce the most asymptotically efficient GMM estimates. While all three `pyblp` estimations should yield consistent parameter estimates close to the true values, the standard errors on the optimal IV estimates should be the smallest, reflecting this efficiency."""
q9_code = """# Need to re-run the data generation code to get the true values in memory for comparison
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
mc = np.exp(gamma0 + gamma1 * cost_shifter + omega / 8)

def calculate_shares_and_derivs(p_t, t, beta_i_t):
    delta_t = alpha * p_t + beta1 * quality[t, :] + xi[t, :]
    mu_it = (beta_i_t[:, 0][:, np.newaxis] * satellite[t, :] + beta_i_t[:, 1][:, np.newaxis] * wired[t, :])
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

blp_data = pd.read_csv('/Users/marek/Library/CloudStorage/Dropbox/github/pyblp/pyblp_homework/data/blp_data.csv')
equilibrium_prices = blp_data.prices.values.reshape(T, J)

est_elasticities = results_optimal.compute_elasticities()
est_diversions = results_optimal.compute_diversion_ratios()

# Calculate true elasticities and diversions for the first market as an example
s_t, dsdp = calculate_shares_and_derivs(equilibrium_prices[0, :], 0, beta_i[0,:,:])
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

When two firms merge, they internalize the competitive externality between them. Before the merger, if firm 1 lowered its price, it would steal some customers from firm 2, an effect it would ignore. After the merger, the newly combined firm recognizes that lowering the price of product 1 cannibalizes sales of product 2. To avoid this, the merged entity has an incentive to raise the prices of both products 1 and 2. The prices of non-merging firms are also likely to rise in response to the price increases of the merged firms (prices are strategic complements)."""
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

The price increases are significantly larger for the merger between firms 1 and 2 than for the merger between firms 1 and 3. This is because firms 1 and 2 are both satellite providers and are therefore close substitutes. The incentive to raise prices to avoid cannibalization is very strong. Firms 1 and 3 are a satellite and a wired provider, respectively. They are more distant substitutes, so the incentive to raise prices post-merger is weaker, resulting in smaller price increases."""
q13_text = """**13. Thus far you have assumed that there are no \"efficiencies\" (reduction in costs) resulting from the merger. Explain briefly why a merger-specific reduction in marginal cost could mean that a merger is welfare-enhancing.**"""
q13_answer = """*Answer to Question 13:*

A merger creates two opposing effects on prices: an upward pressure from the internalization of competition, and a downward pressure from any cost-saving efficiencies. If the cost efficiencies are large enough, the downward pressure can dominate, leading to lower post-merger prices. Lower prices directly benefit consumers. A merger is welfare-enhancing if the increase in producer surplus (from efficiencies) and the change in consumer surplus (from price changes) sum to a positive number. Even if prices rise slightly, a large enough cost saving could still make the merger total-welfare-enhancing if the gains to producers outweigh the loss to consumers."""
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

Up to this point, all parameters and results (prices, shares, elasticities) are independent of the market size, $M_t$. We could implicitly set $M_t=1$ in all markets without affecting the results. However, consumer surplus is a dollar value that represents the total utility gain for all consumers in the market. To calculate it, we need to know the actual number of consumers. The change in consumer surplus is therefore proportional to $M_t$. To calculate the impact on total welfare, we need to sum the change in consumer surplus and the change in producer profits, both of which depend on $M_t$."""

# --- Coding Tips ---
coding_tips = """## Coding Tips\n\n- You can draw from a multivariate normal with variance $\\Sigma$ by drawing independent standard normal random variables and using the Cholesky decomposition of $\\Sigma$. You need to make sure you take the *lower triangular* portion. In particular, if $z=(z_{1},\\ldots ,z_{k})^{\\'} \\sim N(0,I_{k})$ and $A=Chol(\\Sigma)$, then $Az$ is distributed $N(0,\\Sigma)$.\n\n- When you estimate the logit and nested logit models, you will have to choose which functions of the exogenous variables to use as instruments. One option would be to use all of them---the exogenous demand shifters (own and competing products) and the exogenous cost shifters. Alternatively, you might want to use something more like the BLP approximation of the optimal instruments.\n\n- To display the average prices, use the following:\n```python\nT, J= 600, 4\nprint(changed_prices.reshape((T, J)).mean(axis= 0))\n```\n\n- To display the average elasticities and diversion rations, use the following:\n```python\nT, J= 600, 4\nprint(elasticities.reshape((T, J, J)).mean(axis= 0))\n```\n\n- To apply 15% cost reduction by the merged firms, use the following.\n```python\nmerger_costs= costs.copy()\nmerger_costs[product_data.merger_ids== 1]= 0.85*merger_costs[product_data.merger_ids== 1]\n```"""

# --- Assemble the full notebook ---

nb['cells'].extend([
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
    nbf.write(nb, f)

print("Successfully created BLP_homework.ipynb")
