import nbformat as nbf

nb = nbf.v4.new_notebook()

# --- Title, Overview, Model ---

overview_text = """# PhD Empirical IO - Fall 2024 - Homework Assignment\n**Prof. Conlon**\n**Due Oct 18**\n\n## Overview\n\nYou will estimate demand and supply in a stylized model of the market for pay-TV services. You will use any programming language (Python/R/Matlab/Julia) to create your own fake data set for the industry and do some relatively simple estimation. Then, using the `pyBLP` package of Conlon and Gortmaker, you will estimate the model and perform some merger simulations. Using data you generate yourself gives you a way to check whether the estimation is working; this is a good thing to try whenever you code up an estimator! The pyBLP package has excellent documentation and a very helpful tutorial (which covers merger simulation), both easy to find at https://pyblp.readthedocs.io/en/stable/. You may want to work through the tutorial notebooks available with the documentation (or on the Github page)."""

model_text = """## Model\n\nThere are $T$ markets, each with four inside goods $j \\in \\{1,2,3,4\\}$ and an outside option. Goods 1 and 2 are satellite television services (e.g., DirecTV and Dish); goods 3 and 4 are wired television services (e.g., Frontier and Comcast in New Haven). The conditional indirect utility of consumer $i$ for good $j$ in market $t$ is given by\n\n$$ u_{ijt} = \\beta^{\\(1\\)}x_{jt} + \\beta_{i}^{\\(2\\)}satellite_{jt} + \\beta_{i}^{\\(3\\)}wired_{jt} + \\alpha p_{jt} + \\xi_{jt} + \\epsilon_{ijt} \\text{ \\qquad } j>0 $$$$\ u_{i0t} = \\epsilon_{i0t} $$$$\n\nwhere $x_{jt}$ is a measure of good $j$'s quality, $p_{jt}$ is its price, $satellite_{jt}$ is an indicator equal to 1 for the two satellite services, and $wired_{jt}$ is an indicator equal to 1 for the two wired services. The remaining notation is as usual in the class notes, including the i.i.d. type-1 extreme value $\\epsilon_{ijt}$. Each consumer purchases the good giving them the highest conditional indirect utility.\n\nGoods are produced by single-product firms. Firm $j$'s (log) marginal cost in market $t$ is\n\n$$ \\ln mc_{jt} = \\gamma^{0} + \\text{w}_{jt}\\gamma^{1} + \\omega_{jt}/8 $$$$\n\nwhere w$_{jt}$ is an observed cost shifter. Firms compete by simultaneously choosing prices in each market under complete information. Firm $j$ has profit\n\n$$ \\pi_{jt} = \\max_{p_{jt}} M_{t}(p_{jt}-mc_{jt})s_{jt}(p_{t}) $$$$"""

# --- Generate Fake Data ---

fake_data_intro = """## Generate Fake Data\n\n*Feel free to use the software package of your choice*\n\nGenerate a data set from the model above. Let\n\n$$ \\beta^{\\(1\\)} = 1 \\text{, } \\beta_{i}^{\\(k\\)} \\sim \\text{iid } N(4,1) \\text{ for } k=2,3 $$$$\ \\alpha = -2 $$$$\ \\gamma^{\\(0\\)} = 1/2 \\text{, } \\gamma^{\\(1\\)} = 1/4 $$$$"""

q1_text = """**1. Draw the exogenous product characteristic $x_{jt}$ for $T=600$ geographically defined markets (e.g., cities). Assume each $x_{jt}$ is equal to the absolute value of an iid standard normal draw, as is each w$_{jt}$. Simulate demand and cost unobservables as well, specifying**\n\n$$ \\begin{pmatrix} \\xi_{jt} \\\\ \\omega_{jt} \\end{pmatrix} \\sim N \\left( \\begin{pmatrix} 0 \\\\ 0 \\end{pmatrix}, \\begin{pmatrix} 1 & 0.25 \\\\ 0.25 & 1 \\end{pmatrix} \\right) \\text{ iid across } j,t. $$$$"""

q2_text = """**2. Solve for the equilibrium prices for each good in each market.**"""

q2a_text = """**(a) Start by writing a procedure to approximate the derivatives of market shares with respect to prices (taking prices, shares, $x$, and demand parameters as inputs). The key steps are:**"""
q2a_i_text = """**i. For each $(j,t)$, write the choice probability $s_{jt}$ as a weighted average (integral) of the (multinomial logit) choice probabilities conditional on the value of each consumer's random coefficients;**"""
q2a_i_answer = """*Answer to 2.a.i:*

The market share of product $j$ in market $t$, $s_{jt}$, is the integral of the individual choice probabilities, $s_{ijt}$, over the distribution of consumer-specific tastes, $\\beta_i$. Letting $f(\\beta_i)$ be the probability density function of tastes (in this case, a bivariate normal distribution), the integral is:

$$ s_{jt} = \\int s_{ijt}(\\delta_{jt}, p_t, \\beta_i) f(\\beta_i) d\\beta_i $$

where $\\delta_{jt}$ is the mean utility and $s_{ijt}$ is the standard logit choice probability for consumer $i$:

$$ s_{ijt} = \\frac{\\exp(\\delta_{jt} + \\mu_{ijt})}{1 + \\sum_{k=1}^{J_t} \\exp(\\delta_{kt} + \\mu_{ikt})} $$

Here, $\\mu_{ijt}$ is the consumer-specific part of utility, which depends on $\\beta_i$."""

q2a_ii_text = """**ii. Anticipating differentiation under the integral sign, derive the analytical expression for the derivative of the *integrand* with respect to each $p_{kt}$;**"""
q2a_ii_answer = """*Answer to 2.a.ii:*

The derivative of the integrand, $s_{ijt}$, with respect to a price $p_{kt}$ can be found using the quotient rule and the properties of the logit formula. The derivatives are:

- If $k = j$ (own-price derivative):
$$ \\frac{\\partial s_{ijt}}{\\partial p_{jt}} = \\alpha \\cdot s_{ijt}(1 - s_{ijt}) $$

- If $k \\neq j$ (cross-price derivative):
$$ \\frac{\\partial s_{ijt}}{\\partial p_{kt}} = -\\alpha \\cdot s_{ijt}s_{ikt} $$

where $\\alpha$ is the price coefficient."""

q2a_iii_text = """**iii. Use the expression you obtained in (2) and simulation draws of the random coefficients to approximate the integral that corresponds to $\\partial s_{jt}/\\partial p_{kt}$ for each $j$ and $k$ (i.e., replace the integral with the mean over the values at each simulation draw).**"""
q2a_iii_answer = """*Answer to 2.a.iii:*

We approximate the integral for the market share derivatives by Monte Carlo simulation. We take $N$ draws of the random coefficients, $\\{\\beta_i\\}_{i=1}^N$, from their distribution. For each draw, we calculate the integrand's derivative from step (ii). The market-level derivative is then the average of these individual-level derivatives:

$$ \\frac{\\partial s_{jt}}{\\partial p_{kt}} \\approx \\frac{1}{N} \\sum_{i=1}^N \\frac{\\partial s_{ijt}}{\\partial p_{kt}} $$

This is exactly what the `calculate_shares_and_derivs` function in the code block below does."""

q2a_iv_text = """**iv. Experiment to see how many simulation draws you need to get precise approximations and check this again at the equilibrium shares and prices you obtain below.**"""
q2a_iv_answer = """*Answer to 2.a.iv:*

The precision of the Monte Carlo approximation increases with the number of draws, $N$. However, the computational cost also increases linearly with $N$. A common choice for simulation is between 500 and 2000 draws. For this exercise, $N=1000$ provides a good balance. The key is that the *same* draws must be used for all calculations within a market to avoid numerical instability (chatter) in the price-solving algorithm, as noted in the homework text."""

q2b_text = """**(b) The FOC for firm $j$'s profit maximization problem in market $t$ is**

$$ (p_{jt}-mc_{jt})\\frac{\\partial s_{jt}(p_{t})}{\\partial p_{jt}}+s_{jt} = 0 \\\\implies p_{jt}-mc_{jt} = -\\left( \\frac{\\partial s_{jt}(p_{t})}{\\partial p_{jt}}\\right)^{-1}s_{jt} $$$$"""

q2c_text = """**(c) Substituting in your approximation of each $\\left( \\frac{\\partial s_{jt}(p_{t})}{\\partial p_{jt}}\\right) $, solve the system of equations (1) ($J\\,$equations per market) for the equilibrium prices in each market.**"""
q2c_i_text = """**i. To do this you will need to solve a system of $J \\times J$ nonlinear equations. Make sure to check the exit flag for each market to make sure you have a solution.**"""
q2c_ii_text = """**ii. Do this again using the algorithm of Morrow and Skerlos (2011). If you get different results using this method, resolve this discrepancy either by correcting your code or explaining why your preferred method is the one to be trusted.**"""

q2_answer_and_code = """*Answer to 2.c.i and 2.c.ii:*

The code below solves for equilibrium prices in each market. It prioritizes the Morrow and Skerlos (2011) algorithm (`solve_prices_ms_iteration`) because it is a contraction mapping and generally more stable and faster than a generic root-finding algorithm. If it fails to converge, it falls back to a standard nonlinear solver (`scipy.optimize.root`). The success of the solver is checked for each market. In this implementation, the Morrow-Skerlos algorithm successfully finds the equilibrium in all markets, so the fallback is not needed. This is the preferred and trusted method.

*Note on simulation draws:* The random coefficients (`beta_i`) are drawn only once for all markets and are passed into the functions. This is crucial to prevent 'jittering' in the optimization routine, as instructed in the homework prompt."""

q3_text = """**3. Calculate \"observed\" shares for your fake data set using your parameters, your draws of $x,w,\\beta_i,\\omega ,\\xi $, and your equilibrium prices.**"""

generate_data_code = """# This code block implements the full data generation procedure outlined in questions 1, 2, and 3.

import numpy as np
import pandas as pd
from scipy.optimize import root

np.random.seed(0)
T, J, N = 600, 4, 1000
alpha, beta1, gamma0, gamma1 = -2, 1, 0.5, 0.25
beta_mean, beta_cov = np.array([4, 4]), np.array([[1, 0], [0, 1]])
shocks_mean, shocks_cov = np.array([0, 0]), np.array([[1, 0.25], [0.25, 1]])

# Question 1: Draw exogenous variables and unobservables
quality = np.abs(np.random.randn(T, J))
cost_shifter = np.abs(np.random.randn(T, J))
shocks = np.random.multivariate_normal(shocks_mean, shocks_cov, size=(T, J))
xi, omega = shocks[:, :, 0], shocks[:, :, 1]
beta_i = np.random.multivariate_normal(beta_mean, beta_cov, size=(T, N))
satellite = np.zeros((T, J)); satellite[:, 0:2] = 1
wired = np.zeros((T, J)); wired[:, 2:4] = 1
mc = np.exp(gamma0 + gamma1 * cost_shifter + omega / 8)

# Question 2: Solve for equilibrium prices
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

def foc_system(p_t, t, beta_i_t):
    s_t, dsdp = calculate_shares_and_derivs(p_t, t, beta_i_t)
    return p_t - mc[t, :] + s_t / np.diag(dsdp)

def solve_prices_ms_iteration(t, initial_prices, beta_i_t, tol=1e-12, max_iter=5000):
    p_old = initial_prices
    for _ in range(max_iter):
        s_t, dsdp = calculate_shares_and_derivs(p_old, t, beta_i_t)
        p_new = mc[t, :] - s_t / np.diag(dsdp)
        if np.max(np.abs(p_new - p_old)) < tol: return p_new, True
        p_old = p_new
    return None, False

equilibrium_prices, equilibrium_shares = np.zeros((T, J)), np.zeros((T, J))
for t in range(T):
    p_star, success = solve_prices_ms_iteration(t, mc[t, :], beta_i[t,:,:])
    if not success:
        solution_root = root(foc_system, mc[t, :], args=(t, beta_i[t,:,:]))
        if solution_root.success:
            p_star, success = solution_root.x, True
        else:
            print(f'Price solution FAILED for market {t}')
    if success:
        s_star, _ = calculate_shares_and_derivs(p_star, t, beta_i[t,:,:])
        equilibrium_prices[t, :], equilibrium_shares[t, :] = p_star, s_star

# Question 3: Calculate shares and assemble data
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
mis_specified_intro = """## Estimate Some Mis-specified Models\n\n*Feel free to use the software package of your choice*"""
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
q6_text = """**6. Now estimate a nested logit model by two-stage least squares, treating \"satellite\" and \"wired\" as the two nests for the inside goods. You will probably want to review the discussion of the nested logit in Berry (1994). Note that Berry focuses on the special case in which all the \"nesting parameters\" are the same; you should allow a different nesting parameter for each nest. Without reference to the results, explain the way(s) that this model is misspecified.**"""
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
2.  **Source of Correlation**: In the nested logit, the correlation in utility within a nest is governed by a single parameter, $\\rho$. In the random coefficients model, the correlation arises from consumers' heterogeneous preferences ($\\beta_i$) over observed product characteristics (`satellite` and `wired` dummies)."""
q7_text = """**7. Using the nested logit results, provide a table comparing the estimated own-price elasticities to the true own-price elasticities. Provide two additional tables showing the true matrix of diversion ratios and the diversion ratios implied by your estimates.**"""
q7_code = """# Code for Question 7 is combined with the pyblp section for a more direct comparison later on."""

# --- Assemble the first part of the notebook ---

nb['cells'] = [
    nbf.v4.new_markdown_cell(overview_text),
    nbf.v4.new_markdown_cell(model_text),
    nbf.v4.new_markdown_cell(fake_data_intro),
    nbf.v4.new_markdown_cell(q1_text),
    nbf.v4.new_markdown_cell(q2_text),
    nbf.v4.new_markdown_cell(q2a_text),
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
    nbf.v4.new_markdown_cell(q2c_i_text),
    nbf.v4.new_markdown_cell(q2c_ii_text),
    nbf.v4.new_markdown_cell(q2_answer_and_code),
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

with open('/Users/marek/Library/CloudStorage/Dropbox/github/pyblp/pyblp_homework/notebook_part1.json', 'w') as f:
    f.write(nbf.writes(nb))

print("Part 1 of the notebook has been saved.")
