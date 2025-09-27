## Introduction

This is the first of three exercises that will give you a solid foundation for doing BLP-style estimation. The running example is the same as in lecture: what if we halved an important product's price?

## Setup

Most of the computational heavy-lifting in these exercises will be done by the open-source Python package [PyBLP](https://github.com/jeffgortmaker/pyblp). It is easiest to use PyBLP in Python, and the hints/solutions for the exercises will be given in Python. But for those who are more familiar with R, it is straightforward to [call PyBLP from R](https://github.com/jeffgortmaker/pyblp#other-languages) with the [reticulate](https://rstudio.github.io/reticulate/) package. It is technically possible to call PyBLP from other languages like Julia and MATLAB, but most users either use Python or R.

You should install PyBLP on top of the [Anaconda Distribution](https://www.anaconda.com/). Anaconda comes pre-packaged with all of PyBLP's dependencies and many more Python packages that are useful for statistical computing. Steps:

1. [Install Anaconda](https://docs.anaconda.com/free/anaconda/install/) if you haven't already. You may wish to [create a new environment](https://docs.anaconda.com/free/anacondaorg/user-guide/work-with-environments/) for just these exercises, but this isn't strictly necessary.
2. [Install PyBLP](https://github.com/jeffgortmaker/pyblp#installation). On the Anaconda command line, you can run the command `pip install pyblp`.

If you're using Python, you have two broad options for how to do the coding exercises.

- Use a [Jupyter Notebook](https://jupyter.org/install#jupyter-notebook). The solutions to each exercise will be in a notebook. In general, notebooks are a good way to weave text and code for short exercises, and to distribute quick snippets of code with others.
- Use an integrated development environment (IDE). Once you get beyond a few hundred lines of code, I strongly recommend using an IDE and not notebooks. For Python, I recommend [VS Code](https://code.visualstudio.com/) or [PyCharm](https://www.jetbrains.com/pycharm/). The former is free and the latter has a free community edition with all the features you'll need for standard Python development. Both [integrate well](https://docs.anaconda.com/free/anaconda/ide-tutorials/) with Anaconda.

If using a notebook, you can right click and save the following notebook template: [notebook.ipynb](https://github.com/Mixtape-Sessions/Demand-Estimation/raw/main/Exercises/Templates/notebook.ipynb). If using an IDE, you can right click and save the following script template: [script.py](https://github.com/Mixtape-Sessions/Demand-Estimation/raw/main/Exercises/Templates/script.py). Both import various packages used throughout the exercise.

```python
import pyblp
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
```

The notebook additionally configures these packages to reduce the amount of information printed to the screen.

```python
pyblp.options.digits = 3
pyblp.options.verbose = False
pd.options.display.precision = 3
pd.options.display.max_columns = 50

import IPython.display
IPython.display.display(IPython.display.HTML('<style>pre { white-space: pre !important; }</style>'))
```

Finally, both show how to load the data that we'll be using today.

## Data

Today, you'll use the [`products.csv`](https://github.com/Mixtape-Sessions/Demand-Estimation/raw/main/Exercises/Data/products.csv) dataset, which is a simplified version of [Nevo's (2000)](https://nbviewer.org/github/Mixtape-Sessions/Demand-Estimation/raw/main/Readings/5-Nevo-2000.pdf) fake cereal data with less information and fewer derived columns. The data were motivated by real grocery store scanner data, but due to the proprietary nature of this type of data, the provided data are not entirely real. This dataset has been used as a standard example in much of the literature on BLP estimation.

Compared to typical datasets you might use in your own work, the number of observations in this example dataset is quite small. This helps with making these exercises run very fast, but in practice one would want more data than just a couple thousand data points to estimate a flexible model of demand. Typical datasets will also include many more product characteristics. This one only includes a couple to keep the length of the exercises manageable.

The data contains information about 24 breakfast cereals across 94 markets. Each row is a product-market pair. Each market has the same set of breakfast cereals, although with different prices and quantities. The columns in the data are as follows.

Column              | Data Type | Description
------------------- | --------- | -----------
`market`            | String    | The city-quarter pair that defines markets $t$ used in these exercises. The data were motivated by real cereal purchase data across 47 US cities in the first 2 quarters of 1988.
`product`           | String    | The firm-brand pair that defines products $j$ used in these exercises. Each of 5 firms produces between 1 and 9 brands of cereal.
`mushy`             | Binary    | A dummy product characteristic equal to one if the product gets soggy in milk.
`servings_sold`     | Float     | Total quantity $q_{jt}$ of servings of the product sold in a market, which will be used to compute market shares.
`city_population`   | Float     | Total population of the city, which will be used to define a market size.
`price_per_serving` | Float     | The product's price $p_{jt}$ used in these exercises.
`price_instrument`  | Float     | An instrument to handle price endogeneity in these exercises. Think of it as a cost-shifter, a Hausman instrument, or any other valid IV that we discussed in class.

Throughout the exercises, we use these data to estimate an increasingly flexible BLP-style model of demand for cereal. We will use predictions from this model to see how our running example, cutting the price of one cereal, affects demand for that cereal and for its substitutes.

## Questions

### 1. Describe the data

You can download `products.csv` from [this link](https://github.com/Mixtape-Sessions/Demand-Estimation/raw/main/Exercises/Data/products.csv). To load it, you can use [`pd.read_csv`](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html). To look at a random sample of its rows, you can use [`.sample`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sample.html). To compute summary statistics for different columns, you can use [`.describe`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html). Throughout these exercises, you'll be given links to functions and methods that can be used to answer the questions. If you're unsure about how to use them, you should click on the link, where there is typically example code lower down on the page.

### 2. Compute market shares

To transform observed quantities $q_{jt}$ into market shares $s_{jt} = q_{jt} / M_t$, we first need to define a market size $M_t$. We'll assume that the potential number of servings sold in a market is the city's total population multiplied by 90 days in the quarter. Create [a new column](https://pandas.pydata.org/docs/getting_started/intro_tutorials/05_add_columns.html) called `market_size` equal to `city_population` times `90`. Note that this assumption is somewhat reasonable but also somewhat arbitrary. Perhaps a sizable portion of the population in a city would never even consider purchasing cereal. Or perhaps those who do tend to want more than one serving per day. In the third exercise, we'll think more about how to discipline our market size assumption with data.

Next, compute a new column `market_share` equal to `servings_sold` divided by `market_size`. This gives our market shares $s_{jt}$. We'll also need the outside share $s_{0t} = 1 - \sum_{j \in J_t} s_{jt}$. Create a new column `outside_share` equal to this expression. You can use [`.groupby`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html) to group by market and [`.transform('sum')`](https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.transform.html) to compute the within-market sum of inside shares. Compute summary statistics for your inside and outside shares. If you computed market shares correctly, the smallest outside share should be $s_{0t} \approx 0.305$ and the largest should be $s_{0t} \approx 0.815$.

### 3. Estimate the pure logit model with OLS

Recall the pure logit estimating equation: $\log(s_{jt} / s_{0t}) = \delta_{jt} = \alpha p_{jt} + x_{jt}' \beta + \xi_{jt}$. First, create a new column `logit_delta` equal to the left-hand side of this expression. You can use [`np.log`](https://numpy.org/doc/stable/reference/generated/numpy.log.html) to compute the log.

Then, use the package of your choice to run an OLS regression of `logit_delta` on a constant, `mushy`, and `price_per_serving`. There are many packages for running OLS regressions in Python. One option is to use the [formula interface for `statsmodels`](https://www.statsmodels.org/stable/example_formulas.html#ols-regression-using-formulas). To use robust standard errors, you can specify `cov_type='HC0'` in [`OLS.fit`](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.fit.html).

Interpret your estimates. Your coefficient on `price_per_serving` should be around `-7.48`. In particular, can you re-express your estimate on `mushy` in terms of how much consumers are willing to pay for `mushy`, using your estimated price coefficient?

### 4. Run the same regression with PyBLP

For the rest of the exercises, we'll use PyBLP do to our demand estimation. This isn't necessary for estimating the pure logit model, which can be done with linear regressions, but using PyBLP allows us to easily run our price cut counterfactual and make the model more flexible in subsequent days' exercises.

PyBLP requires that some key columns have specific names. You can use [`.rename`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rename.html) to rename the following columns so that they can be understood by PyBLP.

- `market` --> `market_ids`
- `product` --> `product_ids`
- `market_share` --> `shares`
- `price_per_serving` --> `prices`

By default, PyBLP treats `prices` as endogenous, so it won't include them in its matrix of instruments. But the "instruments" for running an OLS regression are the same as the full set of regressors. So when running an OLS regression and not account for price endogeneity, we'll "instrument" for `prices` with `prices` themselves. We can do this by creating a new column `demand_instruments0` equal to `prices`. PyBLP will recognize all columns that start with `demand_instruments` and end with `0`, `1`, `2`, etc., as "excluded" instruments to be stacked with the exogenous characteristics to create the full set of instruments.

With the correct columns in hand, we can initialize our [`pyblp.Problem`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.Problem.html). To specify the same R-style formula for our regressors, use [`pyblp.Formulation`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.Formulation.html). The full code should look like the following.

```python
ols_problem = pyblp.Problem(pyblp.Formulation('1 + mushy + prices'), product_data)
```

If you `print(ols_problem)`, you'll get information about the configured problem. There should be 94 markets (`T`), 2256 observations (`N`), 3 product characteristics (`K1`), and 3 total instruments (`MD`). You can verify that these instruments are simply the regressors by looking at `ols_problem.products.X1` and `ols_problem.products.ZD`, comparing these with `mushy` and `prices` in your dataframe. For the full set of notation used by PyBLP, which is very close to the notation used in the lectures, see [this page](https://pyblp.readthedocs.io/en/stable/notation.html).

To estimate the configured problem, use [`.solve`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.Problem.solve.html). Use `method='1s'` to just do 1-step GMM instead of the default 2-step GMM. In this case, this will just run a simple linear OLS regression. The full code should look like the following.

```python
ols_results = ols_problem.solve(method='1s')
```

Again, if you `print(ols_results)`, you'll get estimates from the logit model. Make sure that your estimates are the same as those you got from your OLS regression. If you used `'HC0'` standard errors like suggested above, you standard errors should also be the same.

### 5. Add market and product fixed effects

Since we expect price $p_{jt}$ to be correlated with unobserved product quality $\xi_{jt}$, we should be worried that our estimated $\hat{\alpha}$ on price is biased. Since we have multiple observations per market and product, and prices vary both across and within markets, it is feasible for us to add both market and product fixed effects. If $\xi_{jt} = \xi_j + \xi_t + \Delta\xi_{jt}$ and most of the correlation between $p_{jt}$ and $\xi_{jt}$ is due to correlation between $p_{jt}$ and either $\xi_j$ (product fixed effects) or $\xi_t$ (market fixed effects), then explicitly accounting for these fixed effects during estimation should help reduce the bias of our $\hat{\alpha}$.

The simplest way to add fixed effects is as dummy variables. We won't do this today, but for your own reference, you could do this either by making a separate column for each possible market and product fixed effects and adding these to your formulation, or you could use the shorthand `mushy + prices + C(market_ids) + C(product_ids)`. See [`pyblp.Formulation`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.Formulation.html) for different shorthands you can use. Since there are only 24 products and 94 markets for a total of 118 fixed effects, this approach is actually feasible in this case. But in a more realistic dataset with hundreds or thousands of products and markets, running an OLS regression with this many dummy variables starts to become computationally infeasible.

The alternative, which we'll do today, is to "absorb" the fixed effects. For a single fixed effect, we could just de-mean our outcome variable and each of our regressors within the fixed effects levels, and then run our regression. For multiple fixed effects, we need to *iteratively* de-mean. PyBLP does this automatically if you specify `absorb='C(market_ids) + C(product_ids)'` in your formulation instead of adding these as dummy variables.

Since `mushy` is always either 1 or 0 for the same product across different markets, it's collinear with product fixed effects, and you can drop it from your formula. Similarly, you can drop the constant. After dropping these, re-create your problem with absorbed fixed effects and re-solve it. Compare the new $\hat{\alpha}$ estimate with the last one. You should now be getting a coefficient on price of around `-28.6`. Does its change suggest that price was positively or negatively correlated with unobserved product-level/market-level quality?

### 6. Add an instrument for price

Adding market and product fixed effects can be helpful, but since unobserved quality typically varies by both product *and* market, we really want to instrument for prices. The data comes with a column `price_instrument` that we should interpret as a valid instrument for price that satisfies the needed exclusion restriction. It could be a cost-shifter, a valid Hausman instrument, or similar.

Before using it, we should first run a first-stage regression to make sure that it's a relevant instrument for price. To do so, use the same package you used above to run an OLS regression to run a second OLS regression of prices on `price_instrument` and your market and product fixed effects. If using the [formula interface for `statsmodels`](https://www.statsmodels.org/stable/example_formulas.html#ols-regression-using-formulas), you can use the same fixed effect shorthand as in PyBLP, with your full formula looking like `prices ~ price_instrument + C(market_ids) + C(product_ids)`. Does `price_instrument` seem like a relevant instrument for `prices`?

Now that we've checked relevance, we can set our `demand_instruments0` column equal to `price_instrument`, re-create the problem, and re-solve it. You should get a new coefficient on price of around `-30.6`. Does the change in $\hat{\alpha}$ suggest that price was positively or negatively correlated with $\Delta\xi_{jt}$ in $\xi_{jt} = \xi_j + \xi_t + \Delta\xi_{jt}$?

### 7. Cut a price in half and see what happens

Now that we have our pure logit model estimated, we can run our counterfactual of interest: what if we halved an important product's price? We'll select a single market, the most recent quarter in the first city: `C01Q2`. Create a new dataframe called `counterfactual_data` by [selecting](https://pandas.pydata.org/docs/getting_started/intro_tutorials/03_subset_data.html) data for just that market and inspect the data. We'll pretend that we're firm one, and deciding whether we want to cut the price of our brand four's product `F1B04`. In particular, we might be worried about *cannibalization*, i.e. how much this price cut will result in consumers of our other 8 brands of cereal in this market just substituting from their old choice to the new, cheaper cereal. Alternatively, we could be a regulator or academic interested in how taxing that product would affect demand in the market.

In your new dataframe with just data from `C01Q2`, create a `new_prices` column that is the same as `prices` but with the price of `F1B04` cut in half. To do this, you could use [`DataFrame.loc`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.loc.html). Then, use [`.compute_shares`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.ProblemResults.compute_shares.html) on your results from the last question, passing `market_id='C01Q2'` to only compute new market shares for our market of interest, and passing `prices=counterfactual_data['new_prices']` to specify that prices should be set to the new prices. This function will re-compute market shares at the changed prices implied by the model's estimates. Store them in a `new_shares` column.

Compute the percent change in shares for each product in the market. From firm one's perspective, do the estimates of cannibalization make sense. That is, do the signs on the percent changes for product `F1B04` and for other products make sense? Would you normally expect percent changes for other products to be different depending on how other products compare to the one whose price is being changed?

### 8. Compute demand elasticities

To better understand what's going on, use [`.compute_elasticities`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.ProblemResults.compute_elasticities.html), again specifying `market_id='C01Q2'`, to compute price elasticities for our market of interest. These measure what the model predicts will happen to demand in percentage terms when there's a 1% change in price of a product. The diagonal elements are own-price elasticities and the off-diagonal elements are cross-price elasticities. Does demand seem very elastic? Do the cross-price elasticities seem particularly reasonable?

## Supplemental Questions

These questions will not be directly covered in lecture, but will be useful to think about when doing BLP-style estimation in your own work.

### 1. Try different standard errors

By default, PyBLP computed standard errors that are robust to heteroskedasticity. But we may be concerned that unobserved quality $\xi_{jt}$ is systematically correlated across markets for a given product $j$, or across products for a given market $t$. Choose which one you think is more likely and try clustering your standard errors by that dimension. You can do this with `se_type='clustered'` in [`.solve`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.Problem.solve.html), for which you'll need a `clustering_ids` column in your product data. See how your standard error for $\alpha$ changes.

### 2. Compute confidence intervals for your counterfactual

Your estimate of $\hat{\alpha}$ comes with a standard error, but your counterfactual demand predictions don't. Ideally we'd like to not only have a point estimate for a counterfactual prediction, but also a measure (up to model misspecification) of how confident we are in these predictions. The easiest way to do this is with a "parametric bootstrap." The intuition is we can draw from the estimated asymptotic distribution of our $\hat{\alpha}$, and for each draw, re-compute demand, and see how demand responds to the same price cut.

You can do a parametric bootstrap with the [`.bootstrap`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.ProblemResults.bootstrap.html) method. Start with just a few draws (e.g., `draws=100`) and remember to set your `seed` so that you get the same draws every time you run the code. When new parameters are drawn, you get new [`.boostrapped_shares`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.BootstrappedResults.html#pyblp.BootstrappedResults.bootstrapped_shares), which take the place of your old `shares`. You can use the same [`.compute_shares`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.ProblemResults.compute_shares.html) method on the [`BootstrapedResults`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.BootstrappedResults.html) class, although you'll have to pass a `prices` argument with prices replicated along a new axis by as many draws as you have.

Once you have some bootstrapped shares, compute the same percent changes, and compute the 2.5th and 97.5th percentiles of these changes for each product. Are these 95% confidence intervals for your predictions particularly wide?

### 3. Impute marginal costs from pricing optimality

The canonical demand side of the BLP model assumes firms set prices in static Bertrand-Nash equilibrium. See [this section](https://pyblp.readthedocs.io/en/stable/background.html#supply) for a quick summary using PyBLP notation. Given an estimated demand model and such assumptions about pricing, we can impute marginal costs `c_{jt}`.

To do so, you first need to tell PyBLP what firms own what products. Create a new `firm_ids` column in your data, re-initialize your problem, and re-solve it. Then, you should be able to run the [`.compute_costs`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.ProblemResults.compute_costs.html) method to impute firms' marginal cost of producing each cereal. Do these marginal costs look particularly reasonable? How might limitations of your demand model and supply model bias them? What would they and observed prices imply about firms' markups and economic profits?

### 4. Check your code by simulating data

Even experienced software developers make a lot of mistakes when writing code. Writing "unit tests" or "integration tests" that check whether the code you've written seems to be working properly is incredibly important when writing complicated code to estimate demand. Perhaps the most useful test you can write when doing demand estimation (or most other types of structural estimation) is the following.

1. Simulate fake data under some true parameters.
2. Estimate your model on the simulated data and make sure that you can recover the true parameters, up to sampling error.

If you do these steps many times, the resulting Monte Carlo experiment will also give you a good sense for the finite sample statistical properties of your estimator.

PyBLP's [`Simulation`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.Simulation.html) class makes simulating data fairly straightforward. Its interface is similar to [`Problem`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.Problem.html), but you also specify your parameter estimates and structural errors. In addition to checking your code, you can also use this class for more complicated counterfactuals. After initializing your simulation, you can use [`.replace_endogenous`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.Simulation.replace_endogenous.html) to have PyBLP replace the prices $p_{jt}$ and market shares $s_{jt}$ with those that rationalize the chosen true parameters and other parts of the simulation. It does so by solving the pricing equilibrium. You'll have to pass your imputed marginal costs via the `costs` argument.

Initialize a simulation of the pure logit model with the same `product_data` and the same estimated `xi` but with an $\alpha$ somewhat different than the one you estimated. Make sure your chosen $\alpha$ is negative, otherwise demand will be upward sloping and PyBLP will have trouble solving for equilibrium prices. To the estimated [`.xi`](https://pyblp.readthedocs.io/en/latest/_api/pyblp.ProblemResults.html#pyblp.ProblemResults.xi) you can add the estimated fixed effects [`.xi_fe`](https://pyblp.readthedocs.io/en/latest/_api/pyblp.ProblemResults.html#pyblp.ProblemResults.xi_fe), since the simulation class does not support fixed effects absorption.

Have PyBLP solve for prices and market shares, and use the resulting data to re-estimate your pure logit regression. See if you can get an estimated $\hat{\alpha}$ close to the true one.
