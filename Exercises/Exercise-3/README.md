## Introduction

This is the last of three exercises that will give you a solid foundation for doing BLP-style estimation. We'll continue with the same running example: what if we halved an important product's price? Our goal today is to further relax some of the unrealistic substitution patterns that we were forced to bake-in to our model because of limited cross-market variation. To do so, we will use results from consumer surveys to introduce within-market variation.

## Setup

We'll be continuing where we left off after the [second exercise](https://github.com/Mixtape-Sessions/Demand-Estimation/blob/main/Exercises/Exercise-2/README.md). You should just keep adding to your code, using [its solutions](https://github.com/Mixtape-Sessions/Demand-Estimation/blob/main/Exercises/Exercise-2/Solutions.ipynb) if you had trouble with some parts.

## Data

Today, you won't be including any new datasets into estimation, just a few carefully-chosen summary statistics from a couple of imaginary consumer surveys. These could have been from industry reports, from surveys administered by you, the researcher, or from any number of other places.

The first survey randomly sampled consumers who purchased breakfast cereal in markets `C01Q1` and `C01Q2`, i.e. during the first two quarters in the first city covered by the product data. It elicited information about income. The second survey was another random sample of the same consumers, but it asked questions about second choice diversion.

Survey Name | Observations | Statistics
----------- | ------------ | ----------
"Income"    | 100          | Estimated mean log of quarterly income was `7.9`.
"Diversion" | 200          | When asked what they would have done had their chosen cereal not been available, `28%` said they would have purchased no cereal in the covered product data. Out of all respondents, `31%` both purchased a mushy cereal and would have purchased another mushy cereal had it not been available.

In today's exercise, we will match these three statistics to introduce some additional dimensions of heterogeneity into our BLP model of demand for cereal and see how our counterfactual changes.

## Questions

### 1. Use the income statistic to match a parameter on log income

In the last exercise, we were unable to estimate a parameter on log income $y_{it}$ alone because market fixed effects eliminate needed cross-market income variation. Instead, we simply assumed that this parameter is zero, i.e. that income does not shift individuals' preference for cereal in general one way or another. Today, we'll incorporate a micro moment $\mathbb{E}[y_{it} | j > 0] = 7.9$ to estimate this parameter and see whether this assumption was reasonable.

To do so, we'll need first re-create our problem with a constant in our formulation for `X2`. This is because the parameter we want to add will be on the interaction between a constant from the product data and log income in the agent data. Your `X2` formulation should now be `1 + mushy + prices`.

After re-initializing our problem with the extra constant, we need to configure our micro moment. To do so, you'll have to configure three objects in the following order.

1. Define a [`pyblp.MicroDataset`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.MicroDataset.html) $d$.
2. Define a [`pyblp.MicroPart`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.MicroPart.html) $p$.
3. Define a [`pyblp.MicroMoment`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.MicroPart.html) $m$.

First, define a [`pyblp.MicroDataset`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.MicroDataset.html) $d$ to represent the "income" survey. Choose a useful `name`, set the number of `observations=100` from the survey, and set the `market_ids` to the list of market IDs that the survey covers. You also need to specify a function `compute_weights`, which defines how to compute sampling weights $w_{dijt}$. You should look at the [`pyblp.MicroDataset`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.MicroDataset.html) documentation to understand how this function works. Within the two surveyed markets, the survey did not differentially select consumers except to only select those who purchased a cereal, $j \neq 0$. So our sampling weights should be $w_{dijt} = 1\\{j \neq 0\\}$. To implement this, you should define a function that for market $t$ returns a $I_t \times J_t$ matrix of ones. Equivalently, your function could return a $I_t \times (1 + J_t)$ matrix with the first column being zeros, corresponding to $j = 0$, and the final $J_t$ columns being ones, corresponding to $j \neq 0$. Not having a first column is just convenient PyBLP shorthand for setting it to zeros. Your function should look like

```python
compute_weights=lambda t, p, a: np.ones((a.size, p.size))
```

The arguments required for your function are `t`, the market ID, `p`, the [`Products`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.Products.html#pyblp.Products) subsetted to this market, and `a`, the [`Agents`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.Agents.html#pyblp.Agents) subsetted to this market. We'll just return a matrix of the size required by the `compute_weights` argument.

Next, define a [`pyblp.MicroPart`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.MicroPart.html) $p$ to represent the expectation $\mathbb{E}[y_{it} | j > 0]$. Choose a useful `name`, specify the configured `dataset`, and also specify a second function `compute_values`, which defines how to compute micro values $v_{pijt}$. The function has the same arguments and output size as `compute_weights` above, except here we want to have a matrix where the same $y_{it}$ value is repeated $J_t$ times for each column. One convenient way to do this is with the [`np.einsum`](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html) function, although there are many other ways, such as using [`np.repeat`](https://numpy.org/doc/stable/reference/generated/numpy.repeat.html) or [`np.tile`](https://numpy.org/doc/stable/reference/generated/numpy.tile.html).

```python
compute_values=lambda t, p, a: np.einsum('i,j', a.demographics[:, 0], np.ones(p.size))
```

Here, we're just selecting the first (and only) column of demographics, log income, and repeating it for as many products as there are in the market.

Finally, define a [`pyblp.MicroMoment`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.MicroMoment.html) $m$. Choose a useful `name`, specify the observed `value=7.9` that we want to match, and for the `parts` argument, you can just pass the above configured micro part. This will specify the identity function $f_m(v) = v$. You would use the other arguments `compute_value` and `compute_gradient` if you wanted to specify a more complicated function $f_m(\cdot)$ and its gradient. In this case, you could specify a list of micro `parts` and these additional functions would select from these.

Given our micro moment, say `income_moment`, we can just pass a list with it as the only element to `micro_moments=[income_moment]` in [`.solve`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.Problem.solve.html). Since our `X2` configuration has an additional column for the constant, our `sigma` matrix needs another column/row, and our `pi` matrix needs another row. Set the new elements in both equal to zeros, except for the one in $\Pi$ corresponding to the interaction between the new constant in `X2` and log income, which you can set to some nonzero initial value, say `1`. In practice, you'll want to try out multiple random starting values.

Again, we're just identified, so we should get an approximately zero objective at the optimum. If we saw "marching down the gradient" and have a near-zero gradient norm and positive Hessian eigenvalues at the optimum, we can look at our estimates. You should get a new parameter estimate of around `-0.331`. Does the new parameter estimate suggest that the original assumption of it being zero was fairly okay, or not?

### 2. Use the diversion statistics to estimate unobserved preference heterogeneity for a constant and mushy

In the last exercise, we were also unable to estimate parameters in $\Sigma$ on a constant and `mushy` because there was no cross-market variation in the number of products or `mushy`. Instead, we again simply assumed these parameters were zero, and all the preference heterogeneity was from income. We'll incorporate the second choice moments $\mathbb{P}(k = 0 | j > 0) = 0.28$ and $\mathbb{P}(\text{mushy}_j \text{ and } \text{mushy}_k | j > 0) = 0.31$ to estimate these parameters.

Since these numbers come from a different survey, you should define a new [`pyblp.MicroDataset`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.MicroDataset.html). If the latter two statistics were instead based on the same responses, we should be defining all our micro moments on the same dataset. When defining `compute_weights`, the one difference from the income micro moment is that to include information about second choices, your function now needs to return an array with three dimensions, not a two-dimensional matrix, in order to define weights $w_{dijkt}$, which now have an additional index $k$. The last dimension is what defines $k$. To allow for selecting individuals who select $k = 0$, the final dimension should be of size $1 + J_t$, for an array of dimensions $I_t \times J_t \times (1 + J_t)$. Again, it should just be an array of all ones, since beyond the two markets, there was no differential selection of consumers, except that they chose cereal.

Then, you should define two new [`pyblp.MicroPart`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.MicroPart.html)s, one for each of the statistics, and two new [`pyblp.MicroMoment`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.MicroMoment.html)s in the same way as before. Both are also just averages on the micro dataset, not complicated functions, so these should look similar to the last micro moment. The main difference is that `compute_values` will now return a $I_t \times J_t \times (1 + J_t)$ array, with ones and zeros to choose micro values $v_{pijkt}$ that implement the desired statistics.

When solving the problem, we just append the two new micro moments to our `micro_moments` list, set the new parameters in `sigma` to nonzero initial values, and re-optimize. Second choice computations can take some time, up to a few minutes.

After confirming that optimization seemed to have been successful, interpret the new parameters. How large is the unobserved preference heterogeneity for the inside (or equivalently, the outside) option. How large is it for the mushy characteristic?

### 3. Evaluate changes to the price cut counterfactual

Using the new estimates, re-run the same price cut counterfactual from the last two exercises. Re-compute percent changes and compare with those from day 2. Do substitution patterns and cannibalization estimates now look more reasonable?

## Supplemental Questions

These additional questions will go beyond just defining micro moments, and will be useful to think about when doing BLP-style estimation in your own work.

### 1. See how your market size assumption affects results

In the first exercise, we made a somewhat arbitrary assumption about the size of the market. Vary this assumption, for example by assuming that the potential market size is *two* servings per day per individual in the city, instead of just one. Re-compute your market shares and re-estimate the model. See how the results of your price counterfactual change when you have a parameter in $\Sigma$ on the constant, and when you assume that parameter is zero. In particular, compute percent changes to the counterfactual outside share $s_{0t}$ and see how that changes.

In general, outside diversion will scale with the assumed potential market size, unless we include sufficient preference heterogeneity, particularly for the outside option. Directly matching a moment to pin down diversion to the outside option is a fairly clear way to estimate what diversion to the outside option should look like in a counterfactual.

### 2. Simulate some micro data and use it to match optimal micro moments

This exercise doesn't come with a full dataset of consumer demographics and their choices, only with a few summary statistics, but we can simulate some to get a sense for how one might use all the information in a full micro dataset via optimal micro moments. To simulate some micro data, take one of your estimated models and use the [`.simulate_micro_data`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.ProblemResults.simulate_micro_data.html) method, using your configured "Income" `dataset` and setting a `seed`. You may want to use [`pyblp.data_to_dict`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.data_to_dict.html) to get the simulated data into a format that you can pass to create a more easily-usable [`pd.DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html).

The simulated data should look like the assumed micro data in the "Income" survey, with one exception. The `agent_indices` column corresponds to the within-market row index of individual types $i_n$ in your `agent_data`. This includes information about unobserved preference heterogenity, which the real micro dataset wouldn't have. Compute the same `agent_indices` in your agent data using [`.groupby`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html) and [`.cumcount`](https://pandas.pydata.org/pandas-docs/version/1.3/reference/api/pandas.core.groupby.GroupBy.cumcount.html), and merge `log_income` into your simulated micro dataset. You can then drop the `agent_indices` column. The `choice_indices` column just represents the within-market row index of the respondent's choice $j_nt$, which is presumably observed in the "Income" dataset.

The resulting data should be in the same format as needed by [`.compute_micro_scores`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.ProblemResults.compute_micro_scores.html). You'll just need to tell PyBLP how to over the unobserved preference heterogeneity you just dropped. One option is to use the `integrate` argument, but you can also replicate each row in your micro data for as many draws as you want, add a `weights` column equal to one over the number of draws, and add `nodes0`, `nodes1`, and `nodes2` columns with standard normal draws. These two options will do the same thing, if you use the `monte_carlo` specification when configuring your integration configuration.

Given an estimated model and some micro data, this function computes the score (the derivative of the log likelihood of each micro observation with respect to the model's nonlinear parameters) of each micro data observation. Specifically, the scores are returned as a list, one element for each parameter in [`.theta`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.ProblemResults.html#pyblp.ProblemResults.theta). For each element of `theta`, create a new micro moment that matches the mean score in the micro data.

In order to do so, you'll have to specify `compute_values` in the corresponding [`pyblp.MicroPart`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.MicroPart.html). Each $v_{pijt}$ should equal the score of a consumer of type $i$ who chooses $j$ in market $t$. PyBLP has another convenient function for computing these: [`.compute_agent_scores`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.ProblemResults.compute_agent_scores.html). After passing your micro dataset to this function (and also configuring `integration`), it also return a list, one for each element of `theta`. Each element in the list is a dictionary mapping market IDs to the matrix that you need to pass to `compute_values`. In this function, you can use the `t` argument to directly select the right matrix.

One approach would be to replace the single $\mathbb{E}[y_{it} | j > 0]$ micro moment from this dataset with all the optimal micro moments from this dataset. But to keep results comparable with before (and to maintain a just-identified model), try replacing this sub-optimal micro moment with the optimal micro moment based on scores for the parameter in $\Pi$ that this original micro moment was supposed to target. Do results change much? What does this indicate?

### 3. Use a within-firm diversion ratio to estimate a nesting parameter

One dimension of preference heterogeneity that we have not modelled is within firm. In our price cut counterfactual, beyond mushyness and prices, we do not see more substitution within firm than across firms. However, there are good reasons to think that we might see more substitution within firm in reality. Consumers tend to be loyal to firms, and may prefer some firms to others for reasons that aren't captured by our other observed characteristics.

Typically, a good way to estimate preference heterogeneity for a categorical variable is to have a separate random coefficient on a dummy variable for each category. We have done this for the categorical mushy category. However, for some categorical variables with *many* different categories, adding this many random coefficients would be computationally prohibitive, and there may not be enough variation in the data to do so. In our data, a dummy on each firm would be a lot of additional random coefficients.

Instead, a common choice is to estimate a parameter that governs *within category* correlation of the idiosyncratic preferences $\varepsilon_{ijt}$. These categories are called nests, $h$ in PyBLP notation, and the correlation parameter is called a nesting parameter, $\rho$ in PyBLP notation. Without any random coefficients, we have the nested logit model. With random coefficients, we have the random coefficients nested logit (RCNL) model. See the [RCNL](https://pyblp.readthedocs.io/en/stable/background.html#random-coefficients-nested-logit) part of the PyBLP documentation for more details.

Using aggregate variation, it is common to target a nesting parameter with an instrument that, for each product $j$, counts the number of other products in the same nest in the same market $t$. However, since we have no cross-market variation in this instrument, this is not a particularly credible way to identify $\rho$, much in the same way that without cross-market variation, we have little hope of credibly identifying the parameters in $\Sigma$. Indeed, a nesting structure is just a very particular type of random coefficient!

Instead, we will match a within-firm diversion ratio. Assume that in our diversion survey, we have a third statistic: $\mathbb{P}(f(j) = f(k)) = 0.35$. That is, in the survey, 35% of respondents said they would select a product made by the same firm had their first choice cereal been unavailable. When setting up your problem, create a new column `nesting_ids` equal to firm IDs in your `product_ids`. Then when solving the problem, add an additional micro moment that matches this share, and set some nonzero initial value for the `rho` parameter (it needs to be between 0 and 1). Optimization may take a while because many of the numerical tricks PyBLP uses to make estimation fast don't work when there's a nesting parameter (particularly with second choices). Re-run the counterfactual and see whether it seems more reasonable, paying close attention to changes to within-firm cannibalization vs. cross-firm substitution.
