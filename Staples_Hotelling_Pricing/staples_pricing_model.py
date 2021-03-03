import pandas as pd
from linearmodels import OLS
import numpy as np

# import data
data = pd.read_csv("StaplesData.csv", index_col=0)

# set marginal cost paramaters
c = 10

print(data.head())

# to find monopoly price, take derivative of expected profit / consumer
# with respect to price

covariates = ['NotNearCompetingStore', 'WealthyZipCode', 'Weekday', 'ExistingCustomer']

reg_formula = 'Purchase ~ 1 + Price * ({})'.format(' + '.join(covariates))
print('regression formula', reg_formula, '\n')

model = OLS.from_formula(reg_formula, data)
res = model.fit()
print(res.summary)

# store coeffs for later
beta_0 = res.params.Intercept
beta_1 = res.params[covariates]
beta_2 = res.params.Price
beta_3 = res.params[['Price:' + cov for cov in covariates]]

# consumer not near competing store
print(1-(beta_2 + beta_3['Price:NotNearCompetingStore'])/beta_2)
print(beta_2)
print(beta_2 + beta_3['Price:NotNearCompetingStore'])

"""
we see that conditional on other variables,
being far from a competing store reduces price responsiveness by
37.55% per dollar of price increase. This makes sense becasue the
further away you are from an alternative,
the more likely you are to purchase online.
"""

# consumer in wealthy zip code
print(1-(beta_2 + beta_3['Price:WealthyZipCode'])/beta_2)
print(beta_2)
print(beta_2 + beta_3['Price:WealthyZipCode'])

"""
we see that conditional on other variables,
being in a wealthy zip code reduces price responsiveness
by 29.54% per dollar of price increase. 
This makes sense as consumers with a relativly high income will
see a smaller change in their wealth given a 1 dollar change in price
than those consumers with lower incomes: The change affects
a higher proportion of your wealth. However, depending on how
we define wealth ares with change the magnitude of this change.
"""

# profit maximizing prices
data['monopoly_price'] = c/2-(beta_0+np.dot(data[covariates], beta_1))/(2*(beta_2+np.dot(data[covariates], beta_3)))
print(data.monopoly_price.describe())

# expected profit per consumer
# compute purchase probability
data['purchase_prob'] = beta_0+np.dot(data[covariates], beta_1)+data.monopoly_price*(beta_2+np.dot(data[covariates], beta_3))
# insert probs and prices to profit fn
data['profit'] = data.purchase_prob*(data.monopoly_price-c)
print(data[['purchase_prob', 'profit']].describe())

# to est avg demand take derivative of expected profit per consumer wrt price
model = OLS.from_formula('Purchase ~ 1 + Price', data)
res = model.fit()
print(res.summary)

# Store coeffs for late
beta_0h = res.params.Intercept
beta_2h = res.params.Price

# alpha hat 0 shows quantity demanded at p = 0
# alpha hat 1 shows quantity demanded for unit increase in p
# profit max unifrom price
uniform_monop_price = c/2 - beta_0h/(2*beta_2h)
print(uniform_monop_price)

# expected profit under uniform price
data['purchase_prob_unif_price'] = beta_0+np.dot(data[covariates], beta_1) + uniform_monop_price*(beta_2+np.dot(data[covariates], beta_3))

# insert purchase prob and prices into profit fn
data['profit_unif_price'] = data.purchase_prob_unif_price*(uniform_monop_price -c)

print(data[['purchase_prob_unif_price', 'profit_unif_price']].describe())

# profit loss
mean_monop = data['profit'].mean()
mean_unif = data['profit_unif_price'].mean()
loss = 40000*(mean_monop - mean_unif)
print(loss)

'''
Another variable to study may be how long ago a customers last purchase was.
For example if it was before amazon prime became the leading online shopping
location might have an effect on existing customers. Perhaps we could also look at
if the customer is located in a business center or residential area.
Maybe a business has a higher need for stapelers than households and this may
effect the purchase probability
'''









