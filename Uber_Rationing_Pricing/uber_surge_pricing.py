import pandas as pd
import statsmodels.formula.api as smf
from linearmodels import OLS, IV2SLS
import seaborn as sns
import matplotlib.pyplot as plt

# read data

data = pd.read_csv('UberData.csv')
print(data.head())

# set marginal cost
c = 10

# estimate demand function
demand_model = OLS.from_formula('AcceptedRides ~ 1 + Price + AppOpens', data = data)

fitted_model = demand_model.fit()
print(fitted_model.summary)

# store alpha coeffs
alpha_0 = fitted_model.params.Intercept
alpha_1 = fitted_model.params.Price
alpha_2 = fitted_model.params.AppOpens

# prices that maximize social surplus
true_model = alpha_0 + (alpha_1 * c) + (alpha_2 * data.AppOpens)

is_capacity_binding = 1. * (true_model > data.VehiclesAvailable)

equilibrium_price_if_binding = ((data.VehiclesAvailable - alpha_0 - (alpha_2 * data.AppOpens))/alpha_1)

equilibrium_price = (is_capacity_binding * equilibrium_price_if_binding) + ((1 - is_capacity_binding) * c)

print(equilibrium_price.describe())

# social surplus in equilibrium
accepted_rides = data.AcceptedRides

max_value = equilibrium_price.max()

consumer_surplus = (((alpha_0 - alpha_2 * data.AppOpens / alpha_1) - equilibrium_price - c) * data.VehiclesAvailable * 0.5)
                    
producer_surplus = (equilibrium_price - c) * data.VehiclesAvailable

equilibrium_social_surplus = consumer_surplus + producer_surplus

sum(equilibrium_social_surplus)

# social surplus with price at marginal cost and rationing
rationing_average_value = (maximum_value - c) / 2

# using social surplus formula
rationing_social_surplus = (rationing_average_value - c) * data.VehiclesAvailable

# diff in social surplus over the week
print(sum(equilibrium_social_surplus - rationing_social_surplus))

'''
a rationing price is better becasue it starts by allocation those with this highest WTP for the good.
A random rationing allocation does not start by selecting those consumers with the highest WTP and
therefore by the time the equilibrium QD has been met the consumer surplus will be less.
1F CS = 3492714 and 1H CS = 1499140 demonstrates this.
'''

