# Import some packages for data science
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge

# Modify some settings
plt.rcParams['figure.figsize'] = (10, 7)
plt.rcParams['font.size'] = 16
plt.rcParams['figure.dpi'] = 150

pd.options.display.max_rows = 20
pd.options.display.max_columns = 15

np.random.seed(47)

# Import some packages to help with configuration
import os, sys
from IPython.display import Image

# read data sets
training_data = pd.read_csv("training.csv")
testing_data = pd.read_csv("testing.csv")

# generate raincloud plots
# combine histogram w density 
fig, axs = plt.subplots(nrows=2)

sns.distplot(
    training_data['SalePrice'],
    ax=axs[0]
)
sns.stripplot(
    training_data['SalePrice'],
    jitter=0.4,
    size=3,
    ax=axs[1],
    alpha=0.3
)
sns.boxplot(
    training_data['SalePrice'],
    width=0.3,
    ax=axs[1],
    showfliers=False,
)
# Align axes
spacer = np.max(training_data['SalePrice']) * 0.05
xmin = np.min(training_data['SalePrice']) - spacer
xmax = np.max(training_data['SalePrice']) + spacer
axs[0].set_xlim((xmin, xmax))
axs[1].set_xlim((xmin, xmax))

# Remove some axis text
axs[0].xaxis.set_visible(False)
axs[0].yaxis.set_visible(False)
axs[1].yaxis.set_visible(False)

# Put the two plots together
plt.subplots_adjust(hspace=0)

# Adjust boxplot fill to be white
axs[1].artists[0].set_facecolor('white')

# see summary stats
training_data['SalePrice'].describe()

# add column to data representing total bathrooms in a property
def add_total_bathrooms(data):
    '''
    in:
        data: table containing at least four columns of numbers
        Bsmt_Full_Bath, Full_Bath, Bsmt_Half_Bath, and Half_Bath
    out:
        copy of table with additional col TotalBathrooms
    '''
    # copy data
    with_bathrooms = data.copy()

    # fill missing vals w 0
    bath_vars = ['Bsmt_Full_Bath', 'Full_Bath', 'Bsmt_Half_Bath', 'Half_Bath']
    with_bathrooms = with_bathrooms.fillna({var: 0 for var in bath_vars})

    # add to TotalBathrooms col
    weights = np.array([1, 1, 0.5, 0.5])

    with_bathrooms['TotalBathrooms'] = with_bathrooms['Bsmt_Full_Bath']*weights[0] + with_bathrooms['Full_Bath']*weights[1] + with_bathrooms['Bsmt_Half_Bath']*weights[2] + with_bathrooms['Half_Bath']*weights[3]
    return with_bathrooms

training_data = add.total_bathrooms(training_data)

# look at box plots to see price ranges for diff number of bathrooms
x = 'TotalBathrooms'
y = 'SalePrice'
data = training_data
sns.boxplot(y=y, x=x, data=data)
plt.title('SalePrice distribution for each value of TotalBathrooms');

# now create new features out of old features
# first lets look at the relationship of above ground living area to sale price
sns.jointplot(
    x='Gr_Liv_Area',
    y='SalePrice',
    data=training_data,
    stat_func=None,
    kind="reg",
    ratio=4,
    space=0,
    scatter_kws={
        's': 3,
        'alpha': 0.25
    },
    line_kws={
        'color': 'black'
    }
);
# now lets look at the relationship of garage area to sale price because above does not take into acount garage space
sns.jointplot(
    x='Garage_Area',
    y='SalePrice',
    data=training_data,
    stat_func=None,
    kind="reg",
    ratio=4,
    space=0,
    scatter_kws={
        's': 3,
        'alpha': 0.25
    },
    line_kws={
        'color': 'black'
    }
);

#
def add_power(data, column_name, degree):
    """
    Input:
        data : a table containing column called column_name column_name : a string indicating a column in the table degree: positive integer
    Output:
        copy of data containing a column called column_name<degree> with entr
    """
    with_power = data.copy()
    
    new_column_name = column_name + str(degree)
    new_column_values = with_power[column_name]**(degree)
    
    with_power[new_column_name] = new_column_values
    
    return with_power

training_data = add_power(training_data, "Garage_Area", 2)
training_data = add_power(training_data, "Gr_Liv_Area", 2)

# now check which variable has highest corr with sale price
a = training_data['Gr_Liv_Area'].corr(training_data['SalePrice'])
b = training_data['Gr_Liv_Area2'].corr(training_data['SalePrice'])
c = training_data['Garage_Area'].corr(training_data['SalePrice'])
d = training_data['Garage_Area2'].corr(training_data['SalePrice'])
print(a, b, c, d)
# highest corr:
highest_variable = 'Gr_Liv_Area'

# now we examine relationship of neighbourhood with sale price
fig, axs = plt.subplots(nrows=2)

sns.boxplot(
    x='Neighborhood',
    y='SalePrice',
    data=training_data.sort_values('Neighborhood'),
    ax=axs[0]
)

sns.countplot(
    x='Neighborhood',
    data=training_data.sort_values('Neighborhood'),
    ax=axs[1]
)

# Draw median price
axs[0].axhline(
    y=training_data['SalePrice'].median(),
    color='red',
    linestyle='dotted'
)

# Label the bars with counts
for patch in axs[1].patches:
    x = patch.get_bbox().get_points()[:, 0]
    y = patch.get_bbox().get_points()[1, 1]
    axs[1].annotate(f'{int(y)}', (x.mean(), y), ha='center', va='bottom')

# Format x-axes
axs[1].set_xticklabels(axs[1].xaxis.get_majorticklabels(), rotation=90)
axs[0].xaxis.set_visible(False)

# Narrow the gap between the plots
plt.subplots_adjust(hspace=0.01)

'''
We find a lot of variation in prices across neighborhoods.
Moreover, the amount of data available is not uniformly distributed among neighborhoods.
North Ames, for example, comprises almost 15% of the training data while Green Hill has only 2 observations in this data set.
One way we can deal with the lack of data from some neighborhoods is to create a new feature that bins neighborhoods together.
Let's categorize our neighborhoods in a crude way:
we'll take the top 3 neighborhoods measured by median SalePrice and identify them as "expensive neighborhoods";
the other neighborhoods are not marked.
'''

def find_expensive_neighborhoods(data, n, summary_statistic):
    """
    Input:
        data : table containing at a column Neighborhood and a column SalePri
        n : integer indicating the number of neighborhood to return
        summary_statistic : function used for aggregating the data in each ne
    Output:
      a list of the top n richest neighborhoods as measured by the summary
    """
    neighborhoods = (training_data.groupby("Neighborhood")
                     .agg({"SalePrice" : summary_statistic})
                     .sort_Values("SalePrice", ascending = False)
                     .index[:n])
                     
    return list(neighborhoods)

# top five neighborhoods on average price
find_expensive_neighborhoods(training_data, 5, np.mean)

# top three neighborhoods on meadian price
find_expensive_neighborhoods(training_data, 3, np.median)

expensive_neighborhood_1 = 'StoneBr'
expensive_neighborhood_2 = 'NridgHt'
expensive_neighborhood_3 = 'NoRidge'

# save these
expensive_neighborhoods = [expensive_neighborhood_1, expensive_neighborhood_2, expensive_neighborhood_3]


# lets add the feature in_expensive_neighborhood to training set
# write a function that adds 1 or 0 to indicate this

def add_expensive_neighborhood(data, neighborhoods):
    """
    Input:
        data : a table containing a 'Neighborhood' column neighborhoods : list of strings with names of neighborhoods
    Outtput:
        A copy of the table with an additional column in_expensive_neighborhood
    """
    with_additional_column = data.copy()
    
    withth_additional_column['in_expensive_neighborhood'] = data['Neighborhood'].isin(neighborhoods)

    return with_additional_column

# add to data
training_data = add_expensive_neighborhood(training_data, expensive_neighborhood)

'''
we need to normalize features for regularization.
If the features have different scales,
then regularization will unduly shrink the weights for features with smaller scales.
Write a function called normalize that inputs
either a 1 dimensional array or a 2 dimensional array Z of numbers and outputs a copy of
Z where the columns have been transformed to have mean 0 and standard deviation 1.
'''
# eg
Z = training_data[['Garage_Area','Gr_Liv_Area']].values
Z_normalized = (Z - Z.mean(axis = 0)) / Z.std(axis = 0)

# normalize function
def standardize(Z):
    """
    Input:
       Z: 1 dimensional or 2 dimensional array
    Outuput
       copy of Z with columns having mean 0 and variance 1
    """
    Z_normalized = (Z - Z.mean(axis = 0)) / Z.std(axis = 0)
    return Z_normalized

# TEST
Z = training_data[['Garage_Area','Gr_Liv_Area']].values
assert np.all(np.isclose(standardize(Z).mean(axis = 0), [0,0]))

'''
Let's split the training set into a training set and a validation set.
We will use the training set to fit our model's parameters.
We will use the validation set to estimate how well our model will perform on unseen data.
If we used all the data to fit our model,
we would not have a way to estimate model performance on unseen data.
'''

#Run to make a copy of the original training set
training_data_copy = pd.read_csv("training.csv")

'''
We will split the data in training_data_copy into
two tables named training_data and validating_data .
First we need to shuffle the indices of the table.
Note that the training set has 1998 rows.
We want to generate an array containing the number 0,1,...,1997 in random order.
'''

length_of_training_data = len(training_data_copy)
RANDOM_STATE = 47
shuffled_indices = np.random.RandomState(seed=RANDOM_STATE).permutation(length_of_training_data)

# split validate and train indicies into 80 / 20

train_indices = shuffled_indices[:int(length_of_training_data * 0.8)]
validate_indices = shuffled_indices[int(length_of_training_data * 0.2):]

# 
training_data = training_data_copy.iloc[train_indices]
validating_data = training_data_copy.iloc[validate_indices]

# to try a few different models lets create a reusable pipeline

def select_columns(data, columns):
    """Select only columns passed as arguments."""
    return data.loc[:, columns]
def process_data(data):
    """Process the data for a guided model."""
    nghds = find_expensive_neighborhoods(data, n=3, summary_statistic=np.mean)
    data = ( data.pipe(add_total_bathrooms)
                 .pipe(add_power,'Gr_Liv_Area', 2)
                 .pipe(add_power,'Garage_Area', 2)
                 .pipe(add_expensive_neighborhood, nghds)
                 .pipe(select_columns, ['SalePrice',
                                        'Gr_Liv_Area',
                                        'Garage_Area',
                                        'Gr_Liv_Area2',
                                        'Garage_Area2',
                                        'TotalBathrooms',
                                        'in_expensive_neighborhood']) )

    data.dropna(inplace = True)
    X = data.drop(['SalePrice'], axis = 1)
    X = standardize(X)
    y = data.loc[:, 'SalePrice']
    y = standardize(y)
    return X, y

'''
Note that we split our data into a table of explantory variables X and an array of response variables y .
We can use process_data to transform the training set and validation set from
earlier along with the testing set.
'''
# transformations
X_train, y_train = process_data(training_data)
X_validate, y_validate = process_data(validating_data)
X_test, y_test = process_data(testing_data)

'''
implement Ridge Regression.
Note that alpha is the extra parameter needed to specify the emphasis on regularization.
Large values of alpha mean greater emphasis on regularization.
'''

ridge_regression_model = Ridge(alpha = 1)
ridge_regression_model.fit(X_train, y_train)
ridge_regression_model.coef_

# try some different values for extra parameter
models = dict()
alphas = np.logspace(-4,4,10)
for alpha in alphas:
    ridge_regression_model = Ridge(alpha = alpha)
    models[alpha] = ridge_regression_model


# fit each model to training data
# each key is the value of the extra parameter alpha
# each value is a ridge reg model corresponding to that alpha
for alpha, model in models.items():
    model.fit(X_train, y_train)

# plot data for each alpha
labels = ['Gr_Liv_Area',
          'Garage_Area',
          'Gr_Liv_Area2',
          'Garage_Area2',
          'TotalBathrooms',
          'in_rich_neighborhood']
coefs = []
for alpha, model in models.items():
    coefs.append(model.coef_)
    
coefs = zip(*coefs)

fig, ax = plt.subplots(ncols=1, nrows=1)

for coef, label in zip(coefs, labels):
    plt.plot(alphas, coef, label = label)
    
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim())
# reverse axis plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge Regression Weights')
plt.legend();

# lets measure the quality of our model by calculating MSE between predicted and observed house prices
def mse(observed, predicted):
    """
    Calculates RMSE from actual and predicted values
    Input:
      observed (1D array): vector of actual values
      predicted (1D array): vector of predicted/fitted values
    Output:
    a float, the root-mean square error
    """
    return np.sqrt(np.mean((observed - predicted)**2))

# For each alpha , we use mse to calculate the training error and validating error.
mse_training = dict()
mse_validating = dict()

for alpha, model in models.items():
    y_predict = model.predict(X_train)
    mse_training[alpha] = mse(y_predict, y_train)
    
    y_predict = model.predict(X_validate)
    mse_validating[alpha] = mse(y_predict, y_validate)

# store in dict where key is alpha and value is mse of model
# find min
alpha_training_min = min(mse_training.keys(), key=(lambda k: mse_training[k]))
alpha_validating_min = min(mse_validating.keys(), key=(lambda k: mse_validating[k]))

model = models[alpha_validating_min]
y_predict = model.predict(X_test)

# use residual plot to test appropriatness of model
residuals = y_test - y_predict

plt.axhline(y = 0, color = "red", linestyle = "dashed")
plt.scatter(y_test, residuals, alpha=0.5);

plt.xlabel('Sale Price (Test Data)')
plt.ylabel('Residuals (Actual Price - Predicted Price)')
plt.title("Residuals vs. Sale Price on Test Data")

"""
to improve model:
first we should increase the size of our data.
Due to law of large numbers, with enough data our predicted Y values
would begin to approach the true Y vale Second we should build more variables into our model.
We could look at different combinations of variables in order to mitigate specification bias.
"""







