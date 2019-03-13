#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

#--------Running Python From R ------------------#

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# python path : C:\Users\xxxxx\AppData\Local\Programs\Python\Python37\Scripts

# use cmd to install packages for windows using pip

# FOr this tutorial I have used numpy 

# Import Reticulate

library(reticulate)



# Step 1 : Load Python Environment
repl_python()

#Import Numpy and Matplotlib and sklearn
# Numpy for data generation
# matplotlib for plots
#skelarn for generating polynomial regression model


#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

#--------Example 1 : Simple Linear Regression Model------------------#

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
import operator
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures



np.random.seed(0)  # generate same reproducible  random data using seed parameter which is 0

x = 2 - 3 * np.random.normal(0, 1, 20)  # generate independent variable

y = x - 2 * (x ** 2) + 0.5 * (x ** 3) + np.random.normal(-3, 3, 20) # generate dependant variable

# create a 2-D matrix from vectors
x = x[:, np.newaxis]
y = y[:, np.newaxis]

# Create a linear regression model
model = LinearRegression() 
model.fit(x, y)
y_pred = model.predict(x)


# Computing RMSE and r^2
rmse = np.sqrt(mean_squared_error(y,y_pred))
r2 = r2_score(y,y_pred)


print(rmse)
print(r2)


# show the plots of containing 
plt.scatter(x, y, s=10)
plt.plot(x, y_pred, color='r')
plt.show()



#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

#--------Example 2 : Polynomial Regression Model------------------#

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# Creating a Polynomial feature of independant variable x -> x^2
polynomial_features= PolynomialFeatures(degree=15)
x_poly = polynomial_features.fit_transform(x)
x_poly[0:5,]

# Running the model on transformed data.
model = LinearRegression()
model.fit(x_poly, y)
y_poly_pred = model.predict(x_poly)

# Computing RMSE and r^2
rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
r2 = r2_score(y,y_poly_pred)


print(rmse)
print(r2)

plt.scatter(x, y, s=10)
# sort the values of x before line plot

# Plot the results
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(x,y_poly_pred), key=sort_axis)
x, y_poly_pred = zip(*sorted_zip)
plt.plot(x, y_poly_pred, color='m')
plt.show()





#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

#--------Example 3 : Polynomial Regression on Boston Housing Dataset------------------#

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Import Libraries

import numpy as np
import matplotlib.pyplot as plt 

import pandas as pd  
import seaborn as sns 



# Load the Boston Housing DataSet from scikit-learn

from sklearn.datasets import load_boston
boston_dataset = load_boston()

# boston_dataset is a dictionary,let's check what it contains
boston_dataset.keys()

#Load the data into pandas dataframe
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston.head()


# The target values is missing from the data. 
#Create a new column of target values and add it to dataframe

boston['MEDV'] = boston_dataset.target

# check for missing values in all the columns
boston.isnull().sum()


# Exploratory Data Analysis
# set the size of the figure
sns.set(rc={'figure.figsize':(11.7,8.27)})

# plot a histogram showing the distribution of the target values
sns.distplot(boston['MEDV'], bins=30)
plt.show()


# Prepare the data for training
X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns = ['LSTAT','RM'])
Y = boston['MEDV']

# Split the data into training and testing sets

from sklearn.model_selection import train_test_split

# splits the training and test data set in 80% : 20%
# assign random_state to any value.This ensures consistency.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# Linear Regression Model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)

# model evaluation for training set

y_train_predict = lin_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2 = r2_score(Y_train, y_train_predict)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set

y_test_predict = lin_model.predict(X_test)
# root mean square error of the model
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))

# r-squared score of the model
r2 = r2_score(Y_test, y_test_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))


# We can see that LSTAT doesn't vary exactly in a linear way.
# Let's apply the Polynomial Regression with degree 2 and test.

quit

#source the function
source_python('poly_regression.py')



# Step 1 : Load Python Environment
repl_python()


create_polynomial_regression_model(2)

