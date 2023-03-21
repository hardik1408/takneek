import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Loading the csv file
data=np.genfromtxt('IPL 2022 Batters.csv', delimiter=',')
target=data[1:, 11]
runs=data[1:, 4]
X = np.array(target).reshape((-1, 1)) #converting 1D array into 2D array
y = np.array(runs)

# Spliting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Creating a linear regression model
model = LinearRegression()

# Training the model
model.fit(X_train, y_train)

# Testing the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Printing the results
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error:', mse)

