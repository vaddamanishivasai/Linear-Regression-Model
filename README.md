# Linear-Regression-Model 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# Generate synthetic data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
# Visualize the data
plt.scatter(X, y)
plt.xlabel("Input feature (X)")
plt.ylabel("Output (y)")
plt.title("Generated Data")
plt.show()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
# Make predictions on the test data
y_pred = model.predict(X_test)
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Model coefficients (slope and intercept):", model.coef_, model.intercept_)
print("Mean squared error:", mse)
# Visualize the regression line
plt.scatter(X, y, color="blue")
plt.plot(X, model.predict(X), color="red", linewidth=2)
plt.xlabel("Input feature (X)")
plt.ylabel("Output (y)")
plt.title("Linear Regression Fit")
plt.show()
