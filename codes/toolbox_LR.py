# Importing necessary libraries
# y = intercept + coefficient × X
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Generating sample data
X = np.array([[1], [2], [3], [4], [5]])  # Independent variable (features)
y = np.array([2, 3.5, 3.7, 5, 6])         # Dependent variable (target)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a linear regression model
model = LinearRegression()

# Training the model
model.fit(X_train, y_train)

# Making predictions on the test data
y_pred = model.predict(X_test)

# Evaluating the model
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Mean squared error:", np.mean((y_pred - y_test) ** 2))
