from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# Generate sample data
np.random.seed(0)
X = np.random.rand(100, 3)  # Generate 100 samples with 3 features each
coefficients = np.array([2, 3, 4])  # Coefficients for each feature
y = np.dot(X, coefficients) + np.random.randn(100)  # Generate corresponding y values with noise

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a multi-linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Print the coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Print the mean squared error
mse = np.mean((y_pred - y_test) ** 2)
print("Mean Squared Error:", mse)
