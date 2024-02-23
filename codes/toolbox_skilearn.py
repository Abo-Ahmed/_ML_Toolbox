
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

# Example 1: Model Selection
X, y = np.random.rand(100, 5), np.random.rand(100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Example 2: Preprocessing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Example 3: Model Building
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Example 4: Model Evaluation
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Example 5: Dimensionality Reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

# Example 6: Clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(X_train_scaled)
