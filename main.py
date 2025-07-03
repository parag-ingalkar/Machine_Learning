import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from ml_package.linear_regression import LinearRegression

X, Y = datasets.make_regression(n_samples = 100, n_features = 1, noise= 20, random_state= 10)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state= 1234)
fig = plt.figure(figsize=(8, 6))
plt.scatter(X, Y)
# plt.savefig("LinearRegression_Data.png")

reg = LinearRegression(lr=0.01)

reg.fit(X_train, Y_train)
predictions = reg.predict(X_test)

def mean_squared_error(y, y_pred):
    return np.mean((y - y_pred) ** 2 )

error = mean_squared_error(Y_test, predictions)
print(f"Error on Training set = {error}")

y_pred_line = reg.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8, 6))
m1 = plt.scatter(X_train, Y_train, color = cmap(0.9), s = 10)
m1 = plt.scatter(X_test, Y_test, color = cmap(0.5), s = 10)
plt.plot(X, y_pred_line, color="black", linewidth=2, label="Prediction")
# plt.savefig("LinearRegression_Model_fit.png")
