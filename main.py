import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from ml_package.LinearRegression import LinearRegression
from ml_package.LogisticRegression import LogisticRegression
from ml_package.KNN import KNN


# LINEAR REGRESSION
'''
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
'''

# LOGISTIC REGRESSION
'''
X, y = datasets.load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

clf = LogisticRegression()
scaler = StandardScaler()

X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)


clf.fit(X_train_s, y_train)
y_pred = clf.predict(X_test_s)

accu = accuracy_score(y_test, y_pred)

print(f"Test Accuracy = {accu}")
'''

# KNN

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

cmap = ListedColormap([ '#FF0000' , '#00FF00' , '#0000FF'])
plt.figure()
plt.scatter(X[:,2], X[:,3], c = y, cmap = cmap, edgecolors='k', s =20)
# plt.savefig("implementation_notebooks/KNN_iris.png")

clf = KNN(k=5)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

acc = accuracy_score(y_test, predictions)
print(acc)