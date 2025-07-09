import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def evaluate_metrics(yt, yp):
    results_pos = {}
    results_pos['accuracy'] = accuracy_score(yt, yp)
    precision, recall, f_beta, _ = precision_recall_fscore_support(yt, yp, average='binary')
    results_pos['recall'] = recall
    results_pos['precision'] = precision
    results_pos['f1score'] = f_beta
    return results_pos


# LINEAR REGRESSION
'''
from ml_package.LinearRegression import LinearRegression

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
from ml_package.LogisticRegression import LogisticRegression

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
'''
from ml_package.KNN import KNN

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
'''

# SVM
'''
from ml_package.SVM import SVM

X, y = datasets.make_blobs(n_samples=50, n_features = 2, centers = 2, cluster_std=1.25, random_state=1010)
y = np.where(y == 0, -1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=40)

clf = SVM()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

acc = accuracy_score(y_test, predictions)
print(acc)

def visualize_svm():
    def get_hyperplane(x, w, b, offset):
        return (-w[0] * x + b + offset) / w[1]
    
    fig = plt.figure()
    ax = fig.add_subplot(1 ,1, 1)
    plt.scatter(X[:,0], X[:,1], marker='o', c=y)

    x0_1 = np.amin(X[:, 0])
    x0_2 = np.amax(X[:, 0])

    x1_1 = get_hyperplane(x0_1, clf.weights, clf.bias, 0)
    x1_2 = get_hyperplane(x0_2, clf.weights, clf.bias, 0)

    x1_1_m = get_hyperplane(x0_1, clf.weights, clf.bias, -1)
    x1_2_m = get_hyperplane(x0_2, clf.weights, clf.bias, -1)

    x1_1_p = get_hyperplane(x0_1, clf.weights, clf.bias, 1)
    x1_2_p = get_hyperplane(x0_2, clf.weights, clf.bias, 1)

    ax.plot([x0_1, x0_2], [x1_1, x1_2], 'k')
    ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], 'y--')
    ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], 'y--')

    x1_min = np.amin(X[:, 1])
    x1_max = np.amax(X[:, 1])
    ax.set_ylim([x1_min-3, x1_max+3])

    plt.savefig('implementation_notebooks/svm.png')

visualize_svm()
'''

# Decision Tree
'''
from ml_package.DecisionTree import DecisionTree

# X, y = datasets.load_breast_cancer(return_X_y=True)

tumor_df = pd.read_csv('IBM_Machine_Learning/datafiles/tumor.csv')
# Get the input features
X = tumor_df.iloc[:, :-1].to_numpy()
# Get the target variable
y = tumor_df.iloc[:, -1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

clf = DecisionTree(max_depth=10, min_samples_split=2)
clf.fit(X_train, y_train.values.ravel())
predictions = clf.predict(X_test)

metrics = evaluate_metrics(y_test, predictions)
print(metrics)
'''

# Random Forest

from ml_package.RandomForest import RandomForest

tumor_df = pd.read_csv('IBM_Machine_Learning/datafiles/tumor.csv')
# Get the input features
X = tumor_df.iloc[:, :-1].to_numpy()
# Get the target variable
y = tumor_df.iloc[:, -1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

clf = RandomForest(n_tress= 15, max_depth=5)
clf.fit(X_train, y_train.values.ravel())
predictions = clf.predict(X_test)

metrics = evaluate_metrics(y_test, predictions)
print(metrics)