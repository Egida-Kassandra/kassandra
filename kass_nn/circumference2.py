from sympy import var, solve, Eq
from sympy.plotting import plot, plot_implicit, plot_parametric
import math
import matplotlib.pyplot as plt
import sympy
import numpy as np
import eif as iso
import pandas as pd

def load_data_float(data_pandas):
    """
    Returns numpy array of float from pandas data frame
    :param data_pandas: pandas data frame
    """
    return data_pandas.to_numpy().astype(np.float)
def plot_data(data_train, data_test, col_X, colY, anomaly_scores, clf):
    """
    Plots 2D data set training and testing
    :param data_train: pandas data frame train
    :param data_test: pandas data frame test
    :param col_X: int, x axis
    :param col_Y: int, y axis
    :param anomaly_scores: array of anomaly scores
    """

    #xx, yy = np.meshgrid(np.linspace(np.min(data_train[col_X]), np.max(data_train[colY]), 50), np.linspace(np.min(data_train[col_X]), np.max(data_train[colY]), 50))
    xx, yy = np.meshgrid(np.linspace(-2500, 2500, 50),
                         np.linspace(-2500, 2500, 50))
    S0 = clf.compute_paths(np.c_[xx.ravel(), yy.ravel()])
    S0 = S0.reshape(xx.shape)
    ss0 = np.argsort(anomaly_scores)
    f = plt.figure(figsize=(15, 6))
    ax1 = f.add_subplot(121)
    levels = np.linspace(np.min(S0), np.max(S0), 5)
    CS = ax1.contourf(xx, yy, S0, levels, cmap=plt.cm.YlOrRd)


    # Train data
    x = np.array(data_train[col_X].tolist())
    y = np.array(data_train[colY].tolist())
    plt.scatter(x, y, s=60, c='g', edgecolor='g')
    # Test data
    x = np.array(data_test[col_X].tolist())
    y = np.array(data_test[colY].tolist())
    plt.scatter(x, y, s=20, c='b', edgecolor='b')
    plt.scatter(x[ss0[-3:]], y[ss0[-3:]], s=20, c='r')
    plt.scatter(x[ss0[:3]], y[ss0[:3]], s=20, c='k')
    plt.title('extended')
    plt.show()

def get_hour_points():
    points = []
    for i in range (0, 12):
        x, y = var('x y')
        radius = 1000000
        f1 = Eq((x ** 2) + (y ** 2) - radius, 0)
        f2 = Eq(math.tan(math.radians(i*15)) * x - y, 0)

        sols = solve((f1, f2), (x, y))
        points.append(sols[0])
        points.append(sols[1])
    return points

def get_test_points_A(radius, fromI, toI):
    points = []
    for i in range (fromI, toI):
        x = math.sqrt(radius/(1 + math.tan(math.radians(i*0.25)) ** 2))
        y = math.tan(math.radians(i*0.25))*x
        points.append([x,y])
    return points

def get_test_points_B(radius, fromI, toI):
    points = []
    for i in range (fromI, toI):
        x = math.sqrt(radius/(1 + math.tan(math.radians(i*0.25)) ** 2))
        y = math.tan(math.radians(i*0.25))*x
        points.append([-x,y])
    return points

radius1 = 1000000
radius2 = 4000000
points = get_test_points_A(radius1, 0, 100) +  get_test_points_A(radius2, 100, 300) + get_test_points_A(radius1, 300, 600) + get_test_points_A(radius2, 600, 700) \
         + get_test_points_B(radius1, 300, 600) + get_test_points_B(radius2, 600, 700) + get_test_points_B(radius2, 700, 1000)

x, y = np.array(points).T
#plt.scatter(x,y)
#plt.show()
"""
p1 = plot_implicit(f1, show=False)
p2 = plot_implicit(f2, show=False)
p1.append(p2[0])

#p1.show()
"""

# Columns used for training and testing
columns = [0, 5]

# Loading training data
data_pandas = points
data_pandas = pd.DataFrame(data_pandas)
X_train = load_data_float(data_pandas)
np.random.seed(123)

# pca.fit(X_train)
# X_train = pca.transform(X_train)

# Loading testing data
points = get_test_points_A(radius1, 0, 1) +  get_test_points_A(radius2, 100, 101) + get_test_points_A(radius1, 300, 301) + get_test_points_A(radius2, 600, 601) \
         + get_test_points_B(radius1, 300, 301) + get_test_points_B(radius2, 600, 601) + get_test_points_B(radius2, 700, 701) \
            \
         + get_test_points_B(radius1, 400, 401) + get_test_points_B(radius2, 650, 651) + get_test_points_B(radius2, 850, 851) \
         + get_test_points_A(1000000, 5, 6) +  get_test_points_A(radius2, 200, 201) + get_test_points_A(radius1, 450, 451) + get_test_points_A(radius2, 650, 651) \
            \
         + get_test_points_A(radius2, 5, 6) +  get_test_points_A(radius1, 200, 201) + get_test_points_A(radius2, 450, 451) + get_test_points_A(radius1, 650, 651) \

datatest_pandas = points
datatest_pandas = pd.DataFrame(datatest_pandas)
X_test = load_data_float(datatest_pandas)
# pca.fit(X_test)
# X_test = pca.transform(X_test)



clf = iso.iForest(X_train, ntrees=1000, sample_size=1400, ExtensionLevel=1)  # 2000, 20000, ext 1 ###### cuidaooo



anomaly_scores = clf.compute_paths(X_test)
anomaly_scores_sorted = np.argsort(anomaly_scores)
indices_with_preds = anomaly_scores_sorted[-int(np.ceil(0.9 * X_train.shape[0])):]
print(indices_with_preds)
print((anomaly_scores))

# Plot block
data_pandas = pd.DataFrame(X_train)
datatest_pandas = pd.DataFrame(X_test)
plot_data(data_pandas, datatest_pandas, 0, 1, anomaly_scores, clf)