from sympy import var, solve, Eq, subfactorial
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_model(fig, X_train, X_test, anomaly_scores, clf, mesh, subplot_value, chart_name, n_threads):
    # Plot block

    data_pandas = pd.DataFrame(X_train)
    datatest_pandas = pd.DataFrame(X_test)
    plot_data(fig, data_pandas, datatest_pandas, 0, 1, anomaly_scores, clf, mesh, subplot_value, chart_name, n_threads)


def plot_model_hours(X_train, X_test, anomaly_scores, clf, mesh, n_threads):
    # Plot block
    data_pandas = pd.DataFrame(X_train)
    datatest_pandas = pd.DataFrame(X_test)
    extra_points = get_hour_points()
    plot_data_hours(data_pandas, datatest_pandas, 0, 1, anomaly_scores, clf, mesh, extra_points, n_threads)


def open_plot():
    f = plt.figure(figsize=(8, 8))
    f.subplots_adjust(left=0.09, bottom=0.1, right=0.95, top=0.95, wspace=0.2, hspace=0.2)
    return f


def plot_data(fig, data_train, data_test, col_X, colY, anomaly_scores, clf, mesh, subplot_value, chart_name, n_threads):
    """
    Plots 2D data set training and testing
    :param data_train: pandas data frame train
    :param data_test: pandas data frame test
    :param col_X: int, x axis
    :param col_Y: int, y axis
    :param anomaly_scores: array of anomaly scores
    :param clf: model
    """

    xx, yy = np.meshgrid(np.linspace(-mesh, mesh, 50), np.linspace(-mesh, mesh, 50))
    S0 = clf.compute_paths(np.c_[xx.ravel(), yy.ravel()], n_threads)
    S0 = S0.reshape(xx.shape)
    ss0 = np.argsort(anomaly_scores)

    ax1 = fig.add_subplot(subplot_value[0], subplot_value[1], subplot_value[2])
    levels = np.linspace(np.min(S0), np.max(S0), 5)
    CS = ax1.contourf(xx, yy, S0, levels, cmap=plt.cm.YlOrRd)

    # Train data
    x = np.array(data_train[col_X].tolist())
    y = np.array(data_train[colY].tolist())
    plt.scatter(x, y, s=20, c='g', edgecolor='g')

    # Test data
    x = np.array(data_test[col_X].tolist())
    y = np.array(data_test[colY].tolist())
    plt.scatter(x, y, s=5, c='m', edgecolor='m')
    plt.scatter(x[ss0[-3:]], y[ss0[-3:]], s=5, c='c')
    plt.scatter(x[ss0[:3]], y[ss0[:3]], s=5, c='k')
    plt.title(chart_name)


def close_plot():
    plt.savefig("dangerousness_svg.png", dpi=100)
    plt.show()

def plot_data_hours(data_train, data_test, col_X, colY, anomaly_scores, clf, mesh, extra_points, n_threads=10):
    """
    Plots 2D data set training and testing
    :param data_train: pandas data frame train
    :param data_test: pandas data frame test
    :param col_X: int, x axis
    :param col_Y: int, y axis
    :param anomaly_scores: array of anomaly scores
    :param clf: model
    """

    xx, yy = np.meshgrid(np.linspace(-mesh, mesh, 50), np.linspace(-mesh, mesh, 50))
    S0 = clf.compute_paths(np.c_[xx.ravel(), yy.ravel()], n_treads)
    S0 = S0.reshape(xx.shape)
    ss0 = np.argsort(anomaly_scores)
    f = plt.figure(figsize=(15, 6))
    ax1 = f.add_subplot(121)
    levels = np.linspace(np.min(S0), np.max(S0), 5)
    CS = ax1.contourf(xx, yy, S0, levels, cmap=plt.cm.YlOrRd)

    # Train data
    x = np.array(data_train[col_X].tolist())
    y = np.array(data_train[colY].tolist())
    plt.scatter(x, y, s=20, c='g', edgecolor='g')

    # Test data
    x = np.array(data_test[col_X].tolist())
    y = np.array(data_test[colY].tolist())
    plt.scatter(x, y, s=5, c='m', edgecolor='m')
    plt.scatter(x[ss0[-3:]], y[ss0[-3:]], s=5, c='c')
    plt.scatter(x[ss0[:3]], y[ss0[:3]], s=5, c='k')
    plt.title('extended')

    plot_points(extra_points)

    plt.show()


def plot_points(points):
    x, y = np.array(points).T
    plt.scatter(x,y, s=10)


def get_hour_points():
    print("GET HOUR DATA")
    points = []
    radius = 1000000
    for i in range (0, 100):
        for i in range (0, 12):
            x, y = var('x y')
            f1 = Eq((x ** 2) + (y ** 2) - radius, 0)
            f2 = Eq(math.tan(math.radians(i*15)) * x - y, 0)

            sols = solve((f1, f2), (x, y))
            points.append(sols[0])
            points.append(sols[1])
        radius += i*10000
    return points
