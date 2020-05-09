from sympy import var, solve, Eq
import math
import matplotlib.pyplot as plt
import numpy as np
import eif as iso
import pandas as pd
from parse_logs import LogParser
import pickle


class MinMeth:
    def __init__(self):
        """Constructor"""
        self.clf = None
        self.X_train = None
        self.columns = [1, 3]
        self.logpar = LogParser()

    def load_data_float(self, data_pandas):
        """
        Returns numpy array of float from pandas data frame
        :param data_pandas: pandas data frame
        """
        return data_pandas.to_numpy().astype(np.float)

    def plot_data(self, data_train, data_test, col_X, colY, anomaly_scores, clf, mesh, extra_points):
        """
        Plots 2D data set training and testing
        :param data_train: pandas data frame train
        :param data_test: pandas data frame test
        :param col_X: int, x axis
        :param col_Y: int, y axis
        :param anomaly_scores: array of anomaly scores
        :param clf: model
        """

        #mesh = 1500
        xx, yy = np.meshgrid(np.linspace(-mesh, mesh, 50), np.linspace(-mesh, mesh, 50))
        S0 = self.clf.compute_paths(np.c_[xx.ravel(), yy.ravel()])
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

        self.plot_points(extra_points)

        plt.show()

    def plot_points(self, points):
        x, y = np.array(points).T
        plt.scatter(x,y, s=10)


    def get_scp_point_A(self, minute, radius):
        one_min = math.pi / 720 # constant
        rad = minute * one_min
        x = math.sqrt(radius / (1 + math.tan(rad) ** 2))
        y = math.tan(rad) * x
        return [x,y]


    def get_scp_point_B(self, minute, radius):
        one_min = math.pi / 720
        rad = minute * one_min
        x = math.sqrt(radius / (1 + math.tan(rad) ** 2))
        x = -x
        y = math.tan(rad) * x
        return [x,y]


    def parse_sc_to_scp(self, data_pandas):
        radius1 = 500000
        radius2 = 300000
        points = []
        col1 = self.columns[0]
        col2 = self.columns[1]
        for index, row in data_pandas.iterrows():
            if 0 <= row[col1] < 360 or 1080 < row[1] <= 1440:
                points.append(self.get_scp_point_A(row[col1], radius1 + radius2 * row[col2]))
            elif 360 < row[col1] < 1080:
                points.append(self.get_scp_point_B(row[col1], radius1 + radius2 * row[col2]))
            elif row[1] == 360:
                points.append([0, math.sqrt(radius1 + radius2 * row[col2])])
            elif row[1] == 1080:
                points.append([0, -math.sqrt(radius1 + radius2 * row[col2])])
        return points


    def load_data_pandas(self, filename, is_train):
        """
        Loads log file and returns pandas data frames
        :param filename: name of the file
        :param is_train: boolean, if the file has training or testing logs
        """

        data_train = self.logpar.parse_file(filename, is_train)
        return pd.DataFrame(data_train)[self.columns]


    def train_and_save_model(self, filename):
        # Columns used for training and testing
         # Init in constructor

        # Loading training data
        train = self.load_data_pandas(filename, True)
        data_pandas = pd.DataFrame(self.parse_sc_to_scp(train))
        self.X_train = self.load_data_float(data_pandas)
        np.random.seed(123)

        # Train block
        self.clf = iso.iForest(self.X_train, ntrees=2000, sample_size=1000, ExtensionLevel=1)  # 2000, 15000, ext 1 ###### cuidaooo

        # Save block
        """
        pickle.dumps(self.clf)
        with open('min_vs_meth.kass', 'wb') as model_file:
            pickle.dump(self.clf, model_file)
            """

    def predict(self, X_test, mesh, extra_points):
        print("PREDICTING")
        # Predict block
        anomaly_scores = self.clf.compute_paths(X_test)
        anomaly_scores_sorted = np.argsort(anomaly_scores)
        indices_with_preds = anomaly_scores_sorted[-int(np.ceil(0.9 * self.X_train.shape[0])):]
        print(indices_with_preds)
        print(np.sort(anomaly_scores))

        # Plot block
        data_pandas = pd.DataFrame(self.X_train)
        datatest_pandas = pd.DataFrame(X_test)
        self.plot_data(data_pandas, datatest_pandas, 0, 1, anomaly_scores, self.clf, mesh, extra_points)


    def load_model(self):
        with open('min_vs_meth.kass', 'wb') as model_file:
            model = pickle.load(model_file)
            return model

    def get_hour_points(self):
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

if __name__ == '__main__':
    minmeth = MinMeth()
    minmeth.train_and_save_model("train_logs/sint_data_url_50freq_modified.txt") #'train_logs/access3_features_sint.log'

    # Loading testing data

    test = minmeth.load_data_pandas('test_logs/BIG_TEST_TRANS_min_directory.txt', False)
    datatest_pandas = pd.DataFrame(minmeth.parse_sc_to_scp(test))
    X_test = minmeth.load_data_float(datatest_pandas)

    # Predicting and plotting
    #minmeth.predict(X_test, 4000)


    ## Plot hours
    test = minmeth.get_hour_points()
    test = pd.DataFrame(test)
   # datatest_pandas = pd.DataFrame(minmeth.parse_sc_to_scp(test))
    #X_test = minmeth.load_data_float(test)
    minmeth.predict(X_test, 4000, test)

