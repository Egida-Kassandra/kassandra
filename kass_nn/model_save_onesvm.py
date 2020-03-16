import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
from kass_nn.parse_logs import LogParser
import pandas as pd
import time
from sklearn.neighbors import LocalOutlierFactor
import pickle
# import eif as iso
# from eif_old import iForest as iso
# import eif_old as iso
# https://stackabuse.com/scikit-learn-save-and-restore-models/

from sklearn.ensemble import IsolationForest

if __name__ == '__main__':
    rng = np.random.RandomState(42)

    # Generate train data
    logpar = LogParser()
    data_train = logpar.parse_file('./train_logs/access3_features.log', True)
    # print(data_train[34750]) # line
    print('========================================================')
    data_test = logpar.parse_file('./test_logs/testdata3.log', False)
    # print(data_test)
    data_pandas = pd.DataFrame(data_train)[[0, 1]]
    # print(data_pandas)

    datatest_pandas = pd.DataFrame(data_test)[[0, 1]]
    # print(datatest_pandas)
    data = np.array(data_train)
    data_test = np.array(data_test)
    print(data.shape)
    print(data_test.shape)
    # print(data)

    # X_train = np.array(data_train)
    # X_train.reshape(-1, 1)
    # Generate some regular novel observations

    # X_test = np.array(data[531460:])
    # data_test = parse_logs.parse_file('fool2.log')
    # data_test = data[:531469]
    # X_test = np.array(data_test)
    # Generate some abnormal novel observations

    # X_outliers = np.array(data[len(data)-1])
    # X_outliers = X_outliers.reshape(1, -1)

    # fit the model
    # necesario parametrizar el threshold, depende del dominio
    # para news: 0.03
    # clf = IsolationForest(n_estimators=500, max_samples=1000, contamination=0.03, random_state=0)
    # para tramsilverio: 0.005
    # clf = IsolationForest(n_estimators=37680, max_samples=512, contamination=0.01, random_state=0)
    # clf = IsolationForest(n_estimators=10000, max_samples=37680, contamination=0.01, random_state=rng, max_features=2, n_jobs=4)

    # clf = IsolationForest(n_estimators=2000, max_samples=1000, contamination=0.016, random_state=rng, max_features=5,
    #                     n_jobs=-1)

    # access1
    # dict ip= 1663 -> si
    # 1732 -> anomalia
    # 1703,1702,1701 -> anomalia
    # 1700 -> anomalia
    # 1698 -> anomalia

    # linea 34k -> si
    # 34228 -> si
    # 34689 -> si
    # 34700 -> si
    # 34780 -> si
    # 34800 -> si
    # 34810 -> si
    # 34814 -> si
    # 34815 -> si
    # 34816 -> anom
    # 34818 -> anom
    # 34820 -> anom
    # 34821 -> anom
    # 34822 -> anom
    # 34823 -> anom
    # 34824 -> anom
    # 34825 -> anom
    # 34826 -> anom
    # clf.fit(data_pandas)
    """
    print("DECISION FUNCTION")
    start = time.time()
    scores = clf.decision_function(datatest_pandas)
    end = time.time()
    print(end - start)
    print(scores)

    print("PREDICT")
    start = time.time()
    prediction = clf.predict(datatest_pandas)
    end = time.time()
    print(end - start)
    print(prediction)
    count = sum(map(lambda x: x < 0, prediction))
    print('anomalies: ', count)
    pkl_filename = "pickle_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(clf, file)
    """
    X_train = data_pandas.to_numpy().astype(np.float)
    print(X_train.shape[0])
    print(X_train[0])
    print(data_pandas[0])
    print("TRAIN")
    start = time.time()
    #clf = svm.OneClassSVM(nu=0.01, kernel="rbf", gamma=0.05)
    clf = LocalOutlierFactor(n_neighbors=10000, novelty=True,contamination=0.01)
    #clf.fit(X_train)
    clf.fit(X_train)
    end = time.time()
    print(end - start)
    # aumentar ntrees y sample_size: mejorar y probar en paralelo, max probado= 4000, 15000
    # calculate anomaly scores
    X_test = datatest_pandas.to_numpy().astype(np.float)
    print("PREDICT")
    start = time.time()
    anomaly_scores = clf.predict(X_test)
    end = time.time()
    print(end - start)

    # anomaly_scores = [-r for r in anomaly_scores if r < 0.5]
    #anomaly_scores_sorted = np.argsort(anomaly_scores)
    #indices_with_preds = anomaly_scores_sorted[-int(np.ceil(0.9 * X_train.shape[0])):]
   # print(indices_with_preds)
    print(anomaly_scores)
