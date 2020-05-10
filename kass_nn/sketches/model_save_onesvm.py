import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from kass_nn.util.parse_logs import LogParser
import pandas as pd
import time
# import eif as iso
# from eif_old import iForest as iso
# import eif_old as iso
# https://stackabuse.com/scikit-learn-save-and-restore-models/

from sklearn import preprocessing

if __name__ == '__main__':
    rng = np.random.RandomState(42)

    # Generate train data
    logpar = LogParser()
    data_train = logpar.parse_file('./train_logs/access3_features.log', True)
    # print(data_train[34750]) # line
    print('========================================================')
    data_test = logpar.parse_file('BIG_TEST_TRANS.txt', False)
    data_train = logpar.parse_file('./train_logs/access3_features.log', True)
    # print(data_train[34750]) # line
    print('========================================================')
    data_test = logpar.parse_file('./test_logs/testdata3.log', False)
    # print(data_test)
    data_pandas = pd.DataFrame(data_train)[[0, 4]]
    # print(data_pandas)

    datatest_pandas = pd.DataFrame(data_test)[[0, 4]]
    # print(datatest_pandas)
    data = np.array(data_train)
    data_test = np.array(data_test)
    print(data.shape)
    print(data_test.shape)

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
    X_test = datatest_pandas.to_numpy().astype(np.float)

    #X_train = X_train[:200]

    ## scale
    scaler = preprocessing.StandardScaler().fit(X_train)
    #X_train = scaler.transform(X_train)
    #X_test = scaler.transform(X_test)
    print(X_test)

    print(X_train.shape[0])
    print("TRAIN")
    start = time.time()
    clf = svm.OneClassSVM(nu=0.01, kernel="rbf", gamma="scale")
    #clf = LocalOutlierFactor(n_neighbors=10000, novelty=True,contamination=0.01)
    #clf.fit(X_train)
    clf.fit(X_train)
    end = time.time()
    print(end - start)
    # aumentar ntrees y sample_size: mejorar y probar en paralelo, max probado= 4000, 15000
    # calculate anomaly scores

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

    n_of_samples = len(X_train);
    X_train = X_train[:n_of_samples]
    x = np.array(data_pandas[0].tolist()[:n_of_samples])
    y = np.array(data_pandas[4].tolist()[:n_of_samples])

    #x = np.array(datatest_pandas[0].tolist())
    #y = np.array(datatest_pandas[1].tolist())

    #xx, yy = np.meshgrid(x, y)
    xx, yy = np.meshgrid(np.linspace(-500, 3000, 50), np.linspace(-120, 2000, 50))
    xx = xx.astype(np.float)
    yy = yy.astype(np.float)
    #anomaly_scores = clf.compute_paths(X_in=np.c_[xx.ravel(), yy.ravel()])
    clf.fit(X_train)
    anomaly_scores = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    print(anomaly_scores)
    print(clf.predict(X_test))

    S1 = anomaly_scores.reshape(xx.shape)
    plt.contourf(xx, yy, S1, cmap=plt.cm.YlOrRd_r)
    print("shape: ", X_train.shape)

    b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white',
                     s=20, edgecolor='k')
    #plt.scatter(2801, 0, c='white',s=20, edgecolor='k')

    plt.axis('tight')
    plt.xlim((-500, 3000))
    plt.ylim((-120, 2000))
    #plt.show()

    """
    =========================================
    f = plt.figure(figsize=(15, 6))
    ax2 = f.add_subplot(121)
    levels = np.linspace(np.min(S1), np.max(S1), 10)
    CS = ax2.contourf(x, y, S1, levels, cmap=plt.cm.YlOrRd)
    plt.scatter(x, y, s=15, c='None', edgecolor='k')
    """

    # test data
    ss0 = np.argsort(anomaly_scores)
    x = np.array(datatest_pandas[0].tolist())
    y = np.array(datatest_pandas[4].tolist())
    plt.scatter(x, y, s=5, c='b', edgecolor='b')

    plt.show()
