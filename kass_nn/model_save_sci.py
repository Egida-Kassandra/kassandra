import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
from parse_logs import LogParser
import pandas as pd
import time
import pickle
import eif as iso
# https://stackabuse.com/scikit-learn-save-and-restore-models/

from sklearn.ensemble import IsolationForest
if __name__ == '__main__':
    rng = np.random.RandomState(42)

    # Generate train data
    logpar = LogParser()
    data_train = logpar.parse_file('access3_features.log', True)
   # print(data_train[34750]) # line
    print('========================================================')
    data_test = logpar.parse_file('testdata3.log', False)
    #print(data_test)
    data_pandas = pd.DataFrame(data_train)[[0,1,2,3,4]]
    #print(data_pandas)

    datatest_pandas = pd.DataFrame(data_test)[[0,1,2,3,4]]
    #print(datatest_pandas)
    data = np.array(data_train)
    data_test = np.array(data_test)
    print(data.shape)
    print(data_test.shape)
    #print(data)

    #X_train = np.array(data_train)
    #X_train.reshape(-1, 1)
    # Generate some regular novel observations

    #X_test = np.array(data[531460:])
    #data_test = parse_logs.parse_file('fool2.log')
    #data_test = data[:531469]
    #X_test = np.array(data_test)
    # Generate some abnormal novel observations

    #X_outliers = np.array(data[len(data)-1])
    #X_outliers = X_outliers.reshape(1, -1)

    # fit the model
    # necesario parametrizar el threshold, depende del dominio
    # para news: 0.03
    #clf = IsolationForest(n_estimators=500, max_samples=1000, contamination=0.03, random_state=0)
    # para tramsilverio: 0.005
    #clf = IsolationForest(n_estimators=37680, max_samples=512, contamination=0.01, random_state=0)
    #clf = IsolationForest(n_estimators=10000, max_samples=37680, contamination=0.01, random_state=rng, max_features=2, n_jobs=4)

    #clf = IsolationForest(n_estimators=2000, max_samples=1000, contamination=0.016, random_state=rng, max_features=5,
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
    #clf.fit(data_pandas)
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
    clf = iso.iForest(X_train, ntrees=4000, sample_size=15000, ExtensionLevel=1) # 2000, 10000
    # aumentar ntrees y sample_size: mejorar y probar en paralelo, max probado= 4000, 15000
    # calculate anomaly scores
    X_test = datatest_pandas.to_numpy().astype(np.float)
    print("PREDICT")
    start = time.time()
    anomaly_scores = clf.compute_paths(X_test)
    end = time.time()
    print(end - start)
    #anomaly_scores = [-r for r in anomaly_scores if r < 0.5]
    anomaly_scores_sorted = np.argsort(anomaly_scores)
    indices_with_preds = anomaly_scores_sorted[-int(np.ceil(0.9 * X_train.shape[0])):]
    print(indices_with_preds)
    print(anomaly_scores)
    """
    for i in range(0, len(anomaly_scores)):
        if anomaly_scores[i] > 0.55:
            anomaly_scores[i] = -anomaly_scores[i]
    print(anomaly_scores)
    count = sum(map(lambda x: x < 0, anomaly_scores))
    print('anomalies: ', count)
    """
    """
    # sort the scores
    anomaly_scores_sorted = np.argsort(anomaly_scores)
    # retrieve indices of anomalous observations
    indices_with_preds = anomaly_scores_sorted[-int(np.ceil(anomalies_ratio * X.shape[0])):]
    """
    """
    isoF_outliers_values = X_test[clf.predict(X_test) == -1]
    isoF_outliers_values = isoF_outliers_values.tolist()
    for i in isoF_outliers_values:
        if i in data_test:
            print(data_test.index(i))
    print(isoF_outliers_values)
    
    plt.figure(figsize=(12, 6))
    plt.hist(scores, bins=500)
    plt.show()
    """
    """
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    y_pred_outliers = clf.predict(X_outliers)


    print("Accuracy dev :", list(y_pred_test).count(1)/y_pred_test.shape[0])
    print("Accuracy test:", list(y_pred_outliers).count(-1)/y_pred_outliers.shape[0])

    
    # plot the line, the samples, and the nearest vectors to the plane
    xx, yy, zz = np.meshgrid(np.linspace(-10, 10, 50), np.linspace(-10, 10, 50),  np.linspace(-10, 10, 50))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.title("IsolationForest")
    plt.contourf(xx, yy, zz, Z, cmap=plt.cm.Blues_r)
    
    b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white',
                     s=20, edgecolor='k')
    b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green',
                     s=20, edgecolor='k')
    c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red',
                    s=20, edgecolor='k')
    plt.axis('tight')
    plt.xlim((-10, 10))
    plt.ylim((-10, 10))
    plt.legend([b1, b2, c],
               ["training observations",
                "new regular observations", "new abnormal observations"],
               loc="upper left")
    plt.show()
    """