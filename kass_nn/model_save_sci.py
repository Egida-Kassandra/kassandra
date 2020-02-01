import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
from parse_logs import LogParser
import pandas as pd
import time

from sklearn.ensemble import IsolationForest
if __name__ == '__main__':
    rng = np.random.RandomState(42)

    # Generate train data
    logpar = LogParser()
    data_train = logpar.parse_file('access_news.log', True)
    data_test = logpar.parse_file('testdata.log', False)
    #print(data_test)
    data_pandas = pd.DataFrame(data_train)[[0,7]]
    #print(data_pandas)

    datatest_pandas = pd.DataFrame(data_test)[[0, 7]]
    print(datatest_pandas)
    data = np.array(data_train)
    data_test = np.array(data_test)
    print(data.shape)
    print(data_test.shape)
    #print(data)

    X_train = np.array(data_pandas)
    #X_train.reshape(-1, 1)
    # Generate some regular novel observations

    #X_test = np.array(data[531460:])
    #data_test = parse_logs.parse_file('fool2.log')
    #data_test = data[:531469]
    X_test = np.array(datatest_pandas)
    # Generate some abnormal novel observations

    #X_outliers = np.array(data[len(data)-1])
    #X_outliers = X_outliers.reshape(1, -1)

    # fit the model
    clf = IsolationForest(n_estimators=500, max_samples=1000, contamination=0.03, random_state=0) #0.04
    clf.fit(data_pandas)
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
    """
    isoF_outliers_values = X_test[clf.predict(X_test) == -1]
    isoF_outliers_values = isoF_outliers_values.tolist()
    for i in isoF_outliers_values:
        if i in data_test:
            print(data_test.index(i))
    print(isoF_outliers_values)
    """
    plt.figure(figsize=(12, 6))
    plt.hist(scores, bins=500)
    plt.show()
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