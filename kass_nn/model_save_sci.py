import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
from parse_logs import LogParser
import pandas as pd
import time
import pickle
import eif as iso
#from eif_old import iForest as iso
#import eif_old as iso
# https://stackabuse.com/scikit-learn-save-and-restore-models/

from sklearn.ensemble import IsolationForest
if __name__ == '__main__':
    rng = np.random.RandomState(42)

    # Generate train data
    logpar = LogParser()
    data_train = logpar.parse_file('train_logs/access3_features.log', True)
   # print(data_train[34750]) # line
    print('========================================================')
    data_test = logpar.parse_file('test_logs/BIG_TEST_TRANS.txt', False)
    #print(data_train)
    data_pandas = pd.DataFrame(data_train)[[3,4]]
    #print(data_pandas)

    datatest_pandas = pd.DataFrame(data_test)[[3,4]]
    #print(datatest_pandas)
    data = np.array(data_train)
    data_test = np.array(data_test)
    print(data.shape)
    print(data_test.shape)
    #print(data)


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


    #print(X_train.shape[0])
    #print(X_train[0])
    #print(data_pandas[0])
    print("TRAIN")
    start = time.time()
    clf = iso.iForest(X_train, ntrees=1500, sample_size=15000, ExtensionLevel=1) # 2000, 20000, ext 1 ###### cuidaooo
    end = time.time()
    print(end - start)
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
    print(np.sort(anomaly_scores))


    ss0=np.argsort(anomaly_scores)


    f = plt.figure(figsize=(15,6))
    x = np.array(data_pandas[3].tolist())
    y = np.array(data_pandas[4].tolist())
    plt.scatter(x, y, s=60, c='g', edgecolor='g')
    #plt.scatter(x[ss0[-10:]], y[ss0[-10:]], s=55, c='k')
    #plt.scatter(x[ss0[:10]], y[ss0[:10]], s=55, c='r')

    x = np.array(datatest_pandas[3].tolist())
    y = np.array(datatest_pandas[4].tolist())
    plt.scatter(x,y,s=20,c='b',edgecolor='b')
    plt.scatter(x[ss0[-5:]],y[ss0[-5:]],s=20,c='k')
    plt.scatter(x[ss0[:1]],y[ss0[:1]],s=20,c='r')
    plt.scatter(x[7], y[7], s=20, c='y')
    plt.title('extended')

    plt.show()

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