import time
from kass_nn.kass_main.train_predict import TrainPredict


if __name__ == '__main__':
    train_filename = "../train_logs/min_dir/sint_data_url_50freq_modified.txt" #"../train_logs/access3_features_sint.log"
    test_filename = "../test_logs/min_file_ext/BIG_TEST_TRANS_min_file_ext.txt"
    start = time.time()
    trainpredict = TrainPredict(train_filename)
    trainpredict.train_all()
    end = time.time()
    print(end - start)
    trainpredict.predict_all(test_filename)


