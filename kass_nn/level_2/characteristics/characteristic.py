from kass_nn.level_2.eif import eif


def get_eif(charac):
    # Loading training data
    X_train = eif.load_parsed_data("", True, charac)
    charac.X_train = X_train
    # Training model
    clf = eif.train_model(X_train, charac)
    # Return model
    return clf


def get_prediction(test_filename, charac, clf):
    # Loading testing data
    X_test = eif.load_parsed_data(test_filename, False, charac)
    charac.X_test = X_test
    # Predicting model
    anomaly_scores = eif.predict_wo_train(X_test, clf)
    return anomaly_scores


