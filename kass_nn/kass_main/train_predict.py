from kass_nn.characteristics.min_vs_file_ext import MinFileExt
from kass_nn.characteristics.min_vs_long_req import MinLong
from kass_nn.characteristics.min_vs_meth import MinMeth
from kass_nn.characteristics.min_vs_url_directory import MinDir
import kass_nn.characteristics.characteristic as charac
from kass_nn.util import kass_plotter as plt

from kass_nn.util.parse_logs import LogParser

class TrainPredict:

    def __init__(self, train_filename):
        """Constructor"""
        self.train_filename = train_filename
        self.logpar = LogParser(train_filename)
        self.min_meth = None
        self.min_dir = None
        self.min_file_ext = None
        self.min_long = None
        self.min_meth_clf = None
        self.min_dir_clf = None
        self.min_file_ext_clf = None
        self.min_long_clf = None

    def train_all(self):
        print("Min vs Meth")
        self.min_meth = MinMeth(self.logpar)
        self.min_meth_clf = charac.get_eif(self.min_meth)
        #plt.plot_model(min_meth.X_train, min_meth.X_test, min_meth_pred, min_meth_clf, min_meth.mesh)

        print("Min vs Dir")
        self.min_dir = MinDir(self.logpar)
        self.min_dir_clf = charac.get_eif(self.min_dir)

        print("Min vs FileExt")
        self.min_file_ext = MinFileExt(self.logpar)
        self.min_file_ext_clf = charac.get_eif(self.min_file_ext)

        print("Min vs Long")
        self.min_long = MinLong(self.logpar)
        self.min_long_clf = charac.get_eif(self.min_long)




    def predict_all(self, test_filename):
        min_meth_pred = charac.get_prediction(test_filename, self.min_meth, self.min_meth_clf)[0]
        min_dir_pred = charac.get_prediction(test_filename, self.min_dir, self.min_dir_clf)[0]
        min_file_ext_pred = charac.get_prediction(test_filename, self.min_file_ext, self.min_file_ext_clf)[0]
        min_long_pred = charac.get_prediction(test_filename, self.min_long, self.min_long_clf)[0]
        print("=" * 80)
        print("RESULTS")
        print("\tMin vs Meth: {}".format(min_meth_pred))
        print("\tMin vs Dir: {}".format(min_dir_pred))
        print("\tMin vs FileExt: {}".format(min_file_ext_pred))
        print("\tMin vs Long: {}".format(min_long_pred))

