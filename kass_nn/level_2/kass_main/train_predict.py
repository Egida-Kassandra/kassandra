from kass_nn.level_2.characteristics.min_vs_file_ext import MinFileExt
from kass_nn.level_2.characteristics.min_vs_long_req import MinLong
from kass_nn.level_2.characteristics.min_vs_meth import MinMeth
from kass_nn.level_2.characteristics.min_vs_url_directory import MinDir
import kass_nn.level_2.characteristics.characteristic as charac
from kass_nn.util import kass_plotter as plt
import kass_nn.level_2.danger_labeling.dangerousness as dang
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

    def train_all(self):
        print("Min vs Meth")
        self.min_meth = MinMeth(self.logpar)
        self.min_meth.clf = charac.get_eif(self.min_meth)

        print("Min vs Dir")
        self.min_dir = MinDir(self.logpar)
        self.min_dir.clf = charac.get_eif(self.min_dir)

        print("Min vs FileExt")
        self.min_file_ext = MinFileExt(self.logpar)
        self.min_file_ext.clf = charac.get_eif(self.min_file_ext)

        print("Min vs Long")
        self.min_long = MinLong(self.logpar)
        self.min_long.clf = charac.get_eif(self.min_long)

    def predict_all(self, test_filename):
        min_meth_pred = charac.get_prediction(test_filename, self.min_meth, self.min_meth.clf)[0]
        min_dir_pred = charac.get_prediction(test_filename, self.min_dir, self.min_dir.clf)[0]
        min_file_ext_pred = charac.get_prediction(test_filename, self.min_file_ext, self.min_file_ext.clf)[0]
        min_long_pred = charac.get_prediction(test_filename, self.min_long, self.min_long.clf)[0]

        print("=" * 80)
        print("RESULTS")
        print("\tMin vs Meth: {}".format(min_meth_pred))
        print("\tMin vs Dir: {}".format(min_dir_pred))
        print("\tMin vs FileExt: {}".format(min_file_ext_pred))
        print("\tMin vs Long: {}".format(min_long_pred))

        anomaly_scores = [min_meth_pred, min_dir_pred, min_file_ext_pred, min_long_pred]
        print("=" * 80)
        print(dang.get_dangerousness_label(anomaly_scores))

        self.plot_dangerousness(min_meth_pred, min_dir_pred, min_file_ext_pred, min_long_pred)




    def plot_dangerousness(self, min_meth_pred, min_dir_pred, min_file_ext_pred, min_long_pred):
        fig = plt.open_plot()
        plt.plot_model(fig, self.min_meth.X_train, self.min_meth.X_test, min_meth_pred, self.min_meth.clf,
                       self.min_meth.mesh, [2, 2, 1], "Min vs Meth")
        plt.plot_model(fig, self.min_dir.X_train, self.min_dir.X_test, min_dir_pred, self.min_dir.clf,
                       self.min_dir.mesh, [2, 2, 2], "Min vs Dir")
        if min_file_ext_pred is not None:
            plt.plot_model(fig, self.min_file_ext.X_train, self.min_file_ext.X_test, min_file_ext_pred,
                           self.min_file_ext.clf,
                           self.min_file_ext.mesh, [2, 2, 3], "Min vs FileExt")
        plt.plot_model(fig, self.min_long.X_train, self.min_long.X_test, min_long_pred, self.min_long.clf,
                       self.min_long.mesh, [2, 2, 4], "Min vs Long")

        plt.close_plot()


