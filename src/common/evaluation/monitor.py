import math

import numpy as np

from common.common import safe_make_dir
from variational.layers import VariationalLayer, DenseReparameterisation, Conv2dReparameterization, VariationalGRUCell


BEST_VALUE_INITIALISER = dict()
MONITOR_FUNCTION = dict()

for M in ["accuracy",
          "au_roc",
          "au_pr",
          "pos_precision",
          "macro_precision",
          "micro_precision",
          "pos_recall",
          "macro_recall",
          "micro_recall",
          "pos_f1",
          "macro_f1",
          "micro_f1",
          "mcc",
          "macro_accuracy",
          "weighted_accuracy",
          "macro_au_roc",
          "weighted_au_roc",
          "macro_au_pr",
          "weighted_au_pr",
          "macro_pos_precision",
          "weighted_pos_precision",
          "macro_macro_precision",
          "weighted_macro_precision",
          "macro_micro_precision",
          "weighted_micro_precision",
          "macro_pos_recall",
          "weighted_pos_recall",
          "macro_macro_recall",
          "weighted_macro_recall",
          "macro_micro_recall",
          "weighted_micro_recall",
          "macro_pos_f1",
          "weighted_pos_f1",
          "macro_macro_f1",
          "weighted_macro_f1",
          "macro_micro_f1",
          "weighted_micro_f1",
          "macro_mcc",
          "weighted_mcc"
          ]:

    BEST_VALUE_INITIALISER[M] = -1.0
    MONITOR_FUNCTION[M] = lambda best, new: best < new

for M in ["ece",
          "mce",
          "macro_ece",
          "weighted_ece",
          "macro_mce"
          "weighted_mce"]:

    BEST_VALUE_INITIALISER[M] = math.inf
    MONITOR_FUNCTION[M] = lambda best, new: best > new


class CustomSaverVirtual:
    def __init__(self,
                 output_folder,
                 method_string,
                 monitor_target_to_measures,
                 keras_model_test):
        self.output_folder = output_folder
        self.method_string = method_string
        self.monitor_target_to_measures = monitor_target_to_measures
        self.keras_model_test = keras_model_test
        self.saver_paths = dict()
        self.saver_dict = dict()

        self.method_output_prefix = output_folder + "/" + method_string

        safe_make_dir(self.method_output_prefix)

        for monitor_target in monitor_target_to_measures.keys():
            self.saver_paths[monitor_target] = dict()
            self.saver_dict[monitor_target] = dict()
            safe_make_dir(self.method_output_prefix + "/" + monitor_target)
            for monitor_measure in monitor_target_to_measures[monitor_target]:
                safe_make_dir(self.method_output_prefix + "/" + monitor_target + "/" + monitor_measure)
                self.saver_paths[monitor_target][monitor_measure] = self.method_output_prefix + "/" + monitor_target + "/" + monitor_measure + "/"
                self.saver_dict[monitor_target][monitor_measure] = self.keras_model_test
                # safe_make_dir(self.method_output_prefix + "/" + target + "/" + measure)

    def save_model(self,
                   target,
                   measure):
        # self.saver_dict[target][measure].save(self.saver_paths[target][measure] + "_model")
        self.saver_dict[target][measure].save_weights(self.saver_paths[target][measure] + "_model")

    def load_model(self,
                   target,
                   measure,
                   custom_objects):
        print(custom_objects)

        # self.saver_dict[target][measure] = tf.saved_model.load(self.saver_paths[target][measure] + "_model",
        #                                                               custom_objects=custom_objects)
        # self.saver_dict[target][measure] = tf.keras.models.load_model(self.saver_paths[target][measure] + "_model",
        #                                                               custom_objects=custom_objects)
        self.saver_dict[target][measure].load_weights(self.saver_paths[target][measure] + "_model")
        return self.saver_dict[target][measure]


class PerformanceMonitorVirtual:
    def __init__(self,
                 output_folder,
                 method_string,
                 custom_saver,
                 monitor_target_to_measures,
                 report_target_to_measures,
                 are_test_labels_available,
                 y_pred_names,
                 model_configuration):
        self.output_folder = output_folder
        self.method_string = method_string
        self.custom_saver = custom_saver
        self.monitor_target_to_measures = monitor_target_to_measures
        self.report_target_to_measures = report_target_to_measures
        self.are_test_labels_available = are_test_labels_available
        self.y_pred_names = y_pred_names
        self.model_configuration = model_configuration

        self.method_output_prefix = output_folder + "/" + method_string

        # Contains measure summary for last run.
        self.measures = dict()

        # I may want to monitor multiple performance measures per multiple tasks/targets separately.

        # Contains test items and summary, dependent on target and measure
        self.test_measures_dict = dict()
        self.test_items_dict = dict()
        self.best_performance_dict = dict()
        self.monitor_function_dict = dict()
        for monitor_target in self.monitor_target_to_measures.keys():
            self.best_performance_dict[monitor_target] = dict()
            self.monitor_function_dict[monitor_target] = dict()
            self.test_measures_dict[monitor_target] = dict()
            self.test_items_dict[monitor_target] = dict()
            for monitor_measure in self.monitor_target_to_measures[monitor_target]:
                self.best_performance_dict[monitor_target][monitor_measure] = dict()
                self.monitor_function_dict[monitor_target][monitor_measure] = dict()
                for report_target in self.report_target_to_measures.keys():
                    self.best_performance_dict[monitor_target][monitor_measure][report_target] = dict()
                    self.monitor_function_dict[monitor_target][monitor_measure][report_target] = dict()
                    for report_measure in self.report_target_to_measures[report_target]:
                        self.best_performance_dict[monitor_target][monitor_measure][report_target][report_measure] =\
                            BEST_VALUE_INITIALISER[monitor_measure]
                        self.monitor_function_dict[monitor_target][monitor_measure][report_target][report_measure] =\
                            MONITOR_FUNCTION[monitor_measure]

    def get_measures(self,
                     items,
                     partition):
        raise NotImplementedError

    def report_measures(self,
                        partition,
                        output_channel_targets=None):
        measures = self.measures[partition]

        for report_target in self.report_target_to_measures.keys():
            if output_channel_targets is not None:
                if report_target not in output_channel_targets:
                    continue
            print(partition + " measures on: " + report_target)
            for report_measure in self.report_target_to_measures[report_target]:
                print(report_measure + ":", measures[report_target][report_measure])

    def monitor_improvement(self):
        noticed_improvement = False

        for monitor_target in self.monitor_target_to_measures.keys():
            for monitor_measure in self.monitor_target_to_measures[monitor_target]:
                if self.monitor_function_dict[monitor_target][monitor_measure][monitor_target][monitor_measure](self.best_performance_dict[monitor_target][monitor_measure][monitor_target][monitor_measure],
                                                                                                                self.measures["devel"][monitor_target][monitor_measure]):
                    self.best_performance_dict[monitor_target][monitor_measure] = self.measures["devel"]
                    noticed_improvement = True
                    self.custom_saver.save_model(target=monitor_target,
                                                 measure=monitor_measure)
        return noticed_improvement

    def get_test_measures(self,
                          test_items,
                          target,
                          measure):
        if self.are_test_labels_available:
            self.get_measures(items=test_items,
                              partition="test")
            self.test_measures_dict[target][measure] = self.measures["test"]
        self.test_items_dict[target][measure] = test_items

    def report_best_performance_measures(self,
                                         output_channel_targets=None):
        for monitor_target in self.monitor_target_to_measures.keys():
            for monitor_measure in self.monitor_target_to_measures[monitor_target]:
                print("Model selected on " + monitor_measure + " of " + monitor_target)
                print("Best devel " + monitor_measure + ":", self.best_performance_dict[monitor_target][monitor_measure][monitor_target][monitor_measure])

                if self.are_test_labels_available:
                    for report_target in self.report_target_to_measures.keys():
                        if output_channel_targets is not None:
                            if report_target not in output_channel_targets:
                                continue
                        print("Test measures on: " + report_target)
                        for report_measure in self.report_target_to_measures[report_target]:
                            print(report_measure + ":",
                                  self.test_measures_dict[monitor_target][monitor_measure][report_target][report_measure])

    def get_results_summary(self):
        results = dict()
        items = dict()

        results["method_string"] = self.method_string
        items["method_string"] = self.method_string

        for monitor_target in self.monitor_target_to_measures.keys():
            results[monitor_target] = dict()
            items[monitor_target] = dict()
            for monitor_measure in self.monitor_target_to_measures[monitor_target]:
                results[monitor_target][monitor_measure] = dict()
                items[monitor_target][monitor_measure] = dict()
                for report_target in self.report_target_to_measures.keys():
                    results[monitor_target][monitor_measure][report_target] = dict()

                    for report_measure in self.report_target_to_measures[report_target]:
                        results[monitor_target][monitor_measure][report_target]["best_devel_" + report_measure] = self.best_performance_dict[monitor_target][monitor_measure][report_target][report_measure]

                for y_pred_name in self.y_pred_names:
                    items[monitor_target][monitor_measure][y_pred_name] = dict()

                    items[monitor_target][monitor_measure][y_pred_name]["test_pred"] = \
                        self.test_items_dict[monitor_target][monitor_measure][y_pred_name]["pred"]
                    np.save(
                        self.output_folder + "/" + self.method_string + "/" + monitor_target + "/" + monitor_measure + "/" + y_pred_name + "_" + "test_pred.npy",
                        self.test_items_dict[monitor_target][monitor_measure][y_pred_name]["pred"])

        if self.are_test_labels_available:
            for monitor_target in self.monitor_target_to_measures:
                for monitor_measure in self.monitor_target_to_measures[monitor_target]:
                    # if monitor_target in self.test_measures_dict[monitor_target][monitor_measure].keys():
                    #     for report_measure in self.test_measures_dict[monitor_target][monitor_measure][monitor_target].keys():
                    for report_target in self.report_target_to_measures.keys():
                        for report_measure in self.report_target_to_measures[report_target]:
                            results[monitor_target][monitor_measure][report_target]["test_" + report_measure] = self.test_measures_dict[monitor_target][monitor_measure][report_target][report_measure]

                    for y_pred_name in self.y_pred_names:
                        items[monitor_target][monitor_measure][y_pred_name]["test_true"] = \
                            self.test_items_dict[monitor_target][monitor_measure][y_pred_name]["true"]
                        np.save(
                            self.output_folder + "/" + self.method_string + "/" + monitor_target + "/" + monitor_measure + "/" + y_pred_name + "_" + "test_true.npy",
                            self.test_items_dict[monitor_target][monitor_measure][y_pred_name]["true"])

        return results, items
