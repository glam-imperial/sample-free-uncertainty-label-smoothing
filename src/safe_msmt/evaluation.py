from common.evaluation.monitor import PerformanceMonitorVirtual, CustomSaverVirtual
from common.evaluation.measures import get_multitask_classification_measures
from variational.activations import sigmoid_moments_np


class CustomSaver(CustomSaverVirtual):
    def __init__(self,
                 output_folder,
                 method_string,
                 monitor_target_to_measures,
                 keras_model_test):
        super().__init__(output_folder,
                         method_string,
                         monitor_target_to_measures,
                         keras_model_test)


class PerformanceMonitor(PerformanceMonitorVirtual):
    def __init__(self,
                 output_folder,
                 method_string,
                 custom_saver,
                 monitor_target_to_measures,
                 report_target_to_measures,
                 are_test_labels_available,
                 y_pred_names,
                 model_configuration):
        super().__init__(output_folder,
                         method_string,
                         custom_saver,
                         monitor_target_to_measures,
                         report_target_to_measures,
                         are_test_labels_available,
                         y_pred_names,
                         model_configuration)

    def get_measures(self,
                     items,
                     partition):
        global_pooling = self.model_configuration["global_pooling"]
        if "bayesian" in self.model_configuration.keys():
            if "use_logit_vars" in self.model_configuration["bayesian"].keys():
                use_logit_vars = self.model_configuration["bayesian"]["use_logit_vars"]
            else:
                use_logit_vars = False
        else:
            use_logit_vars = False

        measures = dict()

        for y_pred_name in self.y_pred_names:
            # report_target here is "whinny_single"
            # y_pred_name here also happens to be "whinny_single"
            if global_pooling == "Prediction":
                multitask_measures,\
                per_task_measures = get_multitask_classification_measures(true=items[y_pred_name]["true"],
                                                                          pred=items[y_pred_name]["pred"],
                                                                          are_logits=False)
            else:
                if use_logit_vars:
                    pred = sigmoid_moments_np(items[y_pred_name]["pred"][:, :30],
                                              items[y_pred_name]["pred"][:, 30:])

                    multitask_measures,\
                    per_task_measures = get_multitask_classification_measures(true=items[y_pred_name]["true"],
                                                                              pred=pred,
                                                                              are_logits=False)
                else:
                    multitask_measures, \
                    per_task_measures = get_multitask_classification_measures(true=items[y_pred_name]["true"],
                                                                              pred=items[y_pred_name]["pred"],
                                                                              are_logits=True)
            measures = {k: v for k, v in per_task_measures.items()}
            measures["label"] = multitask_measures

        self.measures[partition] = measures
