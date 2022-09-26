import sklearn
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_curve, precision_recall_curve


def get_multitask_classification_measures(true, pred, are_logits):
    multitask_measures = dict()
    per_task_measures = dict()
    y_measures_list = list()
    y_weight_list = list()

    number_of_classes = true.shape[1]

    for c in range(number_of_classes):
        true_c = true[:, c]
        pred_c = pred[:, c]
        y_measures_list.append(get_binary_classification_measures(true_c,
                                                                  pred_c,
                                                                  are_logits))
        y_weight_list.append(true_c.sum(axis=0))
    y_weight_list = np.array(y_weight_list, dtype=np.float32)
    y_weight_list = y_weight_list / np.sum(y_weight_list)
    y_weight_list = list(y_weight_list)

    for measure_name in ["accuracy",
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
                         "ece",
                         "mce"]:
        multitask_measures["macro_" + measure_name] = np.mean([m[measure_name] for m in y_measures_list])
        multitask_measures["weighted_" + measure_name] = np.sum([m[measure_name] * w for m, w in zip(y_measures_list, y_weight_list)])

    for c in range(number_of_classes):
        per_task_measures["label_" + repr(c)] = y_measures_list[c]

    return multitask_measures, per_task_measures


def get_binary_classification_measures(true, pred, are_logits):
    target_measures = dict()

    true_indicator = true

    if are_logits:
        pred_logits = pred
        pred_logits = np.nan_to_num(pred_logits)
        pred_prob = sigmoid(pred_logits)
    else:
        pred_prob = pred
        pred_prob = np.nan_to_num(pred_prob)

    true_labels = true_indicator
    pred_labels = make_indicator_from_probabilities(pred_prob,
                                                    0.5)

    # Accuracy.
    accuracy = sklearn.metrics.accuracy_score(true_labels, pred_labels, normalize=True, sample_weight=None)
    target_measures["accuracy"] = accuracy

    # AU-ROC.
    au_roc_macro = sklearn.metrics.roc_auc_score(true_indicator, pred_prob, average="macro")
    target_measures["au_roc"] = au_roc_macro

    # AU-PR
    au_prc_macro = sklearn.metrics.average_precision_score(true_indicator, pred_prob, average="macro")
    target_measures["au_pr"] = au_prc_macro

    # Precision, Recall, F1
    precision_classes, recall_classes, f1_classes, _ = sklearn.metrics.precision_recall_fscore_support(
        true_labels,
        pred_labels,
        zero_division=0,
        average=None)
    precision_micro, recall_micro, f1_micro, _ = sklearn.metrics.precision_recall_fscore_support(true_labels,
                                                                                                 pred_labels,
                                                                                                 zero_division=0,
                                                                                                 average="micro")

    # MCC
    mcc = binary_matthews_correlation_coefficient(true_labels,
                                                  pred_labels)

    # Calibration.
    true_labels = np.reshape(true_labels, (true_labels.size, 1))
    true_labels = np.hstack([1.0 - true_labels, true_labels])
    pred_prob = np.reshape(pred_prob, (pred_prob.size, 1))
    pred_prob = np.hstack([1.0 - pred_prob, pred_prob])
    cal = calibration(true_labels, pred_prob, num_bins=10)

    target_measures["pos_precision"] = precision_classes[1]
    target_measures["macro_precision"] = np.mean(precision_classes)
    target_measures["micro_precision"] = precision_micro
    target_measures["pos_recall"] = recall_classes[1]
    target_measures["macro_recall"] = np.mean(recall_classes)
    target_measures["micro_recall"] = recall_micro
    target_measures["pos_f1"] = f1_classes[1]
    target_measures["macro_f1"] = np.mean(f1_classes)
    target_measures["micro_f1"] = f1_micro
    target_measures["mcc"] = mcc
    target_measures["ece"] = cal["ece"]
    target_measures["mce"] = cal["mce"]

    return target_measures


def make_indicator_from_probabilities(y_pred_prob,
                                      threshold):
    y_pred_indicator = np.zeros_like(y_pred_prob)
    y_pred_indicator[y_pred_prob >= threshold] = 1.0
    y_pred_indicator[y_pred_prob < threshold] = 0.0

    return y_pred_indicator


def binary_matthews_correlation_coefficient(y_true,
                                            y_pred):  # These are labels; not probabilities or logits.
    # y_pos_true = y_true[:, 1]
    y_pos_true = y_true
    # y_pos_pred_indicator = y_pred[:, 1]
    y_pos_pred_indicator = y_pred

    TP = np.count_nonzero(np.multiply(y_pos_pred_indicator,
                                      y_pos_true))
    TN = np.count_nonzero(np.multiply((y_pos_pred_indicator - 1.0),
                                      (y_pos_true - 1.0)))
    FP = np.count_nonzero(np.multiply(y_pos_pred_indicator,
                                      (y_pos_true - 1.0)))
    FN = np.count_nonzero(np.multiply((y_pos_pred_indicator - 1.0),
                                      y_pos_true))

    if np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) == 0.0:
        mcc = 0.0
    else:
        mcc = ((TP * TN) - (FP * FN)) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    return mcc


def get_rates_and_thresholds(y_true,
                             y_pred,
                             curve_function_type):
    measures_at_threshold = (dict(), dict())
    thresholds = dict()

    if y_true.shape[-1] != y_pred.shape[-1]:
        raise ValueError("Y true and y pred do not cover the same number of classes.")

    number_of_classes = y_true.shape[-1]

    for i in range(number_of_classes):
        if curve_function_type == "ROC":
            measures_at_threshold[0][i],\
            measures_at_threshold[1][i],\
            thresholds[i] = roc_curve(y_true[:, i],
                                      y_pred[:, i],
                                      drop_intermediate=False)
        elif curve_function_type == "PR":
            measures_at_threshold[0][i],\
            measures_at_threshold[1][i], \
            thresholds[i] = precision_recall_curve(y_true[:, i],
                                                   y_pred[:, i])
        else:
            raise ValueError("Invalid curve function type.")

    return measures_at_threshold,\
           thresholds,\
           number_of_classes


def get_optimal_threshold_per_class(measures_at_threshold,
                                    thresholds,
                                    number_of_classes,
                                    measure_function_type):
    measure_per_class = [None] * number_of_classes
    optimal_threshold_per_class = [None] * number_of_classes

    for i in range(number_of_classes):
        if measure_function_type == "J":
            measure_per_class[i] = youden_j_statistic(fpr=measures_at_threshold[0][i],
                                                      tpr=measures_at_threshold[1][i])
        elif measure_function_type == "F1":
            measure_per_class[i] = f1_measure(precision=measures_at_threshold[0][i],
                                              recall=measures_at_threshold[1][i])
        else:
            raise ValueError("Invalid measure function type.")
        optimal_threshold_per_class[i] = thresholds[i][np.argmax(measure_per_class[i])]

    return measure_per_class, optimal_threshold_per_class


def youden_j_statistic(fpr, tpr):
    return tpr - fpr


def f1_measure(precision, recall):
    return (2 * precision * recall) / (precision + recall)


def stable_softmax(X):
    exps = np.exp(X - np.max(X, 1).reshape((X.shape[0], 1)))
    return exps / np.sum(exps, 1).reshape((X.shape[0], 1))


def sigmoid(x):
    x = np.nan_to_num(x)
    return expit(x)
    # return 1. / (1. + np.exp(-x))


def calibration(y, p_mean, num_bins=10):
    """Compute the calibration. -- https://github.com/google-research/google-research/blob/master/uncertainties/sources/postprocessing/metrics.py
    References:
    https://arxiv.org/abs/1706.04599
    https://arxiv.org/abs/1807.00263
    Args:
        y: one-hot encoding of the true classes, size (?, num_classes)
        p_mean: numpy array, size (?, num_classes)
                containing the mean output predicted probabilities
        num_bins: number of bins
    Returns:
        cal: a dictionary
             {reliability_diag: realibility diagram
        ece: Expected Calibration Error
        mce: Maximum Calibration Error
             }
    """
    # Compute for every test sample x, the predicted class.
    class_pred = np.argmax(p_mean, axis=1)
    # and the confidence (probability) associated with it.
    conf = np.max(p_mean, axis=1)
    # Convert y from one-hot encoding to the number of the class
    y = np.argmax(y, axis=1)
    # Storage
    acc_tab = np.zeros(num_bins)  # empirical (true) confidence
    mean_conf = np.zeros(num_bins)  # predicted confidence
    nb_items_bin = np.zeros(num_bins)  # number of items in the bins
    tau_tab = np.linspace(0, 1, num_bins+1)  # confidence bins
    for i in np.arange(num_bins):  # iterate over the bins
        # select the items where the predicted max probability falls in the bin
        # [tau_tab[i], tau_tab[i + 1)]
        sec = (tau_tab[i + 1] > conf) & (conf >= tau_tab[i])
        nb_items_bin[i] = np.sum(sec)  # Number of items in the bin
        # select the predicted classes, and the true classes
        class_pred_sec, y_sec = class_pred[sec], y[sec]
        # average of the predicted max probabilities
        mean_conf[i] = np.mean(conf[sec]) if nb_items_bin[i] > 0 else np.nan
        # compute the empirical confidence
        acc_tab[i] = np.mean(
          class_pred_sec == y_sec) if nb_items_bin[i] > 0 else np.nan

    # Cleaning
    mean_conf = mean_conf[nb_items_bin > 0]
    acc_tab = acc_tab[nb_items_bin > 0]
    nb_items_bin = nb_items_bin[nb_items_bin > 0]

    # Reliability diagram
    reliability_diag = (mean_conf, acc_tab)

    weights = nb_items_bin.astype(np.float) / np.sum(nb_items_bin)
    if np.sum(weights) == 0.0:
        weights = np.ones_like(nb_items_bin.astype(np.float)) / num_bins

    # Expected Calibration Error
    ece = np.average(
        np.absolute(mean_conf - acc_tab),
        weights=weights)
    # Maximum Calibration Error
    mce = np.max(np.absolute(mean_conf - acc_tab))
    # Saving
    cal = {'reliability_diag': reliability_diag,
           'ece': ece,
           'mce': mce}
    return cal
