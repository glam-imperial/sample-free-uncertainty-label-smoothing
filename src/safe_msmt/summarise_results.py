import statistics
import os.path

from common.common import load_pickle
from safe_msmt.configuration import DATA_FOLDER

OUTPUT_FOLDER = DATA_FOLDER + '/Results'


def trial_average(summary_list, name, ignore_first_t=0, return_list=False):
    value_list = list()
    for s_i, s in enumerate(summary_list):
        if s_i < ignore_first_t:
            continue
        if name in s.keys():
            value_list.append(s[name])
    if len(value_list) > 1:
        m_v = statistics.mean(value_list)
        std_v = statistics.stdev(value_list)
        max_v = max(value_list)
    elif len(value_list) == 0:
        m_v = 0.0
        std_v = 0.0
        max_v = 0.0
    else:
        m_v = value_list[0]
        std_v = 0.0
        max_v = value_list[0]

    if return_list:
        return (m_v, std_v, max_v), value_list
    else:
        return (m_v, std_v, max_v)


results_dict = dict()

ignore_trials = {}

for name in [
    "att-SEResNet20-attention-4",
    "SEResNet20-attention-4",
    "BD-kle-10-fix0-SEResNet20-attention-4",
    "smooth-BD-kle-10-fix0-SEResNet20-attention-4",
    "uniform-smooth-BD-kle-10-fix0-SEResNet20-attention-4",
    "BD-kle-10-attfix0-SEResNet20-attention-4",
    "smooth-BD-kle-10-attfix0-SEResNet20-attention-4",
    "uniform-smooth-BD-kle-10-attfix0-SEResNet20-attention-4",
]:

    print(name)
    if name not in ignore_trials.keys():
        ignore_trials[name] = 0
    trial_summaries = list()

    for t in range(30):
        if not os.path.exists(OUTPUT_FOLDER + "/" + name + "/results_summary_trial" + repr(t) + ".pkl"):
            continue
        print("Trial: ", t)
        filepath = OUTPUT_FOLDER + "/" + name + "/results_summary_trial" + repr(t) + ".pkl"
        try:
            results_summary = load_pickle(filepath)
        except FileNotFoundError:
            continue
        print(t, results_summary["label"]["weighted_au_pr"]["label"]["test_weighted_au_pr"], results_summary["label"]["weighted_au_pr"]["label"]["test_weighted_ece"])
        results_dict[name] = results_summary["label"]["weighted_au_pr"]["label"]
        trial_summaries.append(results_summary["label"]["weighted_au_pr"]["label"])
        # print(results_summary["whinny_single"])
        # try:
        #     print("Best devel POS AU PR:", results_summary["whinny_single"]["best_devel_pos_au_pr"])
        # except KeyError:
        #     pass

        # if True:
        #     print("Test  Macro AU PR:    ", results_summary["whinny_single"]["test_macro_au_pr"])
        #     print("Test  Macro AU ROC:   ", results_summary["whinny_single"]["test_macro_au_roc"])
        #     print("Test  POS AU PR:      ", results_summary["whinny_single"]["test_pos_au_pr"])
        #     print("Test  NEG AU PR:      ", results_summary["whinny_single"]["test_neg_au_pr"])
        #     print("Test  Macro F1:       ", results_summary["whinny_single"]["test_macro_f1"])
        #     print("Test  Macro Recall:   ", results_summary["whinny_single"]["test_macro_recall"])
        #     print("Test  Macro Precision:", results_summary["whinny_single"]["test_macro_precision"])
        #     print("Test  POS F1:         ", results_summary["whinny_single"]["test_pos_f1"])
        #     print("Test  POS Recall:     ", results_summary["whinny_single"]["test_pos_recall"])
        #     print("Test  POS Precision:  ", results_summary["whinny_single"]["test_pos_precision"])

    print("Trial averages.")
    print("Best devel W AU PR:", trial_average(trial_summaries, "best_devel_weighted_au_pr", ignore_trials[name]))

    if True:
        print("Test  W AU PR:    ", trial_average(trial_summaries, "test_weighted_au_pr", ignore_trials[name]))
        print("Test  W AU ROC:   ", trial_average(trial_summaries, "test_weighted_au_roc", ignore_trials[name]))
        print("Test  W MCC:   ", trial_average(trial_summaries, "test_weighted_mcc", ignore_trials[name]))
        print("Test  W F1:   ", trial_average(trial_summaries, "test_weighted_pos_f1", ignore_trials[name]))
        print("Test  W M-F1:   ", trial_average(trial_summaries, "test_weighted_macro_f1", ignore_trials[name]))
        print("Test  W UAR:   ", trial_average(trial_summaries, "test_weighted_macro_recall", ignore_trials[name]))
        print("Test  W ECE:   ", trial_average(trial_summaries, "test_weighted_ece", ignore_trials[name]))
        print("Test  W MCE:   ", trial_average(trial_summaries, "test_weighted_mce", ignore_trials[name]))

