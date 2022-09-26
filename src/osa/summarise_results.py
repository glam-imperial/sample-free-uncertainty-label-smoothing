import statistics
import os.path

from common.common import load_pickle
from osa.configuration import DATA_FOLDER

OUTPUT_FOLDER = DATA_FOLDER + '/Results'


def trial_average(summary_list, name, return_list=False):
    value_list = list()
    for s in summary_list:
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
results_dict_secondary = dict()

for name in [
    "att-SEResNet20-attention-4",
    "SEResNet20-attention-4",
    "BD-attfix0-SEResNet20-attention-4",
    "smooth-BD-attfix0-SEResNet20-attention-4",
    "uniform-smooth-BD-attfix0-SEResNet20-attention-4",
    "BD-fix0-SEResNet20-attention-4",
    "smooth-BD-fix0-SEResNet20-attention-4",
    "uniform-smooth-BD-fix0-SEResNet20-attention-4",
]:

    print(name)
    trial_summaries = list()

    for t in range(40):
        if not os.path.exists(OUTPUT_FOLDER + "/" + name + "/results_summary_trial" + repr(t) + ".pkl"):
            continue
        print("Trial: ", t)

        filepath = OUTPUT_FOLDER + "/" + name + "/results_summary_trial" + repr(t) + ".pkl"
        try:
            results_summary = load_pickle(filepath)
        except FileNotFoundError:
            continue
        print(t, results_summary["whinny_single"]["au_pr"]["whinny_single"]["test_au_pr"])
        results_dict[name] = results_summary["whinny_single"]["au_pr"]["whinny_single"]
        trial_summaries.append(results_summary["whinny_single"]["au_pr"]["whinny_single"])


    print("Trial averages.")
    print("Best devel AU PR:", trial_average(trial_summaries, "best_devel_au_pr"))

    if True:
        print("Test  AU PR:    ", trial_average(trial_summaries, "test_au_pr"))
        print("Test  AU ROC:   ", trial_average(trial_summaries, "test_au_roc"))
        print("Test  MCC:   ", trial_average(trial_summaries, "test_mcc"))
        print("Test  ECE:   ", trial_average(trial_summaries, "test_ece"))
        print("Test  MCE:   ", trial_average(trial_summaries, "test_mce"))
        print("Test  Macro F1:       ", trial_average(trial_summaries, "test_macro_f1"))
        print("Test  Macro Recall:   ", trial_average(trial_summaries, "test_macro_recall"))
        print("Test  Macro Precision:", trial_average(trial_summaries, "test_macro_precision"))
        print("Test  POS F1:         ", trial_average(trial_summaries, "test_pos_f1"))
        print("Test  POS Recall:     ", trial_average(trial_summaries, "test_pos_recall"))
        print("Test  POS Precision:  ", trial_average(trial_summaries, "test_pos_precision"))
