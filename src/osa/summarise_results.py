import statistics
import os.path

from common.common import load_pickle
from osa.configuration import DATA_FOLDER, OUTPUT_FOLDER


def trial_average(summary_list, name, return_list=False):
    value_list = list()
    for s_i, s in enumerate(summary_list):
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
    "WideResNet",
    "max-SEResNet20-attention-4",
    "BNN-max-SEResNet20-attention-4",
    "uniform-smooth-BNN-max-SEResNet20-attention-4",
    "ua-smooth-BNN-max-SEResNet20-attention-4",
    "att-SEResNet20-attention-4",
    "BNN-att-SEResNet20-attention-4",
    "uniform-smooth-BNN-att-SEResNet20-attention-4",
    "ua-smooth-BNN-att-SEResNet20-attention-4"
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
        print("Test  Macro F1:       ", trial_average(trial_summaries, "test_macro_f1"))
        print("Test  Macro Recall:   ", trial_average(trial_summaries, "test_macro_recall"))
        print("Test  ECE:   ", trial_average(trial_summaries, "test_ece"))
