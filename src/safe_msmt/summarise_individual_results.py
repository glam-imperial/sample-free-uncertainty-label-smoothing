import statistics
import os.path
import numpy as np

from common.common import load_pickle
from safe_msmt.configuration import DATA_FOLDER, OUTPUT_FOLDER
from common.evaluation.measures import get_binary_classification_measures, get_multitask_classification_measures

iucn_status = \
{
0: ("little spiderhunter", "least concern"),
1: ("bushy-crested hornbill", "near threatened"),
2: ("banded bay cuckoo", "least concern"),
3: ("grey-headed babbler", "least concern"),
4: ("chestnut-backed scimitar-babbler", "least concern"),
5: ("brown fulvetta", "near threatened"),
6: ("blue-eared barbet", "least concern"),
7: ("rhinoceros hornbill", "vulnerable"),
8: ("rufous-tailed shama", "near threatened"),
9: ("rufous tailed tailorbird", "least concern"),
10: ("black-naped monarch", "least concern"),
11: ("slender-billed crow", "least concern"),
12: ("buff-vented bulbul", "near threatened"),
13: ("ferruginous babbler", "least concern"),
14: ("black-capped babbler", "least concern"),
15: ("chestnut-rumped babbler", "near threatened"),
16: ("yellow-vented bulbul", "least concern"),
17: ("fluffy-backed tit-babbler", "near threatened"),
18: ("bornean gibbon", "endangered"),
19: ("dark-necked tailorbird", "least concern"),
20: ("rufous-fronted babbler", "least concern"),
21: ("ashy tailorbird", "least_concern"),
22: ("pied fantail", "least concern"),
23: ("short-tailed babbler", "near threatened"),
24: ("plaintive cuckoo", "least concern"),
25: ("sooty-capped babbler", "near threatened"),
26: ("spectacled bulbul", "least concern"),
27: ("chestnut-winged babbler", "least concern"),
28: ("bold-striped tit-babbler", "least concern"),
29: ("black-headed bulbul", "least concern"),
}

endangered_classes = [1, 5, 7, 8, 12, 15, 17, 18, 23, 25]
endangered_classes = np.array(endangered_classes, dtype=np.int32)

num_calls = [70, 23, 24, 229, 15, 13, 93, 55, 10, 19, 19, 10, 57, 77, 61, 56, 33, 11,  4, 36, 63, 57, 60, 48, 31, 110, 25, 74, 99, 16]
classes =   [ 0,  1,  2,   3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,  24, 25, 26, 27, 28, 29]
class_names = ['little spiderhunter',  # 0
               'bushy-crested hornbill',  # 1
               'banded bay cuckoo',  # 2
               'grey-headed babbler',  # 3
               'chestnut-backed scimitar-babbler',  # 4
               'brown fulvetta',  # 5
               'blue-eared barbet',  # 6
               'rhinoceros hornbill',  # 7
               'rufous-tailed shama',  # 8
               'rufous-tailed tailorbird',  # 9
               'black-naped monarch',  # 10
               'slender-billed crow',  # 11
               'buff-vented bulbul',  # 12
               'ferruginous babbler',  # 13
               'black-capped babbler',  # 14
               'chestnut-rumped babbler',  # 15
               'yellow-vented bulbul',  # 16
               'fluffy-backed tit-babbler',  # 17
               'bornean gibbon',  # 18
               'dark-necked tailorbird',  # 19
               'rufous-fronted babbler',  # 20
               'ashy tailorbird',  # 21
               'pied fantail',  # 22
               'short-tailed babbler',  # 23
               'plaintivecuckoo',  # 24
               'sooty-capped babbler',  # 25
               'spectacled bulbul',  # 26
               'chestnut-winged babbler',  # 27
               'bold-striped tit-babbler',  # 28
               'black-headed bulbul']  # 29

num_calls = np.array(num_calls)
classes = np.array(classes)
class_names = np.array(class_names)

i = np.argsort(num_calls)

print(num_calls[i])
print(classes[i])


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

individual_results_dict = dict()
endangered_results_dict = dict()

ignore_trials = {}

names = [
    "WideResNet",
    "max-SEResNet20-attention-4",
    "BNN-max-SEResNet20-attention-4",
    "uniform-smooth-BNN-max-SEResNet20-attention-4",
    "ua-smooth-BNN-max-SEResNet20-attention-4",
    "att-SEResNet20-attention-4",
    "BNN-att-SEResNet20-attention-4",
    "uniform-smooth-BNN-att-SEResNet20-attention-4",
    "ua-smooth-BNN-att-SEResNet20-attention-4",
]

for c in range(30):
    individual_results_dict[c] = dict()
    for n in names:
        individual_results_dict[c][n] = list()

for n in names:
    endangered_results_dict[n] = list()

for name in names:
    print(name)
    if name not in ignore_trials.keys():
        ignore_trials[name] = 0

    for t in range(30):
        if not os.path.exists(OUTPUT_FOLDER + "/" + name + "/items_summary_trial" + repr(t) + ".pkl"):
            continue
        print("Trial: ", t)
        filepath = OUTPUT_FOLDER + "/" + name + "/items_summary_trial" + repr(t) + ".pkl"
        try:
            results_summary = load_pickle(filepath)
        except FileNotFoundError:
            continue

        test_pred = results_summary["label"]["weighted_au_pr"]["label"]["test_pred"]
        print(results_summary["label"]["weighted_au_pr"]["label"])
        test_true = results_summary["label"]["weighted_au_pr"]["label"]["test_true"]

        measures, _ = get_multitask_classification_measures(true=test_true[:, endangered_classes],
                                                            pred=test_pred[:, endangered_classes],
                                                            are_logits=True)
        measures = {"test_" + k: v for k, v in measures.items()}
        endangered_results_dict[name].append(measures)

        number_of_classes = test_true.shape[1]

        for c in range(number_of_classes):
            test_pred_c = test_pred[:, c]
            test_true_c = test_true[:, c]

            measures = get_binary_classification_measures(true=test_true_c,
                                                          pred=test_pred_c,
                                                          are_logits=True)
            measures = {"test_" + k: v for k, v in measures.items()}
            individual_results_dict[c][name].append(measures)

print("Endangered.")
for name in names:
    print(name)
    print("Test  W AU PR:    ", trial_average(endangered_results_dict[name], "test_weighted_au_pr"))
    print("Test  W AU ROC:   ", trial_average(endangered_results_dict[name], "test_weighted_au_roc"))
    print("Test  W M-F1:   ", trial_average(endangered_results_dict[name], "test_weighted_macro_f1"))
    print("Test  W UAR:   ", trial_average(endangered_results_dict[name], "test_weighted_macro_recall"))
    print("Test  W ECE:   ", trial_average(endangered_results_dict[name], "test_weighted_ece"))

fp = open("individual_results.txt", "w")
for n, c in zip(num_calls[i], classes[i]):
    fp.write("Class: " + repr(iucn_status[c]) + " " + repr(n))
    fp.write("\n")
    for name in names:
        fp.write(name)
        fp.write("\n")
        fp.write("Test  W AU PR:    " + repr(trial_average(individual_results_dict[c][name], "test_au_pr")))
        fp.write("\n")
        fp.write("Test  W AU ROC:   " + repr(trial_average(individual_results_dict[c][name], "test_au_roc")))
        fp.write("\n")
        fp.write("Test  W M-F1:     " + repr(trial_average(individual_results_dict[c][name], "test_macro_f1")))
        fp.write("\n")
        fp.write("Test  W UAR:      " + repr(trial_average(individual_results_dict[c][name], "test_macro_recall")))
        fp.write("\n")
        fp.write("Test  W ECE:      " + repr(trial_average(individual_results_dict[c][name], "test_ece")))
        fp.write("\n")

fp.close()
