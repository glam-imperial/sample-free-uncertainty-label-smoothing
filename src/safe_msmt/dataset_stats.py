import numpy as np
import pandas as pd

unique_classes = ['little spiderhunter', 'bushy-crested hornbill', 'banded bay cuckoo', 'grey-headed babbler', 'chestnut-backed scimitar-babbler', 'brown fulvetta', 'blue-eared barbet', 'rhinoceros hornbill', 'rufous-tailed shama', 'rufous-tailed tailorbird', 'black-naped monarch', 'slender-billed crow', 'buff-vented bulbul', 'ferruginous babbler', 'black-capped babbler', 'chestnut-rumped babbler', 'yellow-vented bulbul', 'fluffy-backed tit-babbler', 'bornean gibbon', 'dark-necked tailorbird', 'rufous-fronted babbler', 'ashy tailorbird', 'pied fantail', 'short-tailed babbler', 'plaintivecuckoo', 'sooty-capped babbler', 'spectacled bulbul', 'chestnut-winged babbler', 'bold-striped tit-babbler', 'black-headed bulbul']

annotation_file = "/data/Data/XinWen/positive_multilabel.txt"

data_df = pd.read_csv(annotation_file, " ")

data_np = data_df.values[:, 3:]

print(data_np.shape)

data_np_sum_0 = data_np.sum(axis=0)

print("Total calls in dataset:", data_np.sum())
print("Rarest and most common class:", data_np_sum_0.min(), data_np_sum_0.max())
data_np_sum_1 = data_np.sum(axis=1)
print("Calls in overlap:", data_np_sum_1[data_np_sum_1 > 1].shape, data_np_sum_1[data_np_sum_1 > 1].shape / data_np.sum())

index = np.argsort(data_np_sum_0)[::-1]

print(index)
data_np_sum_0_sorted = data_np_sum_0[index]
unique_classes_sorted = list(np.array(unique_classes)[index])

data_np_sorted = data_np[:, index]

data_np_sum_0_sorted_overlapped = np.zeros_like(data_np_sum_0_sorted)
counts = data_np_sorted.sum(axis=1)#  .reshape((data_np_sorted.shape[0], 1))
# print(counts)
for c_i, cls in enumerate(unique_classes_sorted):
    overlapped_clips = data_np_sorted[counts > 1][:, c_i].sum()
    data_np_sum_0_sorted_overlapped[c_i] = overlapped_clips

import matplotlib.pyplot as plt
plt.figure(figsize=(6.4, 4.8))
plt.bar(unique_classes_sorted, list(data_np_sum_0_sorted), width=0.8, align='center', color="b")
plt.bar(unique_classes_sorted, list(data_np_sum_0_sorted_overlapped), width=0.8, align='center', color="r")
plt.xticks(range(len(unique_classes_sorted)), unique_classes_sorted, rotation=90)
names = ['single label calls', 'multiple label calls']
plt.legend(names)
plt.tight_layout()
plt.savefig("overlap_barchart.pdf")

print("Total calls that are in overlap:", data_np_sum_0_sorted_overlapped.sum())
print("Total calls that are in overlap (%):", data_np_sum_0_sorted_overlapped.sum() / data_np.sum())
print("Number of multilabel clips:", data_np_sorted[counts > 1].shape[0])
print("Number of multilabel clips (%):", data_np_sorted[counts > 1].shape[0] / data_np.shape[0])
