from openpyxl import load_workbook
import numpy as np
# import matplotlib.pyplot as plt
from safe_msmt.time_intervals import find_overlaps, find_overlaps_v2


def identify_classes_to_use(xlsx_filepath,
                            threshold,
                            verbose):
    print("Reading the XLSX file.")
    file_list,\
    start_list,\
    end_list,\
    y_labels_list,\
    classes = read_xlsx_annotation(xlsx_filepath,
                                   verbose)
    print("The set of unique classes:", y_labels_list)
    print("The set of unique classes:", classes)


    # print("Identifying species that appear more than", threshold, "times.")
    # over3,\
    # negative = get_species(y_labels_list,
    #                        threshold)
    # print("over3", over3)
    # print("negative", negative)

    # print("Identifying files with identified classes.")
    # fileover3 = find_files(over3, y_labels_list, file_list)
    # print(fileover3)

    # print("Identifying files with no annotated species.")
    # filenegative = find_files(negative, y_labels_list, file_list)
    # print(filenegative)

    # print("Identifying species that appear in more than", threshold, "files.")
    # over3dic, over3flist, occurrence = get_species_multiple_files(over3, threshold, y_labels_list, file_list)
    # print(over3dic)
    # print(over3flist)
    # print(occurrence)


    print("Identifying species that appear in more than", threshold, "files.")
    usable_classes_to_file,\
    usable_classes,\
    usable_classes_to_count, \
    negative_files = get_species_multiple_files(classes,
                                                threshold,
                                                y_labels_list,
                                                file_list)
    print("usable_classes_to_file", usable_classes_to_file)
    print("usable_classes", usable_classes)
    print("usable_classes_to_count", usable_classes_to_count)
    print("negative_files", negative_files)

    # sorted_dic = {k: v for k, v in sorted(usable_classes_to_count.items(), key=lambda item: item[1])}
    sorted_dic = sorted(usable_classes_to_count.items(), key=lambda item: item[1])
    print("sorted_dic", sorted_dic)
    sorted_usable_classes = [k for k, v in sorted_dic]
    print("sorted_usable_classes", sorted_usable_classes)
    # plot_occurrence(sorted_dic)

    # if verbose:
    #     # print('Files that contain species with more than 3 annotation', len(fileover3))
    #     # print('Files that contain Negative files', len(filenegative))
    #     # print('Species that appear in at least 3 files: ', len(over3flist))
    #     # overlap,\
    #     # overlapcls,\
    #     # overlapped_duration,\
    #     # totaldic = find_overlaps(file_list,
    #     #                          start_list,
    #     #                          end_list,
    #     #                          y_labels_list,
    #     #                          negative_files)
    #
    #     unique_classes = ['little spiderhunter', 'bushy-crested hornbill', 'banded bay cuckoo', 'grey-headed babbler', 'chestnut-backed scimitar-babbler', 'brown fulvetta', 'blue-eared barbet', 'rhinoceros hornbill', 'rufous-tailed shama', 'rufous-tailed tailorbird', 'black-naped monarch', 'slender-billed crow', 'buff-vented bulbul', 'ferruginous babbler', 'black-capped babbler', 'chestnut-rumped babbler', 'yellow-vented bulbul', 'fluffy-backed tit-babbler', 'bornean gibbon', 'dark-necked tailorbird', 'rufous-fronted babbler', 'ashy tailorbird', 'pied fantail', 'short-tailed babbler', 'plaintivecuckoo', 'sooty-capped babbler', 'spectacled bulbul', 'chestnut-winged babbler', 'bold-striped tit-babbler', 'black-headed bulbul']
    #
    #     overlap, \
    #     overlapcls, \
    #     overlapped_duration, \
    #     totaldic = find_overlaps_v2(file_list,
    #                              start_list,
    #                              end_list,
    #                              y_labels_list,
    #                              negative_files,
    #                              unique_classes)
    #
    #     oo = list(overlapped_duration.values())
    #     print('Average Overlap: ', np.mean(oo))
    #     print('std: ', np.std(oo))
    #     print('min: ', min(oo))
    #     print('max: ', max(oo))
    #
    #     file_list_eff = list()
    #     for fi, cls in zip(file_list, y_labels_list):
    #         if cls in unique_classes:
    #             file_list_eff.append(fi)
    #
    #     # report_overlap(file_list, overlapcls, overlapped_duration,
    #     #                len(negative_files), len(np.unique(file_list)) - len(negative_files), len(classes))
    #     report_overlap(file_list, overlapcls, overlapped_duration,
    #                    len(negative_files), len(np.unique(file_list_eff)) - len(negative_files), len(unique_classes))
    #     # targetann = get_ann(usable_classes, file_list, start_list, end_list, y_labels_list)
    #     # overlap, overlapcls, overlapped_duration, totaldic = find_overlaps(targetann[:, 0], targetann[:, 1],
    #     #                                                                    targetann[:, 2], targetann[:, 3],
    #     #                                                                    negative_files)
    #     # report_overlap(targetann[:, 0], overlapcls, overlapped_duration, len(negative_files), len(fileover3),
    #     #                len(usable_classes))

    return usable_classes, sorted_usable_classes, negative_files, file_list, start_list, end_list, y_labels_list


def read_xlsx_annotation(xlsx_filepath,
                         verbose):
    # Load the entire workbook.
    wb = load_workbook(xlsx_filepath,
                       data_only=True)
    ws = wb['Sheet1']
    all_rows = list(ws.rows)

    y_labels = []
    file = []
    start = []
    end = []
    for line in all_rows[21:]:
        file.append(line[0].value)
        start.append(line[1].value)
        end.append(line[2].value)
        label = line[3].value
        if 'Blue-eared Barbet' in label:
            label = 'Blue-eared Barbet'
        if 'Bushy-crested Hornbill' in label or 'Busy-crested Hornibll' in label:
            label = 'Bushy-crested Hornbill'
        if 'Maroon Woodpecker' in label:
            label = 'Maroon Woodpecker'
        if 'Plaintive Cuckoo' in label or label == 'Plaintive Cukcoo':
            label = 'Plaintive Cuckoo'
        if 'Rhinoceros Hornbill' in label:
            label = 'Rhinoceros Hornbill'
        if label == 'Green Iroa':
            label = 'green iora'
        if label == 'Fluff-backed Tit-babbler':
            label = 'Fluffy-backed Tit-babbler'
        if label == 'Slender-billed Crown':
            label = 'slender-billed crow'
        if label == 'Specatacled Bulbul':
            label = 'spectacled bulbul'
        if 'Orange-bellied' in label:
            label = 'Orange-bellied Flowerpecker'
        if label in ['Chestnut-backed Scimitar Babbler  ', 'Chestnut-backed Scimitar-babbler']:
            label = 'Chestnut-backed Scimitar-babbler'
        if label in ['Pied Fanrtail', 'Pied Fantail', 'Pied Fantail   ', 'Pied Fantial', 'Pied fantail']:
            label = 'Pied Fantail'
        if label in ['Black-headed  Bulbul',  'Black-headed Bulbil','Black-headed Bulbul', 'Black-headed Bululb']:
            label = 'Black-headed Bulbul'
        if label in ['Black-headed Babbler', 'Black-heaed Babbler']:
            label = 'Black-headed Babbler'
        if label == 'Olive-backed Woodpecker?':
            label = 'Olive-backed Woodpecker'
        y_labels.append(label.lower().rstrip())

    [classes, y] = np.unique(y_labels, return_inverse=True)
    f = np.unique(file)
    y = np.array(y)
    if verbose:
        print('Number of files in annotation:', len(f))
        print('Number of species in annotation:', len(classes))
        print('Number of instances:', len(y))
    duration = []
    zerocount = 0
    for i, s, e, cls in zip(file, start, end, y_labels):
        d = e - s
        if d == 0 and cls != 'nothing':
            zerocount += 1
        if d >= 0 and cls != 'nothing':
            duration.append(d)

    if verbose:
        meand = np.mean(duration)
        print('Average Duration: ', meand)
        print('std: ', np.std(duration))
        print('q10: ', np.quantile(duration, .1))
        print('q20: ', np.quantile(duration, .2))
        print('q30: ', np.quantile(duration, .3))
        print('q40: ', np.quantile(duration, .4))
        print('q50: ', np.quantile(duration, .5))
        print('q60: ', np.quantile(duration, .6))
        print('q70: ', np.quantile(duration, .7))
        print('q80: ', np.quantile(duration, .8))
        print('q90: ', np.quantile(duration, .9))
        print('min: ', min(duration))
        print('max: ', max(duration))
        print('Number of calls with a duration of 0', zerocount)
        # fig1, ax1 = plt.subplots()
        # ax1.boxplot(duration)
        # plt.show()
    return file, start, end, y_labels, classes


def get_ann(species, files,starts, ends, labels):
    ann = []
    for i in range(len(labels)):
        if labels[i] in species:
            ann.append([files[i], starts[i], ends[i], labels[i]])
    ann = np.array(ann)
    return ann


def get_short_ann(files,starts, ends, labels, call_threshold):
    ann = []
    for i in range(len(labels)):
        if ends[i] - starts[i] < call_threshold:
            ann.append([files[i], starts[i], ends[i], labels[i]])
    ann = np.array(ann)
    return ann


def get_species(y_labels, count_threshold):
    classes, counts = np.unique(y_labels, return_counts=True)
    result = []
    negative = []
    for cls, count in zip(classes, counts):
        if count >= count_threshold and cls != 'nothing' and cls != 'unknown':
            result.append(cls)
        elif cls == 'nothing':
            negative.append(cls)
    # ignore Unkown
    print('Species with more than 3 annotation: ', len(result))
    return result, negative


def find_files(targetlabels, alllabels, files):
    targetfiles = []
    for i in range(len(alllabels)):
        if alllabels[i] in targetlabels and files[i] not in targetfiles:
            targetfiles.append(files[i])
    return targetfiles


def get_species_multiple_files(classes,
                               threshold,
                               y_labels_list,
                               file_list):
    usable_classes_to_file = {}
    usable_classes = []
    usable_classes_to_count = {}

    negative_files = []
    for c in classes:
        if c == "nothing":
            for cls, fi in zip(y_labels_list, file_list):
                if cls == c:
                    negative_files.append(fi)
        else:
            val = []
            for cls, fi in zip(y_labels_list, file_list):
                if cls == c:
                    val.append(fi)
            f = np.unique(val)
            if len(f) >= threshold:
                usable_classes_to_file[c] = val
                # usable_classes_to_count[c] = len(val)
                usable_classes_to_count[c] = len(set(val))
                usable_classes.append(c)

    return usable_classes_to_file,\
           usable_classes,\
           usable_classes_to_count,\
           negative_files


# def plot_occurrence(sorted_dic):
#     plt.figure(figsize=(15, 6))
#     plt.bar(range(len(sorted_dic)), list(sorted_dic.values()), align='center')
#     plt.xticks(range(len(sorted_dic)), list(sorted_dic.keys()), rotation=90)
#     plt.show()


def report_overlap(files, overlapcls, overlapped_duration,
                   num_neg, num_pos, num_cls, file_length=45):
    oinstance = 0
    overlapped = 0
    totalduration = 0
    cls = []
    for f in np.unique(files):
        if f in overlapcls.keys():
            oinstance += len(overlapcls[f])
            overlapped += overlapped_duration[f]
            totalduration += 45
            cls += overlapcls[f]

    print("Number of overlapped instance: ", oinstance)
    print('Proportion of overlapped instance: ', oinstance / len(files))
    print('Total overlapped duration in seconds: ', overlapped)
    print('Proportion of overlapped duration in seconds including negatives: ',
          overlapped / ((num_pos + num_neg) * file_length))
    print('Proportion of overlapped duration in seconds excluding negatives: ',
          overlapped / (num_pos * file_length))
    print('Total call length (simple concatenation):', totalduration)
    print('Total file length that contain positive examples: ', num_pos * file_length)
    print('Proportion of overlapped duration in seconds wrt to total file length: ',
          overlapped / (num_pos * file_length))
    print('Number of species with overlapped calls: ', len(set(cls)))
    print('Proportion of species with overlapped calls: ', len(set(cls)) / num_cls)
    print(num_cls)
