import os
import collections
import yaml

from common.metadata_util import process_metadata

# Edit these.
PROJECT_FOLDER = '/path/to/sample-free-uncertainty-label-smoothing/'

DATA_FOLDER = '/path/to/osa_smw_dataset/'

# No need to edit these.
TFRECORDS_FOLDER = DATA_FOLDER + "/tfrecords_osa"

OUTPUT_FOLDER = PROJECT_FOLDER + '/ResultsRevisionOSA'

YAML_CONFIGURATION_FOLDER = PROJECT_FOLDER + "/src/osa/experiment_configurations"

PARTITIONS = {"train": ["Luna",  "Other", "Catappa",  "Osa 1", "Osa 2", "Live-recording", "Piro-new calls"],
              "devel": ["Tape", "Shady", "Eleanor"],
              "test": ["Corcovado Calls", "new", "Will", "Corcovado Round 2 positives"]}


def get_name_to_metadata(tf_names):
    name_to_metadata = dict()
    for name in [
                 "support",
                 "logmel_spectrogram",
                 "segment_id",
                 "version_id",
                 "whinny_single",
                 ]:
        name_to_metadata[name] = dict()

    name_to_metadata["support"]["numpy_shape"] = (48000, 1)
    name_to_metadata["logmel_spectrogram"]["numpy_shape"] = (300, 128)
    name_to_metadata["segment_id"]["numpy_shape"] = (1, )
    name_to_metadata["version_id"]["numpy_shape"] = (1, )
    name_to_metadata["whinny_single"]["numpy_shape"] = (1, )

    name_to_metadata["support"]["variable_type"] = "support"
    name_to_metadata["logmel_spectrogram"]["variable_type"] = "x"
    name_to_metadata["segment_id"]["variable_type"] = "id"
    name_to_metadata["version_id"]["variable_type"] = "id"
    name_to_metadata["whinny_single"]["variable_type"] = "y"

    name_to_metadata = process_metadata(name_to_metadata)

    name_to_metadata = {k: name_to_metadata[k] for k in tf_names}

    return name_to_metadata


def filter_names(all_path_list,
                 pos_variations,
                 neg_variations):
    path_dict = collections.defaultdict(list)

    for path in all_path_list:
        path_split = path[:-10].split("_")
        segment_id = int(path_split[-2])
        version_id = int(path_split[-1])
        name = "_".join(path_split[1:4])
        path_dict[name].append(path)

    all_path_list_new = list()
    for k, v in path_dict.items():
        if "pos" in k:
            if pos_variations is None:
                number_of_variations = len(v)
            else:
                number_of_variations = pos_variations
        elif "neg" in k:
            if neg_variations is None:
                number_of_variations = len(v)
            else:
                number_of_variations = neg_variations
        else:
            raise ValueError
        for i, vv in enumerate(v):
            if i < number_of_variations:
                all_path_list_new.append(v[i])
    return all_path_list_new


def get_dataset_info(tfrecords_folder):
    partitions = ["train",
                  "devel",
                  "test"]
    path_list_dict = dict()
    partition_size_dict = dict()
    for partition in partitions:
        partition_eff = partition

        all_path_list = os.listdir(tfrecords_folder + "/" + partition_eff)

        if partition_eff == "train":
            all_path_list = filter_names(all_path_list,
                                         pos_variations=5,  # Multiple versions per positive sample exist -- offline random time shift.
                                         neg_variations=None)  # Get all negatives.
        elif partition_eff in ["devel", "test"]:
            all_path_list = filter_names(all_path_list,
                                         pos_variations=1,   # Only 1 version per positive sample exists (and should exist).
                                         neg_variations=None)  # Get all negatives.
        else:
            raise ValueError

        all_path_list = [tfrecords_folder + "/" + partition_eff + "/" + name for name in all_path_list]

        path_list_dict[partition] = all_path_list

        partition_size_dict[partition] = len(all_path_list)
    return path_list_dict, partition_size_dict


def get_config_dict_from_yaml(file_name):
    # Read the parameters from the YAML file.
    stream = open(YAML_CONFIGURATION_FOLDER + "/" + file_name + ".yaml", 'r')
    # CONFIG_DICT = yaml.load(stream)
    CONFIG_DICT = yaml.safe_load(stream)
    stream.close()

    # Get the list of TFRECORDS file paths per partition.
    PATH_LIST_DICT, \
    PARTITIONS_SIZE_DICT = get_dataset_info(TFRECORDS_FOLDER)

    CONFIG_DICT["tfrecords_folder"] = TFRECORDS_FOLDER
    CONFIG_DICT["output_folder"] = OUTPUT_FOLDER

    CONFIG_DICT["path_list_dict"] = PATH_LIST_DICT
    CONFIG_DICT["train_size"] = PARTITIONS_SIZE_DICT["train"]
    CONFIG_DICT["devel_size"] = PARTITIONS_SIZE_DICT["devel"]
    CONFIG_DICT["test_size"] = PARTITIONS_SIZE_DICT["test"]

    CONFIG_DICT["model_configuration"]["name_to_metadata"] = get_name_to_metadata(CONFIG_DICT["model_configuration"]["input_type_list"] +
                                                                                  CONFIG_DICT["model_configuration"]["output_type_list"])
    CONFIG_DICT["results_summary_path"] = OUTPUT_FOLDER + "/" + CONFIG_DICT["method_string"] + "/results_summary"
    CONFIG_DICT["items_summary_path"] = OUTPUT_FOLDER + "/" + CONFIG_DICT["method_string"] + "/items_summary"

    CONFIG_DICT["y_pred_names"] = CONFIG_DICT["model_configuration"]["output_type_list"]
    CONFIG_DICT["model_configuration"]["global_pooling_configuration"]["outputs_list"] = (1, )

    CONFIG_DICT["monitor_target_to_measures"] = {"whinny_single":
                                                     ["au_pr", ]}
    CONFIG_DICT["report_target_to_measures"] = {"whinny_single":
                                                    ["accuracy",
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
                                                     "mce"]}

    CONFIG_DICT["output_channel_targets"] = ["whinny_single", ]

    import osa.losses as losses
    import osa.evaluation as evaluation

    CONFIG_DICT["losses_module"] = losses
    CONFIG_DICT["evaluation_module"] = evaluation

    return CONFIG_DICT
