from common.trials import run_experiments
from safe_msmt.configuration import get_config_dict_from_yaml


def make_config_dict_list():
    config_dict_list = list()

    for name in [
        "SEResNet20-attention-4",
        "att-SEResNet20-attention-4",
        "BD-fix0-SEResNet20-attention-4",
        "BD-attfix0-SEResNet20-attention-4",
        "smooth-BD-attfix0-SEResNet20-attention-4",
        "smooth-BD-fix0-SEResNet20-attention-4",
        ]:  # These are the names of the YAML files in folder: experiment_configurations.
        config_dict = get_config_dict_from_yaml(name)
        config_dict_list.append(config_dict)

    return config_dict_list


if __name__ == '__main__':
    run_experiments(make_config_dict_list())
