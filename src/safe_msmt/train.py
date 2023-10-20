from common.trials import run_experiments
from safe_msmt.configuration import get_config_dict_from_yaml


def make_config_dict_list():
    config_dict_list = list()

    for name in [
        # "WideResNet",
        # "max-SEResNet20-attention-4",
        # "BNN-max-SEResNet20-attention-4",
        # "uniform-smooth-BNN-max-SEResNet20-attention-4",
        # "ua-smooth-BNN-max-SEResNet20-attention-4",
        # "att-SEResNet20-attention-4",
        # "BNN-att-SEResNet20-attention-4",
        # "uniform-smooth-BNN-att-SEResNet20-attention-4",
        "ua-smooth-BNN-att-SEResNet20-attention-4"
        ]:  # These are the names of the YAML files in folder: experiment_configurations.
        config_dict = get_config_dict_from_yaml(name)
        config_dict_list.append(config_dict)

    return config_dict_list


if __name__ == '__main__':
    run_experiments(make_config_dict_list())
