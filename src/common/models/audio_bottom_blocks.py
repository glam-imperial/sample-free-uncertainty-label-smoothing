import tensorflow as tf


def get_Identity_block(input_layer_list,
                       config_dict):
    net_train = input_layer_list[0]
    net_test = input_layer_list[0]
    return net_train, net_test
