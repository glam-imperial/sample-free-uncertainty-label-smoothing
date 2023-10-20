import tensorflow as tf
import numpy as np


class AttentionGlobalPooling:
    def __init__(self,
                 number_of_heads,
                 number_of_features,
                 sequence_length,
                 outputs_list):
        self.number_of_heads = number_of_heads
        self.number_of_features = number_of_features
        self.sequence_length = sequence_length
        self.outputs_list = outputs_list

        self.heads_list = list()
        for head_i in range(self.number_of_heads):
            self.heads_list.append(tf.keras.layers.Dense(1, use_bias=False))

        self.dense_layer_list = list()
        for t, output_units in enumerate(self.outputs_list):
            self.dense_layer_list.append(tf.keras.layers.Dense(output_units,
                                                               activation=None))
            # self.dense_layer_list.append(tf.keras.layers.Dense(2,
            #                                                    activation=None))

    def __call__(self, x, training):
        net = tf.reshape(x, (-1, self.number_of_features))  # [bs*sequence_length, number_of_features]

        energy_list = list()
        for head_i in range(self.number_of_heads):
            energy = tf.reshape(self.heads_list[head_i](net, training=training),
                                (-1, self.sequence_length, 1, 1))

            energy_list.append(energy)

        attention_weights_list = list()
        for head_i in range(self.number_of_heads):
            attention_weights_list.append(tf.nn.softmax(energy_list[head_i], axis=1))

        if self.number_of_heads > 1:
            attention_weights = tf.concat(attention_weights_list,
                                          axis=3)  # [-1, sequence_length, 1, number_of_heads]
        elif self.number_of_heads == 1:
            attention_weights = attention_weights_list[0]  # [-1, sequence_length, 1, 1]
        else:
            raise ValueError("Invalid number of heads.")

        net = tf.reshape(net, (-1, self.sequence_length, self.number_of_features, 1))  # [bs, sequence_length, number_of_features, 1]

        mean_hidden_list = list()
        for head_i in range(self.number_of_heads):
            mean_hidden = tf.reduce_sum(tf.multiply(net, attention_weights_list[head_i]), axis=1,
                                        keepdims=True)
            mean_hidden_list.append(mean_hidden)  # [bs, 1, number_of_features, 1]

        for head_i in range(self.number_of_heads):
            mean_hidden = tf.reshape(mean_hidden_list[head_i], (-1, self.number_of_features))
            mean_hidden_list[head_i] = mean_hidden

        if len(mean_hidden_list) > 1:
            mean_hidden = tf.concat(mean_hidden_list, axis=1)
        elif len(mean_hidden_list) == 1:
            mean_hidden = mean_hidden_list[0]
        else:
            raise ValueError("Invalid number of heads.")

        mean_hidden = tf.reshape(mean_hidden, (-1, self.number_of_features * self.number_of_heads))

        prediction_single = list()
        for t in range(len(self.outputs_list)):
            prediction_single.append(self.dense_layer_list[t](mean_hidden, training=training))

        return prediction_single, attention_weights


def get_attention_global_pooling(net_train,
                                 net_test,
                                 y_pred_names,
                                 global_pooling_configuration):
    number_of_heads = global_pooling_configuration["number_of_heads"]
    number_of_features = global_pooling_configuration["number_of_features"]
    sequence_length = global_pooling_configuration["sequence_length"]
    outputs_list = global_pooling_configuration["outputs_list"]

    attention_global_pooling = AttentionGlobalPooling(number_of_heads=number_of_heads,
                                                      number_of_features=number_of_features,
                                                      sequence_length=sequence_length,
                                                      outputs_list=outputs_list)

    prediction_single_train,\
    attention_weights_train = attention_global_pooling(x=net_train,
                                                       training=True)
    prediction_single_test,\
    attention_weights_test = attention_global_pooling(x=net_test,
                                                      training=False)

    prediction_train = dict()
    prediction_test = dict()
    for i, y_pred_name in enumerate(y_pred_names):
        prediction_train[y_pred_name] = prediction_single_train[i]
        prediction_train[y_pred_name + "_prob"] = tf.nn.sigmoid(prediction_single_train[i])

        prediction_test[y_pred_name] = prediction_single_test[i]
        prediction_test[y_pred_name + "_prob"] = tf.nn.sigmoid(prediction_single_test[i])

    return prediction_train, prediction_test
    # return prediction_test
