import tensorflow as tf
import numpy as np

from variational.layers import DenseReparameterisation
from variational.activations import sigmoid_moments


class VariationalAttentionGlobalPooling(tf.keras.layers.Layer):
    def __init__(self,
                 number_of_heads,
                 number_of_features,
                 sequence_length,
                 outputs_list,
                 **kwargs):
        super(VariationalAttentionGlobalPooling, self).__init__(**kwargs)
        self.number_of_heads = number_of_heads
        self.number_of_features = number_of_features
        self.sequence_length = sequence_length
        self.outputs_list = outputs_list

    def build(self, input_shape):
        self.heads_list = list()
        for head_i in range(self.number_of_heads):
            self.heads_list.append(DenseReparameterisation(units=1,
                                                           variance_parameterisation_type="layer_wise",
                                                           activation=None,
                                                           trainable=True,
                                                           use_bias=False,
                                                           weight_initializer="glorot_uniform",
                                                           bias_initializer="zeros",
                                                           name=self.name + "_head_" + repr(head_i),
                                                           dtype=tf.float32,
                                                           dynamic=False,
                                                           reuse=None))

        self.dense_layer_list = list()
        for t, output_units in enumerate(self.outputs_list):
            self.dense_layer_list.append(DenseReparameterisation(units=output_units,
                                                                 # weight_prior="ard",
                                                                 variance_parameterisation_type="layer_wise",
                                                                 activation=None,
                                                                 trainable=True,
                                                                 use_bias=True,
                                                                 weight_initializer="glorot_uniform",
                                                                 bias_initializer="zeros",
                                                                 name=self.name + "_output_" + repr(t),
                                                                 dtype=tf.float32,
                                                                 dynamic=False,
                                                                 reuse=None))

        super(VariationalAttentionGlobalPooling, self).build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'number_of_heads': self.number_of_heads,
            'number_of_features': self.number_of_features,
            'sequence_length': self.sequence_length,
            'outputs_list': self.outputs_list,
        })
        return config

    def call(self, inputs, training, x_var, support=None):
        x_shape = tf.shape(inputs)

        sequence_length = self.sequence_length
        number_of_features = self.number_of_features

        net = tf.reshape(inputs, (-1, number_of_features))  # [bs*sequence_length, number_of_features]
        net_var = tf.reshape(x_var, (-1, number_of_features))  # [bs*sequence_length, number_of_features]

        energy_list = list()
        # energy_var_list = list()
        for head_i in range(self.number_of_heads):
            energy = net
            energy_var = net_var
            energy, \
            energy_var = self.heads_list[head_i](inputs=energy,
                                                 training=training,
                                                 inputs_variances=energy_var)
            energy = tf.reshape(energy,
                                (-1, sequence_length, 1, 1))
            energy_var = tf.reshape(energy_var,
                                    (-1, sequence_length, 1, 1))

            energy_list.append(energy)
            # energy_var_list.append(energy_var)

        attention_weights_list = list()
        for head_i in range(self.number_of_heads):
            attention_weights_list.append(tf.nn.softmax(energy_list[head_i],
                                                        axis=1))

        if self.number_of_heads > 1:
            attention_weights = tf.concat(attention_weights_list,
                                          axis=3)  # [-1, sequence_length, 1, number_of_heads]
        elif self.number_of_heads == 1:
            attention_weights = attention_weights_list[0]  # [-1, sequence_length, 1, 1]
        else:
            raise ValueError("Invalid number of heads.")

        net = tf.reshape(net, (-1,
                               sequence_length,
                               number_of_features,
                               1))  # [bs, sequence_length, number_of_features, 1]
        net_var = tf.reshape(net_var, (-1,
                                       sequence_length,
                                       number_of_features,
                                       1))  # [bs, sequence_length, number_of_features, 1]

        mean_hidden_list = list()
        mean_hidden_var_list = list()
        for head_i in range(self.number_of_heads):
            mean_hidden = tf.reduce_sum(tf.multiply(net,
                                                    attention_weights_list[head_i]),
                                        axis=1,
                                        keepdims=True)
            mean_hidden_list.append(mean_hidden)  # [bs, 1, number_of_features, 1]

            # mean_hidden_var = tf.reduce_sum(tf.multiply(net_var,
            #                                             attention_weights_list[head_i]),
            #                                 axis=1,
            #                                 keepdims=True)
            mean_hidden_var = tf.reduce_sum(tf.multiply(net_var,
                                                        tf.pow(attention_weights_list[head_i], 2.0)),
                                            axis=1,
                                            keepdims=True)
            mean_hidden_var_list.append(mean_hidden_var)  # [bs, 1, number_of_features, 1]

        for head_i in range(self.number_of_heads):
            mean_hidden = tf.reshape(mean_hidden_list[head_i], (-1, number_of_features))
            mean_hidden_var = tf.reshape(mean_hidden_var_list[head_i], (-1, number_of_features))

            mean_hidden_list[head_i] = mean_hidden
            mean_hidden_var_list[head_i] = mean_hidden_var

        if len(mean_hidden_list) > 1:
            mean_hidden = tf.concat(mean_hidden_list, axis=1)
            mean_hidden_var = tf.concat(mean_hidden_var_list, axis=1)
        elif len(mean_hidden_list) == 1:
            mean_hidden = mean_hidden_list[0]
            mean_hidden_var = mean_hidden_var_list[0]
        else:
            raise ValueError("Invalid number of heads.")

        mean_hidden = tf.reshape(mean_hidden, (-1, number_of_features * self.number_of_heads))
        mean_hidden_var = tf.reshape(mean_hidden_var, (-1, number_of_features * self.number_of_heads))

        prediction_single = list()
        prediction_single_var = list()
        for t in range(len(self.outputs_list)):
            prediction_single_t, \
                prediction_single_var_t = self.dense_layer_list[t](inputs=mean_hidden,
                                                                   training=training,
                                                                   inputs_variances=mean_hidden_var)

            print(prediction_single_t.shape)
            print(prediction_single_var_t.shape)
            prediction_single.append(prediction_single_t)
            prediction_single_var.append(prediction_single_var_t)

        return prediction_single, prediction_single_var, attention_weights

    def get_kl_loss(self):
        self.kl = 0.0

        for head_i in range(self.number_of_heads):
            self.kl = self.kl + self.heads_list[head_i].get_kl_loss()

        for t, output_units in enumerate(self.outputs_list):
            self.kl = self.kl + self.dense_layer_list[t].get_kl_loss()

        return self.kl


def get_variational_attention_global_pooling(net_train,
                                             net_test,
                                             net_var_train,
                                             net_var_test,
                                             y_pred_names,
                                             global_pooling_configuration):
    number_of_heads = global_pooling_configuration["number_of_heads"]
    number_of_features = global_pooling_configuration["number_of_features"]
    sequence_length = global_pooling_configuration["sequence_length"]
    outputs_list = global_pooling_configuration["outputs_list"]

    attention_global_pooling = VariationalAttentionGlobalPooling(number_of_heads=number_of_heads,
                                                                 number_of_features=number_of_features,
                                                                 sequence_length=sequence_length,
                                                                 outputs_list=outputs_list,
                                                                 name="att_emb")

    prediction_single_train,\
    prediction_single_train_var,\
    attention_weights_train = attention_global_pooling(inputs=net_train,
                                                       x_var=net_var_train,
                                                       training=True)
    prediction_single_test,\
    prediction_single_test_var,\
    attention_weights_test = attention_global_pooling(inputs=net_test,
                                                      x_var=net_var_test,
                                                      training=False)

    kl_loss = attention_global_pooling.get_kl_loss()

    prediction_train = dict()
    prediction_test = dict()
    for i, y_pred_name in enumerate(y_pred_names):
        prediction_train[y_pred_name] = prediction_single_train[i]
        prediction_train[y_pred_name + "_prob"] = tf.nn.sigmoid(prediction_single_train[i])
        prediction_train[y_pred_name + "_var"] = prediction_single_train_var[i]
        y, y_var = sigmoid_moments(prediction_single_train[i], prediction_single_train_var[i])
        prediction_train[y_pred_name + "_moments_prob"] = y
        prediction_train[y_pred_name + "_moments_var"] = y_var

        prediction_test[y_pred_name] = prediction_single_test[i]
        prediction_test[y_pred_name + "_prob"] = tf.nn.sigmoid(prediction_single_test[i])
        prediction_test[y_pred_name + "_var"] = prediction_single_test_var[i]
        y, y_var = sigmoid_moments(prediction_single_test[i], prediction_single_test_var[i])
        prediction_test[y_pred_name + "_moments_prob"] = y
        prediction_test[y_pred_name + "_moments_var"] = y_var

    return prediction_train, prediction_test, kl_loss
    # return prediction_test, kl_loss
