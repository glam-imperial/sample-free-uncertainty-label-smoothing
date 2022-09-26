import tensorflow as tf
import numpy as np

from variational.layers import DenseReparameterisation
from variational.activations import sigmoid_moments


class VariationalAttentionGlobalPooling(tf.keras.layers.Layer):
    def __init__(self,
                 number_of_heads,
                 use_temporal_std,
                 pool_heads,
                 auto_pooling,
                 uncertainty_handling,
                 number_of_features,
                 sequence_length,
                 use_auto_array,
                 outputs_list,
                 **kwargs):
        super(VariationalAttentionGlobalPooling, self).__init__(**kwargs)
        self.number_of_heads = number_of_heads
        self.use_temporal_std = use_temporal_std
        self.pool_heads = pool_heads
        self.auto_pooling = auto_pooling
        self.uncertainty_handling = uncertainty_handling
        self.number_of_features = number_of_features
        self.sequence_length = sequence_length
        self.use_auto_array = use_auto_array
        self.outputs_list = outputs_list

        if self.uncertainty_handling not in ["propagate",
                                             "concatenate",
                                             "curiosity"]:
            raise NotImplementedError

        if self.uncertainty_handling == "curiosity":
            self.curiosity_value_list = list()
            for head_i in range(self.number_of_heads):
                self.curiosity_value_list.append(self.add_weight(name=self.name + "_curio" + repr(head_i),
                                                                 shape=1,
                                                                 dtype=tf.float32,
                                                                 initializer="ones",
                                                                 regularizer=None,
                                                                 trainable=self.trainable,
                                                                 constraint=None,
                                                                 partitioner=None,
                                                                 use_resource=None))

        self.heads_list = list()
        if self.use_auto_array:
            raise NotImplementedError
        else:
            for head_i in range(self.number_of_heads):
                self.heads_list.append(DenseReparameterisation(units=1,
                                                               variance_parameterisation_type="layer_wise",
                                                               use_clt=True,
                                                               activation=None,
                                                               uncertainty_propagation_type="ral",
                                                               trainable=True,
                                                               use_bias=False,
                                                               weight_initializer="glorot_uniform",
                                                               bias_initializer="zeros",
                                                               name=self.name + "_head_" + repr(head_i),
                                                               dtype=tf.float32,
                                                               dynamic=False,
                                                               reuse=None))

        if self.auto_pooling == "Auto":
            raise NotImplementedError
        elif self.auto_pooling == "MultiResolution":
            raise NotImplementedError

        if self.pool_heads == "gating":
            raise NotImplementedError
        elif self.pool_heads == "gating_old":
            raise NotImplementedError
        elif self.pool_heads in ["attention", "attention_auto"]:
            raise NotImplementedError

        self.dense_layer_list = list()
        for t, output_units in enumerate(self.outputs_list):
            self.dense_layer_list.append(DenseReparameterisation(units=output_units,
                                                                 variance_parameterisation_type="layer_wise",
                                                                 use_clt=True,
                                                                 activation=None,
                                                                 uncertainty_propagation_type="ral",
                                                                 trainable=True,
                                                                 use_bias=True,
                                                                 weight_initializer="glorot_uniform",
                                                                 bias_initializer="zeros",
                                                                 name=self.name + "_output_" + repr(t),
                                                                 dtype=tf.float32,
                                                                 dynamic=False,
                                                                 reuse=None))

    def get_config(self):
        config = super().get_config()
        config.update({
            'number_of_heads': self.number_of_heads,
            'use_temporal_std': self.use_temporal_std,
            'pool_heads': self.pool_heads,
            'auto_pooling': self.auto_pooling,
            'uncertainty_handling': self.uncertainty_handling,
            'number_of_features': self.number_of_features,
            'sequence_length': self.sequence_length,
            'use_auto_array': self.use_auto_array,
            'outputs_list': self.outputs_list,
        })
        return config

    def call(self, inputs, training, x_var):
        net = tf.reshape(inputs, (-1, self.number_of_features))  # [bs*sequence_length, number_of_features]
        net_var = tf.reshape(x_var, (-1, self.number_of_features))  # [bs*sequence_length, number_of_features]

        energy_list = list()
        # energy_var_list = list()
        for head_i in range(self.number_of_heads):
            if self.use_auto_array:
                raise NotImplementedError
            else:
                if self.uncertainty_handling == "propagate":
                    energy = net
                    energy_var = net_var
                elif self.uncertainty_handling == "concatenate":
                    energy = tf.concat([net, net_var], axis=1)
                    energy_var = tf.concat([net_var, tf.zeros_like(net_var)], axis=1)
                elif self.uncertainty_handling == "curiosity":
                    energy = net + self.curiosity_value_list[head_i] * tf.sqrt(1e-8 + net_var)
                    energy_var = net_var
                else:
                    raise NotImplementedError
                energy,\
                _,\
                energy_var = self.heads_list[head_i](inputs=energy,
                                                     training=training,
                                                     n_samples=1,
                                                     sample_type="Sample",
                                                     inputs_variances=energy_var)
                energy = tf.reshape(energy,
                                    (-1, self.sequence_length, 1, 1))
                energy_var = tf.reshape(energy_var,
                                        (-1, self.sequence_length, 1, 1))

            energy_list.append(energy)
            # energy_var_list.append(energy_var)

        if self.auto_pooling == "Auto":
            raise NotImplementedError
        elif self.auto_pooling == "MultiResolution":
            raise NotImplementedError
        elif self.auto_pooling == "no_auto":
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
        else:
            raise ValueError("Invalid auto pooling type.")

        net = tf.reshape(net, (-1, self.sequence_length, self.number_of_features, 1))  # [bs, sequence_length, number_of_features, 1]
        net_var = tf.reshape(net_var, (-1, self.sequence_length, self.number_of_features, 1))  # [bs, sequence_length, number_of_features, 1]

        mean_hidden_list = list()
        mean_hidden_var_list = list()
        for head_i in range(self.number_of_heads):
            mean_hidden = tf.reduce_sum(tf.multiply(net, attention_weights_list[head_i]), axis=1,
                                        keep_dims=True)
            mean_hidden_list.append(mean_hidden)  # [bs, 1, number_of_features, 1]

            # mean_hidden_var = tf.reduce_sum(tf.multiply(net_var, attention_weights_list[head_i]), axis=1,
            #                                 keep_dims=True)
            mean_hidden_var = tf.reduce_sum(tf.multiply(net_var,
                                                        tf.pow(attention_weights_list[head_i], 2.0)),
                                            axis=1,
                                            keep_dims=True)
            mean_hidden_var_list.append(mean_hidden_var)  # [bs, 1, number_of_features, 1]

        if self.use_temporal_std:
            raise NotImplementedError

        if self.pool_heads == "gating":
            raise NotImplementedError
        elif self.pool_heads == "gating_old":
            raise NotImplementedError

        elif self.pool_heads in ["attention", "attention_auto"]:
            raise NotImplementedError
        elif self.pool_heads == "no_pool":
            for head_i in range(self.number_of_heads):
                mean_hidden = tf.reshape(mean_hidden_list[head_i], (-1, self.number_of_features))
                mean_hidden_var = tf.reshape(mean_hidden_var_list[head_i], (-1, self.number_of_features))

                mean_hidden_list[head_i] = mean_hidden
                mean_hidden_var_list[head_i] = mean_hidden_var
                if self.use_temporal_std:
                    raise NotImplementedError
        else:
            raise ValueError("Invalid head pooling method.")

        if len(mean_hidden_list) > 1:
            mean_hidden = tf.concat(mean_hidden_list, axis=1)
            mean_hidden_var = tf.concat(mean_hidden_var_list, axis=1)
        elif len(mean_hidden_list) == 1:
            mean_hidden = mean_hidden_list[0]
            mean_hidden_var = mean_hidden_var_list[0]
        else:
            raise ValueError("Invalid number of heads.")

        if self.pool_heads == "attention":
            raise NotImplementedError
        else:
            if self.use_temporal_std:
                raise NotImplementedError
            else:
                mean_hidden = tf.reshape(mean_hidden, (-1, self.number_of_features * self.number_of_heads))
                mean_hidden_var = tf.reshape(mean_hidden_var, (-1, self.number_of_features * self.number_of_heads))

        prediction_single = list()
        prediction_single_var = list()
        for t in range(len(self.outputs_list)):
            prediction_single_t, \
            _, \
            prediction_single_var_t = self.dense_layer_list[t](inputs=mean_hidden,
                                                               training=training,
                                                               n_samples=1,
                                                               sample_type="Sample",
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
                                             net_train_var,
                                             net_test_var,
                                             y_pred_names,
                                             global_pooling_configuration):
    number_of_heads = global_pooling_configuration["number_of_heads"]
    use_temporal_std = global_pooling_configuration["use_temporal_std"]
    pool_heads = global_pooling_configuration["pool_heads"]
    auto_pooling = global_pooling_configuration["auto_pooling"]
    number_of_features = global_pooling_configuration["number_of_features"]
    sequence_length = global_pooling_configuration["sequence_length"]
    use_auto_array = global_pooling_configuration["use_auto_array"]
    uncertainty_handling = global_pooling_configuration["uncertainty_handling"]
    outputs_list = global_pooling_configuration["outputs_list"]

    attention_global_pooling = VariationalAttentionGlobalPooling(number_of_heads=number_of_heads,
                                                                 use_temporal_std=use_temporal_std,
                                                                 pool_heads=pool_heads,
                                                                 auto_pooling=auto_pooling,
                                                                 uncertainty_handling=uncertainty_handling,
                                                                 number_of_features=number_of_features,
                                                                 sequence_length=sequence_length,
                                                                 use_auto_array=use_auto_array,
                                                                 outputs_list=outputs_list,
                                                                 name="att_emb")

    prediction_single_train,\
    prediction_single_train_var,\
    attention_weights_train = attention_global_pooling(inputs=net_train,
                                                       x_var=net_train_var,
                                                       training=True)
    prediction_single_test,\
    prediction_single_test_var,\
    attention_weights_test = attention_global_pooling(inputs=net_test,
                                                      x_var=net_test_var,
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
