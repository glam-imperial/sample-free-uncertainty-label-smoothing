import tensorflow as tf
import numpy as np


class AttentionGlobalPooling:
    def __init__(self,
                 number_of_heads,
                 use_temporal_std,
                 pool_heads,
                 auto_pooling,
                 number_of_features,
                 sequence_length,
                 use_auto_array,
                 outputs_list):
        self.number_of_heads = number_of_heads
        self.use_temporal_std = use_temporal_std
        self.pool_heads = pool_heads
        self.auto_pooling = auto_pooling
        self.number_of_features = number_of_features
        self.sequence_length = sequence_length
        self.use_auto_array = use_auto_array
        self.outputs_list = outputs_list

        self.heads_list = list()
        if self.use_auto_array:
            for head_i in range(self.number_of_heads):
                self.heads_list.append(tf.keras.layers.Dense(self.number_of_features, use_bias=False))
        else:
            for head_i in range(self.number_of_heads):
                self.heads_list.append(tf.keras.layers.Dense(1, use_bias=False))

        if self.auto_pooling == "Auto":
            self.auto_pool_variable_list = list()
            for head_i in range(self.number_of_heads):
                auto_init = 1 / max(1., np.floor((float(head_i) - 1.) / 2.) * 5)
                self.auto_pool_variable_list.append(tf.Variable(auto_init, dtype=tf.float32))
        elif self.auto_pooling == "MultiResolution":
            if self.number_of_heads == 1:
                raise ValueError("One head does not make sense for multiresolution prediction pooling.")
            self.auto_pool_constant_list = list()
            for head_i in range(self.number_of_heads):
                auto_init = 1 / max(1., np.floor((float(head_i) - 1.) / 2.) * 5)
                self.auto_pool_constant_list.append(tf.constant(auto_init, dtype=tf.float32))

        if self.pool_heads == "gating":
            self.mean_glu_layer = tf.keras.layers.Dense(self.number_of_features)
            if self.use_temporal_std:
                self.std_glu_layer = tf.keras.layers.Dense(self.number_of_features)
        elif self.pool_heads == "gating_old":
            if self.use_temporal_std:
                self.glu_layer = tf.keras.layers.Dense(2 * self.number_of_features * self.number_of_heads)
            else:
                self.glu_layer = tf.keras.layers.Dense(self.number_of_features * self.number_of_heads)
        elif self.pool_heads in ["attention", "attention_auto"]:
            if self.number_of_heads == 1:
                raise NotImplementedError
            self.pool_heads_layer = tf.keras.layers.Dense(1, use_bias=False)

            if self.pool_heads == "attention_auto":
                self.pool_heads_auto_pool_variable = tf.Variable(1.0, dtype=tf.float32)

        self.dense_layer_list = list()
        for t, output_units in enumerate(self.outputs_list):
            self.dense_layer_list.append(tf.keras.layers.Dense(output_units,
                                                               activation=None))

    def __call__(self, x, training):
        net = tf.reshape(x, (-1, self.number_of_features))  # [bs*sequence_length, number_of_features]

        energy_list = list()
        for head_i in range(self.number_of_heads):
            if self.use_auto_array:
                energy = tf.reshape(self.heads_list[head_i](net, training=training),
                                    (-1, self.sequence_length, self.number_of_features, 1))
            else:
                energy = tf.reshape(self.heads_list[head_i](net, training=training),
                                    (-1, self.sequence_length, 1, 1))

            energy_list.append(energy)

        if self.auto_pooling == "Auto":
            attention_weights_list = list()
            for head_i in range(self.number_of_heads):
                attention_weights_list.append(
                    tf.nn.softmax(self.auto_pool_variable_list[head_i] * energy_list[head_i], axis=1))

            if self.number_of_heads > 1:
                attention_weights = tf.concat(attention_weights_list,
                                              axis=3)  # [-1, sequence_length, 1, number_of_heads]
            elif self.number_of_heads == 1:
                attention_weights = attention_weights_list[0]  # [-1, sequence_length, 1, 1]
            else:
                raise ValueError("Invalid number of heads.")
        elif self.auto_pooling == "MultiResolution":
            if self.number_of_heads == 1:
                raise ValueError("One head does not make sense for multiresolution prediction pooling.")

            attention_weights_list = list()
            for head_i in range(self.number_of_heads):
                attention_weights_list.append(
                    tf.nn.softmax(self.auto_pool_constant_list[head_i] * energy_list[head_i], axis=1))

            if self.number_of_heads > 1:
                attention_weights = tf.concat(attention_weights_list,
                                              axis=3)  # [-1, sequence_length, 1, number_of_heads]
            elif self.number_of_heads == 1:
                attention_weights = attention_weights_list[0]  # [-1, sequence_length, 1, 1]
            else:
                raise ValueError("Invalid number of heads.")
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

        mean_hidden_list = list()
        for head_i in range(self.number_of_heads):
            mean_hidden = tf.reduce_sum(tf.multiply(net, attention_weights_list[head_i]), axis=1,
                                        keep_dims=True)
            mean_hidden_list.append(mean_hidden)  # [bs, 1, number_of_features, 1]

        if self.use_temporal_std:
            std_hidden_list = list()
            for head_i in range(self.number_of_heads):
                a = tf.multiply(tf.multiply(net, net),
                                attention_weights_list[head_i])

                b = tf.multiply(mean_hidden_list[head_i],
                                mean_hidden_list[head_i])

                std_hidden = tf.reduce_sum(a, axis=1, keep_dims=True) - b
                std_hidden = std_hidden + tf.abs(tf.reduce_min(std_hidden)) + tf.constant(0.000001,
                                                                                          dtype=tf.float32)
                std_hidden = tf.math.sqrt(std_hidden)
                std_hidden_list.append(std_hidden)  # [bs, 1, number_of_features, 1]

        if self.pool_heads == "gating":
            for head_i in range(self.number_of_heads):
                mean_hidden = tf.reshape(mean_hidden_list[head_i], (-1, self.number_of_features))
                if self.use_temporal_std:
                    std_hidden = tf.reshape(std_hidden_list[head_i], (-1, self.number_of_features))

                mean_hidden_list[head_i] = tf.multiply(tf.math.sigmoid(self.mean_glu_layer(mean_hidden,
                                                                                           training=training)),
                                                       mean_hidden)  # [bs, number_of_features]
                if self.use_temporal_std:
                    std_hidden_list[head_i] = tf.multiply(tf.math.sigmoid(self.std_glu_layer(std_hidden,
                                                                                             training=training)),
                                                          std_hidden)

                    mean_hidden_list[head_i] = tf.concat([mean_hidden_list[head_i],
                                                          std_hidden_list[head_i]],
                                                         axis=1)  # [bs, 2*number_of_features]
        elif self.pool_heads == "gating_old":
            for head_i in range(self.number_of_heads):
                mean_hidden_list[head_i] = tf.reshape(mean_hidden_list[head_i], (-1, self.number_of_features))
                if self.use_temporal_std:
                    std_hidden_list[head_i] = tf.reshape(std_hidden_list[head_i], (-1, self.number_of_features))

            if self.use_temporal_std:
                mean_hidden_list.extend(std_hidden_list)

            mean_hidden = tf.concat(mean_hidden_list, axis=1)

            mean_hidden = tf.multiply(tf.math.sigmoid(self.glu_layer(mean_hidden,
                                                                     training=training)),
                                      mean_hidden)

            mean_hidden_list = list()
            mean_hidden_list.append(mean_hidden)  # [bs, number_of_heads*number_of_features] -- consider STD

        elif self.pool_heads in ["attention", "attention_auto"]:
            if self.number_of_heads == 1:
                raise NotImplementedError

            pool_heads_hidden_list = list()
            pool_heads_energy_list = list()

            for head_i in range(self.number_of_heads):
                mean_hidden = tf.reshape(mean_hidden_list[head_i], (-1, self.number_of_features))
                if self.use_temporal_std:
                    std_hidden = tf.reshape(std_hidden_list[head_i], (-1, self.number_of_features))
                    pool_heads_hidden_list.append(tf.concat([mean_hidden,
                                                             std_hidden], axis=1))
                else:
                    pool_heads_hidden_list.append(mean_hidden)
                pool_heads_energy = tf.reshape(self.pool_heads_layer(pool_heads_hidden_list[head_i], training=training),
                                               (-1, 1))
                pool_heads_energy_list.append(pool_heads_energy)
            pool_heads_energy = tf.concat(pool_heads_energy_list, axis=1)

            if self.pool_heads == "attention":
                pool_heads_attention_weights = tf.reshape(tf.nn.softmax(pool_heads_energy, axis=1),
                                                          (-1, self.number_of_heads))
            elif self.pool_heads == "attention_auto":
                pool_heads_attention_weights = tf.reshape(tf.nn.softmax(self.pool_heads_auto_pool_variable * pool_heads_energy,
                                                                        axis=1),
                                                          (-1, self.number_of_heads))
            else:
                raise NotImplementedError

            pooled_hidden = pool_heads_hidden_list[0] * tf.expand_dims(pool_heads_attention_weights[:, 0], axis=1)
            for head_i in range(1, self.number_of_heads):
                pooled_hidden = pooled_hidden + pool_heads_hidden_list[head_i] * tf.expand_dims(
                    pool_heads_attention_weights[:, head_i], axis=1)

            mean_hidden_list = list()
            mean_hidden_list.append(pooled_hidden)  # [bs, number_of_heads*number_of_features] -- consider STD
        elif self.pool_heads == "no_pool":
            for head_i in range(self.number_of_heads):
                mean_hidden = tf.reshape(mean_hidden_list[head_i], (-1, self.number_of_features))
                if self.use_temporal_std:
                    std_hidden = tf.reshape(std_hidden_list[head_i], (-1, self.number_of_features))
                    mean_hidden_list[head_i] = tf.concat([mean_hidden,
                                                          std_hidden], axis=1)
                else:
                    mean_hidden_list[head_i] = mean_hidden
        else:
            raise ValueError("Invalid head pooling method.")

        if len(mean_hidden_list) > 1:
            mean_hidden = tf.concat(mean_hidden_list, axis=1)
        elif len(mean_hidden_list) == 1:
            mean_hidden = mean_hidden_list[0]
        else:
            raise ValueError("Invalid number of heads.")

        if self.pool_heads == "attention":
            if self.use_temporal_std:
                mean_hidden = tf.reshape(mean_hidden, (-1, 2 * self.number_of_features))
            else:
                mean_hidden = tf.reshape(mean_hidden, (-1, self.number_of_features))
        else:
            if self.use_temporal_std:
                mean_hidden = tf.reshape(mean_hidden, (-1, 2 * self.number_of_features * self.number_of_heads))
            else:
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
    use_temporal_std = global_pooling_configuration["use_temporal_std"]
    pool_heads = global_pooling_configuration["pool_heads"]
    auto_pooling = global_pooling_configuration["auto_pooling"]
    number_of_features = global_pooling_configuration["number_of_features"]
    sequence_length = global_pooling_configuration["sequence_length"]
    use_auto_array = global_pooling_configuration["use_auto_array"]
    outputs_list = global_pooling_configuration["outputs_list"]

    attention_global_pooling = AttentionGlobalPooling(number_of_heads=number_of_heads,
                                                      use_temporal_std=use_temporal_std,
                                                      pool_heads=pool_heads,
                                                      auto_pooling=auto_pooling,
                                                      number_of_features=number_of_features,
                                                      sequence_length=sequence_length,
                                                      use_auto_array=use_auto_array,
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
