import tensorflow as tf

from variational.layers import DenseReparameterisation


def dice_sorensen_coefficient(x_avg, x):
    dsc = 2 * tf.divide(tf.multiply(x_avg, x), (tf.pow(x_avg, 2.0) + tf.pow(x, 2.0)))
    return dsc


class AdaPool2D(tf.keras.layers.Layer):
    def __init__(self,
                 pool_size,
                 strides,
                 padding='valid',
                 type='adapool',  # ["ada", "em", "edscw"]
                 **kwargs):
        # From: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9982650
        super(AdaPool2D, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self.type = type

        if self.type not in ["ada", "em", "edscw"]:
            raise ValueError("Invalid pooling type.")

    def build(self,
              input_shape):  # [batch_size, image_length, image_width, filters]
        self.mask = self.add_weight(name=self.name + "_temp",
                                    shape=1,
                                    dtype=tf.float32,
                                    initializer="ones",
                                    regularizer=None,
                                    trainable=True,
                                    constraint=None,
                                    partitioner=None,
                                    use_resource=None)

        super(AdaPool2D, self).build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'pool_size': self.pool_size,
            'strides': self.strides,
            'padding': self.padding,
            'type': self.type,
        })
        return config

    def call(self, inputs, training, **kwargs):
        _, _, _, c = inputs.get_shape().as_list()
        # net_shape = [tf.shape(net_train)[k] for k in range(4)]
        b, w, h, _ = [tf.shape(inputs)[k] for k in range(4)]

        pooled_data = inputs

        pooled_data = tf.image.extract_patches(images=pooled_data,
                                               sizes=[1, self.pool_size[0], self.pool_size[1], 1],
                                               strides=[1, self.strides[0], self.strides[1], 1],
                                               rates=[1, 1, 1, 1],
                                               padding='VALID')
        pooled_data = tf.reshape(pooled_data, [-1,
                                               w // self.strides[0],
                                               h // self.strides[1],
                                               self.pool_size[0] * self.pool_size[1],
                                               c])
        # Exponential maximum pooling (emPool / SoftPool)
        if self.type in ["ada", "em"]:
            em_pool = tf.nn.softmax(pooled_data, axis=3)
            em_pool = tf.reduce_sum(tf.multiply(pooled_data,
                                                em_pool),
                                    axis=3)

        # Exponential Dice-Sorensen Coefficient Weighting Pooling (eDSCWPool)
        if self.type in ["ada", "edscw"]:
            dsc_pool = tf.reduce_mean(pooled_data, axis=3, keepdims=True)

            dsc_pool = tf.nn.softmax(dice_sorensen_coefficient(dsc_pool, pooled_data), axis=3)
            dsc_pool = tf.reduce_sum(tf.multiply(pooled_data,
                                                 dsc_pool),
                                     axis=3)

        if self.type == "em":
            pooled_data = em_pool
        elif self.type == "edscw":
            pooled_data = dsc_pool
        elif self.type == "ada":
            pooled_data = em_pool * self.mask + dsc_pool * (1 - self.mask)
        else:
            raise ValueError("Invalid pooling type.")

        return pooled_data


class AttMaxPool2D(tf.keras.layers.Layer):
    def __init__(self,
                 pool_size,
                 strides,
                 padding='valid',
                 **kwargs):
        super(AttMaxPool2D, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding

    def build(self,
              input_shape):  # [batch_size, image_length, image_width, filters]

        # self.projection = tf.keras.layers.Dense(1, use_bias=False)

        # self.temperature = self.add_weight(name=self.name + "_temp",
        #                                    shape=1,
        #                                    dtype=tf.float32,
        #                                    initializer="ones",
        #                                    regularizer=None,
        #                                    trainable=True,
        #                                    constraint=None,
        #                                    partitioner=None,
        #                                    use_resource=None)

        super(AttMaxPool2D, self).build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'pool_size': self.pool_size,
            'strides': self.strides,
            'padding': self.padding,
        })
        return config

    def call(self, inputs, training, **kwargs):
        h = self.max_of_means_call(inputs,
                                   training)

        return h

    def max_of_means_call(self,
                          inputs,
                          training):
        _, _, _, c = inputs.get_shape().as_list()
        # net_shape = [tf.shape(net_train)[k] for k in range(4)]
        b, w, h, _ = [tf.shape(inputs)[k] for k in range(4)]

        pooled_data = inputs

        pooled_data = tf.image.extract_patches(images=pooled_data,
                                               sizes=[1, self.pool_size[0], self.pool_size[1], 1],
                                               strides=[1, self.strides[0], self.strides[1], 1],
                                               rates=[1, 1, 1, 1],
                                               padding='VALID')
        pooled_data = tf.reshape(pooled_data, [-1,
                                               w // self.strides[0],
                                               h // self.strides[1],
                                               self.pool_size[0] * self.pool_size[1],
                                               c])
        # temperature = tf.log(1. + tf.exp(self.temperature))
        # temperature = self.temperature
        # attention_weights = self.projection(pooled_data)
        attention_weights = tf.nn.softmax(pooled_data, axis=3)

        # pooled_data = tf.image.extract_patches(images=inputs,
        #                                        sizes=[1, self.pool_size[0], self.pool_size[1], 1],
        #                                        strides=[1, self.strides[0], self.strides[1], 1],
        #                                        rates=[1, 1, 1, 1],
        #                                        padding='VALID')
        # pooled_data = tf.reshape(pooled_data, [-1,
        #                                        w // self.strides[0],
        #                                        h // self.strides[1],
        #                                        self.pool_size[0] * self.pool_size[1],
        #                                        c])
        pooled_data = tf.reduce_sum(tf.multiply(pooled_data,
                                                attention_weights),
                                    axis=3)

        return pooled_data


class VariationalMaxPool2D(tf.keras.layers.Layer):
    def __init__(self,
                 pool_size,
                 strides,
                 padding='valid',
                 # curiosity_initial_value=1.0,
                 curiosity_type="fixed",
                 **kwargs):
        super(VariationalMaxPool2D, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        # self.curiosity_initial_value = curiosity_initial_value
        self.curiosity_type = curiosity_type
        # self.name = name

        # if self.strides[0] > 1 or self.strides[1] > 1:
        #     raise NotImplementedError

        if self.curiosity_type not in ["fixed",
                                       "attentive-fixed"]:
            raise NotImplementedError

    def build(self,
              input_shape):  # [batch_size, image_length, image_width, filters]
        if self.curiosity_type == "fixed":
            pass
            # self.curiosity_value = tf.constant(self.curiosity_initial_value,
            #                                    dtype=tf.float32)
        elif self.curiosity_type == "attentive-fixed":
            pass
            # self.projection = DenseReparameterisation(units=1,
            #                                           variance_parameterisation_type="layer_wise",
            #                                           activation=None,
            #                                           trainable=True,
            #                                           use_bias=False,
            #                                           weight_initializer="glorot_uniform",
            #                                           bias_initializer="zeros",
            #                                           name=self.name + "_attpoolprojection",
            #                                           dtype=tf.float32,
            #                                           dynamic=False,
            #                                           reuse=None)
            # self.curiosity_value = tf.constant(self.curiosity_initial_value,
            #                                    dtype=tf.float32)
            # self.temperature = self.add_weight(name=self.name + "_temp",
            #                                    shape=1,
            #                                    dtype=tf.float32,
            #                                    initializer="ones",
            #                                    regularizer=None,
            #                                    trainable=True,
            #                                    constraint=None,
            #                                    partitioner=None,
            #                                    use_resource=None)
        else:
            # No extreme value distributions.
            # Moments known for two Gaussians: Exact Distribution of the Max/Min of Two Gaussian Random Variables
            # For more than two: On the distribution of maximum of multivariate normal random vectors
            # Helpful stackoverflow for large number of normal distributions: https://stats.stackexchange.com/questions/34418/asymptotic-distribution-of-maximum-order-statistic-of-iid-random-normals
            raise NotImplementedError

        super(VariationalMaxPool2D, self).build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'pool_size': self.pool_size,
            'strides': self.strides,
            'padding': self.padding,
            # 'curiosity_initial_value': self.curiosity_initial_value,
            'curiosity_type': self.curiosity_type,
        })
        return config

    def call(self, inputs, training, inputs_variances, **kwargs):
        h_mean, h_var = self.max_of_means_call(inputs,
                                               training,
                                               inputs_variances)

        if training:
            self.get_kl_loss()

        return h_mean, \
               h_var

    def max_of_means_call(self,
                          inputs,
                          training,
                          inputs_variances):
        if self.curiosity_type in ["fixed", ]:
            # pooled_data = inputs + self.curiosity_value * tf.sqrt(1e-8 + inputs_variances)
            pooled_data = inputs

            _, indices = tf.nn.max_pool_with_argmax(pooled_data,
                                                    ksize=[1, self.pool_size[0], self.pool_size[1], 1],
                                                    strides=[1, self.strides[0], self.strides[1], 1],
                                                    padding='VALID',
                                                    include_batch_in_index=True)

            b, w, h, c = inputs.get_shape().as_list()
            pooled_data = tf.gather(tf.reshape(inputs, shape=[-1, ]), indices)
            h_var = tf.gather(tf.reshape(inputs_variances, shape=[-1, ]), indices)
        elif self.curiosity_type in ["attentive-fixed",]:
            b, w, h, c = inputs.get_shape().as_list()
            # pooled_data = inputs + self.curiosity_value * tf.sqrt(1e-8 + inputs_variances)
            pooled_data = inputs

            pooled_data = tf.image.extract_patches(images=pooled_data,
                                                   sizes=[1, self.pool_size[0], self.pool_size[1], 1],
                                                   strides=[1, self.strides[0], self.strides[1], 1],
                                                   rates=[1, 1, 1, 1],
                                                   padding='VALID')
            # pooled_data = tf.reshape(pooled_data, [-1,
            #                                        c])

            h_var = tf.image.extract_patches(images=inputs_variances,
                                             sizes=[1, self.pool_size[0], self.pool_size[1], 1],
                                             strides=[1, self.strides[0], self.strides[1], 1],
                                             rates=[1, 1, 1, 1],
                                             padding='VALID')
            # h_var = tf.reshape(h_var, [-1,
            #                            c])

            # temperature = tf.log(1. + 1e-8 + tf.exp(self.temperature))
            # temperature = self.temperature
            # attention_weights,\
            #     attention_weights_var = self.projection(inputs=pooled_data,
            #                                             training=training,
            #                                             inputs_variances=h_var)
            #
            attention_weights = pooled_data
            attention_weights_var = h_var

            attention_weights = tf.reshape(attention_weights, [-1,
                                                               w // self.strides[0],
                                                               h // self.strides[1],
                                                               self.pool_size[0] * self.pool_size[1],
                                                               c])

            attention_weights_var = tf.reshape(attention_weights_var, [-1,
                                                               w // self.strides[0],
                                                               h // self.strides[1],
                                                               self.pool_size[0] * self.pool_size[1],
                                                               c])

            attention_weights = tf.nn.softmax(attention_weights, axis=3) + 0.0 * attention_weights_var

            pooled_data = tf.reshape(pooled_data, [-1,
                                                   w // self.strides[0],
                                                   h // self.strides[1],
                                                   self.pool_size[0] * self.pool_size[1],
                                                   c])
            h_var = tf.reshape(h_var, [-1,
                                       w // self.strides[0],
                                       h // self.strides[1],
                                       self.pool_size[0] * self.pool_size[1],
                                       c])

            # pooled_data = tf.image.extract_patches(images=inputs,
            #                                        sizes=[1, self.pool_size[0], self.pool_size[1], 1],
            #                                        strides=[1, self.strides[0], self.strides[1], 1],
            #                                        rates=[1, 1, 1, 1],
            #                                        padding='VALID')
            # pooled_data = tf.reshape(pooled_data, [-1,
            #                                        w // self.strides[0],
            #                                        h // self.strides[1],
            #                                        self.pool_size[0] * self.pool_size[1],
            #                                        c])
            pooled_data = tf.reduce_sum(tf.multiply(pooled_data,
                                                    attention_weights),
                                        axis=3)

            # h_var = tf.reduce_sum(tf.multiply(h_var,
            #                                   attention_weights),
            #                       axis=3)
            h_var = tf.reduce_sum(tf.multiply(h_var,
                                              tf.pow(attention_weights, 2.0)),
                                  axis=3)
        else:
            raise NotImplementedError

        return pooled_data, \
               h_var

    def get_kl_loss(self):
        self.kl = 0.0

        return self.kl
