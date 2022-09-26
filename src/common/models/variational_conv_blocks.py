import math

import tensorflow as tf

import variational.layers as variational_layers
import variational.activations as variational_activations


class VariationalConvBlock():
    def __init__(self, filters,
                 use_bias,
                 max_pool_size,
                 max_pool_strides,
                 num_layers,
                 use_max_pool,
                 max_pool_curiosity_initial_value,
                 max_pool_curiosity_type,
                 se_uncertainty_handling,
                 use_se,
                 use_ds,
                 ratio=4,
                 name=None,
                 **kwargs):
        # super(VariationalConvBlock, self).__init__(**kwargs)
        self.filters = filters
        self.use_bias = use_bias
        self.max_pool_size = max_pool_size
        self.max_pool_strides = max_pool_strides
        self.num_layers = num_layers
        self.use_max_pool = use_max_pool
        self.max_pool_curiosity_initial_value = max_pool_curiosity_initial_value
        self.max_pool_curiosity_type = max_pool_curiosity_type
        self.se_uncertainty_handling = se_uncertainty_handling
        self.use_se = use_se
        self.use_ds = use_ds
        self.ratio = ratio
        self.name = name

        if self.use_ds:
            raise NotImplementedError
        else:
            self.conv2d_function = variational_layers.Conv2dReparameterization

        self.layer_list = list()
        for n in range(self.num_layers):
            self.layer_list.append(
                self.conv2d_function(filters=self.filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                     data_format=None, dilation_rate=(1, 1),# weight_prior=variational_priors.WeightPriorARD(),
                                     variance_parameterisation_type="layer_wise", use_clt=True, activation="relu",
                                     uncertainty_propagation_type="ral", trainable=True, use_bias=self.use_bias,
                                     # uncertainty_propagation_type="sample", trainable=True, use_bias=self.use_bias,
                                     weight_initializer="glorot_uniform", bias_initializer="zeros",
                                     name=self.name + "conv_" + repr(n), reuse=None, dtype=tf.float32, dynamic=False))

        if self.use_se:
            self.layer_list.append(VariationalSEBlock(filters=self.filters,
                                                      ratio=self.ratio,
                                                      type="cse",
                                                      uncertainty_handling=self.se_uncertainty_handling,
                                                      name=self.name + "se_block"))

        if self.use_max_pool:
            self.layer_list.append(VariationalMaxPool2D(pool_size=self.max_pool_size,
                                                        strides=self.max_pool_strides,
                                                        padding='valid',
                                                        curiosity_initial_value=self.max_pool_curiosity_initial_value,
                                                        curiosity_type=self.max_pool_curiosity_type,
                                                        name=self.name + "conv_max_pool"))

    # def get_config(self):
    #
    #     config = super().get_config()
    #     config.update({
    #         'filters': self.filters,
    #         'use_bias': self.use_bias,
    #         'max_pool_size': self.max_pool_size,
    #         'max_pool_strides': self.max_pool_strides,
    #         'num_layers': self.num_layers,
    #         'use_max_pool': self.use_max_pool,
    #         'max_pool_curiosity_initial_value': self.max_pool_curiosity_initial_value,
    #         'max_pool_curiosity_type': self.max_pool_curiosity_type,
    #         'use_se': self.use_se,
    #         'use_ds': self.use_ds,
    #         'ratio': self.ratio,
    #     })
    #     return config

    def __call__(self, inputs, training, inputs_variances, **kwargs):
        h_mean,\
        h_list,\
        h_var = self.layer_list[0](inputs=inputs,
                                   training=training,
                                   n_samples=1,
                                   sample_type="Sample",
                                   inputs_variances=inputs_variances)
        for l in self.layer_list[1:]:
            h_mean, \
            h_list, \
            h_var = l(inputs=h_mean,
                      training=training,
                      n_samples=1,
                      sample_type="Sample",
                      inputs_variances=h_var)

        if training:
            self.get_kl_loss()
        return h_mean, h_list, h_var

    def get_kl_loss(self):
        self.kl = 0.0

        for l in self.layer_list:
            get_kl_loss_op = getattr(l, "get_kl_loss", None)
            if get_kl_loss_op is not None:
                self.kl = self.kl + l.get_kl_loss()
        return self.kl


class VariationalMaxPool2D(tf.keras.layers.Layer):
    def __init__(self,
                 pool_size,
                 strides,
                 padding='valid',
                 curiosity_initial_value=1.0,
                 curiosity_type="fixed",
                 # name=None,
                 **kwargs):
        super(VariationalMaxPool2D, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self.curiosity_initial_value = curiosity_initial_value
        self.curiosity_type = curiosity_type
        # self.name = name

        # if self.strides[0] > 1 or self.strides[1] > 1:
        #     raise NotImplementedError

        if self.curiosity_type not in ["fixed",
                                       "attentive-adaptive",
                                       "attentive-fixed",
                                       "attentive_means",
                                       "attentive_distributional"]:
            raise NotImplementedError

        if self.curiosity_type == "fixed":
            self.curiosity_value = tf.constant(self.curiosity_initial_value,
                                               dtype=tf.float32)
        elif self.curiosity_type == "attentive-fixed":
            self.curiosity_value = tf.constant(self.curiosity_initial_value,
                                               dtype=tf.float32)
            # self.temperature = tf.Variable(1.0,
            #                                dtype=tf.float32,
            #                                trainable=True,
            #                                name=self.name + "_temp")
            self.temperature = self.add_weight(name=self.name + "_temp",
                                               shape=1,
                                               dtype=tf.float32,
                                               initializer="ones",
                                               regularizer=None,
                                               trainable=True,
                                               constraint=None,
                                               partitioner=None,
                                               use_resource=None)
        elif self.curiosity_type == "attentive-adaptive":
            self.curiosity_value = self.add_weight(name=self.name + "_curio",
                                                   shape=1,
                                                   dtype=tf.float32,
                                                   initializer="ones",  # TODO: Initial curiosity value.
                                                   regularizer=None,
                                                   trainable=True,
                                                   constraint=None,
                                                   partitioner=None,
                                                   use_resource=None)

            self.temperature = self.add_weight(name=self.name + "_temp",
                                               shape=1,
                                               dtype=tf.float32,
                                               initializer="ones",
                                               regularizer=None,
                                               trainable=True,
                                               constraint=None,
                                               partitioner=None,
                                               use_resource=None)
        else:
            # No extreme value distributions.
            # Moments known for two Gaussians: Exact Distribution of the Max/Min of Two Gaussian Random Variables
            # For more than two, open problem: On the distribution of maximum of multivariate normal random vectors
            # Helpful stackoverflow: https://stats.stackexchange.com/questions/34418/asymptotic-distribution-of-maximum-order-statistic-of-iid-random-normals
            raise NotImplementedError

    def get_config(self):
        config = super().get_config()
        config.update({
            'pool_size': self.pool_size,
            'strides': self.strides,
            'padding': self.padding,
            'curiosity_initial_value': self.curiosity_initial_value,
            'curiosity_type': self.curiosity_type,
        })
        return config

    def call(self, inputs, training, inputs_variances, **kwargs):
        h_mean, h_list, h_var = self.max_of_means_call(inputs,
                                                       training,
                                                       inputs_variances)

        if training:
            self.get_kl_loss()

        return h_mean, \
               h_list, \
               h_var

    def max_of_means_call(self,
                          inputs,
                          training,
                          inputs_variances):
        if self.curiosity_type in ["fixed", ]:
            pooled_data = inputs + self.curiosity_value * tf.sqrt(1e-8 + inputs_variances)

            _, indices = tf.nn.max_pool_with_argmax(pooled_data,
                                                    ksize=[1, self.pool_size[0], self.pool_size[1], 1],
                                                    strides=[1, self.strides[0], self.strides[1], 1],
                                                    padding='VALID',
                                                    include_batch_in_index=True)

            b, w, h, c = inputs.get_shape().as_list()
            pooled_data = tf.gather(tf.reshape(inputs, shape=[-1, ]), indices)
            h_var = tf.gather(tf.reshape(inputs_variances, shape=[-1, ]), indices)
            h_list = list()
            h_list.append(pooled_data)
        elif self.curiosity_type in ["attentive-fixed",
                                     "attentive-adaptive"]:
            b, w, h, c = inputs.get_shape().as_list()
            if self.curiosity_type == "attentive-fixed":
                pooled_data = inputs + self.curiosity_value * tf.sqrt(1e-8 + inputs_variances)
            elif self.curiosity_type == "attentive-adaptive":
                pooled_data = inputs + tf.log(1. + tf.exp(self.curiosity_value)) * tf.sqrt(
                    1e-8 + inputs_variances)
            else:
                raise ValueError

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
            # temperature = tf.log(1. + 1e-8 + tf.exp(self.temperature))
            temperature = self.temperature
            attention_weights = tf.nn.softmax(pooled_data * temperature, axis=3)

            pooled_data = tf.image.extract_patches(images=inputs,
                                                   sizes=[1, self.pool_size[0], self.pool_size[1], 1],
                                                   strides=[1, self.strides[0], self.strides[1], 1],
                                                   rates=[1, 1, 1, 1],
                                                   padding='VALID')
            pooled_data = tf.reshape(pooled_data, [-1,
                                                   w // self.strides[0],
                                                   h // self.strides[1],
                                                   self.pool_size[0] * self.pool_size[1],
                                                   c])
            pooled_data = tf.reduce_sum(tf.multiply(pooled_data,
                                                    attention_weights),
                                        axis=3)
            # pooled_data = tf.reduce_sum(tf.multiply(pooled_data,
            #                                         tf.pow(attention_weights, 2.0)),
            #                             axis=3)
            h_var = tf.image.extract_patches(images=inputs_variances,
                                             sizes=[1, self.pool_size[0], self.pool_size[1], 1],
                                             strides=[1, self.strides[0], self.strides[1], 1],
                                             rates=[1, 1, 1, 1],
                                             padding='VALID')
            h_var = tf.reshape(h_var, [-1,
                                       w // self.strides[0],
                                       h // self.strides[1],
                                       self.pool_size[0] * self.pool_size[1],
                                       c])
            # h_var = tf.reduce_sum(tf.multiply(h_var,
            #                                   attention_weights),
            #                       axis=3)
            h_var = tf.reduce_sum(tf.multiply(h_var,
                                              tf.pow(attention_weights, 2.0)),
                                  axis=3)
            h_list = list()
            h_list.append(pooled_data)
        else:
            raise NotImplementedError

        return pooled_data, \
               h_list, \
               h_var

    def get_kl_loss(self):
        self.kl = 0.0

        return self.kl


class VariationalResBlock:
    def __init__(self,
                 filters,
                 use_se,
                 use_ds,
                 se_uncertainty_handling,
                 ratio=4,
                 se_type="cse",
                 name=None,
                 **kwargs):
        # super(VariationalResBlock, self).__init__(**kwargs)
        self.filters = filters
        self.use_se = use_se
        self.use_ds = use_ds
        self.se_uncertainty_handling = se_uncertainty_handling
        self.ratio = ratio
        self.se_type = se_type
        self.name = name

        if self.use_ds:
            raise NotImplementedError
        else:
            self.conv2d_function = variational_layers.Conv2dReparameterization

        self.conv2d_1 = self.conv2d_function(filters=self.filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                             data_format=None, dilation_rate=(1, 1),# weight_prior = variational_priors.WeightPriorARD(),
                                             variance_parameterisation_type="layer_wise", use_clt=True, activation="relu",
                                             uncertainty_propagation_type="ral", trainable=True, use_bias=False,
                                             # uncertainty_propagation_type="sample", trainable=True, use_bias=False,
                                             weight_initializer="glorot_uniform", bias_initializer="zeros",
                                             name=self.name + "conv_1", reuse=None, dtype=tf.float32, dynamic=False)

        self.conv2d_2 = self.conv2d_function(filters=self.filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                             data_format=None, dilation_rate=(1, 1),# weight_prior = variational_priors.WeightPriorARD(),
                                             variance_parameterisation_type="layer_wise", use_clt=True, activation=None,
                                             uncertainty_propagation_type="ral", trainable=True, use_bias=False,
                                             # uncertainty_propagation_type="sample", trainable=True, use_bias=False,
                                             weight_initializer="glorot_uniform", bias_initializer="zeros",
                                             name=self.name + "conv_2", reuse=None, dtype=tf.float32, dynamic=False)

        self.conv_res = self.conv2d_function(filters=self.filters, kernel_size=(1, 1), strides=(1, 1), padding='same',
                                             data_format=None, dilation_rate=(1, 1),# weight_prior = variational_priors.WeightPriorARD(),
                                             variance_parameterisation_type="layer_wise", use_clt=True, activation=None,
                                             uncertainty_propagation_type="ral", trainable=True, use_bias=False,
                                             # uncertainty_propagation_type="sample", trainable=True, use_bias=False,
                                             weight_initializer="glorot_uniform", bias_initializer="zeros",
                                             name=self.name + "conv_res", reuse=None, dtype=tf.float32, dynamic=False)

        if self.use_se:
            self.se_block = VariationalSEBlock(filters=self.filters,
                                               ratio=self.ratio,
                                               type=self.se_type,
                                               uncertainty_handling=self.se_uncertainty_handling,
                                               name=self.name + "se_block")

    # def get_config(self):
    #     config = super().get_config()
    #     config.update({
    #         'filters': self.filters,
    #         'use_se': self.use_se,
    #         'use_ds': self.use_ds,
    #         'ratio': self.ratio,
    #         'se_type': self.se_type,
    #     })
    #     return config

    def __call__(self, inputs, training, inputs_variances, **kwargs):
        cnv2d_mean, \
        cnv2d_list, \
        cnv2d_var = self.conv2d_1(inputs=inputs,
                                  training=training,
                                  n_samples=1,
                                  sample_type="Sample",
                                  inputs_variances=inputs_variances)

        cnv2d_mean, \
        cnv2d_list, \
        cnv2d_var = self.conv2d_2(inputs=cnv2d_mean,
                                  training=training,
                                  n_samples=1,
                                  sample_type="Sample",
                                  inputs_variances=cnv2d_var)

        res_mean, \
        res_list, \
        res_var = self.conv_res(inputs=inputs,
                                training=training,
                                n_samples=1,
                                sample_type="Sample",
                                inputs_variances=inputs_variances)

        cnv2d_mean = cnv2d_mean + res_mean
        cnv2d_list = [a + b for a, b in zip(cnv2d_list, res_list)]
        cnv2d_var = cnv2d_var + res_var

        cnv2d_mean, \
        cnv2d_var = variational_activations.relu_moments(x=cnv2d_mean,
                                                         x_var=cnv2d_var)

        if self.use_se:
            cnv2d_mean, \
            cnv2d_list, \
            cnv2d_var = self.se_block(cnv2d_mean,
                                      training=training,
                                      inputs_variances=cnv2d_var,
                                      n_samples=1,
                                      sample_type="Sample")

        if training:
            self.get_kl_loss()

        return cnv2d_mean, cnv2d_list, cnv2d_var

    def get_kl_loss(self):
        self.kl = self.conv2d_1.get_kl_loss()
        self.kl = self.kl + self.conv2d_2.get_kl_loss()
        self.kl = self.kl + self.conv_res.get_kl_loss()
        if self.use_se:
            self.kl = self.kl + self.se_block.get_kl_loss()
        return self.kl


class VariationalSEBlock(tf.keras.layers.Layer):
    "Concurrent Spatial and Channel ‘Squeeze & Excitation’ in Fully Convolutional Networks"
    def __init__(self,
                 filters,
                 ratio=4,
                 type="cse",
                 uncertainty_handling="propagate",
                 # name=None,
                 **kwargs):
        super(VariationalSEBlock, self).__init__(**kwargs)
        self.filters = filters
        self.ratio = ratio
        self.type = type
        self.uncertainty_handling = uncertainty_handling
        # self.name = name

        if self.uncertainty_handling not in ["propagate",
                                             "concatenate",
                                             "curiosity"]:
            raise NotImplementedError

        if self.type not in ["cse", "sse", "scse"]:
            raise ValueError("Invalid SE Block type.")

        if self.type in ["cse", "scse"]:
            self.cse_pooling = tf.keras.layers.GlobalAveragePooling2D()
            if self.uncertainty_handling == "concatenate":
                squeeze_filters = self.filters // (self.ratio * 2)
            else:
                squeeze_filters = self.filters // self.ratio

            if self.uncertainty_handling == "curiosity":
                self.curiosity_value = self.add_weight(name=self.name + "curiosity_value",
                                                       shape=1,
                                                       dtype=tf.float32,
                                                       initializer="ones",
                                                       regularizer=None,
                                                       trainable=True,
                                                       constraint=None,
                                                       partitioner=None,
                                                       use_resource=None)
            self.cse_fc_1 = variational_layers.DenseReparameterisation(units=squeeze_filters,
                                                                       variance_parameterisation_type="layer_wise",
                                                                       use_clt=True,
                                                                       activation="relu",
                                                                       uncertainty_propagation_type="ral",
                                                                       trainable=True,
                                                                       use_bias=True,
                                                                       weight_initializer="glorot_uniform",
                                                                       bias_initializer="zeros",
                                                                       name=self.name + "se_dense_1",
                                                                       dtype=tf.float32,
                                                                       dynamic=False,
                                                                       reuse=None)
            self.cse_fc_2 = variational_layers.DenseReparameterisation(units=self.filters,
                                                                       variance_parameterisation_type="layer_wise",
                                                                       use_clt=True,
                                                                       activation="sigmoid",
                                                                       uncertainty_propagation_type="ral",
                                                                       trainable=True,
                                                                       use_bias=True,
                                                                       weight_initializer="glorot_uniform",
                                                                       bias_initializer="zeros",
                                                                       name=self.name + "se_dense_2",
                                                                       dtype=tf.float32,
                                                                       dynamic=False,
                                                                       reuse=None)
            self.cse_reshape = tf.keras.layers.Reshape((1, 1, self.filters))
        if self.type in ["sse", "scse"]:
            raise NotImplementedError

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'ratio': self.ratio,
            'type': self.type,
            'uncertainty_handling': self.uncertainty_handling,
        })
        return config

    def call(self, inputs, training, inputs_variances, **kwargs):
        if self.type not in ["cse", "sse", "scse"]:
            raise ValueError("Invalid SE Block type.")

        if self.type in ["cse", "scse"]:
            cse_se = self.cse_pooling(inputs, training=training)
            cse_se_var = self.cse_pooling(inputs_variances, training=training)

            if self.uncertainty_handling == "concatenate":
                cse_se = tf.concat([cse_se, cse_se_var], axis=1)
                cse_se_var = tf.concat([cse_se_var, tf.zeros_like(cse_se_var)], axis=1)
            elif self.uncertainty_handling == "propagate":
                pass
            elif self.uncertainty_handling == "curiosity":
                cse_se = cse_se + self.curiosity_value * tf.sqrt(1e-8 + cse_se_var)
                # cse_se_var = cse_se_var
            else:
                raise ValueError

            cse_se,\
            cse_out_list, \
            cse_se_var = self.cse_fc_1(cse_se,
                                       training=training,
                                       n_samples=1,
                                       sample_type="Sample",
                                       inputs_variances=cse_se_var)
            cse_se, \
            cse_out_list, \
            cse_se_var = self.cse_fc_2(cse_se,
                                       training=training,
                                       n_samples=1,
                                       sample_type="Sample",
                                       inputs_variances=cse_se_var)
            cse_se = self.cse_reshape(cse_se, training=training)
            cse_se_var = self.cse_reshape(cse_se_var, training=training)

            cse_out = inputs * cse_se
            # cse_out_var = inputs_variances * cse_se
            # cse_out_var = tf.multiply(inputs_variances,
            #                           tf.pow(cse_se, 2.0))
            cse_out_var = tf.multiply(tf.pow(inputs, 2.0) + inputs_variances,
                                      tf.pow(cse_se, 2.0) + cse_se_var) - tf.multiply(tf.pow(inputs, 2.0),
                                                                                      tf.pow(cse_se, 2.0))
            cse_out_var = tf.nn.relu(cse_out_var)
            # [8,64,1,8] vs. [8,64,150,64]
        if self.type in ["sse", "scse"]:
            raise NotImplementedError

        if self.type == "cse":
            pass
            # out = cse_out
            # out_list = list()
            # out_list.append(out)
            # out_var = cse_out_var
        elif self.type == "sse":
            raise NotImplementedError
        elif self.type == "scse":
            raise NotImplementedError
        else:
            raise ValueError("Invalid SE Block type.")

        if training:
            self.get_kl_loss()

        return cse_out, cse_out_list, cse_out_var
        # return out, out_list, out_var

    def get_kl_loss(self):
        self.kl = self.cse_fc_1.get_kl_loss()
        self.kl = self.kl + self.cse_fc_2.get_kl_loss()
        return self.kl
