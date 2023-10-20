import tensorflow as tf

import variational.layers as variational_layers
import variational.activations as variational_activations

from common.models.local_pooling import VariationalMaxPool2D


class VariationalConvBlock(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 use_bias,
                 max_pool_size,
                 max_pool_strides,
                 num_layers,
                 use_max_pool,
                 max_pool_curiosity_initial_value,
                 max_pool_curiosity_type,
                 use_se,
                 ratio=4,
                 **kwargs):
        super(VariationalConvBlock, self).__init__(**kwargs)
        self.filters = filters
        self.use_bias = use_bias
        self.max_pool_size = max_pool_size
        self.max_pool_strides = max_pool_strides
        self.num_layers = num_layers
        self.use_max_pool = use_max_pool
        self.max_pool_curiosity_initial_value = max_pool_curiosity_initial_value
        self.max_pool_curiosity_type = max_pool_curiosity_type
        self.use_se = use_se
        self.ratio = ratio

        self.conv2d_function = variational_layers.Conv2dReparameterization

    def build(self,
              input_shape):  # [batch_size, image_length, image_width, filters]
        self.layer_list = list()
        for n in range(self.num_layers):
            self.layer_list.append(
                self.conv2d_function(filters=self.filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                     data_format=None, dilation_rate=(1, 1),
                                     # weight_prior=variational_priors.WeightPriorARD(),
                                     variance_parameterisation_type="layer_wise", activation="relu",
                                     trainable=True, use_bias=self.use_bias,
                                     # uncertainty_propagation_type="sample", trainable=True, use_bias=self.use_bias,
                                     weight_initializer="glorot_uniform", bias_initializer="zeros",
                                     name=self.name + "conv_" + repr(n), reuse=None, dtype=tf.float32,
                                     dynamic=False))

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
                                                        # curiosity_initial_value=self.max_pool_curiosity_initial_value,
                                                        curiosity_type=self.max_pool_curiosity_type,
                                                        name=self.name + "conv_max_pool"))

        super(VariationalConvBlock, self).build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'use_bias': self.use_bias,
            'max_pool_size': self.max_pool_size,
            'max_pool_strides': self.max_pool_strides,
            'num_layers': self.num_layers,
            'use_max_pool': self.use_max_pool,
            'max_pool_curiosity_initial_value': self.max_pool_curiosity_initial_value,
            'max_pool_curiosity_type': self.max_pool_curiosity_type,
            'use_se': self.use_se,
            'ratio': self.ratio,
        })
        return config

    def call(self, inputs, training, inputs_variances, **kwargs):
        h_mean,\
        h_var = self.layer_list[0](inputs=inputs,
                                   training=training,
                                   inputs_variances=inputs_variances)
        for l in self.layer_list[1:]:
            h_mean, \
            h_var = l(inputs=h_mean,
                      training=training,
                      inputs_variances=h_var)

        if training:
            self.get_kl_loss()
        return h_mean, h_var

    def get_kl_loss(self):
        self.kl = 0.0

        for l in self.layer_list:
            get_kl_loss_op = getattr(l, "get_kl_loss", None)
            if get_kl_loss_op is not None:
                self.kl = self.kl + l.get_kl_loss()
        return self.kl


class VariationalResBlock(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 use_se,
                 ratio=4,
                 se_type="cse",
                 **kwargs):
        super(VariationalResBlock, self).__init__(**kwargs)
        self.filters = filters
        self.use_se = use_se
        self.ratio = ratio
        self.se_type = se_type

        self.conv2d_function = variational_layers.Conv2dReparameterization

    def build(self,
              input_shape):  # [batch_size, image_length, image_width, filters]
        self.conv2d_1 = self.conv2d_function(filters=self.filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                             data_format=None, dilation_rate=(1, 1),
                                             # weight_prior = variational_priors.WeightPriorARD(),
                                             variance_parameterisation_type="layer_wise",
                                             activation="relu",
                                             trainable=True, use_bias=False,
                                             # uncertainty_propagation_type="sample", trainable=True, use_bias=False,
                                             weight_initializer="glorot_uniform", bias_initializer="zeros",
                                             name=self.name + "conv_1", reuse=None, dtype=tf.float32, dynamic=False)
        # Use of BN reduces performance on OSA-SMW by a lot.
        # self.bn_1 = MomentBatchNormalization()

        self.conv2d_2 = self.conv2d_function(filters=self.filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                             data_format=None, dilation_rate=(1, 1),
                                             # weight_prior = variational_priors.WeightPriorARD(),
                                             variance_parameterisation_type="layer_wise", activation=None,
                                             trainable=True, use_bias=False,
                                             # uncertainty_propagation_type="sample", trainable=True, use_bias=False,
                                             weight_initializer="glorot_uniform", bias_initializer="zeros",
                                             name=self.name + "conv_2", reuse=None, dtype=tf.float32, dynamic=False)
        # Use of BN reduces performance on OSA-SMW by a lot.
        # self.bn_2 = MomentBatchNormalization()

        self.conv_res = self.conv2d_function(filters=self.filters, kernel_size=(1, 1), strides=(1, 1), padding='same',
                                             data_format=None, dilation_rate=(1, 1),
                                             # weight_prior = variational_priors.WeightPriorARD(),
                                             variance_parameterisation_type="layer_wise", activation=None,
                                             trainable=True, use_bias=False,
                                             # uncertainty_propagation_type="sample", trainable=True, use_bias=False,
                                             weight_initializer="glorot_uniform", bias_initializer="zeros",
                                             name=self.name + "conv_res", reuse=None, dtype=tf.float32, dynamic=False)
        # Use of BN reduces performance on OSA-SMW by a lot.
        # self.bn_res = MomentBatchNormalization()

        if self.use_se:
            self.se_block = VariationalSEBlock(filters=self.filters,
                                               ratio=self.ratio,
                                               type=self.se_type,
                                               name=self.name + "se_block")

        super(VariationalResBlock, self).build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'use_se': self.use_se,
            'ratio': self.ratio,
            'se_type': self.se_type,
        })
        return config

    def call(self, inputs, training, inputs_variances, **kwargs):
        cnv2d_mean, \
        cnv2d_var = self.conv2d_1(inputs=inputs,
                                  training=training,
                                  inputs_variances=inputs_variances)
        # cnv2d_mean, \
        # cnv2d_var = self.bn_1(inputs=cnv2d_mean,
        #                       training=training,
        #                       inputs_variances=cnv2d_var)

        cnv2d_mean, \
        cnv2d_var = self.conv2d_2(inputs=cnv2d_mean,
                                  training=training,
                                  inputs_variances=cnv2d_var)
        # cnv2d_mean, \
        # cnv2d_var = self.bn_2(inputs=cnv2d_mean,
        #                       training=training,
        #                       inputs_variances=cnv2d_var)

        res_mean, \
        res_var = self.conv_res(inputs=inputs,
                                training=training,
                                inputs_variances=inputs_variances)
        # res_mean, \
        # res_var = self.bn_res(inputs=res_mean,
        #                         training=training,
        #                         inputs_variances=res_var)

        cnv2d_mean = cnv2d_mean + res_mean
        cnv2d_var = cnv2d_var + res_var

        cnv2d_mean, \
        cnv2d_var = variational_activations.relu_moments(x=cnv2d_mean,
                                                         x_var=cnv2d_var)

        if self.use_se:
            cnv2d_mean, \
            cnv2d_var = self.se_block(cnv2d_mean,
                                      training=training,
                                      inputs_variances=cnv2d_var)

        if training:
            self.get_kl_loss()

        return cnv2d_mean, cnv2d_var

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
                 **kwargs):
        super(VariationalSEBlock, self).__init__(**kwargs)
        self.filters = filters
        self.ratio = ratio
        self.type = type
        # self.name = name

    def build(self,
              input_shape):  # [batch_size, image_length, image_width, filters]
        if self.type not in ["cse", "sse", "scse"]:
            raise ValueError("Invalid SE Block type.")

        if self.type in ["cse", "scse"]:
            self.cse_pooling = tf.keras.layers.GlobalAveragePooling2D()
            squeeze_filters = self.filters // self.ratio

            self.cse_fc_1 = variational_layers.DenseReparameterisation(units=squeeze_filters,
                                                                       variance_parameterisation_type="layer_wise",
                                                                       activation="relu",
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
                                                                       activation="sigmoid",
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

        super(VariationalSEBlock, self).build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'ratio': self.ratio,
            'type': self.type,
        })
        return config

    def call(self, inputs, training, inputs_variances, **kwargs):
        if self.type not in ["cse", "sse", "scse"]:
            raise ValueError("Invalid SE Block type.")

        if self.type in ["cse", "scse"]:
            cse_se = self.cse_pooling(inputs, training=training)
            cse_se_var = self.cse_pooling(inputs_variances, training=training)

            cse_se,\
            cse_se_var = self.cse_fc_1(cse_se,
                                       training=training,
                                       inputs_variances=cse_se_var)
            cse_se, \
            cse_se_var = self.cse_fc_2(cse_se,
                                       training=training,
                                       inputs_variances=cse_se_var)
            cse_se = self.cse_reshape(cse_se, training=training)
            cse_se_var = self.cse_reshape(cse_se_var, training=training)

            cse_out = tf.multiply(inputs,
                                  cse_se)
            # cse_out = inputs * cse_se

            # cse_out_var = inputs_variances * cse_se
            # cse_out_var = tf.multiply(inputs_variances,
            #                           tf.pow(cse_se, 2.0))
            cse_out_var = tf.multiply(tf.pow(inputs, 2.0) + inputs_variances,
                                      tf.pow(cse_se, 2.0) + cse_se_var) - tf.multiply(tf.pow(inputs, 2.0),
                                                                                      tf.pow(cse_se, 2.0))
            # cse_out_var = tf.nn.relu(cse_out_var)
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

        return cse_out, cse_out_var

    def get_kl_loss(self):
        self.kl = self.cse_fc_1.get_kl_loss()
        self.kl = self.kl + self.cse_fc_2.get_kl_loss()
        return self.kl
