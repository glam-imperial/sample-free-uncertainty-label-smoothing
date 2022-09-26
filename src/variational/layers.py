import math

import tensorflow as tf

from variational.bbb_utils import *
import variational.activations as variational_activations


class VariationalLayer(tf.keras.layers.Layer):
    def __init__(self,
                 # weight_prior,
                 variance_parameterisation_type,
                 use_clt,
                 activation=None,
                 uncertainty_propagation_type=None,
                 # trainable=True,
                 use_bias=True,
                 weight_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 # name=None,
                 reuse=None,
                 **kwargs):
        super(VariationalLayer, self).__init__(
            # trainable=trainable,
            #                                    name=name,
            #                                    dtype=tf.float32,
            #                                    dynamic=False,
                                               **kwargs)
        # self.weight_prior = weight_prior
        self.variance_parameterisation_type = variance_parameterisation_type
        self.use_clt = use_clt
        self.activation = activation
        self.uncertainty_propagation_type = uncertainty_propagation_type
        # self.trainable = trainable
        self.use_bias = use_bias
        self.weight_initializer = tf.keras.initializers.get(weight_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        # self.name = name
        self.reuse = reuse

        #######################################################################################

        if variance_parameterisation_type not in ["additive",
                                                  "weight_wise",
                                                  "neuron_wise",
                                                  "layer_wise"]:
            raise NotImplementedError("Invalid variance parameterisation type.")

        self.rho_init = tf.random_uniform_initializer(-4.6, -3.9)

        self.built_q_network = False

        self.kl_loss = None

        if self.use_clt:
            if self.activation is None:
                # Doesn't matter whatever uncertainty propagation type it is.
                if (self.uncertainty_propagation_type is None) or (self.uncertainty_propagation_type == "sample"):
                    self.clt_activation_function = get_clt_activation_function(activation_type="sample",
                                                                               is_conv=self.is_conv)
                elif self.uncertainty_propagation_type == "ral":
                    self.clt_activation_function = get_clt_activation_function(activation_type="ral",
                                                                               is_conv=self.is_conv)
            elif self.activation in ["relu", "sigmoid"]:
                if (self.uncertainty_propagation_type is None) or (self.uncertainty_propagation_type == "sample"):
                    # We will sample and propagate the sample and variance.
                    self.clt_activation_function = get_clt_activation_function(activation_type="sample",
                                                                               is_conv=self.is_conv)
                elif self.uncertainty_propagation_type == "ral":
                    # We will propagate approximated moments.
                    self.clt_activation_function = get_clt_activation_function(activation_type="ral",
                                                                               is_conv=self.is_conv)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        else:
            # Doesn't matter whatever uncertainty propagation type it is.
            self.uncertainty_propagation_type = None
            if self.activation is None:
                self.noclt_activation_function = get_noclt_activation_function(activation_type="sample",
                                                                               is_conv=self.is_conv)
            else:
                self.noclt_activation_function = get_noclt_activation_function(activation_type="sample",
                                                                               is_conv=self.is_conv)

        # Variational parameters
        # Georgios: Keeping same names for posterior sharpening if possible.
        # self.W_mu = None
        # self.W_rho = None
        # if self.use_bias:
        #     self.b_mu = None
        #     self.b_rho = None

        # Georgios TODO: Am unsure if the below are used now. Could leave.
        # self.eps_out = tf.placeholder(tf.float32, (None, self.b_dims), name='eps_out')
        # self.eps_out_sample = get_random((self._batch_size, self.b_dims), avg=0., std=1.)

    def get_config(self):
        config = super().get_config()
        config.update({
            # 'weight_prior': self.weight_prior,
            'variance_parameterisation_type': self.variance_parameterisation_type,
            'use_clt': self.use_clt,
            'activation': self.activation,
            'uncertainty_propagation_type': self.uncertainty_propagation_type,
            # 'trainable': self.trainable,
            'use_bias': self.use_bias,
            'weight_initializer': self.weight_initializer,
            'bias_initializer': self.bias_initializer,
            'reuse': self.reuse,
        })

        return config

    def call(self,  # Georgios The args were remade to somewhat resemble keras layers.
             inputs,
             training,
             n_samples,
             sample_type,  # "Sample", "MAP", "Variance"
             inputs_variances=None,
             **kwargs):

        if n_samples < 1:
            raise ValueError("We assume at least 1 sample.")

        self.kl_loss = None

        input_tensor = inputs

        # The following are needed to calculate KL-div of variational posterior with prior.
        var_par = dict()
        var_par["use_bias"] = self.use_bias
        var_par["w_dims"] = self.w_dims
        var_par["b_dims"] = self.b_dims
        var_par["W_mu"] = None
        var_par["W_var"] = None
        var_par["W_log_alpha"] = self.W_log_alpha
        var_par["W_rho"] = None
        var_par["W_sample_list"] = []
        if self.use_bias:
            var_par["b_mu"] = None
            var_par["b_var"] = None
            var_par["b_log_alpha"] = self.b_log_alpha
            var_par["b_rho"] = None
            var_par["b_sample_list"] = []

        if self.variance_parameterisation_type == "layer_wise":
            out_mean, out_var, W_mu, W_var, b_mu, b_var = self.layer_wise_variance(input_tensor,
                                                                                   inputs_variances)
        elif self.variance_parameterisation_type == "neuron_wise":
            if self.is_conv:
                raise NotImplementedError
            out_mean, out_var, W_mu, W_var, b_mu, b_var = self.layer_wise_variance(input_tensor,
                                                                                   inputs_variances)
        elif self.variance_parameterisation_type == "weight_wise":
            if self.is_conv:
                raise NotImplementedError
            out_mean, out_var, W_mu, W_var, b_mu, b_var = self.layer_wise_variance(input_tensor,
                                                                                   inputs_variances)
        elif self.variance_parameterisation_type == "additive":
            out_mean, out_var, W_mu, W_var, b_mu, b_var = self.additive_variance(input_tensor,
                                                                                 inputs_variances)
        else:
            raise ValueError("Invalid variance parameterisation type.")

        out_mean = tf.where(tf.is_nan(out_mean), tf.zeros_like(out_mean), out_mean)
        out_var = tf.where(tf.is_nan(out_var), tf.zeros_like(out_var), out_var)

        if sample_type == "Sample":
            var_par["W_mu"] = W_mu
            var_par["W_var"] = W_var
            if self.use_bias:
                var_par["b_mu"] = b_mu
                var_par["b_var"] = b_var
        elif sample_type == "Variance":
            var_par["W_mu"] = tf.zeros_like(W_mu)
            var_par["W_var"] = W_var
            if self.use_bias:
                var_par["b_mu"] = tf.zeros_like(b_mu)
                var_par["b_var"] = b_var
        elif sample_type == "MAP":
            if n_samples > 1:
                raise ValueError("MAP is not sampling.")
            var_par["W_mu"] = W_mu
            var_par["W_var"] = tf.zeros_like(W_var)
            if self.use_bias:
                var_par["b_mu"] = b_mu
                var_par["b_var"] = tf.zeros_like(b_var)
        else:
            raise NotImplementedError("Invalid sampling type.")

        # Local Reparameterization
        if self.use_clt:
            h_mean, h_list, h_var = self.clt_activation_function(out_mean=out_mean,
                                                                 out_var=out_var,
                                                                 n_samples=n_samples,
                                                                 sample_type=sample_type,
                                                                 activation=self.activation)

        # Regular reparameterization.
        else:
            h_mean, h_list, h_var = self.noclt_activation_function(input_tensor=input_tensor,
                                                                   var_par=var_par,
                                                                   w_dims=self.w_dims,
                                                                   b_dims=self.b_dims,
                                                                   use_bias=self.use_bias,
                                                                   n_samples=n_samples,
                                                                   sample_type=sample_type,
                                                                   activation=self.activation)

            # # W_stacked = tf.stack(W_sample_list)
            # # b_stacked = tf.stack(b_sample_list)
            # # self.W = tf.reduce_sum(W_stacked, axis=0) / self._n_samples
            # # self.b = tf.reduce_sum(b_stacked, axis=0) / self._n_samples

        if training:
            self.calculate_kl_loss(var_par=var_par)

        return h_mean, h_list, h_var

    def layer_wise_variance(self, x, x_var):
        W_mu = self.W_mu
        W_var = tf.multiply(tf.exp(self.W_log_alpha),
                            tf.pow(self.W_mu, 2.0))

        if self.is_conv:
            out_mean, out_var = self.conv2d_operation_with_uncertainty(x, x_var, W_mu, W_var)
        else:
            out_mean, out_var = self.dense_operation_with_uncertainty(x, x_var, W_mu, W_var)

        if self.use_bias:
            b_mu = self.b_mu
            b_var = tf.multiply(tf.exp(self.b_log_alpha),
                                tf.pow(self.b_mu, 2.0))

            out_mean = out_mean + b_mu
            out_var = out_var + b_var
        else:
            b_mu = None
            b_var = None
        return out_mean, out_var, W_mu, W_var, b_mu, b_var

    def additive_variance(self, x, x_var):
        W_mu = self.W_mu
        W_var = tf.pow(softplus(self.W_rho), 2.0)

        if not self.is_conv:
            out_mean, out_var = self.conv2d_operation_with_uncertainty(x, x_var, W_mu, W_var)
        else:
            out_mean, out_var = self.dense_operation_with_uncertainty(x, x_var, W_mu, W_var)

        if self.use_bias:
            # RAL did not have bias variance.
            b_mu = self.b_mu
            b_var = tf.pow(softplus(self.b_rho), 2.0)

            out_mean = out_mean + b_mu
            out_var = out_var + b_var
        else:
            b_mu = None
            b_var = None
        return out_mean, out_var, W_mu, W_var, b_mu, b_var

    def conv2d_operation_with_uncertainty(self, x, x_var, W_mu, W_var):
        if self.padding == "same":
            padding = "SAME"
        elif self.padding == "valid":
            padding = "VALID"
        else:
            padding = self.padding

        out_mean = tf.nn.conv2d(input=x,
                                filter=W_mu,
                                strides=self.strides,
                                padding=padding,
                                data_format=self.data_format,
                                dilations=self.dilation_rate)

        if x_var is None:
            out_var = tf.nn.conv2d(input=tf.pow(x, 2.0),
                                   filter=W_var,
                                   strides=self.strides,
                                   padding=padding,
                                   data_format=self.data_format,
                                   dilations=self.dilation_rate)
        else:
            out_var = tf.nn.conv2d(input=x_var + tf.pow(x, 2.0),
                                   filter=W_var,
                                   strides=self.strides,
                                   padding=padding,
                                   data_format=self.data_format,
                                   dilations=self.dilation_rate) +\
                      tf.nn.conv2d(input=x_var,
                                   filter=tf.pow(W_mu, 2.0),
                                   strides=self.strides,
                                   padding=padding,
                                   data_format=self.data_format,
                                   dilations=self.dilation_rate)
        return out_mean, out_var

    def dense_operation_with_uncertainty(self, x, x_var, W_mu, W_var):
        out_mean = tf.matmul(x, W_mu)

        if x_var is None:
            out_var = tf.matmul(tf.pow(x, 2.0), W_var)
            # var_f = F.linear(mu_h ** 2, var_s)
        else:
            out_var = tf.matmul(x_var + tf.pow(x, 2.0), W_var) +\
                      tf.matmul(x_var, tf.pow(W_mu, 2.0))
            # var_f = F.linear(var_h + mu_h.pow(2), var_s) + F.linear(var_h, M.pow(2))
        return out_mean, out_var

    def calculate_kl_loss(self,
                          var_par):
        W_mu = var_par["W_mu"]
        W_log_alpha = var_par["W_log_alpha"]
        use_bias = var_par["use_bias"]
        if use_bias:
            b_mu = var_par["b_mu"]
            b_log_alpha = var_par["b_log_alpha"]

        # This is known in closed form.
        # tf.log(1.0 + (1.0 / (1e-8 + tf.exp(W_log_alpha))))
        W_kl = tf.reduce_sum(tf.log(1.0 + (1.0 / (1e-8 + tf.exp(W_log_alpha)))) * tf.ones_like(W_mu))
        kl = W_kl
        if use_bias:
            b_kl = tf.reduce_sum(tf.log(1.0 + (1.0 / (1e-8 + tf.exp(b_log_alpha)))) * tf.ones_like(b_mu))
            kl = kl + b_kl
        # W_kl = tf.reduce_sum(tf.log(1.0 + (1.0 / 1e-8 + tf.exp(W_log_alpha))) * tf.ones_like(W_mu))
        # kl = W_kl
        # if use_bias:
        #     b_kl = tf.reduce_sum(tf.log(1.0 + (1.0 / 1e-8 + tf.exp(b_log_alpha))) * tf.ones_like(b_mu))
        #     kl = kl + b_kl
        # return kl

        # kl = self.weight_prior.calculate_kl(var_par=var_par)

        self.kl = kl

    def get_kl_loss(self):
        return self.kl


class DenseReparameterisation(VariationalLayer):
    def __init__(self, units,
                 # weight_prior,
                 # variance_parameterisation_type,
                 # use_clt,
                 # activation=None,
                 # uncertainty_propagation_type=None,  # ["ral", ]
                 # trainable=True,
                 # use_bias=True,
                 # weight_initializer="glorot_uniform",
                 # bias_initializer="zeros",
                 # name=None,
                 # reuse=None,
                 **kwargs):
        self.is_conv = False
        super(DenseReparameterisation, self).__init__(
            # weight_prior=weight_prior,
            #                                           variance_parameterisation_type=variance_parameterisation_type,
            #                                           use_clt=use_clt,
            #                                           activation=activation,
            #                                           uncertainty_propagation_type=uncertainty_propagation_type,
            #                                           trainable=trainable,
            #                                           use_bias=use_bias,
            #                                           weight_initializer=weight_initializer,
            #                                           bias_initializer=bias_initializer,
            #                                           name=name,
            #                                           reuse=reuse,
                                                      **kwargs)
        self.units = units

    def get_config(self):

        config = super().get_config()
        config.update({
            'units': self.units,
        })
        return config

    def build(self, input_shape):

        if len(input_shape) != 2:
            raise ValueError("Invalid number of input tensor dimensions. Needs to be [batch_size, input_units].")

        self.w_dims = (int(input_shape[-1]), self.units)
        self.b_dims = (1, self.units)

        self.W_mu = self.add_weight(name=self.name + "_W_mu",
                                    shape=self.w_dims,
                                    dtype=tf.float32,
                                    initializer=self.weight_initializer,
                                    regularizer=None,
                                    trainable=self.trainable,
                                    constraint=None,
                                    partitioner=None,
                                    use_resource=None)
        if self.variance_parameterisation_type == 'layer_wise':
            self.W_rho = None
            self.W_log_alpha = self.add_weight(name=self.name + "_W_alpha",
                                               shape=(1, ),
                                               dtype=tf.float32,
                                               initializer=tf.constant_initializer(-4.0),
                                               regularizer=None,
                                               trainable=self.trainable,
                                               constraint=None,
                                               partitioner=None,
                                               use_resource=None)
        elif self.variance_parameterisation_type == 'neuron_wise':
            self.W_rho = None
            self.W_log_alpha = self.add_weight(name=self.name + "_W_alpha",
                                               shape=self.b_dims,
                                               dtype=tf.float32,
                                               initializer=tf.constant_initializer(-4.0),
                                               regularizer=None,
                                               trainable=self.trainable,
                                               constraint=None,
                                               partitioner=None,
                                               use_resource=None)
        elif self.variance_parameterisation_type == 'weight_wise':
            self.W_rho = None
            self.W_log_alpha = self.add_weight(name=self.name + "_W_alpha",
                                               shape=self.w_dims,  # self.w_dims,
                                               dtype=tf.float32,
                                               initializer=tf.constant_initializer(-4.0),
                                               regularizer=None,
                                               trainable=self.trainable,
                                               constraint=None,
                                               partitioner=None,
                                               use_resource=None)
        elif self.variance_parameterisation_type == 'additive':
            self.W_log_alpha = None
            self.W_rho = self.add_weight(name=self.name + "_W_rho",
                                         shape=self.w_dims,
                                         dtype=tf.float32,
                                         initializer=self.rho_init,
                                         regularizer=None,
                                         trainable=self.trainable,
                                         constraint=None,
                                         partitioner=None,
                                         use_resource=None)
        else:
            raise ValueError("Invalid variance parameterisation type.")
        if self.use_bias:
            self.b_mu = self.add_weight(name=self.name + "_b_mu",
                                        shape=self.b_dims,
                                        dtype=tf.float32,
                                        initializer=self.bias_initializer,
                                        regularizer=None,
                                        trainable=self.trainable,
                                        constraint=None,
                                        partitioner=None,
                                        use_resource=None)

            if self.variance_parameterisation_type == 'layer_wise':
                self.b_rho = None
                self.b_log_alpha = self.add_weight(name=self.name + "_b_alpha",
                                                   shape=(1, ),
                                                   dtype=tf.float32,
                                                   initializer=tf.constant_initializer(-4.0),
                                                   regularizer=None,
                                                   trainable=self.trainable,
                                                   constraint=None,
                                                   partitioner=None,
                                                   use_resource=None)
            elif self.variance_parameterisation_type in ['neuron_wise', 'weight_wise']:
                self.b_rho = None
                self.b_log_alpha = self.add_weight(name=self.name + "_b_alpha",
                                                   shape=self.b_dims,
                                                   dtype=tf.float32,
                                                   initializer=tf.constant_initializer(-4.0),
                                                   regularizer=None,
                                                   trainable=self.trainable,
                                                   constraint=None,
                                                   partitioner=None,
                                                   use_resource=None)
            elif self.variance_parameterisation_type == 'additive':
                self.b_log_alpha = None
                self.b_rho = self.add_weight(name=self.name + "_b_rho",
                                             shape=self.b_dims,
                                             dtype=tf.float32,
                                             initializer=self.rho_init,
                                             regularizer=None,
                                             trainable=self.trainable,
                                             constraint=None,
                                             partitioner=None,
                                             use_resource=None)
            else:
                raise ValueError("Invalid variance parameterisation type.")

        # if self._use_posterior_sharpening:
        #     self.phi_W = tf.Variable(np.full(self.w_dims, 0.01), name=self._name + '_phi_W', shape=self.w_dims,
        #                                  dtype=tf.float32)
        #     self.phi_b = tf.Variable(np.full(self.b_dims, 0.01), name=self._name + '_phi_b', shape=self.b_dims,
        #                                  dtype=tf.float32)
        super().build(input_shape)


class Conv2dReparameterization(VariationalLayer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides,  #(1, 1),
                 padding,  #'valid',
                 data_format,  #=None,
                 dilation_rate,  #=(1, 1),
                 # weight_prior,
                 # variance_parameterisation_type,
                 # use_clt,
                 # activation=None,
                 # uncertainty_propagation_type=None,
                 # trainable=True,
                 # use_bias=True,
                 # weight_initializer="glorot_uniform",
                 # bias_initializer="zeros",
                 # name=None,
                 # reuse=None,
                 **kwargs):
        self.is_conv = True
        super(Conv2dReparameterization, self).__init__(
            # weight_prior=weight_prior,
            #                                            variance_parameterisation_type=variance_parameterisation_type,
            #                                            use_clt=use_clt,
            #                                            activation=activation,
            #                                            uncertainty_propagation_type=uncertainty_propagation_type,
            #                                            trainable=trainable,
            #                                            use_bias=use_bias,
            #                                            weight_initializer=weight_initializer,
            #                                            bias_initializer=bias_initializer,
            #                                            name=name,
            #                                            reuse=reuse,
                                                       **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate

    def get_config(self):

        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
        })
        return config

    def build(self, input_shape):

        if len(input_shape) != 4:
            raise ValueError("Invalid number of input tensor dimensions. Needs to be [batch_size, L, W, C].")

        self._batch_size = input_shape[0].value
        self._in_channels = input_shape[-1].value

        self.length = self.kernel_size[0]
        self.width = self.kernel_size[1]

        self.w_dims = [self.kernel_size[0], self.kernel_size[1], self._in_channels, self.filters]
        self.b_dims = [1, 1, 1, self.filters]

        self.W_mu = self.add_weight(name=self.name + "_k_mu",
                                    shape=self.w_dims,
                                    dtype=tf.float32,
                                    initializer=self.weight_initializer,
                                    regularizer=None,
                                    trainable=self.trainable,
                                    constraint=None,
                                    partitioner=None,
                                    use_resource=None)

        if self.variance_parameterisation_type == 'layer_wise':
            self.W_rho = None
            self.W_log_alpha = self.add_weight(name=self.name + "_k_alpha",
                                               shape=(1, ),
                                               dtype=tf.float32,
                                               initializer=tf.constant_initializer(-4.0),
                                               regularizer=None,
                                               trainable=self.trainable,
                                               constraint=None,
                                               partitioner=None,
                                               use_resource=None)
        elif self.variance_parameterisation_type == 'neuron_wise':
            raise NotImplementedError
        elif self.variance_parameterisation_type == 'weight_wise':
            raise NotImplementedError
        elif self.variance_parameterisation_type == 'additive':
            self.W_log_alpha = None
            self.W_rho = self.add_weight(name=self.name + "_k_rho",
                                         shape=self.w_dims,
                                         dtype=tf.float32,
                                         initializer=self.rho_init,
                                         regularizer=None,
                                         trainable=self.trainable,
                                         constraint=None,
                                         partitioner=None,
                                         use_resource=None)

        else:
            raise ValueError("Invalid variance parameterisation type.")
        if self.use_bias:
            self.b_mu = self.add_weight(name=self.name + "_b_mu",
                                        shape=self.b_dims,
                                        dtype=tf.float32,
                                        initializer=self.bias_initializer,
                                        regularizer=None,
                                        trainable=self.trainable,
                                        constraint=None,
                                        partitioner=None,
                                        use_resource=None)

            if self.variance_parameterisation_type == 'layer_wise':
                self.b_rho = None
                self.b_log_alpha = self.add_weight(name=self.name + "_b_alpha",
                                                   shape=(1, ),
                                                   dtype=tf.float32,
                                                   initializer=tf.constant_initializer(-4.0),
                                                   regularizer=None,
                                                   trainable=self.trainable,
                                                   constraint=None,
                                                   partitioner=None,
                                                   use_resource=None)
            elif self.variance_parameterisation_type in ['neuron_wise', 'weight_wise']:
                raise NotImplementedError
            elif self.variance_parameterisation_type == 'additive':
                self.b_log_alpha = None
                self.b_rho = self.add_weight(name=self.name + "_b_rho",
                                             shape=self.b_dims,
                                             dtype=tf.float32,
                                             initializer=self.rho_init,
                                             regularizer=None,
                                             trainable=self.trainable,
                                             constraint=None,
                                             partitioner=None,
                                             use_resource=None)
            else:
                raise ValueError("Invalid variance parameterisation type.")

        super().build(input_shape)


def get_clt_activation_function(activation_type,
                                is_conv):
    if activation_type == "sample":
        def dense_clt_activation_function(out_mean,
                                          out_var,
                                          n_samples,
                                          sample_type,
                                          activation):
            out_std = tf.sqrt(1e-8 + out_var)
            # The forward pass through one layer
            h_sum = 0.0
            h_list = []
            for s in range(n_samples):
                out_mean_offset = tf.multiply(out_std, get_random(tf.shape(out_std), avg=0., std=1.))

                if sample_type == "Sample":
                    h_sample = out_mean + out_mean_offset
                    h_mean = out_mean
                    h_std = out_std
                elif sample_type == "Variance":
                    h_sample = out_mean_offset
                    h_mean = tf.zeros_like(out_mean)
                    h_std = out_std
                elif sample_type == "MAP":
                    if n_samples > 1:
                        raise ValueError("MAP is not sampling.")
                    h_sample = out_mean
                    h_mean = out_mean
                    h_std = tf.zeros_like(out_std)
                else:
                    raise NotImplementedError("Invalid sampling type.")

                if activation == "relu":
                    h_sample = tf.nn.relu(h_sample)
                    h_mean = tf.nn.relu(h_mean)  # TODO: This is incorrect.
                elif activation == "sigmoid":
                    h_sample = tf.nn.sigmoid(h_sample)
                    h_mean = tf.nn.sigmoid(h_mean)  # TODO: This is incorrect.
                
                h_sum += h_sample
                h_list.append(h_sample)
            h_var = tf.pow(h_std, 2.0)
            return h_mean, h_list, h_var
    elif activation_type == "ral":
        def dense_clt_activation_function(out_mean,
                                          out_var,
                                          n_samples,
                                          sample_type,
                                          activation):
            # The forward pass through one layer
            h_list = []

            # If not activation, return out_mean, out_std
            if activation is None:
                out_mean = out_mean
                out_var = out_var
            elif activation == "relu":
                out_mean,\
                out_var = variational_activations.relu_moments(x=out_mean,
                                                               x_var=out_var)
            elif activation == "sigmoid":
                out_mean, \
                out_var = variational_activations.sigmoid_moments(x=out_mean,
                                                                  x_var=out_var)
            else:
                raise NotImplementedError

            h_mean = out_mean
            h_list.append(out_mean)
            h_var = out_var

            return h_mean, h_list, h_var

    else:
        raise NotImplementedError

    return dense_clt_activation_function


def get_noclt_activation_function(activation_type,
                                  is_conv):
    if activation_type == "sample":
        def dense_noclt_activation_function(input_tensor,
                                            var_par,
                                            w_dims,
                                            b_dims,
                                            use_bias,
                                            n_samples,
                                            sample_type,
                                            activation):
            # The forward pass through one layer
            h_sum = 0.0
            h_list = []

            raise NotImplementedError

            for s in range(n_samples):
                if sample_type == "Sample":
                    W_sample = var_par["W_mu"] + tf.multiply(var_par["W_sigma"],
                                                             get_random(tuple(w_dims), avg=0., std=1.))
                    if use_bias:
                        b_sample = var_par["b_mu"] + tf.multiply(var_par["b_sigma"],
                                                                 get_random(tuple(b_dims), avg=0., std=1.))
                elif sample_type == "Variance":
                    W_sample = tf.multiply(var_par["W_sigma"],
                                           get_random(tuple(var_par["w_dims"]), avg=0., std=1.))
                    if use_bias:
                        b_sample = tf.multiply(var_par["b_sigma"],
                                               get_random((b_dims,), avg=0., std=1.))
                elif sample_type == "MAP":
                    if n_samples > 1:
                        raise ValueError("MAP is not sampling.")
                    W_sample = var_par["W_mu"]
                    if use_bias:
                        b_sample = var_par["b_mu"]
                else:
                    raise NotImplementedError("Invalid sampling type.")
                var_par["W_sample_list"].append(W_sample)
                if use_bias:
                    var_par["b_sample_list"].append(b_sample)

            for s in range(n_samples):
                W_sample = var_par["W_sample_list"][s]
                h_sample = tf.matmul(input_tensor, W_sample)
                h_sample = tf.where(tf.is_nan(h_sample), tf.zeros_like(h_sample), h_sample)

                b_sample = var_par["b_sample_list"][s]
                b_sample = tf.where(tf.is_nan(b_sample), tf.zeros_like(b_sample), b_sample)

                h_sample = h_sample + b_sample

                if activation is None:
                    pass
                elif activation == "relu":
                    h_sample = tf.keras.layers.Activation(tf.keras.activations.relu)(h_sample)
                else:
                    raise NotImplementedError

                h_sum += h_sample
                h_list.append(h_sample)

            h_sum = tf.where(tf.is_nan(h_sum), tf.zeros_like(h_sum), h_sum)
            h_mean = h_sum / n_samples

            h_std = calc_h_std_from_samples(h_mean, h_list)
            return h_mean, h_list, h_std

    else:
        raise NotImplementedError

    return dense_noclt_activation_function


def calc_h_std_from_samples(h_mean, h_list):
    if len(h_list) > 1:
        h_std = tf.pow(h_mean - h_list[0], 2.0)

        for s in range(1, len(h_list)):
            h_std = h_std + tf.pow(h_mean - h_list[s], 2.0)

        h_std = h_std / (len(h_list) - 1.0)
        h_std = tf.sqrt(1e-8 + h_std)
    else:
        h_std = tf.zeros_like(h_mean)

    return h_std
