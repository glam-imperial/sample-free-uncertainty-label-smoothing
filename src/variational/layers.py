import tensorflow as tf

import variational.activations as variational_activations


class VariationalLayer(tf.keras.layers.Layer):
    def __init__(self,
                 # weight_prior,
                 variance_parameterisation_type,
                 activation=None,
                 use_bias=True,
                 weight_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 reuse=None,
                 **kwargs):
        super(VariationalLayer, self).__init__(**kwargs)
        # self.weight_prior = weight_prior
        self.variance_parameterisation_type = variance_parameterisation_type
        self.activation = activation
        self.use_bias = use_bias
        self.weight_initializer = tf.keras.initializers.get(weight_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.reuse = reuse

        #######################################################################################

        if variance_parameterisation_type not in ["additive",
                                                  "weight_wise",
                                                  "neuron_wise",
                                                  "layer_wise"]:
            raise NotImplementedError("Invalid variance parameterisation type.")

        self.rho_init = tf.random_uniform_initializer(-4.6, -3.9)

        # self.kl_loss = None

        self.clt_activation_function = get_clt_activation_function()

    def get_config(self):
        config = super().get_config()
        config.update({
            # 'weight_prior': self.weight_prior,
            'variance_parameterisation_type': self.variance_parameterisation_type,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'weight_initializer': self.weight_initializer,
            'bias_initializer': self.bias_initializer,
            'reuse': self.reuse,
        })

        return config

    def call(self,  # Georgios The args were remade to somewhat resemble keras layers.
             inputs,
             *args,
             **kwargs):

        training = kwargs["training"]
        inputs_variances = kwargs["inputs_variances"]

        # self.kl_loss = None

        input_tensor = inputs

        if self.variance_parameterisation_type == "layer_wise":
            out_mean, out_var = self.layer_wise_variance(input_tensor,
                                                         inputs_variances)
        elif self.variance_parameterisation_type == "neuron_wise":
            if self.is_conv:
                raise NotImplementedError
            out_mean, out_var = self.layer_wise_variance(input_tensor,
                                                         inputs_variances)
        elif self.variance_parameterisation_type == "weight_wise":
            if self.is_conv:
                raise NotImplementedError
            out_mean, out_var = self.layer_wise_variance(input_tensor,
                                                         inputs_variances)
        elif self.variance_parameterisation_type == "additive":
            raise NotImplementedError
            out_mean, out_var = self.additive_variance(input_tensor,
                                                       inputs_variances)
        else:
            raise ValueError("Invalid variance parameterisation type.")

        out_mean = tf.where(tf.is_nan(out_mean), tf.zeros_like(out_mean), out_mean)
        out_var = tf.where(tf.is_nan(out_var), tf.zeros_like(out_var), out_var)

        h_mean, h_var = self.clt_activation_function(out_mean=out_mean,
                                                     out_var=out_var,
                                                     activation=self.activation)

        if training:
            self.calculate_kl_loss()

        return h_mean, h_var

    def layer_wise_variance(self, x, x_var):
        if self.is_conv:
            out_mean, out_var = self.conv2d_operation_with_uncertainty(x, x_var)
        else:
            out_mean, out_var = self.dense_operation_with_uncertainty(x, x_var)

        if self.use_bias:
            out_mean = out_mean + self.b_mu
            out_var = out_var + tf.multiply(tf.exp(self.b_log_alpha),
                                tf.pow(self.b_mu, 2.0))
        else:
            b_mu = None
            b_var = None
        return out_mean, out_var

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
        return out_mean, out_var

    def conv2d_operation_with_uncertainty(self, x, x_var):
        if self.padding == "same":
            padding = "SAME"
        elif self.padding == "valid":
            padding = "VALID"
        else:
            padding = self.padding

        out_mean = tf.nn.conv2d(input=x,
                                filter=self.W_mu,
                                strides=self.strides,
                                padding=padding,
                                data_format=self.data_format,
                                dilations=self.dilation_rate)

        if x_var is None:
            alpha = tf.exp(self.W_log_alpha)
            out_var = tf.nn.conv2d(input=tf.multiply(alpha, tf.pow(x, 2.0)),
                                   filter=tf.pow(self.W_mu, 2.0),
                                   strides=self.strides,
                                   padding=padding,
                                   data_format=self.data_format,
                                   dilations=self.dilation_rate)
        else:
            alpha = tf.exp(self.W_log_alpha)
            out_var = tf.nn.conv2d(input=tf.multiply((1.0 + alpha), x_var) + tf.multiply(alpha, tf.pow(x, 2.0)),
                                   filter=tf.pow(self.W_mu, 2.0),
                                   strides=self.strides,
                                   padding=padding,
                                   data_format=self.data_format,
                                   dilations=self.dilation_rate)
        return out_mean, out_var

    def dense_operation_with_uncertainty(self, x, x_var):
        out_mean = tf.matmul(x, self.W_mu)

        if x_var is None:
            alpha = tf.exp(self.W_log_alpha)
            out_var = tf.matmul(tf.multiply(alpha, tf.pow(x, 2.0)),
                                tf.pow(self.W_mu, 2.0))
        else:
            alpha = tf.exp(self.W_log_alpha)
            out_var = tf.matmul(tf.multiply((1.0 + alpha), x_var) + tf.multiply(alpha, tf.pow(x, 2.0)),
                                tf.pow(self.W_mu, 2.0))
        return out_mean, out_var

    def calculate_kl_loss(self):
        # This is known in closed form.
        W_kl = tf.reduce_sum(tf.log(1.0 + (1.0 / (1e-8 + tf.exp(self.W_log_alpha)))) * tf.ones_like(self.W_mu))
        kl = W_kl
        if self.use_bias:
            b_kl = tf.reduce_sum(tf.log(1.0 + (1.0 / (1e-8 + tf.exp(self.b_log_alpha)))) * tf.ones_like(self.b_mu))
            kl = kl + b_kl

        self.kl = kl

    def get_kl_loss(self):
        return self.kl


class DenseReparameterisation(VariationalLayer):
    def __init__(self, units,
                 **kwargs):
        self.is_conv = False
        super(DenseReparameterisation, self).__init__(**kwargs)
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
                                               shape=(1,),
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
                                                   shape=(1,),
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

        self.calculate_kl_loss()
        super().build(input_shape)


class Conv2dReparameterization(VariationalLayer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides,  # (1, 1),
                 padding,  # 'valid',
                 data_format,  # =None,
                 dilation_rate,  # =(1, 1),
                 **kwargs):
        self.is_conv = True
        super(Conv2dReparameterization, self).__init__(**kwargs)
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

        self._in_channels = input_shape[-1]

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
                                               shape=(1,),
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
                                                   shape=(1,),
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

        self.calculate_kl_loss()
        super().build(input_shape)


def get_clt_activation_function():
    def dense_clt_activation_function(out_mean,
                                      out_var,
                                      activation):
        # If not activation, return out_mean, out_std
        if activation is None:
            out_mean = out_mean
            out_var = out_var
        elif activation == "relu":
            out_mean, \
                out_var = variational_activations.relu_moments(x=out_mean,
                                                               x_var=out_var)
        elif activation == "simplified_activation":
            out_mean, \
                out_var = variational_activations.simplified_activation_moments(x=out_mean,
                                                                                x_var=out_var)
        elif activation == "sigmoid":
            out_mean, \
                out_var = variational_activations.sigmoid_moments(x=out_mean,
                                                                  x_var=out_var)
        else:
            raise NotImplementedError

        h_mean = out_mean
        h_var = out_var

        return h_mean, h_var

    return dense_clt_activation_function


def softplus(x):
    return tf.log(1. + tf.exp(x))
