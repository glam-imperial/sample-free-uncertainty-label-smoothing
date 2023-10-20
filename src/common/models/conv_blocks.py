import tensorflow as tf

from common.models.local_pooling import AttMaxPool2D

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 use_bias,
                 max_pool_size,
                 max_pool_strides,
                 num_layers,
                 use_max_pool,
                 use_se,
                 pool_type,
                 ratio=4,
                 regularisation_factor=None,
                 **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.filters = filters
        self.use_bias = use_bias
        self.max_pool_size = max_pool_size
        self.max_pool_strides = max_pool_strides
        self.num_layers = num_layers
        self.use_max_pool = use_max_pool
        self.use_se = use_se
        self.pool_type = pool_type
        self.ratio = ratio
        self.regularisation_factor = regularisation_factor

        self.conv2d_function = tf.keras.layers.Conv2D

        if self.pool_type in ["attentive", ]:
            self.pooling_class = AttMaxPool2D
        elif self.pool_type in ["max", ]:
            self.pooling_class = tf.keras.layers.MaxPool2D
        else:
            raise ValueError("Invalid max pool type.")

        self.layer_list = list()

    def build(self,
              input_shape):  # [batch_size, image_length, image_width, filters]
        if self.regularisation_factor is not None:
            l2 = tf.keras.regularizers.l2(l=self.regularisation_factor)
            print("regularise l2")
        else:
            l2 = None

        for n in range(self.num_layers):
            self.layer_list.append(
                self.conv2d_function(filters=self.filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                     dilation_rate=(1, 1), activation=None,
                                     use_bias=self.use_bias, kernel_initializer='glorot_uniform',
                                     kernel_regularizer=l2, bias_regularizer=l2))
            self.layer_list.append(tf.keras.layers.Activation(tf.keras.activations.relu))

        if self.use_se:
            self.layer_list.append(SEBlock(filters=self.filters,
                                           ratio=self.ratio,
                                           regularisation_factor=self.regularisation_factor))

        if self.use_max_pool:
            self.layer_list.append(self.pooling_class(pool_size=self.max_pool_size,
                                                      strides=self.max_pool_strides,
                                                      padding='valid'))

        super(ConvBlock, self).build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'use_bias': self.use_bias,
            'max_pool_size': self.max_pool_size,
            'max_pool_strides': self.max_pool_strides,
            'num_layers': self.num_layers,
            'use_max_pool': self.use_max_pool,
            'use_se': self.use_se,
            'pool_type': self.pool_type,
            'ratio': self.ratio,
            'regularisation_factor': self.regularisation_factor,
        })
        return config

    def call(self, x, training):
        net = self.layer_list[0](x, training=training)
        for l in self.layer_list[1:]:
            net = l(net, training=training)
        return net


class ResBlock(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 use_se,
                 ratio=4,
                 se_type="cse",
                 regularisation_factor=None,
                 **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        self.filters = filters
        self.use_se = use_se
        self.ratio = ratio
        self.se_type = se_type
        self.regularisation_factor = regularisation_factor

        self.conv2d_function = tf.keras.layers.Conv2D

    def build(self,
              input_shape):  # [batch_size, image_length, image_width, filters]
        if self.regularisation_factor is not None:
            l2 = tf.keras.regularizers.l2(l=self.regularisation_factor)
            print("regularise l2")
        else:
            l2 = None

        self.conv2d_1 = self.conv2d_function(filters=self.filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                             dilation_rate=(1, 1), activation=None,
                                             use_bias=False, kernel_initializer='glorot_uniform',
                                             kernel_regularizer=l2, bias_regularizer=l2)
        # Use of BN reduces performance on OSA-SMW by a lot.
        # self.bn_1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.relu_1 = tf.keras.layers.Activation(tf.keras.activations.relu)
        # self.dropout_1 = tf.keras.Dropout(rate=0.1)

        self.conv2d_2 = self.conv2d_function(filters=self.filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                             dilation_rate=(1, 1), activation=None,
                                             use_bias=False, kernel_initializer='glorot_uniform',
                                             kernel_regularizer=l2, bias_regularizer=l2)
        # Use of BN reduces performance on OSA-SMW by a lot.
        # self.bn_2 = tf.keras.layers.BatchNormalization(axis=-1)
        self.relu_2 = tf.keras.layers.Activation(tf.keras.activations.relu)

        self.conv_res = self.conv2d_function(filters=self.filters, kernel_size=(1, 1), strides=(1, 1), padding='same',
                                             dilation_rate=(1, 1), activation=None,
                                             use_bias=False, kernel_initializer='glorot_uniform',
                                             kernel_regularizer=l2, bias_regularizer=l2)
        # Use of BN reduces performance on OSA-SMW by a lot.
        # self.bn_res = tf.keras.layers.BatchNormalization(axis=-1)

        if self.use_se:
            self.se_block = SEBlock(filters=self.filters,
                                    ratio=self.ratio,
                                    type=self.se_type)

        super(ResBlock, self).build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'use_se': self.use_se,
            'ratio': self.ratio,
            'se_type': self.se_type,
        })
        return config

    def call(self, x, training):
        cnv2d_1 = self.conv2d_1(x, training=training)
        # bn_1 = self.bn_1(cnv2d_1, training=training)
        relu_1 = self.relu_1(cnv2d_1, training=training)
        # do_1 = self.dropout_1(relu_1, training=training)

        cnv2d_2 = self.conv2d_2(relu_1, training=training)
        # bn_2 = self.bn_2(cnv2d_2, training=training)

        res = self.conv_res(x, training=training)
        # res = self.bn_res(res, training=training)

        bn_2 = cnv2d_2 + res

        relu_2 = self.relu_1(bn_2, training=training)

        if self.use_se:
            relu_2 = self.se_block(relu_2, training=training)

            # relu_2 = relu_2 + res  # Acoustic Scene Classification withSqueeze-Excitation Residual Networks

        return relu_2


class SEBlock(tf.keras.layers.Layer):
    "Concurrent Spatial and Channel ‘Squeeze & Excitation’ in Fully Convolutional Networks"
    def __init__(self,
                 filters,
                 ratio=4,
                 type="cse",
                 regularisation_factor=None,
                 **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.filters = filters
        self.ratio = ratio
        self.type = type
        self.regularisation_factor = regularisation_factor

    def build(self,
              input_shape):  # [batch_size, image_length, image_width, filters]
        if self.type not in ["cse", "sse", "scse"]:
            raise ValueError("Invalid SE Block type.")

        if self.regularisation_factor is not None:
            l2 = tf.keras.regularizers.l2(l=self.regularisation_factor)
            print("regularise l2")
        else:
            l2 = None

        if self.type in ["cse", "scse"]:
            self.cse_pooling = tf.keras.layers.GlobalAveragePooling2D()
            self.cse_fc_1 = tf.keras.layers.Dense(self.filters // self.ratio,
                                                  kernel_regularizer=l2,
                                                  bias_regularizer=l2)
            self.cse_relu_1 = tf.keras.layers.Activation(tf.keras.activations.relu)
            self.cse_fc_2 = tf.keras.layers.Dense(self.filters,
                                                  kernel_regularizer=l2,
                                                  bias_regularizer=l2)
            self.cse_sigmoid_1 = tf.keras.layers.Activation(tf.keras.activations.sigmoid)
            self.cse_reshape = tf.keras.layers.Reshape((1, 1, self.filters))
        if self.type in ["sse", "scse"]:
            self.sse_conv = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), padding='same',
                                                   use_bias=False,
                                                   kernel_regularizer=l2, bias_regularizer=l2)
            self.sse_sigmoid_1 = tf.keras.layers.Activation(tf.keras.activations.sigmoid)

        super(SEBlock, self).build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'ratio': self.ratio,
            'type': self.type,
        })
        return config

    def call(self, x, training):
        if self.type not in ["cse", "sse", "scse"]:
            raise ValueError("Invalid SE Block type.")

        if self.type in ["cse", "scse"]:
            cse_squeeze = self.cse_pooling(x, training=training)
            cse_excitation = self.cse_fc_1(cse_squeeze, training=training)
            cse_excitation = self.cse_relu_1(cse_excitation, training=training)
            cse_excitation = self.cse_fc_2(cse_excitation, training=training)
            cse_excitation = self.cse_sigmoid_1(cse_excitation, training=training)
            cse_excitation = self.cse_reshape(cse_excitation, training=training)

            # cse_out = x * cse_excitation
            cse_out = tf.multiply(x, cse_excitation)

        if self.type in ["sse", "scse"]:
            sse_squeeze = self.sse_conv(x, training=training)
            sse_excitation = self.sse_sigmoid_1(sse_squeeze, training=training)

            # sse_out = x * sse_excitation
            sse_out = tf.multiply(x, sse_excitation)

        if self.type == "cse":
            out = cse_out
        elif self.type == "sse":
            out = sse_out
        elif self.type == "scse":
            out = cse_out + sse_out
        else:
            raise ValueError("Invalid SE Block type.")

        return out
