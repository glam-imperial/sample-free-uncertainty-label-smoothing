import tensorflow as tf


class ConvBlock():
    def __init__(self, filters,
                 use_bias,
                 max_pool_size,
                 max_pool_strides,
                 num_layers,
                 use_max_pool,
                 use_se,
                 use_ds,
                 pool_type,
                 ratio=4):
        self.filters = filters
        self.use_bias = use_bias
        self.max_pool_size = max_pool_size
        self.max_pool_strides = max_pool_strides
        self.num_layers = num_layers
        self.use_max_pool = use_max_pool
        self.use_se = use_se
        self.use_ds = use_ds
        self.pool_type = pool_type
        self.ratio = ratio

        if self.use_ds:
            self.conv2d_function = DepthwiseSeparableConv2D
        else:
            self.conv2d_function = tf.keras.layers.Conv2D

        if self.pool_type in ["attentive", ]:
            self.pooling_class = AttMaxPool2D
        elif self.pool_type in ["max", ]:
            self.pooling_class = tf.keras.layers.MaxPool2D
        else:
            raise ValueError("Invalid max pool type.")

        self.layer_list = list()
        for n in range(self.num_layers):
            self.layer_list.append(
                self.conv2d_function(filters=self.filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                     dilation_rate=(1, 1), activation=None,
                                     use_bias=self.use_bias, kernel_initializer='glorot_uniform'))
            self.layer_list.append(tf.keras.layers.Activation(tf.keras.activations.relu))

        if self.use_se:
            self.layer_list.append(SEBlock(filters=self.filters,
                                           ratio=self.ratio))

        if self.use_max_pool:
            self.layer_list.append(self.pooling_class(pool_size=self.max_pool_size,
                                                      strides=self.max_pool_strides,
                                                      padding='valid'))

    def __call__(self, x, training):
        net = self.layer_list[0](x, training=training)
        for l in self.layer_list[1:]:
            net = l(net, training=training)
        return net


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

        self.temperature = self.add_weight(name=self.name + "_temp",
                                           shape=1,
                                           dtype=tf.float32,
                                           initializer="ones",
                                           regularizer=None,
                                           trainable=True,
                                           constraint=None,
                                           partitioner=None,
                                           use_resource=None)

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
        b, w, h, c = inputs.get_shape().as_list()
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

        return pooled_data


class ResBlock():
    def __init__(self, filters, use_se, use_ds, ratio=4, se_type="cse"):
        self.filters = filters
        self.use_se = use_se
        self.use_ds = use_ds
        self.ratio = ratio
        self.se_type = se_type

        if self.use_ds:
            self.conv2d_function = DepthwiseSeparableConv2D
        else:
            self.conv2d_function = tf.keras.layers.Conv2D

        self.conv2d_1 = self.conv2d_function(filters=self.filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                             dilation_rate=(1, 1), activation=None,
                                             use_bias=False, kernel_initializer='glorot_uniform')
        # self.bn_1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.relu_1 = tf.keras.layers.Activation(tf.keras.activations.relu)
        # self.dropout_1 = tf.keras.Dropout(rate=0.1)

        self.conv2d_2 = self.conv2d_function(filters=self.filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                             dilation_rate=(1, 1), activation=None,
                                             use_bias=False, kernel_initializer='glorot_uniform')
        # self.bn_2 = tf.keras.layers.BatchNormalization(axis=-1)
        self.relu_2 = tf.keras.layers.Activation(tf.keras.activations.relu)

        self.conv_res = self.conv2d_function(filters=self.filters, kernel_size=(1, 1), strides=(1, 1), padding='same',
                                             dilation_rate=(1, 1), activation=None,
                                             use_bias=False, kernel_initializer='glorot_uniform')
        # self.bn_res = tf.keras.layers.BatchNormalization(axis=-1)

        if self.use_se:
            self.se_block = SEBlock(filters=self.filters,
                                    ratio=self.ratio,
                                    type=self.se_type)
            # self.se_pooling = tf.keras.layers.GlobalAveragePooling2D()
            # self.se_fc_1 = tf.keras.layers.Dense(self.filters // self.ratio)
            # self.se_relu_1 = tf.keras.layers.Activation(tf.keras.activations.relu)
            # self.se_fc_2 = tf.keras.layers.Dense(self.filters)
            # self.se_sigmoid_1 = tf.keras.layers.Activation(tf.keras.activations.sigmoid)
            # self.se_reshape = tf.keras.layers.Reshape((1, 1, self.filters))

    def __call__(self, x, training):
        cnv2d_1 = self.conv2d_1(x, training=training)
        # bn_1 = self.bn_1(cnv2d_1, training=training)
        relu_1 = self.relu_1(cnv2d_1, training=training)
        # do_1 = self.dropout_1(relu_1, training=training)

        cnv2d_2 = self.conv2d_2(relu_1, training=training)
        # bn_2 = self.bn_1(cnv2d_2, training=training)

        res = self.conv_res(x, training=training)
        # res = self.bn_res(res, training=training)

        bn_2 = cnv2d_2 + res

        relu_2 = self.relu_1(bn_2, training=training)

        if self.use_se:
            relu_2 = self.se_block(relu_2, training=training)

            # relu_2 = relu_2 + res  # Acoustic Scene Classification withSqueeze-Excitation Residual Networks

        return relu_2


class SEBlock():
    "Concurrent Spatial and Channel ‘Squeeze & Excitation’ in Fully Convolutional Networks"
    def __init__(self, filters, ratio=4, type="cse", use_ti=False):
        self.filters = filters
        self.ratio = ratio
        self.type = type
        self.use_ti = use_ti

        if self.type not in ["cse", "sse", "scse"]:
            raise ValueError("Invalid SE Block type.")

        if self.type in ["cse", "scse"]:
            self.cse_pooling = tf.keras.layers.GlobalAveragePooling2D()
            self.cse_fc_1 = tf.keras.layers.Dense(self.filters // self.ratio)
            self.cse_relu_1 = tf.keras.layers.Activation(tf.keras.activations.relu)
            self.cse_fc_2 = tf.keras.layers.Dense(self.filters)
            self.cse_sigmoid_1 = tf.keras.layers.Activation(tf.keras.activations.sigmoid)
            self.cse_reshape = tf.keras.layers.Reshape((1, 1, self.filters))
        if self.type in ["sse", "scse"]:
            self.sse_conv = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), padding='same',
                                                   use_bias=False)
            self.sse_sigmoid_1 = tf.keras.layers.Activation(tf.keras.activations.sigmoid)

    def __call__(self, x, training):
        if self.type not in ["cse", "sse", "scse"]:
            raise ValueError("Invalid SE Block type.")

        if self.type in ["cse", "scse"]:
            if not self.use_ti:
                cse_squeeze = self.cse_pooling(x, training=training)
                cse_excitation = self.cse_fc_1(cse_squeeze, training=training)
                cse_excitation = self.cse_relu_1(cse_excitation, training=training)
                cse_excitation = self.cse_fc_2(cse_excitation, training=training)
                cse_excitation = self.cse_sigmoid_1(cse_excitation, training=training)
                cse_excitation = self.cse_reshape(cse_excitation, training=training)

                cse_out = x * cse_excitation
            else:
                length = x.shape[1]
                width = x.shape[2]
                channels = x.shape[3]

                cse_squeeze_all = tf.reduce_mean(x, axis=1, keep_dims=False)

                cse_squeeze_all = tf.reshape(cse_squeeze_all, [-1, channels])

                cse_excitation = self.cse_fc_1(cse_squeeze_all, training=training)
                cse_excitation = self.cse_relu_1(cse_excitation, training=training)
                cse_excitation = self.cse_fc_2(cse_excitation, training=training)
                cse_excitation = self.cse_sigmoid_1(cse_excitation, training=training)
                cse_excitation = tf.reshape(cse_excitation, [-1, 1, width, channels])

                cse_out = x * cse_excitation

        if self.type in ["sse", "scse"]:
            sse_squeeze = self.sse_conv(x, training=training)
            sse_excitation = self.sse_sigmoid_1(sse_squeeze, training=training)

            sse_out = x * sse_excitation

        if self.type == "cse":
            out = cse_out
        elif self.type == "sse":
            out = sse_out
        elif self.type == "scse":
            out = cse_out + sse_out
        else:
            raise ValueError("Invalid SE Block type.")

        return out
