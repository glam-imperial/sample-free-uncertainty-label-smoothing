import tensorflow as tf


class WideResNet():
    def __init__(self,
                 spectrogram_length,
                 spectrogram_width,
                 outputs_list):
        self.spectrogram_length = spectrogram_length
        self.spectrogram_width = spectrogram_width
        self.outputs_list = outputs_list

        self.max_pool_size = (2, 1)
        self.max_pool_strides = (2, 1)

        self.input_shape = (self.spectrogram_length,
                            self.spectrogram_width,
                            1)

        # self.output_shape = (self.spectrogram_length // 2 // 2 // 2 // 2 // 2,
        #                      1,
        #                      (self.spectrogram_width // 2 // 2 // 2 // 2 // 2) * 1024)

        self.pooling_class = tf.keras.layers.MaxPool2D

        self.layer_list = list()

        self.layer_list.append(tf.keras.layers.Reshape(self.input_shape))

        self.layer_list.append(ConvBNReLUBlock(filters=32,
                                               kernel_size=(5, 5),
                                               strides=(1, 1),
                                               activation="relu",
                                               dropout_rate=0.0,
                                               padding="same"))
        self.layer_list.append(self.pooling_class(pool_size=self.max_pool_size,
                                                  strides=self.max_pool_strides,
                                                  padding='same'))

        self.layer_list.append(DownsamplingBlock(64))
        self.layer_list.append(ResBlock(64))
        self.layer_list.append(ResBlock(64))

        self.layer_list.append(DownsamplingBlock(128))
        self.layer_list.append(ResBlock(128))
        self.layer_list.append(ResBlock(128))

        self.layer_list.append(DownsamplingBlock(256))
        self.layer_list.append(ResBlock(256))
        self.layer_list.append(ResBlock(256))

        self.layer_list.append(DownsamplingBlock(512))
        self.layer_list.append(ResBlock(512))
        self.layer_list.append(ResBlock(512))

        self.layer_list.append(ConvBNReLUBlock(512,
                                               kernel_size=(7, 8),
                                               strides=(7, 8),
                                               activation="relu",
                                               dropout_rate=0.1,
                                               padding="same"))
        self.layer_list.append(ConvBNReLUBlock(1024,
                                               kernel_size=(1, 1),
                                               strides=(1, 1),
                                               activation="relu",
                                               dropout_rate=0.1,
                                               padding="same"))

        self.output_layer_list = list()
        for t, output_units in enumerate(self.outputs_list):
            self.output_layer_list.append(ConvBNReLUBlock(output_units,
                                               kernel_size=(1, 1),
                                                          strides=(1, 1),
                                               activation=None,
                                               dropout_rate=0.1,
                                               padding="same"))

    def __call__(self, x, training):
        net = self.layer_list[0](x, training=training)
        for l in self.layer_list[1:]:
            net = l(net, training=training)

        prediction_single = list()
        for t, output_units in enumerate(self.outputs_list):
            prediction_single.append(tf.reshape(tf.math.log(tf.reduce_mean(tf.exp(self.output_layer_list[t](net, training=training)), axis=1, keepdims=True)), [-1, output_units]))

        # net = tf.exp(net)
        # net = tf.reduce_mean(net, axis=1, keepdims=True)
        # net = tf.math.log(net)

        return prediction_single


class ConvBNReLUBlock(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides,
                 activation,
                 dropout_rate=0.0,
                 padding="same",
                 **kwargs):
        super(ConvBNReLUBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.padding = padding

        self.conv2d_function = tf.keras.layers.Conv2D

    def build(self,
              input_shape):  # [batch_size, image_length, image_width, filters]

        self.conv2d = self.conv2d_function(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding,
                                           dilation_rate=(1, 1), activation=None,
                                           use_bias=False, kernel_initializer='glorot_uniform')
        # Use of BN reduces performance on OSA-SMW by a lot.
        # self.bn = tf.keras.layers.BatchNormalization(axis=-1)
        self.relu = tf.keras.layers.Activation(self.activation)
        self.dropout = tf.keras.layers.Dropout(rate=self.dropout_rate)

        super(ConvBNReLUBlock, self).build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'activation': self.activation,
            'dropout_rate': self.dropout_rate,
            'padding': self.padding,
        })
        return config

    def call(self, inputs, **kwargs):
        training = kwargs["training"]
        net = self.conv2d(inputs, training=training)
        # net = self.bn(net, training=training)
        net = self.relu(net, training=training)
        net = self.dropout(net, training=training)
        return net


class DownsamplingBlock(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 **kwargs):
        super(DownsamplingBlock, self).__init__(**kwargs)
        self.filters = filters

        self.conv2d_function = tf.keras.layers.Conv2D

    def build(self,
              input_shape):  # [batch_size, image_length, image_width, filters]

        self.conv_1 = self.conv2d_function(filters=self.filters, kernel_size=(1, 1), strides=(1, 1), padding='same',
                                           dilation_rate=(1, 1), activation=None,
                                           use_bias=False, kernel_initializer='glorot_uniform')
        # Use of BN reduces performance on OSA-SMW by a lot.
        # self.bn_1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.relu_1 = tf.keras.layers.Activation(tf.keras.activations.relu)

        self.conv_2 = self.conv2d_function(filters=self.filters, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                           dilation_rate=(1, 1), activation=None,
                                           use_bias=False, kernel_initializer='glorot_uniform')
        # Use of BN reduces performance on OSA-SMW by a lot.
        # self.bn_2 = tf.keras.layers.BatchNormalization(axis=-1)
        self.relu_2 = tf.keras.layers.Activation(tf.keras.activations.relu)

        self.conv_3 = self.conv2d_function(filters=self.filters, kernel_size=(1, 1), strides=(1, 1), padding='same',
                                           dilation_rate=(1, 1), activation=None,
                                           use_bias=False, kernel_initializer='glorot_uniform')
        # Use of BN reduces performance on OSA-SMW by a lot.
        # self.bn_3 = tf.keras.layers.BatchNormalization(axis=-1)

        self.avg_pool = tf.keras.layers.AvgPool2D(pool_size=[2, 2],
                                                  strides=[2, 2],
                                                  padding="same")
        self.conv_res = self.conv2d_function(filters=self.filters, kernel_size=(1, 1), strides=(1, 1), padding='same',
                                             dilation_rate=(1, 1), activation=None,
                                             use_bias=False, kernel_initializer='glorot_uniform')
        # Use of BN reduces performance on OSA-SMW by a lot.
        # self.bn_res = tf.keras.layers.BatchNormalization(axis=-1)

        self.relu_final = tf.keras.layers.Activation(tf.keras.activations.relu)

        super(DownsamplingBlock, self).build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
        })
        return config

    def call(self, inputs, **kwargs):
        training = kwargs["training"]
        net = self.conv_1(inputs, training=training)
        # net = self.bn_1(net, training=training)
        net = self.relu_1(net, training=training)

        net = self.conv_2(net, training=training)
        # net = self.bn_2(net, training=training)
        net = self.relu_2(net, training=training)

        net = self.conv_3(net, training=training)
        # net = self.bn_3(net, training=training)

        res = self.avg_pool(inputs, training=training)
        res = self.conv_res(res, training=training)
        # res = self.bn_res(res, training=training)

        net = net + res

        net = self.relu_final(net)

        return net


class ResBlock(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        self.filters = filters

        self.conv2d_function = tf.keras.layers.Conv2D

    def build(self,
              input_shape):  # [batch_size, image_length, image_width, filters]

        self.conv_1 = self.conv2d_function(filters=self.filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                           dilation_rate=(1, 1), activation=None,
                                           use_bias=False, kernel_initializer='glorot_uniform')
        # Use of BN reduces performance on OSA-SMW by a lot.
        # self.bn_1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.relu_1 = tf.keras.layers.Activation(tf.keras.activations.relu)

        self.conv_2 = self.conv2d_function(filters=self.filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                           dilation_rate=(1, 1), activation=None,
                                           use_bias=False, kernel_initializer='glorot_uniform')
        # Use of BN reduces performance on OSA-SMW by a lot.
        # self.bn_2 = tf.keras.layers.BatchNormalization(axis=-1)
        self.relu_2 = tf.keras.layers.Activation(tf.keras.activations.relu)

        super(ResBlock, self).build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
        })
        return config

    def call(self, inputs, **kwargs):
        training = kwargs["training"]

        net = self.conv_1(inputs, training=training)
        # net = self.bn_1(net, training=training)
        net = self.relu_1(net, training=training)

        net = self.conv_2(net, training=training)
        # net = self.bn_2(net, training=training)

        net = net + inputs

        net = self.relu_2(net, training=training)

        return net


def get_WideResNet_block(net_train,
                         net_test,
                         y_pred_names,
                         global_pooling_configuration):
    outputs_list = global_pooling_configuration["outputs_list"]

    wide_resnet = WideResNet(spectrogram_length=300,
                             spectrogram_width=128,
                             outputs_list=outputs_list)

    prediction_single_train = wide_resnet(x=net_train,
                                          training=True)
    prediction_single_test = wide_resnet(x=net_test,
                                         training=False)

    prediction_train = dict()
    prediction_test = dict()
    for i, y_pred_name in enumerate(y_pred_names):
        prediction_train[y_pred_name] = prediction_single_train[i]
        prediction_train[y_pred_name + "_prob"] = tf.nn.sigmoid(prediction_single_train[i])

        prediction_test[y_pred_name] = prediction_single_test[i]
        prediction_test[y_pred_name + "_prob"] = tf.nn.sigmoid(prediction_single_test[i])

    return prediction_train, prediction_test