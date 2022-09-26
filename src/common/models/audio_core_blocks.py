import tensorflow as tf

from common.models.conv_blocks import ResBlock, ConvBlock, AttMaxPool2D


class ResNet38_PANN():
    def __init__(self,
                 spectrogram_length,
                 spectrogram_width,
                 config_dict,
                 use_se,
                 use_tc,
                 use_ds,
                 pool_type,
                 from_wavegram):
        self.spectrogram_length = spectrogram_length
        self.spectrogram_width = spectrogram_width
        self.use_se = use_se
        self.use_tc = use_tc
        self.use_ds = use_ds
        self.pool_type = pool_type

        if self.use_tc:
            self.max_pool_size = (2, 1)
            self.max_pool_strides = (2, 1)

            if not from_wavegram:
                self.input_shape = (self.spectrogram_length,
                                    1,
                                    self.spectrogram_width)
            else:
                self.input_shape = (self.spectrogram_length // 2,
                                    1,
                                    (self.spectrogram_width // 2) * 96)

            self.output_shape = (self.spectrogram_length // 2 // 2 // 2 // 2 // 2,
                                 1,
                                 1024)

        else:
            self.max_pool_size = (2, 2)
            self.max_pool_strides = (2, 2)

            if not from_wavegram:
                self.input_shape = (self.spectrogram_length,
                                    self.spectrogram_width,
                                    1)
            else:
                self.input_shape = (self.spectrogram_length // 2,
                                    self.spectrogram_width // 2,
                                    96)

            self.output_shape = (self.spectrogram_length // 2 // 2 // 2 // 2 // 2,
                                 1,
                                 (self.spectrogram_width // 2 // 2 // 2 // 2 // 2) * 1024)

        if self.pool_type in ["attentive", ]:
            self.pooling_class = AttMaxPool2D
        elif self.pool_type in ["max", ]:
            self.pooling_class = tf.keras.layers.MaxPool2D
        else:
            raise ValueError("Invalid max pool type.")

        self.layer_list = list()

        if not from_wavegram:
            self.layer_list.append(tf.keras.layers.Reshape(self.input_shape))

            self.layer_list.append(ConvBlock(filters=64,
                                             use_bias=False,
                                             max_pool_size=self.max_pool_size,
                                             max_pool_strides=self.max_pool_strides,
                                             num_layers=2,
                                             use_max_pool=True,
                                             use_se=False,
                                             use_ds=False,
                                             pool_type=self.pool_type,
                                             ratio=4))
        else:
            self.layer_list.append(tf.keras.layers.Reshape(self.input_shape))

        self.layer_list.append(ResBlock(64, use_se=self.use_se, use_ds=self.use_ds))
        self.layer_list.append(ResBlock(64, use_se=self.use_se, use_ds=self.use_ds))
        self.layer_list.append(self.pooling_class(pool_size=self.max_pool_size,
                                                  strides=self.max_pool_strides,
                                                  padding='valid'))

        self.layer_list.append(ResBlock(128, use_se=self.use_se, use_ds=self.use_ds))
        self.layer_list.append(ResBlock(128, use_se=self.use_se, use_ds=self.use_ds))
        self.layer_list.append(ResBlock(128, use_se=self.use_se, use_ds=self.use_ds))
        self.layer_list.append(self.pooling_class(pool_size=self.max_pool_size,
                                                  strides=self.max_pool_strides,
                                                  padding='valid'))

        self.layer_list.append(ResBlock(256, use_se=self.use_se, use_ds=self.use_ds))
        self.layer_list.append(ResBlock(256, use_se=self.use_se, use_ds=self.use_ds))
        self.layer_list.append(ResBlock(256, use_se=self.use_se, use_ds=self.use_ds))
        self.layer_list.append(ResBlock(256, use_se=self.use_se, use_ds=self.use_ds))
        self.layer_list.append(ResBlock(256, use_se=self.use_se, use_ds=self.use_ds))
        self.layer_list.append(self.pooling_class(pool_size=self.max_pool_size,
                                                  strides=self.max_pool_strides,
                                                  padding='valid'))

        self.layer_list.append(ResBlock(512, use_se=self.use_se, use_ds=self.use_ds))
        self.layer_list.append(ResBlock(512, use_se=self.use_se, use_ds=self.use_ds))
        self.layer_list.append(self.pooling_class(pool_size=self.max_pool_size,
                                                  strides=self.max_pool_strides,
                                                  padding='valid'))

        # self.layer_list.append(ConvBlock(filters=2048,
        self.layer_list.append(ConvBlock(filters=1024,
                                         use_bias=False,
                                         max_pool_size=self.max_pool_size,
                                         max_pool_strides=self.max_pool_strides,
                                         num_layers=2,
                                         use_max_pool=False,
                                         use_se=False,
                                         use_ds=False,
                                         pool_type=self.pool_type,
                                         ratio=4))

        self.layer_list.append(tf.keras.layers.Reshape(self.output_shape))

    def __call__(self, x, training):
        net = self.layer_list[0](x, training=training)
        for l in self.layer_list[1:]:
            net = l(net, training=training)
        return net


class FFBlock:
    def __init__(self,
                 outputs_list=(2, )):
        self.outputs_list = outputs_list

        self.layer_list = list()

        self.layer_list.append(tf.keras.layers.Reshape((20*409,)))
        self.layer_list.append(tf.keras.layers.Dense(4096, activation='relu'))
        self.layer_list.append(tf.keras.layers.Dense(4096, activation='relu'))
        self.layer_list.append(tf.keras.layers.Dense(4096, activation='relu'))

        self.dense_layer_list = list()
        for t, output_units in enumerate(self.outputs_list):
            self.dense_layer_list.append(tf.keras.layers.Dense(output_units))

    def __call__(self, x, training):
        net = self.layer_list[0](x, training=training)
        for l in self.layer_list[1:]:
            net = l(net, training=training)

        if len(self.outputs_list) == 1:
            prediction_single = self.dense_layer_list[0](net, training=training)
        else:
            prediction_single = list()
            for t in range(len(self.outputs_list)):
                prediction_single.append(self.dense_layer_list[t](net, training=training))

        return prediction_single


def get_ResNet38_PANN_block(net_train,
                            net_test,
                            config_dict):
    use_se = config_dict["use_se"]
    use_tc = config_dict["use_tc"]
    use_ds = config_dict["use_ds"]
    pool_type = config_dict["pool_type"]

    resnet38_pann = ResNet38_PANN(spectrogram_length=300,
                                  spectrogram_width=128,
                                  config_dict=config_dict,
                                  use_se=use_se,
                                  use_tc=use_tc,
                                  use_ds=use_ds,
                                  pool_type=pool_type,
                                  from_wavegram=False)

    net_train = resnet38_pann(net_train, training=True)
    net_test = resnet38_pann(net_test, training=False)

    return net_train, net_test
