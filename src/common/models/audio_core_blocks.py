import tensorflow as tf

from common.models.conv_blocks import ResBlock, ConvBlock
from common.models.local_pooling import AttMaxPool2D


class ResNet28():
    def __init__(self,
                 spectrogram_length,
                 spectrogram_width,
                 config_dict,
                 use_se,
                 pool_type):
        self.spectrogram_length = spectrogram_length
        self.spectrogram_width = spectrogram_width
        self.use_se = use_se
        self.pool_type = pool_type

        self.max_pool_size = (2, 2)
        self.max_pool_strides = (2, 2)

        self.input_shape = (self.spectrogram_length,
                            self.spectrogram_width,
                            1)

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

        self.layer_list.append(tf.keras.layers.Reshape(self.input_shape))

        self.layer_list.append(ConvBlock(filters=64,
                                         use_bias=False,
                                         max_pool_size=self.max_pool_size,
                                         max_pool_strides=self.max_pool_strides,
                                         num_layers=2,
                                         use_max_pool=True,
                                         use_se=False,
                                         pool_type=self.pool_type,
                                         ratio=4))

        self.layer_list.append(ResBlock(64, use_se=self.use_se))
        self.layer_list.append(ResBlock(64, use_se=self.use_se))
        self.layer_list.append(self.pooling_class(pool_size=self.max_pool_size,
                                                  strides=self.max_pool_strides,
                                                  padding='valid'))

        self.layer_list.append(ResBlock(128, use_se=self.use_se))
        self.layer_list.append(ResBlock(128, use_se=self.use_se))
        # self.layer_list.append(ResBlock(128, use_se=self.use_se))
        self.layer_list.append(self.pooling_class(pool_size=self.max_pool_size,
                                                  strides=self.max_pool_strides,
                                                  padding='valid'))

        self.layer_list.append(ResBlock(256, use_se=self.use_se))
        self.layer_list.append(ResBlock(256, use_se=self.use_se))
        self.layer_list.append(ResBlock(256, use_se=self.use_se))
        # self.layer_list.append(ResBlock(256, use_se=self.use_se))
        # self.layer_list.append(ResBlock(256, use_se=self.use_se))
        self.layer_list.append(self.pooling_class(pool_size=self.max_pool_size,
                                                  strides=self.max_pool_strides,
                                                  padding='valid'))

        self.layer_list.append(ResBlock(512, use_se=self.use_se))
        self.layer_list.append(ResBlock(512, use_se=self.use_se))
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
                                         pool_type=self.pool_type,
                                         ratio=4))

        self.layer_list.append(tf.keras.layers.Reshape(self.output_shape))

    def __call__(self, x, training):
        net = self.layer_list[0](x, training=training)
        for l in self.layer_list[1:]:
            net = l(net, training=training)
        return net


def get_ResNet28_block(net_train,
                       net_test,
                       config_dict):
    use_se = config_dict["use_se"]
    pool_type = config_dict["pool_type"]

    resnet28 = ResNet28(spectrogram_length=300,
                        spectrogram_width=128,
                        config_dict=config_dict,
                        use_se=use_se,
                        pool_type=pool_type)

    net_train = resnet28(net_train, training=True)
    net_test = resnet28(net_test, training=False)

    return net_train, net_test
    # return net
