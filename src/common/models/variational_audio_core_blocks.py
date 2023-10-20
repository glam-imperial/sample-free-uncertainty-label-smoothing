import tensorflow as tf

from common.models.variational_conv_blocks import VariationalResBlock,\
    VariationalConvBlock
from common.models.local_pooling import VariationalMaxPool2D


class VariationalResNet28():
    def __init__(self,
                 spectrogram_length,
                 spectrogram_width,
                 config_dict,
                 use_se,
                 max_pool_curiosity_initial_value,
                 max_pool_curiosity_type):
        self.spectrogram_length = spectrogram_length
        self.spectrogram_width = spectrogram_width
        self.use_se = use_se
        self.max_pool_curiosity_initial_value = max_pool_curiosity_initial_value
        self.max_pool_curiosity_type = max_pool_curiosity_type

        self.max_pool_size = (2, 2)
        self.max_pool_strides = (2, 2)

        self.input_shape = (self.spectrogram_length,
                            self.spectrogram_width,
                            1)

        self.output_shape = (self.spectrogram_length // 2 // 2 // 2 // 2 // 2,
                             1,
                             (self.spectrogram_width // 2 // 2 // 2 // 2 // 2) * 1024)

        self.layer_list = list()

        self.input_reshape_layer = tf.keras.layers.Reshape(self.input_shape,
                                                           name="input_reshape")

        self.layer_list.append(VariationalConvBlock(filters=64,
                                                    use_bias=False,
                                                    max_pool_size=self.max_pool_size,
                                                    max_pool_strides=self.max_pool_strides,
                                                    num_layers=2,
                                                    use_max_pool=True,
                                                    max_pool_curiosity_initial_value=self.max_pool_curiosity_initial_value,
                                                    max_pool_curiosity_type=self.max_pool_curiosity_type,
                                                    use_se=False,
                                                    ratio=4,
                                                    name="conv_block_0"))

        self.layer_list.append(VariationalResBlock(64,
                                                   use_se=self.use_se,
                                                   name="res_block_0"))
        self.layer_list.append(VariationalResBlock(64,
                                                   use_se=self.use_se,
                                                   name="res_block_1"))
        self.layer_list.append(VariationalMaxPool2D(pool_size=self.max_pool_size,
                                                    strides=self.max_pool_strides,
                                                    padding='valid',
                                                    # curiosity_initial_value=self.max_pool_curiosity_initial_value,
                                                    curiosity_type=self.max_pool_curiosity_type,
                                                    name="max_pool_1"))

        self.layer_list.append(VariationalResBlock(128,
                                                   use_se=self.use_se,
                                                   name="res_block_2"))
        self.layer_list.append(VariationalResBlock(128,
                                                   use_se=self.use_se,
                                                   name="res_block_3"))
        # self.layer_list.append(VariationalResBlock(128,
        #                                            use_se=self.use_se,
        #                                            name="res_block_4"))
        self.layer_list.append(VariationalMaxPool2D(pool_size=self.max_pool_size,
                                                    strides=self.max_pool_strides,
                                                    padding='valid',
                                                    # curiosity_initial_value=self.max_pool_curiosity_initial_value,
                                                    curiosity_type=self.max_pool_curiosity_type,
                                                    name="max_pool_4"))

        self.layer_list.append(VariationalResBlock(256,
                                                   use_se=self.use_se,
                                                   name="res_block_5"))
        self.layer_list.append(VariationalResBlock(256,
                                                   use_se=self.use_se,
                                                   name="res_block_6"))
        self.layer_list.append(VariationalResBlock(256,
                                                   use_se=self.use_se,
                                                   name="res_block_7"))
        # self.layer_list.append(VariationalResBlock(256,
        #                                            use_se=self.use_se,
        #                                            name="res_block_8"))
        # self.layer_list.append(VariationalResBlock(256,
        #                                            use_se=self.use_se,
        #                                            name="res_block_9"))
        self.layer_list.append(VariationalMaxPool2D(pool_size=self.max_pool_size,
                                                    strides=self.max_pool_strides,
                                                    padding='valid',
                                                    # curiosity_initial_value=self.max_pool_curiosity_initial_value,
                                                    curiosity_type=self.max_pool_curiosity_type,
                                                    name="max_pool_9"))

        self.layer_list.append(VariationalResBlock(512,
                                                   use_se=self.use_se,
                                                   name="res_block_10"))
        self.layer_list.append(VariationalResBlock(512,
                                                   use_se=self.use_se,
                                                   name="res_block_11"))
        self.layer_list.append(VariationalMaxPool2D(pool_size=self.max_pool_size,
                                                    strides=self.max_pool_strides,
                                                    padding='valid',
                                                    # curiosity_initial_value=self.max_pool_curiosity_initial_value,
                                                    curiosity_type=self.max_pool_curiosity_type,
                                                    name="max_pool_11"))

        # self.layer_list.append(VariationalConvBlock(filters=2048,
        self.layer_list.append(VariationalConvBlock(filters=1024,
                                                    use_bias=False,
                                                    max_pool_size=self.max_pool_size,
                                                    max_pool_strides=self.max_pool_strides,
                                                    num_layers=2,
                                                    use_max_pool=False,
                                                    use_se=False,
                                                    max_pool_curiosity_initial_value=self.max_pool_curiosity_initial_value,
                                                    max_pool_curiosity_type=self.max_pool_curiosity_type,
                                                    ratio=4,
                                                    name="conv_block_last"))

        self.output_reshape_layer = tf.keras.layers.Reshape(self.output_shape)

    def __call__(self, x, training, inputs_variances):
        x = self.input_reshape_layer(inputs=x,
                                     training=training)
        inputs_variances = self.input_reshape_layer(inputs=inputs_variances,
                                                    training=training)

        h_mean, \
        h_var = self.layer_list[0](inputs=x,
                                   training=training,
                                   inputs_variances=inputs_variances)
        for l in self.layer_list[1:]:
            h_mean, \
            h_var = l(inputs=h_mean,
                      training=training,
                      inputs_variances=h_var)

        h_mean = self.output_reshape_layer(inputs=h_mean,
                                           training=training)
        h_var = self.output_reshape_layer(inputs=h_var,
                                          training=training)
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


def get_VariationalResNet28_block(net_train,
                                  net_test,
                                  config_dict):
    use_se = config_dict["use_se"]
    max_pool_curiosity_initial_value = config_dict["max_pool_curiosity_initial_value"]
    max_pool_curiosity_type = config_dict["max_pool_curiosity_type"]

    resnet28 = VariationalResNet28(spectrogram_length=300,
                                   spectrogram_width=128,
                                   config_dict=config_dict,
                                   use_se=use_se,
                                   max_pool_curiosity_initial_value=max_pool_curiosity_initial_value,
                                   max_pool_curiosity_type=max_pool_curiosity_type)

    net_train,\
    var_train = resnet28(net_train,
                         training=True,
                         inputs_variances=tf.zeros_like(net_train))
    net_test,\
    var_test = resnet28(net_test,
                        training=False,
                        inputs_variances=tf.zeros_like(net_test))

    kl_loss = resnet28.get_kl_loss()

    return net_train, net_test, var_train, var_test, kl_loss
    # return net, var_test, kl_loss
