import tensorflow as tf

from common.models.variational_conv_blocks import VariationalResBlock,\
    VariationalConvBlock, VariationalMaxPool2D


class VariationalResNet38_PANN():
    def __init__(self,
                 spectrogram_length,
                 spectrogram_width,
                 max_pool_curiosity_initial_value,
                 max_pool_curiosity_type,
                 se_uncertainty_handling,
                 config_dict,
                 use_se,
                 use_tc,
                 use_ds,
                 from_wavegram):
        self.spectrogram_length = spectrogram_length
        self.spectrogram_width = spectrogram_width
        self.max_pool_curiosity_initial_value = max_pool_curiosity_initial_value
        self.max_pool_curiosity_type = max_pool_curiosity_type
        self.se_uncertainty_handling = se_uncertainty_handling
        self.use_se = use_se
        self.use_tc = use_tc
        self.use_ds = use_ds

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

        self.layer_list = list()

        if not from_wavegram:
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
                                                        se_uncertainty_handling=self.se_uncertainty_handling,
                                                        use_se=False,
                                                        use_ds=False,
                                                        ratio=4,
                                                        name="conv_block_0"))
        else:
            self.input_reshape_layer = tf.keras.layers.Reshape(self.input_shape)

        self.layer_list.append(VariationalResBlock(64,
                                                   use_se=self.use_se,
                                                   use_ds=self.use_ds,
                                                   se_uncertainty_handling=self.se_uncertainty_handling,
                                                   name="res_block_0"))
        self.layer_list.append(VariationalResBlock(64,
                                                   use_se=self.use_se,
                                                   use_ds=self.use_ds,
                                                   se_uncertainty_handling=self.se_uncertainty_handling,
                                                   name="res_block_1"))
        self.layer_list.append(VariationalMaxPool2D(pool_size=self.max_pool_size,
                                                    strides=self.max_pool_strides,
                                                    padding='valid',
                                                    curiosity_initial_value=self.max_pool_curiosity_initial_value,
                                                    curiosity_type=self.max_pool_curiosity_type,
                                                    name="max_pool_1"))

        self.layer_list.append(VariationalResBlock(128,
                                                   use_se=self.use_se,
                                                   use_ds=self.use_ds,
                                                   se_uncertainty_handling=self.se_uncertainty_handling,
                                                   name="res_block_2"))
        self.layer_list.append(VariationalResBlock(128,
                                                   use_se=self.use_se,
                                                   use_ds=self.use_ds,
                                                   se_uncertainty_handling=self.se_uncertainty_handling,
                                                   name="res_block_3"))
        self.layer_list.append(VariationalResBlock(128,
                                                   use_se=self.use_se,
                                                   use_ds=self.use_ds,
                                                   se_uncertainty_handling=self.se_uncertainty_handling,
                                                   name="res_block_4"))
        self.layer_list.append(VariationalMaxPool2D(pool_size=self.max_pool_size,
                                                    strides=self.max_pool_strides,
                                                    padding='valid',
                                                    curiosity_initial_value=self.max_pool_curiosity_initial_value,
                                                    curiosity_type=self.max_pool_curiosity_type,
                                                    name="max_pool_4"))

        self.layer_list.append(VariationalResBlock(256,
                                                   use_se=self.use_se,
                                                   use_ds=self.use_ds,
                                                   se_uncertainty_handling=self.se_uncertainty_handling,
                                                   name="res_block_5"))
        self.layer_list.append(VariationalResBlock(256,
                                                   use_se=self.use_se,
                                                   use_ds=self.use_ds,
                                                   se_uncertainty_handling=self.se_uncertainty_handling,
                                                   name="res_block_6"))
        self.layer_list.append(VariationalResBlock(256,
                                                   use_se=self.use_se,
                                                   use_ds=self.use_ds,
                                                   se_uncertainty_handling=self.se_uncertainty_handling,
                                                   name="res_block_7"))
        self.layer_list.append(VariationalResBlock(256,
                                                   use_se=self.use_se,
                                                   use_ds=self.use_ds,
                                                   se_uncertainty_handling=self.se_uncertainty_handling,
                                                   name="res_block_8"))
        self.layer_list.append(VariationalResBlock(256,
                                                   use_se=self.use_se,
                                                   use_ds=self.use_ds,
                                                   se_uncertainty_handling=self.se_uncertainty_handling,
                                                   name="res_block_9"))
        self.layer_list.append(VariationalMaxPool2D(pool_size=self.max_pool_size,
                                                    strides=self.max_pool_strides,
                                                    padding='valid',
                                                    curiosity_initial_value=self.max_pool_curiosity_initial_value,
                                                    curiosity_type=self.max_pool_curiosity_type,
                                                    name="max_pool_9"))

        self.layer_list.append(VariationalResBlock(512,
                                                   use_se=self.use_se,
                                                   use_ds=self.use_ds,
                                                   se_uncertainty_handling=self.se_uncertainty_handling,
                                                   name="res_block_10"))
        self.layer_list.append(VariationalResBlock(512,
                                                   use_se=self.use_se,
                                                   use_ds=self.use_ds,
                                                   se_uncertainty_handling=self.se_uncertainty_handling,
                                                   name="res_block_11"))
        self.layer_list.append(VariationalMaxPool2D(pool_size=self.max_pool_size,
                                                    strides=self.max_pool_strides,
                                                    padding='valid',
                                                    curiosity_initial_value=self.max_pool_curiosity_initial_value,
                                                    curiosity_type=self.max_pool_curiosity_type,
                                                    name="max_pool_11"))

        # self.layer_list.append(VariationalConvBlock(filters=2048,
        self.layer_list.append(VariationalConvBlock(filters=1024,
                                                    use_bias=False,
                                                    max_pool_size=self.max_pool_size,
                                                    max_pool_strides=self.max_pool_strides,
                                                    num_layers=2,
                                                    use_max_pool=False,
                                                    max_pool_curiosity_initial_value=self.max_pool_curiosity_initial_value,
                                                    max_pool_curiosity_type=self.max_pool_curiosity_type,
                                                    se_uncertainty_handling=self.se_uncertainty_handling,
                                                    use_se=False,
                                                    use_ds=False,
                                                    ratio=4,
                                                    name="conv_block_last"))

        self.output_reshape_layer = tf.keras.layers.Reshape(self.output_shape)

    def __call__(self, x, training, inputs_variances):
        x = self.input_reshape_layer(inputs=x,
                                     training=training)
        inputs_variances = self.input_reshape_layer(inputs=inputs_variances,
                                                    training=training)

        h_mean, \
        h_list, \
        h_var = self.layer_list[0](inputs=x,
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

        h_mean = self.output_reshape_layer(inputs=h_mean,
                                           training=training)
        h_list = [self.output_reshape_layer(inputs=h,
                                            training=training) for h in h_list]
        h_var = self.output_reshape_layer(inputs=h_var,
                                          training=training)
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


def get_VariationalResNet38_PANN_block(net_train,
                                       net_test,
                                       config_dict):
    max_pool_curiosity_initial_value = config_dict["max_pool_curiosity_initial_value"]
    max_pool_curiosity_type = config_dict["max_pool_curiosity_type"]
    uncertainty_handling = config_dict["uncertainty_handling"]
    use_se = config_dict["use_se"]
    use_tc = config_dict["use_tc"]
    use_ds = config_dict["use_ds"]

    resnet38_pann = VariationalResNet38_PANN(spectrogram_length=300,
                                             spectrogram_width=128,
                                             max_pool_curiosity_initial_value=max_pool_curiosity_initial_value,
                                             max_pool_curiosity_type=max_pool_curiosity_type,
                                             se_uncertainty_handling=uncertainty_handling,
                                             config_dict=config_dict,
                                             use_se=use_se,
                                             use_tc=use_tc,
                                             use_ds=use_ds,
                                             from_wavegram=False)

    net_train,\
    list_train,\
    var_train = resnet38_pann(net_train,
                              training=True,
                              inputs_variances=tf.zeros_like(net_train))
    net_test,\
    list_test,\
    var_test = resnet38_pann(net_test,
                             training=False,
                             inputs_variances=tf.zeros_like(net_test))

    kl_loss = resnet38_pann.get_kl_loss()

    return net_train, net_test, var_train, var_test, kl_loss
