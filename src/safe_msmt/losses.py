import tensorflow as tf
import tensorflow.keras.backend as K


def get_loss(pred_train,
             model_configuration,
             y_pred_names,
             other_outputs,
             pos_weights):
    global_pooling = model_configuration["global_pooling"]
    if "loss_type" in model_configuration.keys():
        loss_type = model_configuration["loss_type"]
    else:
        loss_type = None
    if "bayesian" in model_configuration.keys():
        kl_loss_regulariser = model_configuration["bayesian"]["kl_loss_regulariser"]
        if "use_logit_vars" in model_configuration["bayesian"].keys():
            use_logit_vars = model_configuration["bayesian"]["use_logit_vars"]
        else:
            use_logit_vars = False

        if "use_epistemic_smoothing" in model_configuration["bayesian"].keys():
            use_epistemic_smoothing = model_configuration["bayesian"]["use_epistemic_smoothing"]
        else:
            use_epistemic_smoothing = False
    else:
        kl_loss_regulariser = 0.0
        use_logit_vars = False
        use_epistemic_smoothing = False

    target_weight = float(len(y_pred_names))

    def _to_tensor(x, dtype):
        return tf.convert_to_tensor(x, dtype=dtype)

    def _calculate_weighted_multilabel_crossentropy(target,
                                                    output,
                                                    output_var=None,
                                                    from_logits=False):
        if not from_logits:
            # transform back to logits
            _epsilon = _to_tensor(K.epsilon(), output.dtype.base_dtype)
            output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
            output = tf.log(output / (1 - output))
        if output_var is not None:
            output_std = tf.sqrt(output_var)
            output_std = tf.nn.relu(output_std)
            mc_samples = 5
            mc_samples_shape = tf.concat([tf.constant((mc_samples,),
                                                      dtype=tf.dtypes.int32),
                                          tf.shape(output_std)],
                                         axis=0)
            standard_normal_samples = tf.random.normal(mc_samples_shape,
                                                       mean=0.0,
                                                       stddev=1.0,
                                                       dtype=tf.dtypes.float32,
                                                       seed=None,
                                                       name=None)  # [mc_samples, batch_size, n_classes]
            standard_normal_samples = tf.multiply(standard_normal_samples,
                                                  tf.expand_dims(output_std, axis=0))  # [mc_samples, batch_size, n_classes]
            standard_normal_samples = tf.expand_dims(output, axis=0) + standard_normal_samples  # [mc_samples, batch_size, n_classes]
            if use_epistemic_smoothing:
                print("Using epistemic smoothing.")
                output_probabilities_map = tf.nn.sigmoid(output)

                output_probabilities_mc = tf.nn.sigmoid(standard_normal_samples)
                output_probabilities_mc_var = tf.math.reduce_variance(output_probabilities_mc, axis=0)
                output_probabilities_mc = tf.reduce_mean(output_probabilities_mc, axis=0)

                # Manhattan.
                manhattan = tf.abs(output_probabilities_map - output_probabilities_mc)
                # smoothing_probability = tf.reduce_mean(manhattan)
                smoothing_probability = manhattan

                # Tanh variance.
                # smoothing_probability = tf.tanh(output_probabilities_mc_var)

                target_eff = tf.multiply((1.0 - smoothing_probability), target) + smoothing_probability * 0.5
            else:
                target_eff = target
            standard_normal_samples = tf.reshape(standard_normal_samples, (-1, 30))  # [mc_samples * batch_size, n_classes]

            tiled_target = tf.tile(tf.expand_dims(target_eff, axis=0), multiples=(mc_samples, 1, 1))  # [mc_samples, batch_size, n_classes]
            tiled_target = tf.reshape(tiled_target, (-1, 30))  # [mc_samples * batch_size, n_classes]
            lv = 0.0
            for c in range(30):
                lv = lv + tf.nn.weighted_cross_entropy_with_logits(labels=tiled_target[:, c],
                                                                   logits=standard_normal_samples[:, c],
                                                                   pos_weight=pos_weights[c])

            lv = lv / 30.0
        else:
            lv = 0.0
            for c in range(30):
                lv = lv + tf.nn.weighted_cross_entropy_with_logits(labels=target[:, c],
                                                                   logits=output[:, c],
                                                                   pos_weight=pos_weights[c])
            lv = lv / 30.0
        return tf.reduce_mean(lv)

    def loss(y_true, y_pred):
        loss_value = 0.0
        info_loss_value = 0.0

        for y_i, y_pred_name in enumerate(y_pred_names):  # ["whinny_single", ]
            if loss_type is None:
                if use_logit_vars:
                    loss_value = loss_value +\
                                 _calculate_weighted_multilabel_crossentropy(target=y_true,
                                                                                 output=pred_train[y_pred_name],
                                                                                 output_var=pred_train[y_pred_name + "_var"],
                                                                                 from_logits=True) / target_weight
                else:
                    loss_value = loss_value +\
                                 _calculate_weighted_multilabel_crossentropy(target=y_true,
                                                                                 output=pred_train[y_pred_name],
                                                                                 from_logits=True) / target_weight

        kl_loss = other_outputs["kl_loss"]
        loss_value = loss_value + kl_loss_regulariser * kl_loss

        return loss_value

    info_loss = None

    return loss, info_loss
