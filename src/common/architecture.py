import tensorflow as tf

from common.models.audio_core_blocks import get_ResNet28_block
from common.models.wide_resnet import get_WideResNet_block
from common.models.variational_audio_core_blocks import get_VariationalResNet28_block
from common.models.embedding_pooling import get_attention_global_pooling
from common.models.variational_embedding_pooling import get_variational_attention_global_pooling


def get_model(name_to_metadata,
              model_configuration):
    # The below configuration stuff are defined in the YAML files.
    core_model = model_configuration["core_model"]
    core_model_configuration = model_configuration["core_model_configuration"]

    input_type_list = model_configuration["input_type_list"]
    y_pred_names = model_configuration["output_type_list"]

    global_pooling = model_configuration["global_pooling"]
    global_pooling_configuration = model_configuration["global_pooling_configuration"]

    if "bayesian" in model_configuration.keys():
        use_logit_vars = True
        # if "use_logit_vars" in model_configuration["bayesian"].keys():
        #     use_logit_vars = model_configuration["bayesian"]["use_logit_vars"]
        # else:
        #     use_logit_vars = False
    else:
        use_logit_vars = False

    # The below list is required for initialising a Keras model.
    input_layer_list = list()
    for input_type in input_type_list:
        input_layer_list.append(tf.keras.Input(shape=name_to_metadata[input_type]["numpy_shape"]))

    custom_objects = list()

    net_train = input_layer_list[0]
    net_test = input_layer_list[0]

    if core_model == "WideResNet":
        core_kl_loss = 0.0
        pool_kl_loss = 0.0

        prediction_train, \
            prediction_test = get_WideResNet_block(net_train,
                                                   net_test,
                                                   y_pred_names,
                                                   global_pooling_configuration)
    else:
        # The core model, this is where it gets deep.
        core_kl_loss = 0.0
        if core_model == "ResNet28":
            net_train, \
                net_test = get_ResNet28_block(net_train,
                                              net_test,
                                              core_model_configuration)
        elif core_model == "VariationalResNet28":
            net_train, \
            net_test, \
                net_var_train, \
                net_var_test, \
                kl_loss = get_VariationalResNet28_block(net_train,
                                                        net_test,
                                                        core_model_configuration)
            core_kl_loss = core_kl_loss + kl_loss
        else:
            raise ValueError("Invalid core_model type.")

        # For sequential data (audio, speech, video, text), we may need to perform a pooling of the features from all
        # sequence frames, into 1 "frame" that summarises the entire sequence.
        pool_kl_loss = 0.0
        if global_pooling == "Attention":
            prediction_train, \
                prediction_test = get_attention_global_pooling(net_train,
                                                               net_test,
                                                               y_pred_names,
                                                               global_pooling_configuration)
        elif global_pooling == "VariationalAttention":
            prediction_train, \
                prediction_test, \
                kl_loss = get_variational_attention_global_pooling(net_train,
                                                                   net_test,
                                                                   net_var_train,
                                                                   net_var_test,
                                                                   y_pred_names,
                                                                   global_pooling_configuration)

            pool_kl_loss = pool_kl_loss + kl_loss
        else:
            raise ValueError("Invalid global_pooling type.")

    if len(y_pred_names) == 1:
        y_pred_name = y_pred_names[0]
        if use_logit_vars:
            keras_model_train = tf.keras.Model(inputs=input_layer_list, outputs=tf.concat([prediction_train[y_pred_name],
                                                                                           prediction_train[y_pred_name + "_var"]],
                                                                                          axis=1))
            keras_model_test = tf.keras.Model(inputs=input_layer_list, outputs=tf.concat([prediction_test[y_pred_name],
                                                                                          prediction_test[y_pred_name + "_var"]],
                                                                                         axis=1))
        else:
            keras_model_train = tf.keras.Model(inputs=input_layer_list, outputs=prediction_train[y_pred_name])
            keras_model_test = tf.keras.Model(inputs=input_layer_list, outputs=prediction_test[y_pred_name])
    else:
        raise NotImplementedError

    keras_model_test.summary()

    kl_loss = core_kl_loss + pool_kl_loss

    other_outputs = dict()
    other_outputs["kl_loss"] = kl_loss

    return prediction_train, prediction_test, keras_model_train, keras_model_test, other_outputs, custom_objects
    # return None, prediction, None, keras_model_test, other_outputs, custom_objects
