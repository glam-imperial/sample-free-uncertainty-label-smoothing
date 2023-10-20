import os
import math

import tensorflow as tf
if tf.__version__[0] == "2":
    tf = tf.compat.v1

import numpy as np

from common.common import safe_make_dir
from common.batch_generator import BatchGenerator
import common.architecture as architecture


def experiment_run(config_dict):
    tfrecords_folder = config_dict["tfrecords_folder"]
    output_folder = config_dict["output_folder"]
    gpu = config_dict["gpu"]
    are_test_labels_available = config_dict["are_test_labels_available"]
    path_list_dict = config_dict["path_list_dict"]
    train_size = config_dict["train_size"]
    devel_size = config_dict["devel_size"]
    test_size = config_dict["test_size"]
    train_batch_size = config_dict["train_batch_size"]
    devel_batch_size = config_dict["devel_batch_size"]
    test_batch_size = config_dict["test_batch_size"]
    name_to_metadata = config_dict["model_configuration"]["name_to_metadata"]
    method_string = config_dict["method_string"]
    model_configuration = config_dict["model_configuration"]
    initial_learning_rate = config_dict["initial_learning_rate"]
    number_of_epochs = config_dict["number_of_epochs"]
    input_gaussian_noise = config_dict["input_gaussian_noise"]
    val_every_n_epoch = config_dict["val_every_n_epoch"]
    patience = config_dict["patience"]
    y_pred_names = config_dict["y_pred_names"]
    monitor_target_to_measures = config_dict["monitor_target_to_measures"]
    report_target_to_measures = config_dict["report_target_to_measures"]
    output_channel_targets = config_dict["output_channel_targets"]
    input_type_list = model_configuration["input_type_list"]
    output_type_list = model_configuration["output_type_list"]

    losses = config_dict["losses_module"]
    evaluation = config_dict["evaluation_module"]

    method_output_prefix = output_folder + "/" + method_string
    safe_make_dir(method_output_prefix)
    # best_model_chackpoint_path = method_output_prefix + "/best_model.h5"

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = repr(gpu)

    train_steps_per_epoch = math.ceil(train_size / train_batch_size)
    devel_steps_per_epoch = math.ceil(devel_size / devel_batch_size)
    test_steps_per_epoch = math.ceil(test_size / test_batch_size)

    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True

    g = tf.Graph()
    with g.as_default():
        with tf.Session() as sess:
            dataset_train, \
            iterator_train, \
            next_element_train, \
            init_op_train = BatchGenerator(tf_records_folder=tfrecords_folder,
                                           is_training=True,
                                           partition="train",
                                           are_test_labels_available=are_test_labels_available,
                                           name_to_metadata=name_to_metadata,
                                           input_type_list=input_type_list,
                                           output_type_list=output_type_list,
                                           batch_size=train_batch_size,
                                           buffer_size=(train_steps_per_epoch + 1) // 4,
                                           path_list=path_list_dict["train"]).get_tf_dataset()
            dataset_devel, \
            iterator_devel, \
            next_element_devel, \
            init_op_devel = BatchGenerator(tf_records_folder=tfrecords_folder,
                                           is_training=False,
                                           partition="devel",
                                           are_test_labels_available=are_test_labels_available,
                                           name_to_metadata=name_to_metadata,
                                           input_type_list=input_type_list,
                                           output_type_list=output_type_list,
                                           batch_size=devel_batch_size,
                                           buffer_size=(devel_steps_per_epoch + 1) // 4,
                                           path_list=path_list_dict["devel"]).get_tf_dataset()
            dataset_test, \
            iterator_test, \
            next_element_test, \
            init_op_test = BatchGenerator(tf_records_folder=tfrecords_folder,
                                          is_training=False,
                                          partition="test",
                                          are_test_labels_available=are_test_labels_available,
                                          name_to_metadata=name_to_metadata,
                                          input_type_list=input_type_list,
                                          output_type_list=output_type_list,
                                          batch_size=test_batch_size,
                                          buffer_size=(test_steps_per_epoch + 1) // 4,
                                          path_list=path_list_dict["test"]).get_tf_dataset()

            sess.run(init_op_train)
            true_train_np = list()
            for i in range(train_steps_per_epoch):
                batch_data = sess.run(next_element_train)
                true_train_np.append(batch_data[1][0])
                # print(batch_data[0][0].shape)
                # print(batch_data[1][0].shape)
                # plt.imshow(batch_data[0][0][0])
                # plt.savefig("/homes/gr912/Downloads/ff" + repr(i))

            true_train_np = np.vstack(true_train_np)
            # print(true_train_np.shape)
            pos_weights = true_train_np.sum(axis=0)
            pos_weights = true_train_np.shape[0] / pos_weights
            print(pos_weights)

            # Get the true labels for devel and test.
            sess.run(init_op_devel)
            true_devel_np = list()
            for i in range(devel_steps_per_epoch):
                batch_data = sess.run(next_element_devel)
                true_devel_np.append(batch_data[1][0])
                # print(batch_data[0][0].shape)
                # print(batch_data[1][0].shape)
            true_devel_np = np.vstack(true_devel_np)
            # print(true_devel_np.shape)

            sess.run(init_op_test)
            true_test_np = list()
            for i in range(test_steps_per_epoch):
                batch_data = sess.run(next_element_test)
                true_test_np.append(batch_data[1][0])
                # print(batch_data[0][0].shape)
                # print(batch_data[1][0].shape)
            true_test_np = np.vstack(true_test_np)
            # print(true_test_np.shape)

            with tf.variable_scope("Model"):
                model_configuration_effective = {k: v for k, v in model_configuration.items()}

                pred_train,\
                pred_test, \
                keras_model_train,\
                keras_model_test, \
                other_outputs, \
                custom_objects = architecture.get_model(name_to_metadata=name_to_metadata,
                                                        model_configuration=model_configuration_effective)

            loss, \
            _ = losses.get_loss(pred_train=pred_train,
                                            model_configuration=model_configuration,
                                            y_pred_names=y_pred_names,
                                            other_outputs=other_outputs,
                                            pos_weights=pos_weights)
            # loss, \
            # info_loss = losses.get_loss(pred_train=pred_test,
            #                             model_configuration=model_configuration,
            #                             y_pred_names=y_pred_names,
            #                             other_outputs=other_outputs,
            #                             pos_weights=pos_weights)

            optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)

            keras_model_train.compile(optimizer,
                                      loss,
                                      metrics=None)
            keras_model_test.compile(optimizer,
                                     loss,
                                     metrics=None)

            custom_saver = evaluation.CustomSaver(output_folder=output_folder,
                                                  method_string=method_string,
                                                  monitor_target_to_measures=monitor_target_to_measures,
                                                  keras_model_test=keras_model_test)
            current_patience = 0
            performance_monitor = evaluation.PerformanceMonitor(output_folder=output_folder,
                                                                method_string=method_string,
                                                                custom_saver=custom_saver,
                                                                monitor_target_to_measures=monitor_target_to_measures,
                                                                report_target_to_measures=report_target_to_measures,
                                                                are_test_labels_available=are_test_labels_available,
                                                                y_pred_names=y_pred_names,
                                                                model_configuration=model_configuration)

            print("Start training base model.")
            print("Fresh base model.")
            for ee, epoch in enumerate(range(number_of_epochs)):
                print("EPOCH:", epoch + 1)
                history = keras_model_train.fit(x=dataset_train,
                                                epochs=1,
                                                verbose=2,
                                                callbacks=None,
                                                steps_per_epoch=train_steps_per_epoch,
                                                validation_steps=None,
                                                validation_data=None,
                                                workers=8,
                                                use_multiprocessing=True)

                if (ee + 1) % val_every_n_epoch == 0:
                    pred_devel_np = keras_model_test.predict(dataset_devel,
                                                             verbose=2,
                                                             steps=devel_steps_per_epoch,
                                                             callbacks=None,
                                                             workers=8,
                                                             use_multiprocessing=True)
                    # print(pred_devel_np)
                    # print(pred_devel_np.shape)

                    devel_items = dict()
                    devel_items = dict()
                    for target_name in y_pred_names:
                        devel_items[target_name] = dict()
                        devel_items[target_name]["pred"] = pred_devel_np
                        devel_items[target_name]["true"] = true_devel_np

                    performance_monitor.get_measures(items=devel_items,
                                                     partition="devel")
                    performance_monitor.report_measures(partition="devel",
                                                        output_channel_targets=output_channel_targets)

                    noticed_improvement = performance_monitor.monitor_improvement()

                    if noticed_improvement:
                        current_patience = 0
                    else:
                        current_patience += 1
                        if current_patience > patience:
                            break

            for target in monitor_target_to_measures.keys():
                for measure in monitor_target_to_measures[target]:
                    keras_model_test = custom_saver.load_model(target=target,
                                                               measure=measure,
                                                               custom_objects=custom_objects)

                    pred_test_np = keras_model_test.predict(dataset_test,
                                                            verbose=2,
                                                            steps=test_steps_per_epoch,
                                                            callbacks=None,
                                                            workers=8,
                                                            use_multiprocessing=True)

                    test_items = dict()
                    for target_name in y_pred_names:
                        test_items[target_name] = dict()
                        test_items[target_name]["pred"] = pred_test_np
                        test_items[target_name]["true"] = true_test_np

                    performance_monitor.get_test_measures(test_items=test_items,
                                                          target=target,
                                                          measure=measure)
            performance_monitor.report_best_performance_measures(output_channel_targets=output_channel_targets)

            results_summary, items_summary = performance_monitor.get_results_summary()

            return results_summary, items_summary
