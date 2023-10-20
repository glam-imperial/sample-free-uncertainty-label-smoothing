from pathlib import Path

import tensorflow as tf
if tf.__version__[0] == "2":
    tf = tf.compat.v1

from common.augmentation import tf_specaugment


class BatchGenerator:
    def __init__(self,
                 tf_records_folder,
                 is_training,
                 partition,
                 are_test_labels_available,
                 name_to_metadata,
                 input_type_list,
                 output_type_list,
                 batch_size,
                 buffer_size,
                 path_list=None,
                 use_autopad=False):
        self.tf_records_folder = tf_records_folder
        self.is_training = is_training
        self.partition = partition
        self.are_test_labels_available = are_test_labels_available
        self.name_to_metadata = name_to_metadata
        self.input_type_list = input_type_list
        self.output_type_list = output_type_list
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.path_list = path_list
        self.use_autopad = use_autopad

        if (self.path_list is None) or (len(self.path_list) == 0):
            root_path = Path(self.tf_records_folder)
            self.path_list = [str(x) for x in root_path.glob('*.tfrecords')]

        print("Number of files:", len(self.path_list))

    def get_tf_dataset(self):
        dataset = tf.data.TFRecordDataset(self.path_list,
                                          num_parallel_reads=8)

        features_dict = dict()
        for attribute_name, attribute_metadata in self.name_to_metadata.items():
            dtype = attribute_metadata["tfrecords_type"]
            variable_type = attribute_metadata["variable_type"]
            if not ((self.partition == "test") and (not self.are_test_labels_available) and (variable_type == "y")):
                features_dict[attribute_name] = tf.FixedLenFeature([], dtype)

        dataset = dataset.map(lambda x: tf.parse_single_example(x,
                                                                features=features_dict),
                              num_parallel_calls=8)

        def map_func(attribute):
            for attribute_name, attribute_value in attribute.items():
                attribute_metadata = self.name_to_metadata[attribute_name]
                variable_type = attribute_metadata["variable_type"]
                # shape = self.name_to_metadata[attribute_name]["shape"]
                shape = self.name_to_metadata[attribute_name]["numpy_shape"]
                dtype = self.name_to_metadata[attribute_name]["tf_dtype"]

                if variable_type == "id":
                    attribute[attribute_name] = tf.cast(tf.reshape(attribute[attribute_name],
                                                                   shape),
                                                        dtype)
                elif variable_type in ["x", "y", "support"]:
                    attribute[attribute_name] = tf.reshape(tf.decode_raw(attribute[attribute_name],
                                                                         dtype),
                                                           shape)
                    if self.is_training and (variable_type == "x") and (attribute_name == "logmel_spectrogram"):
                        attribute[attribute_name] = tf_specaugment.time_masking(
                            mel_spectrogram=attribute[attribute_name],
                            time_masking_para=24,
                            time_mask_num=2)
                        attribute[attribute_name] = tf_specaugment.frequency_masking(
                            mel_spectrogram=attribute[attribute_name],
                            frequency_masking_para=16,
                            frequency_mask_num=2)

                        attribute[attribute_name] = attribute[attribute_name] + tf.random_normal(shape, mean=.0,
                                                                                                 stddev=0.000001,
                                                                                                 seed=0)
                else:
                    raise ValueError

            input_list = list()
            output_list = list()

            for input_type in self.input_type_list:
                input_list.append(attribute[input_type])

            for output_type in self.output_type_list:
                output_list.append(attribute[output_type])

            return tuple(input_list), tuple(output_list)

        dataset = dataset.map(map_func,
                              num_parallel_calls=8)

        if self.is_training:
            dataset = dataset.shuffle(buffer_size=self.buffer_size)

        input_padded_shapes = list()
        for input_type in self.input_type_list:
            if self.use_autopad:
                padded_shape = self.name_to_metadata[input_type]["padded_shape"]
            else:
                padded_shape = self.name_to_metadata[input_type]["numpy_shape"]
            input_padded_shapes.append(padded_shape)
        output_padded_shapes = list()
        if not ((self.partition == "test") and (not self.are_test_labels_available) and (variable_type == "y")):
            for output_type in self.output_type_list:
                if self.use_autopad:
                    padded_shape = self.name_to_metadata[output_type]["padded_shape"]
                else:
                    padded_shape = self.name_to_metadata[output_type]["numpy_shape"]
                output_padded_shapes.append(padded_shape)

        # print(dataset.output_types,
        #       dataset.output_shapes)

        dataset = dataset.padded_batch(self.batch_size,
                                       padded_shapes=(tuple(input_padded_shapes),
                                                      tuple(output_padded_shapes)))
        dataset = dataset.take(len(self.path_list))
        dataset = dataset.repeat()
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        iterator = tf.data.Iterator.from_structure(dataset.output_types,
                                                   dataset.output_shapes)

        # print(dataset.output_types,
        #       dataset.output_shapes)

        next_element = iterator.get_next()

        init_op = iterator.make_initializer(dataset)

        return dataset, iterator, next_element, init_op
