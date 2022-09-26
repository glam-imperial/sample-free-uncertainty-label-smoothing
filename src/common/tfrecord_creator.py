import tensorflow as tf
import numpy as np


def _int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class TFRecordCreator:
    def __init__(self,
                 tf_records_folder,
                 sample_iterable,
                 are_test_labels_available,
                 is_continuous_time):
        self.tf_records_folder = tf_records_folder
        self.sample_iterable = sample_iterable
        self.are_test_labels_available = are_test_labels_available
        self.is_continuous_time = is_continuous_time

    def create_tfrecords(self):
        for sample in self.sample_iterable:
            self._serialize_sample(sample)

    def _serialize_sample(self, sample):
        id_dict = sample.get_id_dict()
        partition = sample.get_partition()
        x_dict = sample.get_x_dict()
        y_dict = sample.get_y_dict()
        support = sample.get_support()
        composite_name = sample.get_composite_name()

        # Make sure everything is either np float32 or np int64 for tfrecords
        id_dict = {k: np.int64(v) for k, v in id_dict.items()}
        x_dict = {k: np.float32(v) for k, v in x_dict.items()}
        y_dict = {k: np.float32(v) for k, v in y_dict.items()}
        support = np.float32(support)

        # sub_segment_id = id_dict["sub_segment_id"]
        # segment_id = id_dict["segment_id"]
        # sample_id = id_dict["sample_id"]

        writer = tf.io.TFRecordWriter(
            self.tf_records_folder + "/" + partition + "/" + composite_name + '.tfrecords')

        if self.is_continuous_time:
            for step in range(sample.get_number_of_steps()):
                tf_record_dict = dict()

                if step is not None:
                    tf_record_dict["step_id"] = _int_feature(np.int64(step))

                for id_name, id_number in id_dict.items():
                    if id_number is not None:
                        tf_record_dict[id_name] = _int_feature(np.int64(id_number))

                if not ((partition == "test") and (not self.are_test_labels_available)):
                    for y_name, y in y_dict.items():
                        tf_record_dict[y_name] = _bytes_feature(y[step].tobytes())

                for x_name, x in x_dict.items():
                    tf_record_dict[x_name] = _bytes_feature(x[step].tobytes())

                tf_record_dict["support"] = _bytes_feature(support[step].tobytes())

                # Save tf records.
                example = tf.train.Example(features=tf.train.Features(feature=tf_record_dict))

                writer.write(example.SerializeToString())
        else:
            tf_record_dict = dict()

            for id_name, id_number in id_dict.items():
                if id_number is not None:
                    tf_record_dict[id_name] = _int_feature(np.int64(id_number))

            if not ((partition == "test") and (not self.are_test_labels_available)):
                for y_name, y in y_dict.items():
                    tf_record_dict[y_name] = _bytes_feature(y.tobytes())

            for x_name, x in x_dict.items():
                tf_record_dict[x_name] = _bytes_feature(x.tobytes())

            tf_record_dict["support"] = _bytes_feature(support.tobytes())

            example = tf.train.Example(features=tf.train.Features(feature=tf_record_dict))

            writer.write(example.SerializeToString())
