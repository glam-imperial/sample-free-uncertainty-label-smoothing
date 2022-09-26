import tensorflow as tf


def process_metadata(name_to_metadata):
    for variable_name in name_to_metadata.keys():
        numpy_shape = name_to_metadata[variable_name]["numpy_shape"]
        variable_type = name_to_metadata[variable_name]["variable_type"]

        if variable_type == "x":
            tfrecords_type = tf.string
            tf_dtype = tf.float32
            shape = tuple([-1, ] + [d for d in numpy_shape[1:]])
            padded_shape = tuple([None, ] + [d for d in numpy_shape[1:]])
            placeholder_shape = tuple([None, None] + [d for d in numpy_shape[1:]])
        elif variable_type == "y":
            tfrecords_type = tf.string
            tf_dtype = tf.float32
            if len(numpy_shape) > 1:
                shape = tuple([-1, ] + [d for d in numpy_shape[1:]])
                padded_shape = tuple([None, ] + [d for d in numpy_shape[1:]])
                placeholder_shape = tuple([None, None] + [d for d in numpy_shape[1:]])
            else:
                shape = numpy_shape
                padded_shape = numpy_shape
                placeholder_shape = tuple([None, ] + [d for d in numpy_shape[0:]])
        elif variable_type == "support":
            tfrecords_type = tf.string
            tf_dtype = tf.float32
            shape = tuple([-1, ] + [d for d in numpy_shape[1:]])
            padded_shape = tuple([None, ] + [d for d in numpy_shape[1:]])
            placeholder_shape = tuple([None, None] + [d for d in numpy_shape[1:]])
        elif variable_type == "id":
            tfrecords_type = tf.int64
            tf_dtype = tf.int32
            shape = numpy_shape
            padded_shape = numpy_shape
            if len(numpy_shape) > 1:
                placeholder_shape = tuple([None, ] + [d for d in numpy_shape[1:]])
            else:
                placeholder_shape = tuple([None, ] + [1,])
        else:
            raise ValueError("Invalid variable type.")

        name_to_metadata[variable_name]["tfrecords_type"] = tfrecords_type
        name_to_metadata[variable_name]["tf_dtype"] = tf_dtype
        name_to_metadata[variable_name]["shape"] = shape
        name_to_metadata[variable_name]["padded_shape"] = padded_shape
        name_to_metadata[variable_name]["placeholder_shape"] = placeholder_shape

    return name_to_metadata


#     name_to_metadata["support"]["numpy_shape"] = (48000, 1)
#     name_to_metadata["waveform"]["numpy_shape"] = (75, 640)
#     name_to_metadata["logmel_spectrogram"]["numpy_shape"] = (300, 128)
#     name_to_metadata["mfcc"]["numpy_shape"] = (300, 80)
#     name_to_metadata["segment_id"]["numpy_shape"] = (1, )
#     name_to_metadata["version_id"]["numpy_shape"] = (1, )
#     name_to_metadata["whinny_single"]["numpy_shape"] = (2, )
#     name_to_metadata["whinny_continuous"]["numpy_shape"] = (48000, 2)
#
#     name_to_metadata["support"]["variable_type"] = "support"
#     name_to_metadata["waveform"]["variable_type"] = "x"
#     name_to_metadata["logmel_spectrogram"]["variable_type"] = "x"
#     name_to_metadata["mfcc"]["variable_type"] = "x"
#     name_to_metadata["segment_id"]["variable_type"] = "id"
#     name_to_metadata["version_id"]["variable_type"] = "id"
#     name_to_metadata["whinny_single"]["variable_type"] = "y"
#     name_to_metadata["whinny_continuous"]["variable_type"] = "y"
#
#     name_to_metadata = process_metadata(name_to_metadata)

    # name_to_metadata["support"]["tfrecords_type"] = tf.string
    # name_to_metadata["waveform"]["tfrecords_type"] = tf.string
    # name_to_metadata["logmel_spectrogram"]["tfrecords_type"] = tf.string
    # name_to_metadata["mfcc"]["tfrecords_type"] = tf.string
    # name_to_metadata["segment_id"]["tfrecords_type"] = tf.int64
    # name_to_metadata["version_id"]["tfrecords_type"] = tf.int64
    # name_to_metadata["whinny_single"]["tfrecords_type"] = tf.string
    # name_to_metadata["whinny_continuous"]["tfrecords_type"] = tf.string
    #
    # name_to_metadata["support"]["tf_dtype"] = tf.float32
    # name_to_metadata["waveform"]["tf_dtype"] = tf.float32
    # name_to_metadata["logmel_spectrogram"]["tf_dtype"] = tf.float32
    # name_to_metadata["mfcc"]["tf_dtype"] = tf.float32
    # name_to_metadata["segment_id"]["tf_dtype"] = tf.int32
    # name_to_metadata["version_id"]["tf_dtype"] = tf.int32
    # name_to_metadata["whinny_single"]["tf_dtype"] = tf.float32
    # name_to_metadata["whinny_continuous"]["tf_dtype"] = tf.float32
    #
    # name_to_metadata["support"]["shape"] = (-1, 1)
    # name_to_metadata["waveform"]["shape"] = (-1, 640)
    # name_to_metadata["logmel_spectrogram"]["shape"] = (-1, 128)
    # name_to_metadata["mfcc"]["shape"] = (-1, 80)
    # name_to_metadata["segment_id"]["shape"] = (1,)
    # name_to_metadata["version_id"]["shape"] = (1,)
    # name_to_metadata["whinny_single"]["shape"] = (2,)
    # name_to_metadata["whinny_continuous"]["shape"] = (-1, 2)
    #
    # name_to_metadata["support"]["padded_shape"] = (None, 1)
    # name_to_metadata["waveform"]["padded_shape"] = (None, 640)
    # name_to_metadata["logmel_spectrogram"]["padded_shape"] = (None, 128)
    # name_to_metadata["mfcc"]["padded_shape"] = (None, 80)
    # name_to_metadata["segment_id"]["padded_shape"] = (1,)
    # name_to_metadata["version_id"]["padded_shape"] = (1,)
    # name_to_metadata["whinny_single"]["padded_shape"] = (2,)
    # name_to_metadata["whinny_continuous"]["padded_shape"] = (None, 2)
    #
    # name_to_metadata["support"]["placeholder_shape"] = (None, None, 1)
    # name_to_metadata["waveform"]["placeholder_shape"] = (None, None, 640)
    # name_to_metadata["logmel_spectrogram"]["placeholder_shape"] = (None, None, 128)
    # name_to_metadata["mfcc"]["placeholder_shape"] = (None, None, 80)
    # name_to_metadata["segment_id"]["placeholder_shape"] = (None, 1)
    # name_to_metadata["version_id"]["placeholder_shape"] = (None, 1)
    # name_to_metadata["whinny_single"]["placeholder_shape"] = (None, 2)
    # name_to_metadata["whinny_continuous"]["placeholder_shape"] = (None, None, 2)