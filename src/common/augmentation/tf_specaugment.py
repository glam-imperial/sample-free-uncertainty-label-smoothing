import random

import tensorflow as tf


def frequency_masking(mel_spectrogram, frequency_masking_para, frequency_mask_num):
    fbank_size = mel_spectrogram.get_shape().as_list()
    n, v = fbank_size[0], fbank_size[1]

    for i in range(frequency_mask_num):
        f = random.randint(0, frequency_masking_para)
        f0 = random.randint(0, v-f)

        mask = tf.concat([tf.ones(shape=(1, f0), dtype=tf.float32),
                          tf.zeros(shape=(1, f), dtype=tf.float32),
                          tf.ones(shape=(1, v - f - f0), dtype=tf.float32),
                          ], 1)
        mel_spectrogram = tf.multiply(mel_spectrogram, mask)
    return tf.cast(mel_spectrogram, tf.float32)


def time_masking(mel_spectrogram, time_masking_para, time_mask_num):
    fbank_size = mel_spectrogram.get_shape().as_list()
    n, v = fbank_size[0], fbank_size[1]

    for i in range(time_mask_num):
        t = random.randint(0, time_masking_para)
        t0 = random.randint(0, n - t)

        mask = tf.concat([tf.ones(shape=(t0, 1), dtype=tf.float32),
                          tf.zeros(shape=(t, 1), dtype=tf.float32),
                          tf.ones(shape=(n - t - t0, 1), dtype=tf.float32),
                          ], 0)
        mel_spectrogram = tf.multiply(mel_spectrogram, mask)
    return tf.cast(mel_spectrogram, tf.float32)
