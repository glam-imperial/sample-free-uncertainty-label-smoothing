import tensorflow as tf


def frequency_masking(mel_spectrogram, frequency_masking_para, frequency_mask_num):
    fbank_size = mel_spectrogram.get_shape().as_list()
    n, v = fbank_size[0], fbank_size[1]

    for i in range(frequency_mask_num):
        f = tf.cast(tf.random.uniform(shape=(), maxval=frequency_masking_para), tf.int32)
        f0 = tf.cast(tf.random.uniform(shape=(), maxval=v-tf.cast(f, tf.float32)), tf.int32)

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
        t = tf.cast(tf.random.uniform(shape=(), maxval=time_masking_para), tf.int32)
        t0 = tf.cast(tf.random.uniform(shape=(), maxval=n - tf.cast(t, tf.float32)), tf.int32)

        mask = tf.concat([tf.ones(shape=(t0, 1), dtype=tf.float32),
                          tf.zeros(shape=(t, 1), dtype=tf.float32),
                          tf.ones(shape=(n - t - t0, 1), dtype=tf.float32),
                          ], 0)
        mel_spectrogram = tf.multiply(mel_spectrogram, mask)
    return tf.cast(mel_spectrogram, tf.float32)
