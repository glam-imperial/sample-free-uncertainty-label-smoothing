import math

import numpy as np
import tensorflow as tf
from scipy.special import expit


def relu_moments(x, x_var):
    relu_mean = 0.5 * x * (1.0 + tf.math.erf(x / (tf.sqrt(1e-8 + x_var) * math.sqrt(2.0)))) + \
                tf.sqrt(0.5 * (1e-8 + x_var) / math.pi) * tf.exp(-0.5 * tf.pow(x, 2.0) / (1e-8 + x_var))
    relu_var = 0.5 * (x_var + tf.pow(x, 2.0)) * (
               1.0 + tf.math.erf(x / (tf.sqrt(1e-8 + x_var) * math.sqrt(2.0)))) + \
               x * tf.sqrt(0.5 * (1e-8 + x_var) / math.pi) * tf.exp(-0.5 * tf.pow(x, 2.0) / (1e-8 + x_var)) - \
               tf.pow(relu_mean, 2.0)

    x = relu_mean
    x_var = tf.pow(relu_var, 2.0)

    return x, x_var


def sigmoid_moments(x, x_var):
    alpha = 3.0 / np.power(np.pi, 2.0)

    sigmoid_mean = tf.nn.sigmoid(x / tf.sqrt(1.0 + alpha * x_var))
    sigmoid_var = sigmoid_mean * (1.0 - sigmoid_mean) * (1.0 - 1.0 / tf.sqrt(1.0 + alpha * x_var))

    x = sigmoid_mean
    x_var = sigmoid_var

    return x, x_var


def sigmoid(x):
    x = np.nan_to_num(x)
    return expit(x)


def sigmoid_moments_np(x, x_var):
    x_std = np.sqrt(x_var)
    mc_samples = 50
    mc_samples_shape = [mc_samples,] + list(x_var.shape)
    standard_normal_samples = np.random.normal(loc=0.0, scale=1.0, size=mc_samples_shape)
    standard_normal_samples = np.multiply(standard_normal_samples, np.expand_dims(x_std, axis=0))
    standard_normal_samples = np.expand_dims(x,
                                             axis=0) + standard_normal_samples
    standard_normal_samples = sigmoid(standard_normal_samples)
    return np.mean(standard_normal_samples, axis=0)


def sigmoid_moments_mc(x, x_var):
    x_std = np.sqrt(x_var)
    mc_samples = 50
    mc_samples_shape = tf.concat([tf.constant((mc_samples,),
                                              dtype=tf.dtypes.int32),
                                  tf.shape(x_std)],
                                 axis=0)
    standard_normal_samples = tf.random.normal(mc_samples_shape,
                                               mean=0.0,
                                               stddev=1.0,
                                               dtype=tf.dtypes.float32,
                                               seed=None,
                                               name=None)  # [mc_samples, batch_size, n_classes]
    standard_normal_samples = tf.multiply(standard_normal_samples,
                                          tf.expand_dims(x_std, axis=0))  # [mc_samples, batch_size, n_classes]
    standard_normal_samples = tf.expand_dims(x,
                                             axis=0) + standard_normal_samples  # [mc_samples, batch_size, n_classes]
    standard_normal_samples = tf.exp(standard_normal_samples - tf.reduce_logsumexp(standard_normal_samples,
                                                                                   axis=2,
                                                                                   keepdims=True))
    x = tf.reduce_mean(standard_normal_samples, axis=0)

    return x
