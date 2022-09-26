# Imports
import numpy as np
import tensorflow.compat.v1 as tf


# BBB Definitions
def get_random(shape, avg, std, seed=0):
    return tf.random_normal(shape, mean=avg, stddev=std, seed=seed)


# Gaussian prior.
def log_gaussian(x, mu, sigma):
    return -0.5 * np.log(2.0 * np.pi) - 0.5 * tf.log(sigma) - (x - mu) ** 2 / (2 * sigma ** 2)


def gaussian_prior(x, sigma_p):
    pp = log_gaussian(x, 0., sigma_p)
    return tf.reduce_sum(pp)


# Gaussian scale mixture prior.
def gaussian(x, mu, sigma):
    scaling = 1.0 / tf.sqrt(2.0 * np.pi * (sigma ** 2))
    bell = tf.exp(- (x - mu) ** 2 / (2.0 * sigma ** 2))

    return scaling * bell


# This ensures that the first mixture component of the prior is given a larger variance.
def scale_mixture_prior_2(x, sigma_p1, sigma_p2, pi):
    if sigma_p1 > sigma_p2:
        first_gaussian = pi * gaussian(x, 0., sigma_p1)
        second_gaussian = (1 - pi) * gaussian(x, 0., sigma_p2)
    else:
        first_gaussian = pi * gaussian(x, 0., sigma_p2)
        second_gaussian = (1 - pi) * gaussian(x, 0., sigma_p1)

    return tf.reduce_sum(tf.log(first_gaussian + second_gaussian))


def scale_mixture_prior(x, sigma_p1, sigma_p2, pi):
    first_gaussian = pi * gaussian(x, 0., sigma_p1)
    second_gaussian = (1 - pi) * gaussian(x, 0., sigma_p2)

    return tf.reduce_sum(tf.log(first_gaussian + second_gaussian))


def scale_mixture_prior_generalised(x, sigma_prior, mixture_weights):
    gaussian_sum = 0.0
    for s_p, m_w in zip(sigma_prior, mixture_weights):
        gaussian_sum += m_w * gaussian(x, 0., s_p)

    return tf.reduce_sum(tf.log(gaussian_sum))


def softplus(x):
    return tf.log(1. + tf.exp(x))



