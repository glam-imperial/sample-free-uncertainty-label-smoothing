import math
import json

import numpy as np
import tensorflow as tf

from variational.bbb_utils import get_random, scale_mixture_prior_generalised, log_gaussian


class WeightPriorStudent:
    def __init__(self,):
        pass
        # super(WeightPriorStudent, self).__init__()

    def calculate_kl(self,
                     var_par):
        W_mu = var_par["W_mu"]
        W_var = var_par["W_var"]
        W_log_alpha = var_par["W_log_alpha"]
        w_dims = var_par["w_dims"]
        use_bias = var_par["use_bias"]
        if use_bias:
            b_mu = var_par["b_mu"]
            b_var = var_par["b_var"]
            b_log_alpha = var_par["b_log_alpha"]
            b_dims = var_par["b_dims"]

        # This is known in closed form.
        W_kl_a = - 0.5 * tf.log(W_var)
        W_kl_b = tf.sqrt(1e-8 + tf.exp(W_log_alpha)) * get_random(tuple(w_dims), avg=0., std=1.)
        W_kl_c = tf.multiply(W_mu, 1.0 + W_kl_b)
        W_kl_d = tf.pow(W_kl_c, 2.0)
        W_kl_e = (1.0 + 1e-8) * tf.log(1e-8 + W_kl_d) / 2.0
        W_kl = W_kl_a + W_kl_e
        kl = tf.reduce_sum(W_kl)
        if use_bias:
            b_kl_a = - 0.5 * tf.log(b_var)
            b_kl_b = tf.sqrt(1e-8 + tf.exp(b_log_alpha)) * get_random(tuple(b_dims), avg=0., std=1.)
            b_kl_c = tf.multiply(b_mu, 1.0 + b_kl_b)
            b_kl_d = tf.pow(b_kl_c, 2.0)
            b_kl_e = (1.0 + 1e-8) * tf.log(1e-8 + b_kl_d) / 2.0
            b_kl = b_kl_a + b_kl_e
            kl = kl + tf.reduce_sum(b_kl)
        return kl


class WeightPriorARD:
    def __init__(self,):
        pass
        # super(WeightPriorARD, self).__init__()

    # def toJSON(self):
    #     return json.dumps(self, default=lambda o: o.__dict__,
    #                       sort_keys=True, indent=4)

    def calculate_kl(self,
                     var_par):
        W_mu = var_par["W_mu"]
        W_log_alpha = var_par["W_log_alpha"]
        use_bias = var_par["use_bias"]
        if use_bias:
            b_mu = var_par["b_mu"]
            b_log_alpha = var_par["b_log_alpha"]

        # This is known in closed form.
        # tf.log(1.0 + (1.0 / (1e-8 + tf.exp(W_log_alpha))))
        W_kl = tf.reduce_sum(tf.log(1.0 + (1.0 / (1e-8 + tf.exp(W_log_alpha)))) * tf.ones_like(W_mu))
        kl = W_kl
        if use_bias:
            b_kl = tf.reduce_sum(tf.log(1.0 + (1.0 / (1e-8 + tf.exp(b_log_alpha)))) * tf.ones_like(b_mu))
            kl = kl + b_kl
        # W_kl = tf.reduce_sum(tf.log(1.0 + (1.0 / 1e-8 + tf.exp(W_log_alpha))) * tf.ones_like(W_mu))
        # kl = W_kl
        # if use_bias:
        #     b_kl = tf.reduce_sum(tf.log(1.0 + (1.0 / 1e-8 + tf.exp(b_log_alpha))) * tf.ones_like(b_mu))
        #     kl = kl + b_kl
        return kl


class WeightPriorEntropy:
    def __init__(self,):
        pass

    def calculate_kl(self,
                     var_par):
        W_mu = var_par["W_mu"]
        W_log_alpha = var_par["W_log_alpha"]
        use_bias = var_par["use_bias"]
        W_variance = 1e-8 + tf.multiply(tf.exp(W_log_alpha),
                                        tf.pow(W_mu, 2.0))
        if use_bias:
            b_mu = var_par["b_mu"]
            b_log_alpha = var_par["b_log_alpha"]
            b_variance = 1e-8 + tf.multiply(tf.exp(b_log_alpha),
                                            tf.pow(b_mu, 2.0))

        # This is known in closed form.
        W_entropy = tf.reduce_sum(0.5 + 0.5 * tf.log(2 * math.pi * W_variance))
        W_kl = - W_entropy
        kl = W_kl
        if use_bias:
            b_entropy = tf.reduce_sum(0.5 + 0.5 * tf.log(2 * math.pi * b_variance))
            b_kl = - b_entropy
            kl = kl + b_kl
        return kl


class WeightPriorMOG:
    def __init__(self,
                 sigma_prior,  # Should be list, even if just 1 value.
                 mixture_weights,
                 calculation_type):  # ["MC", "Closed"]
        # super(WeightPriorMOG, self).__init__()
        ######################## Initialising prior Gaussian mixture ###########################
        # If there are no mixture weights, initialise at equal mixtures
        if mixture_weights is None:
            mixture_weights = list()
            for i in range(len(sigma_prior)):
                mixture_weights.append(1.0 / len(sigma_prior))

        if len(sigma_prior) != len(mixture_weights):
            raise ValueError('Invalid Gaussian Mixture defined. ')

        for i in mixture_weights:
            if i < 0.0:
                raise ValueError('Invalid Mixture Weights. ')

        # Georgios TODO: Saw both np and statistics mean() in versions. Am keeping commented the other.
        mixture_weights_norm = np.mean(mixture_weights)
        # mixture_weights_norm = statistics.mean(mixture_weights)
        mixture_weights = [m_w / mixture_weights_norm for m_w in mixture_weights]

        self._sigma_prior = sigma_prior
        self._mixture_weights = mixture_weights
        self._calculation_type = calculation_type

        if self._calculation_type not in ["MC", "Closed"]:
            raise ValueError("Invalid KL-div calculation type.")

        if (self._calculation_type == "Closed") and (len(self._mixture_weights) > 1):
            raise NotImplementedError("No approximate closed form calculation of KL-div implemented for MoGs.")

        # if self.hparams.prior_type == 'mixed':
        #     # Georgios: Am a bit unsure what the below does, but looks super cool.
        #     p_s_m = [m_w * np.square(s) for m_w, s in zip(self._mixture_weights, self._sigma_prior)]
        #     p_s_m = np.sqrt(np.sum(p_s_m))
        #
        #     if hparams.prior_target == 0.218:
        #         self.rho_max_init = math.log(math.exp(p_s_m / 2) - 1.0)
        #         self.rho_min_init = math.log(math.exp(p_s_m / 4) - 1.0)
        #     elif hparams.prior_target == 4.25:
        #         self.rho_max_init = math.log(math.exp(p_s_m / 212.078) - 1.0)
        #         self.rho_min_init = math.log(math.exp(p_s_m / 424.93) - 1.0)
        #     elif hparams.prior_target == 6.555:
        #         self.rho_max_init = math.log(math.exp(p_s_m / 3265.708106) - 1.0)
        #         self.rho_min_init = math.log(math.exp(p_s_m / 6507.637711) - 1.0)

    def calculate_kl(self,
                     var_par):
        if self._calculation_type == "MC":
            W_mu = var_par["W_mu"]
            W_sigma = var_par["W_sigma"]
            W_sample_list = var_par["W_sample_list"]
            use_bias = var_par["use_bias"]
            if use_bias:
                b_mu = var_par["b_mu"]
                b_sigma = var_par["b_sigma"]
                b_sample_list = var_par["b_sample_list"]

            # KL-div of gaussian mixtures is intractable and requires an approximation
            # Here, MC-sampling using the weights sampled in call().
            if len(W_sample_list) < 1:
                # We could conceivably perform a new MC-sampling of weights -- too expensive.
                raise NotImplementedError("We have not sampled here -- need to implement alternative approximation.")

            log_prior = 0.
            log_var_posterior = 0.

            for W in W_sample_list:
                log_prior += scale_mixture_prior_generalised(W, self._sigma_prior, self._mixture_weights)
                log_var_posterior += tf.reduce_sum(log_gaussian(W, W_mu, W_sigma))

            if use_bias:
                for b in b_sample_list:
                    log_prior += scale_mixture_prior_generalised(b, self._sigma_prior, self._mixture_weights)
                    log_var_posterior += tf.reduce_sum(log_gaussian(b, b_mu, b_sigma))

            kl = (log_var_posterior - log_prior) / len(W_sample_list)
        elif self._calculation_type == "Closed":
            W_mu = var_par["W_mu"]
            W_sigma = var_par["W_sigma"]
            use_bias = var_par["use_bias"]
            if use_bias:
                b_mu = var_par["b_mu"]
                b_sigma = var_par["b_sigma"]
            # kl = 0.5 * (self.alpha * (self.M.pow(2) + logS.exp()) - logS).sum()
            # This is known in closed form.
            # kl = - 0.5 + tf.log(sigma2) - tf.log(sigma1) + (tf.pow(sigma1, 2) + tf.pow(mu1 - mu2, 2)) / (2 * tf.pow(sigma2, 2))

            sigma_2 = self._sigma_prior[0] + 1e-8

            sigma_1 = W_sigma + tf.constant(1e-8, dtype=tf.float32)
            mu_1 = W_mu

            kl_W = - 0.5 + tf.log(sigma_2) - tf.log(sigma_1) + (tf.pow(sigma_1, 2) + tf.pow(mu_1, 2)) / (
                     2 * tf.pow(sigma_2, 2))

            kl = tf.reduce_sum(kl_W)

            if use_bias:
                sigma_1 = b_sigma + tf.constant(1e-8, dtype=tf.float32)
                mu_1 = b_mu

                kl_b = - 0.5 + tf.log(sigma_2) - tf.log(sigma_1) + (tf.pow(sigma_1, 2) + tf.pow(mu_1, 2)) / (
                         2 * tf.pow(sigma_2, 2))

                kl = kl + tf.reduce_sum(kl_b)
        else:
            raise ValueError("Invalid calculation type.")

        # else:
        #     psm = tf.Variable(4.25)
        #     if self.built:
        #         sigW = tf.debugging.check_numerics(
        #             tf.sqrt(tf.multiply(tf.exp(self.log_alpha), tf.pow(self.W_mu, 2.0)) + 1e-7), 'NaN or inf')
        #         sigb = tf.debugging.check_numerics(softplus(self.b_rho), 'NaN or inf')
        #         klW = tf.log(psm / (sigW + 1e-6)) + (tf.pow(sigW, 2.0) + tf.pow(self.W_mu, 2.0)) / (
        #                 2 * tf.pow(psm, 2.0) + 1e-6) - 0.5
        #         klb = tf.log(psm / (sigb + 1e-6)) + (tf.pow(sigb, 2.0) + tf.pow(self.b_mu, 2.0)) / (
        #                 2 * tf.pow(psm, 2.0) + 1e-6) - 0.5
        #         return tf.reduce_sum(klW) + tf.reduce_sum(klb)
        #     else:
        #         return tf.constant(0.0)

        return kl
