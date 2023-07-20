import source.nn as nn
import tensorflow as tf
from sicore import SelectiveInferenceNorm


class DnnSelectiveInferenceNorm:
    def __init__(self, model, d):
        self.d = d
        self.cam_model = nn.NN(model)

    def set_data(self, data, var):
        self.data = tf.constant(data, dtype=tf.float64, shape=[1, self.d, self.d, 1])
        self.var = var

    def construct_eta(self):
        cam_output = self.cam_model.forward(self.data)
        self.saliency_map = cam_output >= 0

        saliency_flatten = tf.reshape(self.saliency_map, [-1])

        eta_selected = tf.cast(tf.where(saliency_flatten, 1, 0), dtype=tf.float64)
        num_selected_pixels = tf.reduce_sum(eta_selected)

        eta_back = tf.cast(tf.where(saliency_flatten, 0, 1), dtype=tf.float64)
        num_back_pixels = tf.reduce_sum(eta_back)

        if num_back_pixels == 0 or num_selected_pixels == 0:
            raise Exception()

        self.eta = (eta_selected / num_selected_pixels) - (eta_back / num_back_pixels)
        self.data = tf.reshape(self.data, [-1])

        sd = tf.sqrt(tf.reduce_sum(tf.square(self.eta)))
        self.max_tail = 20 * sd

    def algorithm(self, a, b, z):
        a = tf.reshape(a, [1, self.d, self.d, 1])
        b = tf.reshape(b, [1, self.d, self.d, 1])
        bias = tf.zeros([1, self.d, self.d, 1], dtype=tf.float64)
        data = a + b * z
        l = tf.constant(-10000, dtype=tf.float64)
        u = tf.constant(10000, dtype=tf.float64)

        l, u, cam, _ = self.cam_model.forward_si((data, bias, a, b, l, u))
        return (cam >= 0), [[l, u]]

    def model_selector(self, saliency_map):
        return tf.reduce_all(self.saliency_map == saliency_map)

    def inference(self, **kwargs):
        self.si_calculator = SelectiveInferenceNorm(
            self.data, self.var, self.eta, use_tf=True
        )

        result = self.si_calculator.inference(
            self.algorithm, self.model_selector, max_tail=self.max_tail, **kwargs
        )
        return result
