#!/usr/bin/env python3
"""nst_class"""
import numpy as np
import tensorflow as tf


class NST:
    """class NST"""
    style_layers = ['block1_conv1', 'block2_conv1',
                    'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """initialization"""
        tf.executing_eagerly()
        if type(style_image) is not np.ndarray or style_image.ndim != 3\
                or style_image.shape[2] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        if type(content_image) is not np.ndarray or content_image.ndim != 3\
                or content_image.shape[2] != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
        if (type(alpha) is not int and type(alpha) is not float) or alpha < 0:
            raise TypeError('alpha must be a non-negative number')
        if (type(beta) is not int and type(beta) is not float) or beta < 0:
            raise TypeError('beta must be a non-negative number')
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()

    @staticmethod
    def scale_image(image):
        """rescales an image such that its pixels values are
          between 0 and 1 and its largest side is 512 pixels
          """
        if type(image) is not np.ndarray or\
                image.shape[-1] != 3:
            raise TypeError(
                'image must be a numpy.ndarray with shape (h, w, 3)')
        h, w, c = image.shape
        maxd = max(h, w)
        new_h = int(h * (512/maxd))
        new_w = int(w * (512/maxd))
        image = tf.expand_dims(image, axis=0)
        resized = tf.image.resize(image, size=(new_h, new_w), method="bicubic")
        resized /= 255
        return tf.clip_by_value(resized, 0.0, 1.0)

    def load_model(self):
        """
        Load VGG19 model
        :return: The model
        """
        vgg19 = tf.keras.applications.VGG19(include_top=False)
        for layer in vgg19.layers:
            layer.trainable = False
        vgg19.save("model.h5")
        model = tf.keras.models.load_model(
            "model.h5",
            # change MaxPooling2D to AveragePooling2D like in the paper
            custom_objects={
                "MaxPooling2D": tf.keras.layers.AveragePooling2D()
            })

        outputs = ([model.get_layer(layer).output
                   for layer in self.style_layers]
                   + [model.get_layer(self.content_layer).output])
        self.model = tf.keras.models.Model(model.input, outputs)
