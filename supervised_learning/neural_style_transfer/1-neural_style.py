#!/usr/bin/env python3
"""Neural Style Transfer"""
import tensorflow as tf
import numpy as np


class NST:
    """Class of NST"""
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'
    
    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Initialize variables"""
        if not isinstance(style_image, np.ndarray) or style_image.ndim != 3 or style_image.shape[2] != 3:
            raise TypeError("style_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(content_image, np.ndarray) or content_image.ndim != 3 or content_image.shape[2] != 3:
            raise TypeError("content_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")
        
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()
    
    @staticmethod
    def scale_image(image):
        """Static Method hat rescales an image
        such that its pixels values are
        between 0 and 1 and its largest side is 512 pixels"""
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise TypeError("image must be a numpy.ndarray with shape (h, w, 3)")
        
        h, w, _ = image.shape
        if h > w:
            h_new = 512
            w_new = int(w * 512 / h)
        else:
            w_new = 512
            h_new = int(h * 512 / w)
        
        image = tf.image.resize(image, (h_new, w_new), method='bicubic')
        image = image / 255.0
        image = tf.expand_dims(image, axis=0)
        return image
    
    def load_model(self):
        """Load Moodel"""
        vgg = tf.keras.applications.VGG19(include_top=False,
                                          weights='imagenet')
        vgg.trainable = False
        
        x = vgg.input
        for layer in vgg.layers[1:]:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                x = tf.keras.layers.AveragePooling2D(
                    pool_size=layer.pool_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    name=layer.name
                )(x)
            else:
                x = layer(x)
        
        new_vgg = tf.keras.models.Model(inputs=vgg.input, outputs=x)
        
        style_outputs = [new_vgg.get_layer(name).output for name in self.style_layers]
        content_output = new_vgg.get_layer(self.content_layer).output
        model_outputs = style_outputs + [content_output]
        self.model = tf.keras.models.Model(inputs=new_vgg.input, outputs=model_outputs)
