#!/usr/bin/env python3
'''
crop the image
'''

import tensorflow as tf


def crop_image(image, size):
    '''
    image - 3D tf.Tensor containing the image to crop
    size - a tuple containing the size of the crop
    '''
    return tf.image.random_crop(image, size=size)
