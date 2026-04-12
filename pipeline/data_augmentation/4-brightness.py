#!/usr/bin/env python3
'''
change brightness
'''

import tensorflow as tf


def change_brightness(image, max_delta):
    '''
    image - 3D tf.Tensor containing the image to change
    max_delta - the maximum amount of the image should be brightened
    '''
    return tf.image.random_brightness(image, max_delta)
