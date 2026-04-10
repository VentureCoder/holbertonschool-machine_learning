#!/usr/bin/env python3
'''
changing hue
'''

import tensorflow as tf


def change_hue(image, delta):
    '''
    image - 3D tf.Tensor containing the image to change
    delta - the amount the hue should change
    '''
    return tf.image.adjust_hue(image, delta)
