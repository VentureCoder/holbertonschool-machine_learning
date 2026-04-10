#!/usr/bin/env python3
'''
rotate the image
'''

import tensorflow as tf


def rotate_image(image):
    '''
    image - 3D tf.tensor containing the image to rotate
    '''
    return tf.image.rot90(image)
