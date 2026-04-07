#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Convolve Grayscale Padding"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding
    images_padded = np.pad(
        images,
        pad_width=((0, 0), (ph, ph), (pw, pw)),
        mode='constant',
        constant_values=0
    )
    out_h = h + 2 * ph - kh + 1
    out_w = w + 2 * pw - kw + 1
    output = np.zeros((m, out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            image_patch = images_padded[:, i:i + kh, j:j + kw]
            output[:, i, j] = np.sum(image_patch * kernel, axis=(1, 2))
    return output
