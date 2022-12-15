#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Fri  4 Apr 15:33:36 2014 CEST
#


import numpy as np

from bob.bio.face.color import gray_to_rgb, rgb_to_gray


def test_gray():

    # This test verifies that gray-scale conversion works in both directions, for both images and numbers

    gray_image = np.random.random((30, 30))
    color_image = gray_to_rgb(gray_image)

    new_gray_image = rgb_to_gray(color_image)
    assert np.allclose(gray_image, new_gray_image)
