#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def rect_to_bb(rect):
    """
    Helper function for transforming `rect` object (dlib) into, bounding box (x, y, w, h),
    tuple for `cv`

    :param rect: (dlib) object

    :return: (x, y, w, h) tuple
    """

    return (
        rect.left(),
        rect.top(),
        rect.right() - rect.left(),
        rect.bottom() - rect.top(),
    )


def shape_to_np(shape):
    """
    Helper function for transforming found shape (shapes) into numpy array of coordinates

    :param shape: (dlib) object

    :return: numpy array (68x2)
    """

    coords = np.zeros((68, 2), dtype='int')
    for i in xrange(68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords

