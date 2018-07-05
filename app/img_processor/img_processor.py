#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc


class ImgProcessor:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def extract_and_adjust_faces(self, image):
        """
        Method for extracting all faces from given image, and adjusting them (aligning)

        :param image: Image to be parsed
        :return: list of new extracted and adjusted (aligned) face images
        """

        pass

    @abc.abstractmethod
    def mark_up_faces(self, image):
        """
        Method should modify image object, marking up all of the faces that are found on
        the image, and writing up found data on them

        :param image: Image to be marked up
        :return: image
        """

        pass
