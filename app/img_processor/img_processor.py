#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc


class ImgProcessor:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def process_image(self, image):
        pass
