#!/usr/bin/env python
# -*- coding: utf-8 -*-

from app.hog_scanner.VideoParser import VideoParser


if __name__ == '__main__':
    vp = VideoParser()
    vp.start_parsing()
