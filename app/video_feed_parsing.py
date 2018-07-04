#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2

import img_processor.haar_img_processor
import img_processor.hog_img_processor


IMG_PROCESSORS = {
    'hog': img_processor.hog_img_processor.HogImgProcessor(),
    'haar': img_processor.haar_img_processor.HaarImgProcessor()
}


def start_video_feed_parsing(img_processor='hog'):
    """
    Start method

    It captures video feed from machine on which it is started, and starts infinite loop.
    It reads frames from captured feed, and each of them passes through parse function (to extract
    faces), and draw function (to mark extracted faces).

    Loop is finished after user press 'q' button on keyboard. After finishing up loop, it releases
    video feed, and destroys all open-cv displays.
    """

    video_capture = cv2.VideoCapture(0)

    while True:
        _, frame = video_capture.read()

        IMG_PROCESSORS[img_processor].process_image(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
