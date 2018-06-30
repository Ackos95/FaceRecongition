#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import utils.path


class VideoParser(object):
    """
    Class VideoParser

    It contains public method for capturing machines video feed, and starting up infinite processing loop
    """

    def __init__(self):
        """
        Class constructor

        It initializes cascade classifiers which are used for parsing images (frames), and extracting faces,
        and appropriate colors for each used classifier (as two arrays)
        """

        self._face_cascades = [
            cv2.CascadeClassifier(utils.path.in_root_path('./resources/haarcascades/haarcascade_frontalface_alt.xml')),
            cv2.CascadeClassifier(utils.path.in_root_path('./resources/haarcascades/haarcascade_profileface.xml'))
        ]

        self._face_colors = [
            (0, 0, 255),
            (0, 255, 0)
        ]

    def _parse_frame(self, frame):
        """
        Private function for extracting faces from given frame (image)

        It goes through defined cascade classifiers (defined in constructor), and with each
        tries to detect objects (faces) on given image.

        :param frame: One video feed frame (image)

        :return: Array with find results per each cascade classifier (array of arrays)
        """

        ret_values = [[] for _ in xrange(len(self._face_cascades))]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for index, cascade in enumerate(self._face_cascades):
            ret_values[index] = cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(20, 20)
            )

        return ret_values

    def _draw_rectangles(self, frame, found_faces):
        """
        Private function for drawing up rectangles around found objects on frame (image)

        It goes through found objects (faces), per classifier, and draws up rectangle around each
        in color defined for that classifier.

        :param frame: One video feed frame (image), on which to draw rectangle(s)
        :param found_faces: Array with find results per each cascade classifier (array of arrays)
        """

        for index, result_set in enumerate(found_faces):
            for (x, y, w, h) in result_set:
                cv2.rectangle(frame, (x, y), (x + w, y + h), self._face_colors[index], 2)
                cv2.putText(frame, 'Unknown', (x + 5, y + h - 5), cv2.QT_FONT_NORMAL, 1, self._face_colors[index], 1, cv2.LINE_AA)

        cv2.imshow('Video', frame)

    def start_parsing(self):
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

            found_faces = self._parse_frame(frame)
            self._draw_rectangles(frame, found_faces)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()
