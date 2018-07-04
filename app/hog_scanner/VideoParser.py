#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import dlib

import utils.path
import utils.dlib


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

        self._detector = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor(utils.path.in_root_path('resources/HOG_predictors/shape_predictor_68_face_landmarks.dat'))

    def _parse_frame(self, frame):
        """
        Private function for extracting faces from given frame (image)

        It goes through defined cascade classifiers (defined in constructor), and with each
        tries to detect objects (faces) on given image.

        :param frame: One video feed frame (image)

        :return: Array with find results per each cascade classifier (array of arrays)
        """

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found_faces = self._detector(gray, 1)
        shapes = []

        for face_rect in found_faces:
            shapes.append(utils.dlib.shape_to_np(self._predictor(gray, face_rect)))

        return found_faces, shapes

    def _draw_rectangles(self, frame, found_faces, shapes):
        """
        Private function for drawing up rectangles around found objects on frame (image)

        It goes through found objects (faces), per classifier, and draws up rectangle around each
        in color defined for that classifier.

        :param frame: One video feed frame (image), on which to draw rectangle(s)
        :param found_faces: Array with find results per each cascade classifier (array of arrays)
        """

        for index, face_rect in enumerate(found_faces):
            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box
            (x, y, w, h) = utils.dlib.rect_to_bb(face_rect)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # show the face number
            cv2.putText(frame, "Face #{}".format(index + 1), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # render face landmarks
            for (x, y) in shapes[index]:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

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

            found_faces, shapes = self._parse_frame(frame)
            self._draw_rectangles(frame, found_faces, shapes)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()