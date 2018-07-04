#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dlib
import cv2

import img_processor
import utils.path
import utils.dlib


# TODO: Something not right, it is too slow, it blocks video feed
class CnnImgProcessor(img_processor.ImgProcessor):

    def __init__(self):
        """
        Class constructor

        It initializes cascade classifiers which are used for parsing images (frames), and extracting faces,
        and appropriate colors for each used classifier (as two arrays)
        """

        self._detector = dlib.cnn_face_detection_model_v1('resources/HOG_predictors/mmod_human_face_detector.dat')
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
            shapes.append(utils.dlib.shape_to_np(self._predictor(gray, face_rect.rect)))

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
            (x, y, w, h) = utils.dlib.rect_to_bb(face_rect.rect)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # show the face number
            cv2.putText(frame, "Face #{}".format(index + 1), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # render face landmarks
            for (x, y) in shapes[index]:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        cv2.imshow('Video', frame)

    def process_image(self, image):
        """
        Override from ImgProcessor class, method for handling one image

        :param image: image to be parsed
        """

        found_faces, shapes = self._parse_frame(image)
        self._draw_rectangles(image, found_faces, shapes)
