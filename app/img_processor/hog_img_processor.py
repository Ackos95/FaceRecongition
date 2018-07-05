#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dlib
import cv2
import os.path
import pickle

import img_processor
import utils.path
import utils.dlib


class HogImgProcessor(img_processor.ImgProcessor):

    def __init__(self):
        """
        Class constructor

        It initializes cascade classifiers which are used for parsing images (frames), and extracting faces,
        and appropriate colors for each used classifier (as two arrays)
        """

        self._detector = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor(utils.path.in_root_path('resources/HOG_predictors/shape_predictor_68_face_landmarks.dat'))
        self._recognizer = None
        self._recognizer_labels = {}

        if os.path.exists(utils.path.in_root_path('resources/generated/recognitions.yml')):
            self._recognizer = cv2.face.LBPHFaceRecognizer_create()
            self._recognizer.read(utils.path.in_root_path('resources/generated/recognitions.yml'))

        if os.path.exists(utils.path.in_root_path('resources/generated/labels')):
            with open(utils.path.in_root_path('resources/generated/labels'), 'rb') as f:
                self._recognizer_labels = pickle.load(f)

    def _find_faces(self, image):
        """
        Private function for extracting faces from given frame (image)

        It goes through defined cascade classifiers (defined in constructor), and with each
        tries to detect objects (faces) on given image.

        :param image: One video feed frame (image)

        :return: Array with find results per each cascade classifier (array of arrays)
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found_faces = self._detector(gray, 1)
        shapes = []

        for face_rect in found_faces:
            shapes.append(utils.dlib.shape_to_np(self._predictor(gray, face_rect)))

        return found_faces, shapes

    def _draw_rectangles(self, image, found_faces, shapes):
        """
        Private function for drawing up rectangles around found objects on frame (image)

        It goes through found objects (faces), per classifier, and draws up rectangle around each
        in color defined for that classifier.

        :param image: One video feed frame (image), on which to draw rectangle(s)
        :param found_faces: Array with find results per each cascade classifier (array of arrays)
        :param shapes: Array with found face landmarks (array of arrays)
        """

        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        for index, face_rect in enumerate(found_faces):
            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box
            x, y, w, h = utils.dlib.rect_to_bb(face_rect)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            recognized = self._try_to_recognize(grey[y:y + h, x:x + w])

            # show the face name
            cv2.putText(image, recognized, (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # render face landmarks
            for (x, y) in shapes[index]:
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

        return image

    def _try_to_recognize(self, image):
        """
        Helper method for recognizing face(s) on given image

        It uses `self._recognizer` if it is defined, otherwise just returns 'Unrecognized', after
        recognizer has found a match, it looks through category labels (loaded from resources), and
        gives us concrete name

        :param image: Face image to be detected
        :return: Match name for given face
        """

        recognized = 'Unrecognized'
        if image is not None and image.shape[1] != 0 and self._recognizer is not None:
            image = HogImgProcessor._resize_face(image)
            id_, conf = self._recognizer.predict(image)
            person_name = str(id_)
            for name in self._recognizer_labels.keys():
                person_name = name if self._recognizer_labels[name] == id_ else person_name

            # if conf < 50:
            recognized = "{}-{}".format(person_name, conf)

        return recognized

    @staticmethod
    def _resize_face(image, size_=32):
        """
        Helper method for resizing extracted faces to fixed size (32x32)

        :param image: Face image to be resized
        :param size_: Fixed size on which to be resized (default 32)

        :return: Resized image
        """

        if image is None or image.shape[1] == 0:
            return image

        # resize faces to fixed size
        dim = (size_, size_)
        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    def extract_and_adjust_faces(self, image):
        """
        ImgProcessor override, it extracts all faces from image, and runs adjustments methods on it
        (grey color, resize to fixed size)

        :param image: Image to be parsed
        :return: List of adjusted faces for further manipulations
        """

        found_faces, shapes = self._find_faces(image)
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # create list of face images from gray
        ret_list = []
        for face_rect in found_faces:
            x, y, w, h = utils.dlib.rect_to_bb(face_rect)
            grey_face = HogImgProcessor._resize_face(grey[y:y + h, x:x + w])

            ret_list.append(grey_face)

        return ret_list

    def mark_up_faces(self, image):
        """
        ImgProcessor override, modifies image (marks up all found faces with information)

        :param image: Image to be marked up
        :return: marked up image
        """

        found_faces, shapes = self._find_faces(image)
        return self._draw_rectangles(image, found_faces, shapes)
