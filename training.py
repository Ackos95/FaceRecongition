#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import pickle

import app.img_processor.hog_img_processor
import utils.path


def _preprocess_video():
    """
    Helper method for parsing given video(s) into multiple images.

    It takes all video files (`mp4` or `mov` formats) and each of theirs frame extracts and saves
    in same directory as an `.jpg` image

    Useful for training algorithm to recognize yourself (record small video, and extract all frames as images)
    """

    for root, dirs, files in os.walk(utils.path.in_root_path('resources/data/training')):
        for file_ in files:
            if file_.endswith('mp4') or file_.endswith('mov'):
                video_capture = cv2.VideoCapture(utils.path.in_root_path(os.path.join(root, file_)))
                current_frame = 0

                while True:
                    ret, frame = video_capture.read()

                    if frame is None:
                        break

                    name = utils.path.in_root_path(os.path.join(root, "{}.jpg".format(current_frame)))
                    cv2.imwrite(name, frame)
                    current_frame += 1

                video_capture.release()
                cv2.destroyAllWindows()


def _load_data(path):
    """
    Helper method for loading all images under root folder (given by path param), extracting information, and
    preparing arrays for nn training (or testing).

    It goes through folder it received, founds images under sub-folders (sub-folder name is name of person),
    extracts (adjusted) face images from each image, and appends that image (region of interest) with given
    person category (label_map) to x and y arrays.

    :param path: Root folder for loading data

    :return: {
        label_map: { person_name: class_id },
        x: np.array (of encoded images - regions of interest)
        y: np.array (of class_id's)
    }
    """

    img_processor = app.img_processor.hog_img_processor.HogImgProcessor()
    current_id = 0
    label_map = {}

    train_data = {
        'x': [],
        'y': [],
    }

    for root, dirs, files in os.walk(utils.path.in_root_path(path)):
        for file_ in files:
            if file_.endswith('png') or file_.endswith('jpg'):
                person_name = os.path.basename(root)

                if person_name not in label_map:
                    label_map[person_name] = current_id
                    print(person_name, current_id)
                    current_id += 1

                img = cv2.imread(os.path.join(root, file_))
                faces = img_processor.extract_and_adjust_faces(img)
                for face in faces:
                    train_data['x'].append(face)
                    train_data['y'].append(label_map[person_name])

    return label_map, np.array(train_data['x']), np.array(train_data['y'])


def _simple_test_recognizer(recognizer, label_map):
    """
    It loads data from test resources, and for each loaded image, it runs recognizer predict.

    For each prediction, it tries to map predicted key with actual key, and prints out results.

    :param recognizer: LBPHFaceRecongizer
    :param label_map: { person_name: class_id }
    """

    test_label_map, x_test, y_test = _load_data('resources/data/test')
    for i, x in enumerate(x_test):
        id_, conf = recognizer.predict(x)
        found = ''
        real = ''

        for key in test_label_map:
            found = key if test_label_map[key] == id_ else found
        for key in label_map:
            real = key if label_map[key] == y_test[i] else real

        if found == real:
            print('Testing match for: ', found)
        else:
            print('Testing mismmatch for: ', found, ' instead of ', real)


def train(preprocess_video=False):
    """
    Method for training recognizer.

    It preprocess video (if flag is set to true) firstly. After that it loads data from training
    resources, then creates LBPHFaceRecognizer (Local Binary Patterns Histogram), and trains it with
    loaded data.

    In the end results are saved in `resources//generated` folder

    :param preprocess_video: (Boolean flag) whether to run video preprocessing before start
    """

    if preprocess_video:
        _preprocess_video()

    # load data for training
    label_map, x_train, y_train = _load_data('resources/data/training')

    # create recognizer, and train it with loaded data
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(x_train, y_train)

    # save trained data
    recognizer.save(utils.path.in_root_path('resources/generated/recognitions.yml'))
    with open(utils.path.in_root_path('resources/generated/labels'), 'wb') as f:
        pickle.dump(label_map, f)

    # test recognizer
    _simple_test_recognizer(recognizer, label_map)


def test():
    """
    Method for testing, it loads up recognizer data from `generated` resources, and label's map, and
    runs `_simple_test_recognizer`
    """

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(utils.path.in_root_path('resources/generated/recognitions.yml'))
    with open(utils.path.in_root_path('resources/generated/labels'), 'rb') as f:
        label_map = pickle.load(f)

    _simple_test_recognizer(recognizer, label_map)


if __name__ == '__main__':
    # train()
    test()
