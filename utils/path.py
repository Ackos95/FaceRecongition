#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path


def in_root_path(path):
    """
    Helper method for retrieving full path relative to root folder.

    :param path: Relative path (from root folder)
    :return: Absolute path in filesystem
    """

    return os.path.join(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'),  path)
