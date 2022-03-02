import unittest
import pickle
from pathlib import Path

from tests.integration.utils import set_up, tear_down


DATA_URL = 'https://in-vivo-data.s3.eu-west-2.amazonaws.com/M1_D31_short.tar.gz.gpg'
DATA_PASSWORD = 'dcce3c6b53bc8eeb1e73956841f70b9b63435e09101a5dbc7da5f91428e946fe'
SORTING_PATH = '/home/nolanlab/to_sort/'
OUTPUT_PATH = '/ActiveProjects/Klara/VR/M1_D30/'


def setUpModule() -> None:
    """Unittest helper function that is called automatically before tests in this file start"""
    set_up(OUTPUT_PATH, SORTING_PATH, DATA_URL, DATA_PASSWORD)


def tearDownModule() -> None:
    """Unittest helper function that is called automatically after all tests in this file have finished"""
    tear_down(OUTPUT_PATH)

