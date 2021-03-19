import unittest
import pickle
from pathlib import Path
import sys
print(sys.path)
from tests.integration.utils import set_up, tear_down


DATA_URL = 'https://integrationtestdata.blob.core.windows.net/public-integration-test-data/M5_2018-03-06_15-34-44_of.tar.gz.gpg'
DATA_PASSWORD = 'dcce3c6b53bc8eeb1e73956841f70b9b63435e09101a5dbc7da5f91428e946fe'
SORTING_PATH = '/home/nolanlab/to_sort/'
OUTPUT_PATH = '/ActiveProjects/Klara/Open_field_opto_tagging_p038/M5_2018-03-06_15-34-44_of/'


def setUpModule() -> None:
    """Unittest helper function that is called automatically before tests in this file start"""
    set_up(OUTPUT_PATH, SORTING_PATH, DATA_URL, DATA_PASSWORD)


def tearDownModule() -> None:
    """Unittest helper function that is called automatically after all tests in this file have finished"""
    tear_down(OUTPUT_PATH)


class TestDataFrameSpatialFiring(unittest.TestCase):
    """Tests for the spatial_firing.pkl data frame"""

    def setUp(self) -> None:
        df_path = Path(OUTPUT_PATH) / 'MountainSort/DataFrames/spatial_firing.pkl'
        self.df = pickle.load(open(df_path, 'rb'))

    def test_session_id(self) -> None:
        """Test that the correct session ID has been stored"""
        self.assertTrue(
            all(self.df['session_id'] == 'M5_2018-03-06_15-34-44_of')
        )
