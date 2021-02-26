from pathlib import Path
import pandas as pd 
import pytest
from PostSorting.vr_sync_spatial_data import *

@pytest.fixture
def spatial_firing():
    base_path = Path('tests/regression/output/vr/MountainSort/DataFrames/')
    spatial_firing = pd.read_pickle(base_path/'DataFrames'/ 'spatial_firing.pkl')
    return spatial_firing


def setup_module():
    # create the necessary files for running the new pipeline from the old pipeline
    base_path = Path('tests/regression/output/vr/MountainSort/DataFrames/')
    spatial_firing = pd.read_pickle(base_path/ 'spatial_firing.pkl')
    processing_position_data = pd.read_pickle(base_path/'processed_position_data.pkl')
    print(spatial_firing) 


def test():
    pass