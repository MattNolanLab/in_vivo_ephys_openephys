import unittest
import pickle
from pathlib import Path
import pandas as pd
import os
from tests.integration.utils import set_up, tear_down
import pytest


DATA_URL = 'https://in-vivo-data.s3.eu-west-2.amazonaws.com/M1_D31.tar.gz'
test_data_name = 'M1_D31_2018-11-01_12-28-25_sorted'
recording_folder = 'tests/data'

def setUpModule():
    set_up(recording_folder, DATA_URL, test_data_name)

def test_position_files():
    df_pos = pd.read_pickle(Path(recording_folder)/Path(test_data_name)/'processed'/'processed_position.pkl')
    assert len(df_pos) == 45     # check the number of trial
    df_pos.rewarded.sum() == 32 #check the number of rewarded trial


def test_spatial_firing():
    df_spatial = pd.read_pickle(Path(recording_folder)/Path(test_data_name)/'processed'/'spatial_firing_vr.pkl')
    fr = df_spatial.iloc[0].beaconed_firing_rate_map_sem.mean()  
    assert pytest.approx(2.9,0.1) == fr #check the firing rate

def test_ramp_score():
    df_ramp = pd.read_pickle(Path(recording_folder)/Path(test_data_name)/'processed'/'ramp_score.pkl')
    assert pytest.approx(0.88,0.05) == df_ramp.score.max() #check the max ramp score

def tearDownModule():
    tear_down(recording_folder)