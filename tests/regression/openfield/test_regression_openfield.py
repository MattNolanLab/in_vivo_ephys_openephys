from tests.regression.regression_utils import *
from pathlib import Path
import subprocess
import pandas as pd 
import os 
from pytest import fixture


DATA_FOLDER = 'M5_2018-03-06_15-34-44_of'

@fixture
def old_df_path():
    return Path('tests/regression/output/openfield/MountainSort/DataFrames/spatial_firing.pkl')

@fixture
def new_df_path():
    return Path(f'/home/ubuntu/testdata/{DATA_FOLDER}/processed/spatial_firing_of.pkl')

def setup_module(module):
    # Setup up
    print(f'Current directory is {os.getcwd()}')

    # Check if the output file already exists, if not, execute the old pipeline
    if not Path('tests/regression/output/openfield/MountainSort/DataFrames/spatial_firing.pkl').exists():
        subprocess.check_call('tests/regression/openfield/run_old_pipeline.sh',shell=True)


    # Prepare the file to run the new pipeline
    # file will be saved to /home/ubuntu/testdata
    #TODO: allow one to specify the recording folder
    df_old = pd.read_pickle('/home/ubuntu/in_vivo_ephys_openephys/tests/regression/output/openfield/MountainSort/DataFrames/spatial_firing.pkl')
    prepare_files(DATA_FOLDER,df_old,'openfield')

    # Run the new pipeline
    subprocess.check_call(f'./runSnake.py /home/ubuntu/testdata/{DATA_FOLDER}',shell=True)


def test_metrics(new_df_path, old_df_path):
    # compare to see if the metrics matches
    df_new = pd.read_pickle(new_df_path)
    df_old = pd.read_pickle(old_df_path)

    df_new = df_new.reset_index()
    df_old = df_old.reset_index()

    property2check = ['speed_score','hd_score','grid_score']

    for p in property2check:
        pd.testing.assert_series_equal(df_old[p],df_new[p], check_index_type=False,atol=0.05)
        print(f'{p} checked')