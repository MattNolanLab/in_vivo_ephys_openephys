from tests.regression.regression_utils import *
from pathlib import Path
import subprocess
import pandas as pd 
import os 
from pytest import fixture
import numpy as np 

DATA_FOLDER = 'M1_D31_2018-11-01_12-28-25_short'

@fixture
def old_df_path():
    return Path('tests/regression/output/vr/MountainSort/DataFrames/spatial_firing.pkl')

@fixture
def new_df_path():
    return Path(f'/home/ubuntu/testdata/{DATA_FOLDER}/processed/spatial_firing_vr.pkl')

def setup_module(module):
    # Setup up
    print(f'Current directory is {os.getcwd()}')

    old_path = 'tests/regression/output/vr/MountainSort/DataFrames/spatial_firing.pkl'

    # Check if the output file already exists, if not, execute the old pipeline
    if not Path(old_path).exists():
        subprocess.check_call('tests/regression/vr/run_old_pipeline.sh',shell=True)


    # Prepare the file to run the new pipeline
    # file will be saved to /home/ubuntu/testdata
    #TODO: allow one to specify the recording folder
    df_old = pd.read_pickle(old_path)
    prepare_files(DATA_FOLDER, df_old,'VR')

    # Run the new pipeline
    subprocess.check_call(f'tar xvf /home/ubuntu/testdata/M1_D31_short.tar.gz -C /home/ubuntu/testdata',shell=True)
    subprocess.check_call(f'./runSnake.py /home/ubuntu/testdata/{DATA_FOLDER}',shell=True)

def getPosTuningCurve(sr):
    fr = sr[0]
    trial = sr[1].astype(int)
    fr_mean = fr.reshape(trial.max(),-1)
    return fr_mean.mean(axis=0)

def test_pos_tuning_curve(new_df_path, old_df_path):
    # compare to see if the position tuning curves are similar
    df_new = pd.read_pickle(new_df_path)
    df_old = pd.read_pickle(old_df_path)

    df_new = df_new.reset_index()
    df_old = df_old.reset_index()

    for _,row in df_old.iterrows():
        # find the matching cluster
        df_curCluster = df_new[df_new.cluster_id == row.cluster_id]

        c_new = getPosTuningCurve(df_curCluster.iloc[0].spike_rate_on_trials_smoothed)
        c_old  = getPosTuningCurve(row.spike_rate_on_trials_smoothed)

        np.allclose(c_new,c_old,atol=30) # unit of firing rate is in 0.1Hz, e.g. 30 equal to 3Hz

