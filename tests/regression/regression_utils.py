import pandas as pd 
import sys
sys.path.append('/home/ubuntu/in_vivo_ephys_openephys/')
import settings
import subprocess
from pathlib import Path
import yaml
import pickle 
import glob
import os
import shutil

def convertOld2New(df_old):
    df_new = pd.DataFrame()
    df_new['cluster_id'] = df_old['cluster_id']
    df_new['session_id'] = df_old['session_id']
    df_new['sampling_frequency'] = settings.sampling_rate
    df_new['spike_train'] = df_old['firing_times']
    df_new['number_of_spikes'] = df_old['number_of_spikes']
    df_new['noise_overlap'] = df_old['noise_overlap']
    df_new['mean_firing_rate'] = df_old['mean_firing_rate']
    df_new['max_channel'] = df_old['primary_channel']
    df_new['firing_rate'] = df_old['mean_firing_rate']
    return df_new


def make_param_file(recording,expt_type):
    p = {'expt_type':expt_type}
    with open(f'/home/ubuntu/testdata/{recording}/parameters.yaml','w') as f:
        yaml.dump(p,f)

def prepare_files(recording, df_old, expt_type):

    # clean the folder
    shutil.rmtree(f'/home/ubuntu/testdata/{recording}/processed',ignore_errors=True)
    # prepare the folder for testing

    #extract files
    # subprocess.check_call(f'tar xvf ~/testdata/{recording}.tar.gz -C ~/testdata',shell=True)

    #save the pre-sorted results
    df_curated = convertOld2New(df_old)
    if not Path(f'/home/ubuntu/testdata/{recording}/processed/mountainsort4').exists():
        Path(f'/home/ubuntu/testdata/{recording}/processed/mountainsort4').mkdir(parents=True)
        print('New directory created')

        
    with open(f'/home/ubuntu/testdata/{recording}/processed/recording_info.pkl','wb') as f:
        pickle.dump({'module','empty'},f)

    # the sequence of creation must be correct
    df_curated.to_pickle(f'/home/ubuntu/testdata/{recording}/processed/mountainsort4/sorter_df.pkl')
    df_curated.to_pickle(f'/home/ubuntu/testdata/{recording}/processed/mountainsort4/sorter.pkl')
    df_curated.to_pickle(f'/home/ubuntu/testdata/{recording}/processed/mountainsort4/sorter_curated_df.pkl')


    # prepare some files
    make_param_file(recording,expt_type)

    # waveform folders
    Path(f'/home/ubuntu/testdata/{recording}/processed/mountainsort4/waveform/all').mkdir(parents=True,exist_ok=True)
    Path(f'/home/ubuntu/testdata/{recording}/processed/mountainsort4/waveform/curated').mkdir(parents=True, exist_ok=True)


    print(glob.glob(f'/home/ubuntu/testdata/{recording}/processed/*'))