#%%
import pandas as pd 
import sys
sys.path.append('/home/ubuntu/in_vivo_ephys_openephys/')
import setting
import subprocess
from pathlib import Path
import yaml
import pickle 
import glob
import os
import shutil
# %%
# df_old = pd.read_pickle('/home/ubuntu/in_vivo_ephys_openephys/tests/regression/output/openfield/MountainSort/DataFrames/spatial_firing.pkl')
# # %%
# df_new = pd.read_pickle('/home/ubuntu/to_sort/recordings/M2_D32_2021-02-23_16-31-26/processed/mountainsort4/sorter_df.pkl')

#%%
# with open('/home/ubuntu/to_sort/recordings/M2_D32_2021-02-23_16-31-26/processed/recording_info.pkl','rb') as f:
#     info = pickle.load(f)
# %%

# convert the old dataframe to the new dataframe
def convertOld2New(df_old):
    df_new = pd.DataFrame()
    df_new['cluster_id'] = df_old['cluster_id']
    df_new['session_id'] = df_old['session_id']
    df_new['sampling_frequency'] = setting.sampling_rate
    df_new['spike_train'] = df_old['firing_times']
    df_new['number_of_spikes'] = df_old['number_of_spikes']
    df_new['noise_overlap'] = df_old['noise_overlap']
    df_new['mean_firing_rate'] = df_old['mean_firing_rate']
    df_new['max_channel'] = df_old['primary_channel']
    df_new['firing_rate'] = df_old['mean_firing_rate']
    return df_new

df_new = convertOld2New(df_old)

# %%


def make_param_file(recording):
    p = {'expt_type':'openfield'}
    with open(f'/home/ubuntu/testdata/{recording}/parameters.yaml','w') as f:
        yaml.dump(p,f)

def prepare_files(recording, df_old):

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
    make_param_file(recording)

    # waveform folders
    Path(f'/home/ubuntu/testdata/{recording}/processed/mountainsort4/waveform/all').mkdir(parents=True,exist_ok=True)
    Path(f'/home/ubuntu/testdata/{recording}/processed/mountainsort4/waveform/curated').mkdir(parents=True, exist_ok=True)


    print(glob.glob(f'/home/ubuntu/testdata/{recording}/processed/*'))

df_old = pd.read_pickle('/home/ubuntu/in_vivo_ephys_openephys/tests/regression/output/openfield/MountainSort/DataFrames/spatial_firing.pkl')
prepare_files('M5_2018-03-06_15-34-44_of',df_old)


# %% Compare output
recording = 'M5_2018-03-06_15-34-44_of'
df_old = pd.read_pickle('/home/ubuntu/in_vivo_ephys_openephys/tests/regression/output/openfield/MountainSort/DataFrames/spatial_firing.pkl')
df_new = pd.read_pickle(f'/home/ubuntu/testdata/{recording}/processed/spatial_firing_of.pkl')
df_new = df_new.reset_index()
df_old = df_old.reset_index()
# %%
property2check = ['speed_score','hd_score','grid_score']

for p in property2check:
    print(p)
    pd.testing.assert_series_equal(df_old[p],df_new[p], check_index_type=False,atol=0.05)
# %%
