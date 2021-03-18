#%%
import pandas as pd 
import sys
sys.path.append('/home/ubuntu/in_vivo_ephys_openephys/')
import setting
import subprocess
from pathlib import Path
# %%
df_old = pd.read_pickle('/home/ubuntu/in_vivo_ephys_openephys/tests/regression/output/openfield/MountainSort/DataFrames/spatial_firing.pkl')
# %%
df_new = pd.read_pickle('/home/ubuntu/to_sort/recordings/M2_D32_2021-02-23_16-31-26/processed/mountainsort4/sorter_df.pkl')
# %%

# convert the old dataframe to the new dataframe
def convertOld2New(df_old,df_new):
    df_new = pd.DataFrame()
    df_new['cluster_id'] = df_old['cluster_id']
    df_new['session_id'] = df_old['session_id']
    df_new['sampling_frequency'] = setting.sampling_rate
    df_new['spike_train'] = df_old['firing_times']
    df_new['num_of_spikes'] = df_old['number_of_spikes']
    df_new['noise_overlap'] = df_old['noise_overlap']
    df_new['mean_firing_rate'] = df_old['mean_firing_rate']
    df_new['max_channel'] = df_old['primary_channel']
    df_new['firing_rate'] = df_old['mean_firing_rate']
    return df_new

convertOld2New(df_old,df_new)

# %%
def prepare_files(path):
    # prepare the folder for testing

    #extract files
    subprocess.check_call(f'tar xvf ~/testdata/{path}.tar.gz -C ~/testdata')

    #save the pre-sorted results
    df_curated = convertOld2New(df_old,df_new)
    df_curated.to_pickle(f'~/testdata/{path}')


