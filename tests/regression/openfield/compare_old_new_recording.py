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
from tests.regression.regression_utils import *
# %%
df_old = pd.read_pickle('/home/ubuntu/in_vivo_ephys_openephys/tests/regression/output/openfield/MountainSort/DataFrames/spatial_firing.pkl')
# # %%
# df_new = pd.read_pickle('/home/ubuntu/to_sort/recordings/M2_D32_2021-02-23_16-31-26/processed/mountainsort4/sorter_df.pkl')

#%%
# with open('/home/ubuntu/to_sort/recordings/M2_D32_2021-02-23_16-31-26/processed/recording_info.pkl','rb') as f:
#     info = pickle.load(f)
# %%


df_new = convertOld2New(df_old)

df_old = pd.read_pickle('/home/ubuntu/in_vivo_ephys_openephys/tests/regression/output/openfield/MountainSort/DataFrames/spatial_firing.pkl')
prepare_files('M5_2018-03-06_15-34-44_of',df_old,'openfield')


# %% Compare output
recording = 'M5_2018-03-06_15-34-44_of'
df_old = pd.read_pickle('/home/ubuntu/in_vivo_ephys_openephys/tests/regression/output/openfield/MountainSort/DataFrames/spatial_firing.pkl')
df_new = pd.read_pickle(f'/home/ubuntu/testdata/{recording}/processed/spatial_firing_of.pkl')
df_new = df_new.reset_index()
df_old = df_old.reset_index()
# %%
property2check = ['speed_score','hd_score','grid_score']

for p in property2check:
    pd.testing.assert_series_equal(df_old[p],df_new[p], check_index_type=False,atol=0.05)
    print(f'{p} checked')
# %%
