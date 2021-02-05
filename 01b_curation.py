#%% Curation

#%%
import json
import logging
import os
import pickle
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
import spikeinterface as si
import spikeinterface.comparison as sc
import spikeinterface.extractors as se
import spikeinterface.sorters as sorters
import spikeinterface.toolkit as st
import spikeinterface.widgets as sw
from tqdm import tqdm

import setting
import SnakeIOHelper
from PostSorting.make_plots import plot_waveforms
from PreClustering.pre_process_ephys_data import filterRecording
import time 


#%% define input and output
(sinput, soutput) = SnakeIOHelper.getSnake(locals(), 'workflow_vr.smk', [setting.debug_folder+'/processed/mountainsort4/sorter_curated_df.pkl'],
    'curate_clusters')
#%%

'''
Units that had a firing rate > 0.5 Hz, isolation > 0.9, noise overlap < 0.05, 
and peak signal to noise ratio > 1 were accepted for further analysis. 

'''

sorter_df = pd.read_pickle(sinput.sorter_df)
print(f'Total cells before curation: {len(sorter_df)}')

sorter_df['pass_curation'] = ((sorter_df['snr']>5) & 
    # (sorter_df['firing_rate'] > 0.5) &
    ((1-sorter_df['nn_miss_rate']) < 0.1) & # isolation is similar to 1-miss rate
    (sorter_df['noise_overlap'] <0.15) &
    (sorter_df['isi_violation'] <0.1))

#print the origninal spike metrics
print(sorter_df.loc[:,['firing_rate','isi_violation','noise_overlap','snr','nn_miss_rate','pass_curation']])

curated_sorter_df = sorter_df[sorter_df['pass_curation']]
curated_sorter_df.to_pickle(soutput.sorter_curated_df)


#%% Plot spike waveforms
recording = se.load_extractor_from_pickle(sinput.recording_info)

tetrodeNum = np.array(recording.get_channel_ids())//setting.num_tetrodes

plot_waveforms(curated_sorter_df, tetrodeNum, soutput.waveform_figure)

# %%
