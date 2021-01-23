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

import file_utility
import Logger
import setting
import SnakeIOHelper
import spikeinterfaceHelper
from PostSorting.make_plots import plot_waveforms
from PreClustering.pre_process_ephys_data import filterRecording
import time 

logger = logging.getLogger(os.path.basename(__file__)+':'+__name__)

#%% define input and output
(sinput, soutput) = SnakeIOHelper.getSnake(locals(), 'workflow_vr.smk', [setting.debug_folder+'/processed/mountainsort4/sorter_curated_df.pkl'],
    'sort_spikes')

#%% Load data and create recording extractor


signal = file_utility.load_OpenEphysRecording(sinput.recording_to_sort)
geom = pd.read_csv(sinput.tetrode_geom,header=None).values

dead_channel_path =  Path(sinput.recording_to_sort+'/dead_channels.txt')
if dead_channel_path.exists():
    bad_channel = file_utility.getDeadChannel(dead_channel_path)
else:
    bad_channel = []


#%% Remove bad channel and filter the recording
logger.info('Filtering files') #TODO logging not show correctly
Fs = 30000
# recording = se.NumpyRecordingExtractor(signal[:,:Fs*180],setting.sampling_rate,geom)
recording = se.NumpyRecordingExtractor(signal,setting.sampling_rate,geom)

recording = recording.load_probe_file(sinput.probe_file) #load probe definition
recording = st.preprocessing.remove_bad_channels(recording, bad_channel_ids=bad_channel) #remove bad channel
recording = st.preprocessing.bandpass_filter(recording, freq_min=300, freq_max=6000, cache_chunks=True) #need to cache, otherwise the extracting waveform will take along time
recording = st.preprocessing.whiten(recording, seed=0) #must do whitening first, otherwise the waveforms used to calculate cluster metrics will be very noisy
recording = se.CacheRecordingExtractor(recording, save_path=sinput.recording_to_sort+'/processed_data.dat') # cache recording for speedup

tetrodeNum = np.array(recording.get_channel_ids())//setting.num_tetrodes

#%% perform sorting

now = time.time()
with open(sinput.sort_param) as f:
    param = json.load(f)


ms4_params = sorters.get_default_params('mountainsort4')
ms4_params['filter'] = False #have already done this in preprocessing step
ms4_params['whiten'] = False 

sorting_ms4 = sorters.run_mountainsort4(recording=recording,output_folder='sorting_tmp',
    verbose=True, **ms4_params,grouping_property='group', parallel=True)

print('Saving sorting results')
with open(soutput.sorter,'wb') as f:
    pickle.dump(sorting_ms4,f)

#%% Compute quality metrics so that we can re-curate the neurons later
print('Calculating quality metrics...')

start = time.time()
quality_metrics = st.validation.compute_quality_metrics(sorting_ms4, recording,
    max_spikes_per_unit_for_snr = 200, memmap= False,
     metric_names=['snr','isi_violation', 'noise_overlap','nn_miss_rate','firing_rate'],
  recompute_info=True)
print(f'Calculate quality metrics took {time.time()-start}')
 #need to compute metrics before getting the waveforms
    
#%% compute some property of the sorting
print('Computing sorting metrics...')

start = time.time()
st.postprocessing.get_unit_max_channels(recording, sorting_ms4, grouping_property='group',
     save_as_property=True,max_spikes_per_unit=100, seed=0)
print(f'get_unit_max_channels took {time.time()-start}')

start = time.time()
st.postprocessing.get_unit_waveforms(recording, sorting_ms4,save_as_features=True, 
    max_spikes_per_unit=100, memmap=False, seed = 0,recompute_info=True) # disable memmap for speed
print(f'Extracting waveform took {time.time()-start}')


for id in sorting_ms4.get_unit_ids():
    number_of_spikes = len(sorting_ms4.get_unit_spike_train(id))
    mean_firing_rate = number_of_spikes/(recording.get_traces().shape[1]/setting.sampling_rate)
    sorting_ms4.set_unit_property(id,'number_of_spikes',number_of_spikes)
    sorting_ms4.set_unit_property(id, 'mean_firing_rate', mean_firing_rate)


#%% save data

sorting_ms4.dump_to_pickle(soutput.sorter)
session_id = sinput.recording_to_sort.split('/')[-1]
sorter_df=spikeinterfaceHelper.sorter2dataframe(sorting_ms4,session_id)
sorter_df.to_pickle(soutput.sorter_df)



#%% Curation

'''
Units that had a firing rate > 0.5 Hz, isolation > 0.9, noise overlap < 0.05, 
and peak signal to noise ratio > 1 were accepted for further analysis. 

'''

sorter_df['pass_curation'] = ((sorter_df['snr']>2) & 
    (sorter_df['firing_rate'] > 0.5) &
    ((1-sorter_df['nn_miss_rate']) > 0.9) & # isolation is similar to 1-miss rate
    (sorter_df['noise_overlap'] <0.10) &
    (sorter_df['isi_violation'] <0.9))

#print the origninal spike metrics
print(sorter_df.loc[:,['firing_rate','isi_violation','noise_overlap','snr','nn_miss_rate','pass_curation']])

# TODO: implement metrics from mountainsort

curated_sorter_df = sorter_df[sorter_df['pass_curation']]

print(curated_sorter_df.cluster_id)

#%% save curated results
curated_sorter_df.to_pickle(soutput.sorter_curated_df)

#%% Plot spike waveforms
plot_waveforms(curated_sorter_df, tetrodeNum, soutput.waveform_figure)

print(f'Elapsed time {time.time()-now}')

# %%
