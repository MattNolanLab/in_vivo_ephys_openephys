
#%%
import Logger
import setting
from collections import namedtuple
import file_utility
import os
import pandas as pd

import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.toolkit as st
import spikeinterface.sorters as sorters
import spikeinterface.comparison as sc
import spikeinterface.widgets as sw
import json
import pickle

from scipy.signal import butter,filtfilt
from tqdm import tqdm
import numpy as np
from types import SimpleNamespace
#%% define input and output

if 'snakemake' not in locals():
    #Define some variable to run the script standalone
    input = SimpleNamespace()
    output = SimpleNamespace()
    
    input.recording_to_sort = 'testData/M1_D31_2018-11-01_12-28-25/'
    input.probe_file = 'sorting_files/tetrode_16.prb'
    input.sort_param = 'sorting_files/params.json'
    input.tetrode_geom = 'sorting_files/geom_all_tetrodes_original.csv'
    
    sorterPrefix = input.recording_to_sort+'processed/'+setting.sorterName
    output.firings = sorterPrefix+'/firings.mda'
    output.firings_curated = sorterPrefix + '/firings_curated.mda'
    output.cluster_metrics = sorterPrefix + '/cluster_metrics.pkl'
    output.sorter = sorterPrefix +'/sorter.pkl'
    output.sorter_curated = sorterPrefix +'/sorter_curated.pkl'
    output.spike_waveforms = sorterPrefix + '/spike_waveforms.pkl'
else:
    #in snakemake environment, the input and output will be provided by the workflow
    input = snakemake.input
    output = snakemake.output


#%% Load data and create recording extractor
signal = file_utility.load_OpenEphysRecording(input.recording_to_sort)
geom = pd.read_csv(input.tetrode_geom,header=None).values
bad_channel = file_utility.getDeadChannel(input.recording_to_sort+'dead_channels.txt')


#%% Create and filter the recording
recording = se.NumpyRecordingExtractor(signal,setting.sampling_rate,geom)
recording = recording.load_probe_file(input.probe_file) #load probe definition

def filterRecording(recording, sampling_freq, lp_freq=300,hp_freq=6000,order=3):
    fn = sampling_freq / 2.
    band = np.array([lp_freq, hp_freq]) / fn

    b, a = butter(order, band, btype='bandpass')

    if not (np.all(np.abs(np.roots(a)) < 1) and np.all(np.abs(np.roots(a)) < 1)):
        raise ValueError('Filter is not stable')
    
    for i in tqdm(range(recording._timeseries.shape[0])):
        recording._timeseries[i,:] = filtfilt(b,a,recording._timeseries[i,:])

    return recording


filterRecording(recording,setting.sampling_rate) #for faster operation later

#%% Remove some bad channels
recording = st.preprocessing.remove_bad_channels(recording, bad_channel_ids=bad_channel) #remove bad channel
# recording_waveform = st.preprocessing.bandpass_filter(recording, freq_min=300, freq_max=6000,chunk_size=int(setting.sampling_rate*0.02))
# recording_sort = st.preprocessing.bandpass_filter(recording, freq_min=300, freq_max=6000)


#%% perform sorting
param = json.load(open(input.sort_param))
sorting_ms4 = sorters.run_sorter(setting.sorterName,recording, output_folder=sorterPrefix,
    adjacency_radius=param['adjacency_radius'], detect_sign=param['detect_sign'])

#%%
sorting_ms4 = pickle.load(open(input.sorter,'rb'))

#%% compute some property of the sorting
st.postprocessing.get_unit_max_channels(recording, sorting_ms4, max_num_waveforms=100)
st.postprocessing.get_unit_waveforms(recording, sorting_ms4, max_num_waveforms=100)

for id in sorting_ms4.get_unit_ids():
    number_of_spikes = len(sorting_ms4.get_unit_spike_train(id))
    mean_firing_rate = number_of_spikes/(recording._timeseries.shape[1]/setting.sampling_rate)
    sorting_ms4.set_unit_property(id,'number_of_spikes',number_of_spikes)
    sorting_ms4.set_unit_property(id, 'mean_firing_rate', mean_firing_rate)

#%% save data
se.MdaSortingExtractor.write_sorting(sorting_ms4, output.firings)
pickle.dump(sorting_ms4,open(input.sorter,'wb'))

#%% Do some simple curation for now
sorting_ms4_curated = st.curation.threshold_snr(sorting=sorting_ms4, recording = recording,
  threshold =1.2, threshold_sign='less', max_snr_waveforms=100) #remove when less than threshold
print(sorting_ms4_curated.get_unit_ids())

sorting_ms4_curated=st.curation.threshold_firing_rate(sorting_ms4_curated,
    threshold=0.5, threshold_sign='less')
print(sorting_ms4_curated.get_unit_ids())

sorting_ms4_curated=st.curation.threshold_isi_violations(sorting_ms4_curated, threshold = 0.9)
print(sorting_ms4_curated.get_unit_ids())

sorting_ms4_curated = st.curation.threshold_firing_rate(sorting=sorting_ms4_curated,threshold=0.5,threshold_sign='less')
print(sorting_ms4_curated.get_unit_ids())

#%%
#save curated data
sorting_ms4_curated = se.SubSortingExtractor(sorting_ms4,unit_ids=sorting_ms4_curated.get_unit_ids())
pickle.dump(sorting_ms4_curated, open(output.sorter_curated,'wb'))
se.MdaSortingExtractor.write_sorting(sorting_ms4_curated, output.firings_curated)
