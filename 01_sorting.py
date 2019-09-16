
#%%
import pipieline_manager 
from PreClustering import pre_process_ephys_data
import Logger
import setting
import sys
import parameters
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
#%% define input and output

if 'snakemake' not in locals():
    #Define some variable to run the script standalone
    input = namedtuple
    output = namedtuple
    
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

#%%
# recording = st.preprocessing.remove_bad_channels(recording, bad_channel_ids=bad_channel) #remove bad channel
# recording_waveform = st.preprocessing.bandpass_filter(recording, freq_min=300, freq_max=6000,chunk_size=int(setting.sampling_rate*0.02))
# recording_sort = st.preprocessing.bandpass_filter(recording, freq_min=300, freq_max=6000)


#%% perform sorting
param = json.load(open(input.sort_param))
sorting_ms4 = sorters.run_sorter(setting.sorterName,recording, output_folder=sorterPrefix,
    adjacency_radius=param['adjacency_radius'], detect_sign=param['detect_sign'])


#%% save data
se.MdaSortingExtractor.write_sorting(sorting_ms4, output.firings)
pickle.dump(sorting_ms4,open(input.sorter,'wb'))

#%%
sorting_ms4 = pickle.load(open(input.sorter,'rb'))


#%% save sorting waveform
spike_waveform = st.postprocessing.get_unit_waveforms(recording, sorting_ms4,
     max_num_waveforms=100)

pickle.dump(spike_waveform,open(output.spike_waveforms,'wb'))

# #%% Calculate metrics for curation

# metricsCal = st.validation.MetricCalculator(sorting_ms4, recording,verbose=True)
# numWaveform = 100
# metricsName = ['firing_rate', 'num_spikes', 'isi_viol', 'presence_ratio','d_prime','snr']
# metricsCal.compute_all_metric_data(max_num_waveforms=numWaveform, max_num_pca_waveforms=numWaveform)
# metrics = metricsCal.compute_metrics(max_snr_waveforms=numWaveform, max_spikes_for_silhouette=numWaveform,max_spikes_for_unit=numWaveform,
#     max_spikes_for_nn=numWaveform,metric_names=metricsName)

#%% Do some simple curation for now
sorting_ms4_curated = st.curation.threshold_snr(sorting=sorting_ms4, recording = recording,
  threshold =1.2, threshold_sign='less', max_snr_waveforms=100,) #remove when less than threshold
print(sorting_ms4_curated.get_unit_ids())

sorting_ms4_curated=st.curation.threshold_firing_rate(sorting_ms4_curated,
     threshold=0.5, threshold_sign='less')
print(sorting_ms4_curated.get_unit_ids())

st.curation.threshold_isi_violations(sorting_ms4_curated, threshold = 0.9, threshold_sign='greater')
print(sorting_ms4_curated.get_unit_ids())

pickle.dump(sorting_ms4, open(output.sorter_curated,'wb'))
se.MdaSortingExtractor.write_sorting(sorting_ms4, output.firings_curated)


#%%
