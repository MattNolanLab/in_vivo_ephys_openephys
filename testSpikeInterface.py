#%%
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.toolkit as st
import spikeinterface.sorters as sorters
import spikeinterface.comparison as sc
import spikeinterface.widgets as sw
import setting
import OpenEphys
import numpy as np
import pandas as pd
import pickle

#%%
recording_location = 'testData/M1_D31_2018-11-01_12-28-25/'
#%%
def load_OpenEphysRecording(folder):
    signal = []
    for i in range(setting.num_tetrodes*4):
        fname = folder+setting.data_file_prefix+str(i+1)+setting.data_file_suffix+'.continuous'
        x = OpenEphys.loadContinuousFast(fname)['data']
        if i==0:
            #preallocate array on first run
            signal = np.zeros((setting.num_tetrodes*4,x.shape[0]))
        signal[i,:] = x
    return signal

signal = load_OpenEphysRecording(recording_location)

#%% load geometry
geom = pd.read_csv('sorting_files/geom_all_tetrodes_original.csv',header=None).values

#%%
Fs = setting.sampling_rate
recording =se.NumpyRecordingExtractor(signal,setting.sampling_rate,geom)
# recording = se.NumpyRecordingExtractor(signal[:,Fs*120:Fs*300],Fs, geom)
#%%
sw.plot_electrode_geometry(recording)

#%%
fs = recording.get_sampling_frequency()
trace_snippet = recording.get_traces(start_frame=int(fs*0), end_frame=int(fs*2))
w_ts = sw.plot_timeseries(recording)

#%%
channel_ids = recording.get_channel_ids()
fs = recording.get_sampling_frequency()
num_chan = recording.get_num_channels()

print('Channel ids:', channel_ids)
print('Sampling frequency:', fs)
print('Number of channels:', num_chan)



#%%
recording_prb = recording.load_probe_file('sorting_files/tetrode_16.prb')
print('Channels after loading the probe file:', recording_prb.get_channel_ids())
print('Channel groups after loading the probe file:', recording_prb.get_channel_groups())

#%%
def getDeadChannel(deadChannelFile):
    with open(deadChannelFile,'r') as f:
        deadChannels = [int(s) for s in f.readlines()]

    return deadChannels

bad_channel = getDeadChannel(recording_location+'dead_channels.txt')

#%% pre-processing

recording_f = st.preprocessing.bandpass_filter(recording_prb, freq_min=300, freq_max=6000,chunk_size=600)
recording_rm_noise = st.preprocessing.remove_bad_channels(recording_f, bad_channel_ids=bad_channel)
recording_cmr = st.preprocessing.common_reference(recording_rm_noise, reference='median')


#%%
sw.plot_timeseries(recording_f,trange=[60,65])

#%%
sorters.get_default_params('mountainsort4')

#%% load the parameter file
import json
param = json.load(open('sorting_files/params.json'))
sorting_ms4 = sorters.run_mountainsort4(recording_rm_noise, output_folder='mountainsort4',
    adjacency_radius=param['adjacency_radius'], detect_sign=param['detect_sign'])

#%%
import pickle
with open('sorting_ms4.pkl','wb') as f:
    pickle.dump(sorting_ms4,f)

#%%
with open('sorting_ms4.pkl','rb') as f:
    sorting_ms4 = pickle.load(f)

#%%
sw.plot_unit_waveforms(recording_f, sorting_ms4,unit_ids=[1,2,3],max_num_waveforms=50),


#%%
sw.plot_rasters(sorting_ms4,trange=[0,10],color='w')

#%% curation
sorting_ms4_curated = st.curation.threshold_snr(sorting=sorting_ms4, recording = recording_f,
  threshold = , threshold_sign='less',max_snr_waveforms=50)
print(sorting_ms4_curated.get_unit_ids())



#%%
