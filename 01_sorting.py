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
import spikeinterfaceHelper
from tqdm import tqdm
import numpy as np
import setting
from SnakeIOHelper import getSnake
from PreClustering.pre_process_ephys_data import filterRecording
import logging

#for logging
# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(os.path.basename(__file__)+':'+__name__)

#%% define input and output
if 'snakemake' not in locals(): 
    targetname = setting.debug_folder+'/processed/'+setting.sorterName+'/sorter_curated.pkl'
    smk = getSnake('op_workflow.smk',[targetname],
        'sort_spikes' )
    sinput = smk.input
    soutput = smk.output
else:
    sinput = snakemake.input
    soutput = snakemake.output

#%% Load data and create recording extractor

signal = file_utility.load_OpenEphysRecording(sinput.recording_to_sort)
geom = pd.read_csv(sinput.tetrode_geom,header=None).values
bad_channel = file_utility.getDeadChannel(sinput.dead_channel)


#%% Create and filter the recording
logger.info('Filtering files') #TODO logging not show correctly

recording = se.NumpyRecordingExtractor(signal,setting.sampling_rate,geom)
recording = recording.load_probe_file(sinput.probe_file) #load probe definition
filterRecording(recording,setting.sampling_rate) #for faster operation later

#%% Remove some bad channels
recording = st.preprocessing.remove_bad_channels(recording, bad_channel_ids=bad_channel) #remove bad channel

#%% perform sorting
with open(sinput.sort_param) as f:
    param = json.load(f)
sorting_ms4 = sorters.run_sorter(setting.sorterName,recording, output_folder=setting.sorterName,
    adjacency_radius=param['adjacency_radius'], detect_sign=param['detect_sign'],verbose=True)

with open(soutput.sorter,'wb') as f:
    pickle.dump(sorting_ms4,f)
    
#%%
# sorting_ms4 = pickle.load(open(output.sorter,'rb'))

#%% compute some property of the sorting
st.postprocessing.get_unit_max_channels(recording, sorting_ms4, max_spikes_per_unit=100)
st.postprocessing.get_unit_waveforms(recording, sorting_ms4, max_spikes_per_unit=100)

for id in sorting_ms4.get_unit_ids():
    number_of_spikes = len(sorting_ms4.get_unit_spike_train(id))
    mean_firing_rate = number_of_spikes/(recording._recording._timeseries.shape[1]/setting.sampling_rate)
    sorting_ms4.set_unit_property(id,'number_of_spikes',number_of_spikes)
    sorting_ms4.set_unit_property(id, 'mean_firing_rate', mean_firing_rate)

#%% save data
with open(soutput.sorter_curated,'wb') as f:
    pickle.dump(sorting_ms4,f)
sorter_df=spikeinterfaceHelper.sorter2dataframe(sorting_ms4)
sorter_df.to_pickle(soutput.sorter_df)

#%% Do some simple curation for now
sorting_ms4_curated = st.curation.threshold_snr(sorting=sorting_ms4, recording = recording,
  threshold =1.2, threshold_sign='less', max_snr_spikes_per_unit=100, apply_filter=False) #remove when less than threshold
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
curated_sorter_df = spikeinterfaceHelper.sorter2dataframe(sorting_ms4_curated)
curated_sorter_df.to_pickle(soutput.sorter_curated_df)
sorting_ms4_curated = se.SubSortingExtractor(sorting_ms4,unit_ids=sorting_ms4_curated.get_unit_ids())
with open(soutput.sorter_curated,'wb') as f:
    pickle.dump(sorting_ms4_curated, f)


#%% plot the sorted waveform
curated_sorter_df = pd.read_pickle(soutput.sorter_curated_df)

#%%
waveforms = curated_sorter_df.waveforms[0]
waveforms = np.stack([w for w in waveforms if w is not None])
max_channel = curated_sorter_df.max_channel
plt.plot

#%%
