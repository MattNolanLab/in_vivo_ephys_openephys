#%%
import PostSorting.vr_spatial_firing
import PostSorting.vr_firing_rate_maps
import PostSorting.vr_FiringMaps_InTime
import settings
import pandas as pd
from collections import namedtuple
from types import SimpleNamespace
from utils import SnakeIOHelper
from utils.file_utility import *

# #%% define sinput and output
(sinput, soutput) = SnakeIOHelper.getSnake(locals(), 'workflow/workflow_vr.smk', [settings.debug_folder+'/processed/spatial_firing_vr.pkl'],
    'process_expt')

#%% Load data
spike_data = pd.read_pickle(sinput.spatial_firing)
raw_position_data =pd.read_pickle(sinput.raw_position)
processed_position_data = pd.read_pickle(sinput.processed_position_data)
track_length = processed_position_data['track_length'][0]
#%% process firing times

downsample_ratio = settings.sampling_rate / settings.location_ds_rate
_, _, spike_data = PostSorting.vr_spatial_firing.process_spatial_firing(spike_data, raw_position_data, downsample_ratio)
spike_data_vr = PostSorting.vr_spatial_firing.split_spatial_firing_by_trial_type(spike_data)
spike_data_vr = PostSorting.vr_firing_rate_maps.make_firing_field_maps(spike_data, processed_position_data, 
    settings.vr_bin_size_cm, track_length)

spike_data_vr['reward_loc'] = processed_position_data['reward_loc'][0]
spike_data_vr['track_length'] = track_length
#%% save data
spike_data_vr.to_pickle(soutput.spatial_firing_vr)

#%%
