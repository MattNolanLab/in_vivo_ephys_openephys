#%%
import PostSorting.vr_spatial_firing
import PostSorting.vr_firing_rate_maps
import PostSorting.vr_FiringMaps_InTime
import setting
import pandas as pd
from collections import namedtuple
from types import SimpleNamespace
import SnakeIOHelper

# #%% define sinput and output
(sinput, soutput) = SnakeIOHelper.getSnake(locals(), 'workflow/workflow_vr.smk', [setting.debug_folder+'/processed/spatial_firing_vr.pkl'],
    'process_expt')


#%% Load data
spike_data = pd.read_pickle(sinput.spatial_firing)
raw_position_data =pd.read_pickle(sinput.raw_position)
processed_position_data = pd.read_pickle(sinput.processed_position_data)

#%% process firing times
downsample_ratio = setting.sampling_rate / setting.location_ds_rate
_, _, spike_data = PostSorting.vr_spatial_firing.process_spatial_firing(spike_data, raw_position_data, downsample_ratio)
spike_data_vr = PostSorting.vr_spatial_firing.split_spatial_firing_by_trial_type(spike_data)
spike_data_vr = PostSorting.vr_firing_rate_maps.make_firing_field_maps(spike_data, processed_position_data, 
    setting.track_length/setting.location_bin_num, setting.track_length)

#%% save data
spike_data.to_pickle(soutput.spatial_firing_vr)

#%%
