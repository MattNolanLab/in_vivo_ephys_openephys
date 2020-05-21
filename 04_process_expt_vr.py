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
(sinput, soutput) = SnakeIOHelper.getSnake(locals(), 'workflow_vr.smk', [setting.debug_folder+'/processed/spatial_firing_vr.pkl'],
    'process_expt')


#%% Load data
spike_data = pd.read_pickle(sinput.spatial_firing)
raw_position_data =pd.read_pickle(sinput.raw_position)
processed_position_data = pd.read_pickle(sinput.processed_position_data)

#%% process firing times

spike_data_vr = PostSorting.vr_spatial_firing.process_spatial_firing(spike_data, raw_position_data, setting.sampling_rate/setting.location_ds_rate)
spike_data_vr = PostSorting.vr_firing_rate_maps.make_firing_field_maps_all(spike_data, raw_position_data, 
    processed_position_data, soutput.cluster_spike_plot )

#%%
spike_data_vr = PostSorting.vr_FiringMaps_InTime.control_convolution_in_time(spike_data, raw_position_data)

#%% save data
spike_data.to_pickle(soutput.spatial_firing_vr)

#%%
