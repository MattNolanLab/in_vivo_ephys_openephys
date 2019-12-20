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
# if 'snakemake' not in locals():
#     #Define some variable to run the script standalone
#     sinput = SimpleNamespace()
#     output = SimpleNamespace()

#     sinput.recording_to_sort = 'testData/M1_D31_2018-11-01_12-28-25_short'
#     sinput.spatial_firing = sinput.recording_to_sort + '/processed/spatial_firing.hdf'
#     sinput.raw_position = sinput.recording_to_sort + '/processed/raw_position.hdf'
#     sinput.processed_position_data = sinput.recording_to_sort + '/processed/processed_position.hdf'

#     output.spatial_firing_vr = sinput.recording_to_sort + '/processed/spatial_firing_vr.hdf'
#     output.cluster_spike_plot = sinput.recording_to_sort + '/processed/figures/spike_number/'
#     output.spike_data = sinput.recording_to_sort +'/processed/figures/spike_data/'
    

#     SnakeIOHelper.makeFolders(output)
# else:
#     #in snakemake environment, the sinput and output will be provided by the workflow
#     sinput = snakemake.sinput
#     output = snakemake.output


(sinput, soutput) = SnakeIOHelper.getSnake(locals(), 'vr_workflow.smk', [setting.debug_folder+'/processed/spatial_firing_vr.pkl'],
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
