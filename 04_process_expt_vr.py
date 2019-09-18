#%%
import PostSorting.vr_spatial_firing
import PostSorting.vr_firing_rate_maps
import PostSorting.vr_FiringMaps_InTime
import setting
import pandas as pd
from collections import namedtuple

#%% define input and output
if 'snakemake' not in locals():
    #Define some variable to run the script standalone
    input = namedtuple
    output = namedtuple

    input.recording_to_sort = 'testData/M1_D31_2018-11-01_12-28-25'
    input.spatial_firing = input.recording_to_sort + '/processed/spatial_firing.hdf'
    input.raw_position = input.recording_to_sort + '/processed/raw_position.hdf'
    input.processed_position_data = input.recording_to_sort + '/processed/processed_position.hdf'

    output.spatial_firing_vr = input.recording_to_sort + '/processed/spatial_firing_vr.hdf'
else:
    #in snakemake environment, the input and output will be provided by the workflow
    input = snakemake.input
    output = snakemake.output

#%% Load data
spike_data = pd.read_hdf(input.spatial_firing)
raw_position_data =pd.read_hdf(input.raw_position)
processed_position_data = pd.read_hdf(input.processed_position_data)
#%% process firing times

# spike_data = PostSorting.load_snippet_data.get_snippets(spike_data, prm)
spike_data_vr = PostSorting.vr_spatial_firing.process_spatial_firing(spike_data, raw_position_data)

#%%
spike_data_vr = PostSorting.vr_firing_rate_maps.make_firing_field_maps_all(spike_data, raw_position_data, processed_position_data)

#%%
spike_data_vr = PostSorting.vr_FiringMaps_InTime.control_convolution_in_time(spike_data, raw_position_data)

#%% save data
spike_data.to_hdf(output.spatial_firing_vr ,'spatial_firing_vr')
