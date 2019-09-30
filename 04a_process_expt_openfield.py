#%%
import PostSorting.open_field_spatial_firing as open_field_spatial_firing
import PostSorting.speed as speed
import PostSorting.open_field_head_direction as open_field_head_direction
import PostSorting.open_field_firing_maps as open_field_firing_maps
import PostSorting.open_field_grid_cells as open_field_grid_cells
import PostSorting.open_field_firing_fields as open_field_firing_fields
import PostSorting.post_process_sorted_data as post_process_sorted_data
import PostSorting
import setting
import pandas as pd
from collections import namedtuple
from types import SimpleNamespace
import SnakeIOHelper
import pickle
#%% define input and output
if 'snakemake' not in locals():
    #Define some variable to run the script standalone
    input = SimpleNamespace()
    output = SimpleNamespace()

    input.recording_to_sort = 'testData/M1_D27_2018-10-26_13-10-36_of/'
    input.spatial_firing = input.recording_to_sort + 'processed/spatial_firing.hdf'
    input.position = input.recording_to_sort + 'processed/synced_spatial_data.hdf'

    output.spatial_firing_of = input.recording_to_sort + '/processed/spatial_firing_of.hdf'
    output.position_heat_map = input.recording_to_sort +'/processed/position_heat_map.pkl'
    output.hd_histogram = input.recording_to_sort + '/processed/hd_histogram.pkl'

    SnakeIOHelper.makeFolders(output)
else:
    #in snakemake environment, the input and output will be provided by the workflow
    input = snakemake.input
    output = snakemake.output

#%% Load data
spike_data = pd.read_hdf(input.spatial_firing)
synced_spatial_data = pd.read_hdf(input.position)

#%% Proccess spike data together with location data
#TODO curate data
spike_data_spatial = open_field_spatial_firing.add_firing_locations(spike_data, synced_spatial_data)
spike_data_spatial = speed.calculate_speed_score(synced_spatial_data, spike_data_spatial, 250,
        setting.sampling_rate)

#%%
hd_histogram, spatial_firing = open_field_head_direction.process_hd_data(spike_data_spatial, synced_spatial_data)

#%%
position_heat_map, spatial_firing = open_field_firing_maps.make_firing_field_maps(synced_spatial_data, spike_data_spatial)

#%%
spatial_firing = open_field_grid_cells.process_grid_data(spatial_firing)
spatial_firing = open_field_firing_fields.analyze_firing_fields(spatial_firing, synced_spatial_data)
  

#%% Save
spatial_firing.to_hdf(output.spatial_firing_of, 'spatial_firing_of', mode='w')
pickle.dump(hd_histogram,open(output.hd_histogram,'wb'))
pickle.dump(position_heat_map, open(output.position_heat_map,'wb'))
  

#%%
