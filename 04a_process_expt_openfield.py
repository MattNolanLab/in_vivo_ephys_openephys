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
import SnakeIOHelper
import pickle
#%% define input and output

(sinput, soutput) = SnakeIOHelper.getSnake(locals(), 'op_workflow.smk', [setting.debug_folder+'/processed/spatial_firing_of.pkl'],
    'process_expt')
    
#%% Load data
spike_data = pd.read_pickle(sinput.spatial_firing)
synced_spatial_data = pd.read_pickle(sinput.position)

#%% Proccess spike data together with location data
spike_data_spatial = open_field_spatial_firing.add_firing_locations(spike_data, synced_spatial_data)
spike_data_spatial = speed.calculate_speed_score(synced_spatial_data, spike_data_spatial, 250,
        setting.sampling_rate)

#%%
hd_histogram, spatial_firing = open_field_head_direction.process_hd_data(spike_data_spatial, synced_spatial_data)

#%%
position_heat_map, spatial_firing = open_field_firing_maps.make_firing_field_maps(synced_spatial_data, spike_data_spatial)

#%%
spatial_firing = open_field_grid_cells.process_grid_data(spatial_firing)
spatial_firing = open_field_firing_fields.analyze_firing_fields(spatial_firing, synced_spatial_data, soutput.hd_csv)
  
#%% Save
spatial_firing.to_pickle(soutput.spatial_firing_of)
pickle.dump(hd_histogram,open(soutput.hd_histogram,'wb'))
pickle.dump(position_heat_map, open(soutput.position_heat_map,'wb'))
  

#%%
