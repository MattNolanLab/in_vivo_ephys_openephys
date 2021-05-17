#%%
from PostSorting import open_field_spatial_firing
from PostSorting import speed
from PostSorting import open_field_head_direction
from PostSorting import open_field_firing_maps
from PostSorting import open_field_grid_cells
from PostSorting import open_field_firing_fields
from PostSorting import compare_first_and_second_half
import PostSorting
import settings
import pandas as pd
from collections import namedtuple
from utils import SnakeIOHelper
import pickle
#%% define input and output

(sinput, soutput) = SnakeIOHelper.getSnake(locals(), 'workflow/workflow_of.smk', [settings.debug_folder+'/processed/spatial_firing_of.pkl'],
    'process_expt')

#%% Load data
spike_data = pd.read_pickle(sinput.spatial_firing)
synced_spatial_data = pd.read_pickle(sinput.position)

#%% Proccess spike data together with location data
spike_data_spatial = open_field_spatial_firing.process_spatial_firing(spike_data, synced_spatial_data)
spike_data_spatial = speed.calculate_speed_score(synced_spatial_data, spike_data_spatial, 250,
        settings.sampling_rate)

#%% Calculate head direction tuning
hd_histogram, spatial_firing = open_field_head_direction.process_hd_data(spike_data_spatial, synced_spatial_data)

#%% Make firing field heat map
position_heat_map, spatial_firing = open_field_firing_maps.make_firing_field_maps(synced_spatial_data, spike_data_spatial)

#%% Process grid data and analyze firing field
spatial_firing = open_field_grid_cells.process_grid_data(spatial_firing)
spatial_firing = open_field_firing_fields.analyze_firing_fields(spatial_firing, synced_spatial_data, soutput.hd_csv)
spatial_firing = PostSorting.open_field_border_cells.process_border_data(spatial_firing)
spatial_firing = PostSorting.open_field_border_cells.process_corner_data(spatial_firing)

#%% Save
spatial_firing.to_pickle(soutput.spatial_firing_of)
pickle.dump(hd_histogram,open(soutput.hd_histogram,'wb'))
pickle.dump(position_heat_map, open(soutput.position_heat_map,'wb'))
  

#%%
