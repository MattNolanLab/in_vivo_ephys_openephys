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
from SnakeIOHelper import getSnake
import pickle
#%% define input and output
if 'snakemake' not in locals(): 
    #Run the the file from the root project directory
    smk = getSnake('op_workflow.smk',[setting.debug_folder+'/processed/spatial_firing_of.hdf'],
        'process_expt' )
    sinput = smk.input
    soutput = smk.output
else:
    sinput = snakemake.input
    soutput = snakemake.output

#%% Load data
spike_data = pd.read_hdf(sinput.spatial_firing)
synced_spatial_data = pd.read_hdf(sinput.position)

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
spatial_firing.to_hdf(soutput.spatial_firing_of, 'spatial_firing_of', mode='w')
pickle.dump(hd_histogram,open(soutput.hd_histogram,'wb'))
pickle.dump(position_heat_map, open(soutput.position_heat_map,'wb'))
  

#%%
