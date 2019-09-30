#%%
import PostSorting
import setting
import pandas as pd
from collections import namedtuple
from types import SimpleNamespace
import SnakeIOHelper
import pickle
import PostSorting.open_field_make_plots as open_field_make_plots

#%% define input and output
figure_folder ='processed/figures/'


if 'snakemake' not in locals():
    #Define some variable to run the script standalone
    input = SimpleNamespace()
    output = SimpleNamespace()

    input.recording_to_sort = 'testData/M1_D27_2018-10-26_13-10-36_of/'
    input.spatial_firing = input.recording_to_sort + 'processed/spatial_firing_of.hdf'
    input.position = input.recording_to_sort + 'processed/synced_spatial_data.hdf'
    input.spatial_firing_of = input.recording_to_sort + '/processed/spatial_firing_of.hdf'
    input.position_heat_map = input.recording_to_sort +'/processed/position_heat_map.pkl'
    input.hd_histogram = input.recording_to_sort + '/processed/hd_histogram.pkl'

    output.spike_histogram = input.recording_to_sort + figure_folder + 'behaviour/spike_histogram/'
    output.autocorrelogram = input.recording_to_sort + figure_folder + 'behaviour/autocorrelogram/'
    output.spike_trajectories = input.recording_to_sort + figure_folder + 'behaviour/spike_trajectories/'
    output.spike_rate =  input.recording_to_sort + figure_folder + 'behaviour/spike_rate/'
    output.convolved_rate = input.recording_to_sort + figure_folder + 'ConvolvedRates_InTime/'
    output.firing_properties = input.recording_to_sort + figure_folder + 'firing_properties/'
    output.firing_scatter = input.recording_to_sort + figure_folder +'firing_scatters/'
    output.session = input.recording_to_sort + figure_folder +'session/'
    output.rate_maps = input.recording_to_sort + figure_folder +'rate_maps/'
    output.rate_map_autocorrelogram = input.recording_to_sort + figure_folder +'rate_map_autocorrelogram/'
    output.head_direction_plots_2d = input.recording_to_sort + figure_folder +'head_direction_plots_2d/'
    output.head_direction_plots_polar = input.recording_to_sort + figure_folder +'head_direction_plots_polar/'
    output.firing_field_plots = input.recording_to_sort + figure_folder + 'firing_field_plots/'
    output.firing_fields_coloured_spikes = input.recording_to_sort + figure_folder + 'firing_fields_coloured_spikes/'
    output.combined = input.recording_to_sort + figure_folder +'combined/'

    SnakeIOHelper.makeFolders(output)

else:
    #in snakemake environment, the input and output will be provided by the workflow
    input = snakemake.input
    output = snakemake.output

#%% Load data
spatial_firing = pd.read_hdf(input.spatial_firing)
position_data = pd.read_hdf(input.position)
hd_histogram = pickle.load(open(input.hd_histogram,'rb'))
position_heat_map = pickle.load(open(input.position_heat_map,'rb'))

#%% plot figures
# PostSorting.make_plots.plot_waveforms(spatial_firing, prm)
# PostSorting.make_plots.plot_waveforms_opto(spatial_firing, prm)
PostSorting.make_plots.plot_spike_histogram(spatial_firing, output.spike_histogram)

PostSorting.make_plots.plot_firing_rate_vs_speed(spatial_firing, position_data, output.firing_properties)

PostSorting.make_plots.plot_speed_vs_firing_rate(position_data, spatial_firing, setting.sampling_rate, 250, 
    output.firing_properties)

PostSorting.make_plots.plot_autocorrelograms(spatial_firing, output.autocorrelogram)

#%%
open_field_make_plots.plot_spikes_on_trajectory(position_data, spatial_firing, output.firing_scatter)

#%%
open_field_make_plots.plot_coverage(position_heat_map, output.session)

#%%
open_field_make_plots.plot_firing_rate_maps(spatial_firing, output.rate_maps)

#%%
open_field_make_plots.plot_rate_map_autocorrelogram(spatial_firing,  output.rate_map_autocorrelogram )

#%%
open_field_make_plots.plot_hd(spatial_firing, position_data, output.head_direction_plots_2d)

#%%
open_field_make_plots.plot_polar_head_direction_histogram(hd_histogram, spatial_firing, output.head_direction_plots_polar)
#%%
open_field_make_plots.plot_hd_for_firing_fields(spatial_firing, position_data, output.firing_field_plots)

#%%
open_field_make_plots.plot_spikes_on_firing_fields(spatial_firing, output.firing_fields_coloured_spikes)

#%%
open_field_make_plots.make_combined_figure(input.recording_to_sort + figure_folder ,output.combined, spatial_firing)

  
#%%
