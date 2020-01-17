#%%
import PostSorting.make_plots
import setting
import pandas as pd
from collections import namedtuple
from types import SimpleNamespace
import SnakeIOHelper 
import pickle
import PostSorting.open_field_make_plots as open_field_make_plots
import logging

#%% define input and output

(sinput, soutput) = SnakeIOHelper.getSnake(locals(), 'op_workflow.smk', [setting.debug_folder+'/processed/completed.txt'],
'plot_figures')

logger = logging.Logger(__file__)

#%% Load data
spatial_firing = pd.read_pickle(sinput.spatial_firing)
position_data = pd.read_pickle(sinput.position)
hd_histogram = pickle.load(open(sinput.hd_histogram,'rb'))
position_heat_map = pickle.load(open(sinput.position_heat_map,'rb'))

#%% plot figures
logger.info('I will plot spikes vs time for the whole session excluding opto tagging.')
PostSorting.make_plots.plot_spike_histogram(spatial_firing, soutput.spike_histogram)

#%%
logger.info('I will plot spikes vs speed for the whole session excluding opto tagging.')
PostSorting.make_plots.plot_firing_rate_vs_speed(spatial_firing, position_data, soutput.firing_properties)

logger.info('I will plot the speed vs firing rate')
PostSorting.make_plots.plot_speed_vs_firing_rate(position_data, spatial_firing, setting.sampling_rate, 250, 
    soutput.firing_properties)

logger.info('I will plot autocorrelograms for each cluster.')
PostSorting.make_plots.plot_autocorrelograms(spatial_firing, soutput.autocorrelogram)

#%% Plot spike trajectory
logger.info('I will make scatter plots of spikes on the trajectory of the animal.')
open_field_make_plots.plot_spikes_on_trajectory(position_data, spatial_firing, soutput.firing_scatter)

#%% Plot coverage of spatial map
logger.info('I will plot a heat map of the position of the animal to show coverage.')
open_field_make_plots.plot_coverage(position_heat_map, soutput.session)

#%% Plot firing rate map
logger.info('I will make rate map plots.')
open_field_make_plots.plot_firing_rate_maps(spatial_firing, soutput.rate_maps)

#%% Plot autocrrelogram
logger.info('I will make the rate map autocorrelogram grid plots now.')
open_field_make_plots.plot_rate_map_autocorrelogram(spatial_firing,  soutput.rate_map_autocorrelogram )

#%% Plot head-direction
logger.info('I will plot HD on open field maps as a scatter plot for each cluster.')
open_field_make_plots.plot_hd(spatial_firing, position_data, soutput.head_direction_plots_2d)

#%% Plot polar head direction
logger.info('I will make the polar HD plots now.')
open_field_make_plots.plot_polar_head_direction_histogram(hd_histogram, spatial_firing, soutput.head_direction_plots_polar)

#%% Plot head direction for firing fields
logger.info('I will make the polar HD plots for individual firing fields now.')
open_field_make_plots.plot_hd_for_firing_fields(spatial_firing, position_data, soutput.firing_field_plots)

#%% Plot spike on firing fields
logger.info('I will plot detected spikes colour coded in fields.')
open_field_make_plots.plot_spikes_on_firing_fields(spatial_firing, soutput.firing_fields_coloured_spikes)

#%% Combine all figures
logger.info('I will make the combined images now.')
open_field_make_plots.make_combined_figure(soutput.combined, spatial_firing)

#%%
#create the dummy file for output
with open(soutput.result,'w') as f:
    pass
