#%%
import pandas as pd
from PostSorting.vr_sync_spatial_data import *
import PostSorting.vr_spatial_data as vr_spatial_data
import PostSorting.vr_speed_analysis as vr_speed_analysis
import PostSorting.vr_time_analysis as vr_time_analysis
import PostSorting.vr_stop_analysis as vr_stop_analysis
import PostSorting.vr_make_plots as vr_make_plots
import PostSorting
from collections import namedtuple
import setting
from types import SimpleNamespace
import gc
import PostSorting.vr_stop_analysis as vr_stop_analysis
import setting
import scipy.signal as signal
import SnakeIOHelper
#%% Define input and output
(sinput, soutput) = SnakeIOHelper.getSnake(locals(), 'workflow/workflow_vr.smk', [setting.debug_folder+'/processed/processed_position.pkl'],
    'process_position')

#%% Load and downsample the position data
print('Loading location and trial onset files')
recorded_location = get_raw_location(sinput.recording_to_sort, soutput.raw_position_plot) # get raw location from DAQ pin
first_ch = load_first_trial_channel(sinput.recording_to_sort)
second_ch = load_second_trial_channel(sinput.recording_to_sort)

#%%
print('Downsampling')
ds_ratio = int(setting.sampling_rate/setting.location_ds_rate)
recorded_location_ds=downsample(recorded_location, ds_ratio) 
first_ch_ds = downsample(first_ch, ds_ratio)
second_ch_ds = downsample(second_ch, ds_ratio)

#%% Process position data
raw_position_data = pd.DataFrame()
raw_position_data = calculate_track_location(raw_position_data, recorded_location_ds, setting.track_length)
raw_position_data = calculate_trial_numbers(raw_position_data, soutput.trial_figure)
# raw_position_data = smooth_position(raw_position_data, setting.location_ds_rate)

#%% Calculate trial-related information
(raw_position_data,first_ch,second_ch) = calculate_trial_types(raw_position_data, first_ch_ds, second_ch_ds, soutput.trial_type_plot_folder)
raw_position_data = calculate_time(raw_position_data, setting.location_ds_rate)
raw_position_data = calculate_instant_dwell_time(raw_position_data, setting.location_ds_rate)
raw_position_data = calculate_instant_velocity(raw_position_data, soutput.speed_plot, setting.location_ds_rate, lowpass=True)
raw_position_data = get_avg_speed_200ms(raw_position_data, soutput.mean_speed_plot, setting.location_ds_rate)

#%% save data
raw_position_data.to_pickle(soutput.raw_position_data)

#%% bin the position data over trials
processed_position_data = pd.DataFrame() # make dataframe for processed position data
processed_position_data = vr_speed_analysis.calculate_binned_speed(raw_position_data,processed_position_data, setting.track_length)
processed_position_data = vr_time_analysis.calculate_binned_time(raw_position_data,processed_position_data,setting.track_length)


#%% Analysis stops
#TODO: load stop threshold from parameter file
processed_position_data = vr_stop_analysis.get_stops_from_binned_speed(processed_position_data, 4.7)
processed_position_data = vr_stop_analysis.calculate_average_stops(processed_position_data)
processed_position_data = vr_stop_analysis.calculate_first_stops(processed_position_data)
processed_position_data = vr_stop_analysis.calculate_rewarded_stops(processed_position_data)
processed_position_data =vr_stop_analysis.calculate_rewarded_trials(processed_position_data)


#%% plotting position data
vr_make_plots.plot_stops_on_track(processed_position_data, soutput.stop_raster)
vr_make_plots.plot_stop_histogram(processed_position_data, soutput.stop_histogram)
vr_make_plots.plot_speed_histogram(processed_position_data, soutput.speed_histogram)
vr_make_plots.plot_speed_per_trial(processed_position_data, soutput.speed_heat_map, track_length=setting.track_length)

#%% save data
processed_position_data.to_pickle(soutput.processed_position_data)

# %%
