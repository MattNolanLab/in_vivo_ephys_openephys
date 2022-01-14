#%%
'''
This script support variable track length and reward location
The log file from Blender should be in the same folder as other Open Ephys files
'''

from utils.file_utility import get_track_info_log_file
import pandas as pd
from PostSorting.vr_sync_spatial_data import *
from PostSorting import vr_spatial_data
from PostSorting import vr_speed_analysis
from PostSorting import vr_time_analysis
from PostSorting import vr_stop_analysis
from PostSorting import vr_make_plots
from PostSorting import vr_sync_spatial_data
import PostSorting
from PostSorting.open_field_spatial_data import resample_position_data 
from collections import namedtuple
import settings
from types import SimpleNamespace
import gc
import PostSorting.vr_stop_analysis as vr_stop_analysis
import scipy.signal as signal
from utils import SnakeIOHelper
import logging


#%% Define input and output
(sinput, soutput) = SnakeIOHelper.getSnake(locals(), 'workflow/workflow_vr_variable_track.smk', [settings.debug_folder+'/processed/processed_position.pkl'],
    'process_position')

#%% Load and downsample the position data
print('Loading location and trial onset files')
recorded_location = get_raw_location(sinput.recording_to_sort, soutput.raw_position_plot) # get raw location from DAQ pin
first_ch = load_first_trial_channel(sinput.recording_to_sort)
second_ch = load_second_trial_channel(sinput.recording_to_sort)

print('Downsampling')
ds_ratio = int(settings.sampling_rate/settings.location_ds_rate)
recorded_location_ds=downsample(recorded_location, ds_ratio) #filtering may cause
first_ch_ds = downsample(first_ch, ds_ratio)
second_ch_ds = downsample(second_ch, ds_ratio)

#%% Process position data
raw_position_data = pd.DataFrame()
track_length, reward_loc, config_file_name = get_track_info_log_file(sinput.recording_to_sort, sinput.recording_to_sort+'/../../../Session_Parameters/')
raw_position_data = calculate_track_location(raw_position_data, recorded_location_ds, track_length)
raw_position_data = calculate_trial_numbers(raw_position_data, soutput.trial_figure)


# Load and synchronize the blender file
blender_pos_sync, blender_trial_info = calculate_trial_types_from_csv(raw_position_data,
        sinput.recording_to_sort, soutput.blender_sync_plot)


#%%
if 'r10_r20' in config_file_name:
    # mixed environment trial, merge the blender trial info into the original frame
    #TDOO: do not hard code the track length here, need to modify the blender program to write
    # the track length in the log file
    blender_trial_info = vr_sync_spatial_data.clean_up_blender_trial_info(blender_trial_info, 300)
    raw_position_data = raw_position_data.merge(blender_trial_info, on='trial_number')
else:
    raw_position_data = calculate_trial_types(raw_position_data, first_ch_ds, second_ch_ds, soutput.trial_type_plot_folder)


blender_pos_sync.to_pickle(soutput.blender_pos)
blender_trial_info.to_pickle(soutput.blender_trial_info)


#%% Calculate trial-related information
raw_position_data = calculate_time(raw_position_data, settings.location_ds_rate)
raw_position_data = calculate_instant_dwell_time(raw_position_data, settings.location_ds_rate)
raw_position_data = calculate_instant_velocity(raw_position_data, soutput.speed_plot, 
            settings.location_ds_rate, speed_win = 0.1)
raw_position_data = get_avg_speed_200ms(raw_position_data, soutput.mean_speed_plot, settings.location_ds_rate,0.1)

#%% save data
if not 'track_length' in raw_position_data.columns:
    raw_position_data['track_length'] = track_length
    raw_position_data['reward_loc'] = reward_loc
raw_position_data.to_pickle(soutput.raw_position_data)

#%% bin the position data over trials
processed_position_data = pd.DataFrame() # make dataframe for processed position data
processed_position_data = vr_speed_analysis.calculate_binned_speed(raw_position_data,processed_position_data, track_length)
processed_position_data = vr_time_analysis.calculate_binned_time(raw_position_data,processed_position_data, track_length)

#%% Analysis stops
#TODO: load stop threshold from parameter file
processed_position_data = vr_stop_analysis.get_stops_from_binned_speed(processed_position_data, 4.7)
processed_position_data = vr_stop_analysis.calculate_average_stops(processed_position_data)
processed_position_data = vr_stop_analysis.calculate_first_stops(processed_position_data)
processed_position_data = vr_stop_analysis.calculate_rewarded_stops(processed_position_data)
processed_position_data =vr_stop_analysis.calculate_rewarded_trials(processed_position_data)

# merge the track length and reward location from blender file
if 'r10_r20' in config_file_name:
    processed_position_data = processed_position_data.merge(blender_trial_info[['trial_number','track_length','reward_loc']], on='trial_number')
else:
    processed_position_data['track_length'] = track_length
    processed_position_data['reward_loc'] = reward_loc

#%% plotting position data

vr_make_plots.plot_stops_on_track(processed_position_data, soutput.stop_raster, track_length)
vr_make_plots.plot_stop_histogram(processed_position_data, soutput.stop_histogram, track_length)
vr_make_plots.plot_speed_histogram(processed_position_data, soutput.speed_histogram, track_length)
vr_make_plots.plot_speed_per_trial(processed_position_data, soutput.speed_heat_map, track_length=track_length)

#%% save data
processed_position_data.to_pickle(soutput.processed_position_data)

# %%
