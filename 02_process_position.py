#%%
import pandas as pd
import testload
from PostSorting.vr_sync_spatial_data import *
import PostSorting.vr_spatial_data as vr_spatial_data
import PostSorting.vr_make_plots as vr_make_plots
import PostSorting
from collections import namedtuple
import setting
from types import SimpleNamespace
import gc
import PostSorting.vr_stop_analysis as vr_stop_analysis
import setting
import scipy.signal as signal

#%% define input and output
if 'snakemake' not in locals():
    #Define some variable to run the script standalone
    input = SimpleNamespace()
    output = SimpleNamespace()

    input.recording_to_sort = 'testData/M1_D31_2018-11-01_12-28-25'
    
    sorterPrefix = input.recording_to_sort+'/processed/'+setting.sorterName
    
    output.trial_figure = input.recording_to_sort + '/processed/figures/trials.png'
    output.first_trial_ch = input.recording_to_sort + '/processed/figures/trials_type1.png'
    output.second_trial_ch = input.recording_to_sort + '/processed/figures/trials_type2.png'
    output.raw_position_data = input.recording_to_sort +'/processed/raw_position.hdf'
    output.processed_position_data = input.recording_to_sort +'/processed/processed_position.hdf'
    
    figure_folder ='/processed/figures'
    output.stop_raster = input.recording_to_sort +figure_folder + '/behaviour/stop_raster.png'
    output.stop_histogram = input.recording_to_sort +figure_folder + '/behaviour/stop_histogram.png'
    output.speed_histogram = input.recording_to_sort +figure_folder + '/behaviour/speed_histogram.png'
else:
    #in snakemake environment, the input and output will be provided by the workflow
    input = snakemake.input
    output = snakemake.output

#%% Load and downsample the position data
print('Loading location and trial onset files')
recorded_location = get_raw_location(input.recording_to_sort, setting.movement_ch) # get raw location from DAQ pin
first_ch = load_first_trial_channel(input.recording_to_sort)
second_ch = load_second_trial_channel(input.recording_to_sort)

print('Downsampling')
downsample_ratio = int(setting.sampling_rate/setting.location_ds_rate)
recorded_location_ds=downsample(recorded_location, downsample_ratio) #filtering may cause
first_ch_ds = downsample(first_ch, downsample_ratio)
second_ch_ds = downsample(second_ch, downsample_ratio)

#%% Process position data
raw_position_data = pd.DataFrame()
raw_position_data = calculate_track_location(raw_position_data,recorded_location_ds)
raw_position_data = calculate_trial_numbers(raw_position_data, setting.location_ds_rate*0.5)
PostSorting.vr_make_plots.plot_trials(raw_position_data['trial_number'], output.trial_figure)

#%% Calculate trial-related information
(raw_position_data,first_ch,second_ch) = calculate_trial_types(raw_position_data, first_ch_ds, second_ch_ds)
PostSorting.vr_make_plots.plot_trial_channels(first_ch, second_ch, output.first_trial_ch, output.second_trial_ch)

raw_position_data = calculate_time(raw_position_data, setting.location_ds_rate)
raw_position_data = calculate_instant_dwell_time(raw_position_data)
raw_position_data = calculate_instant_velocity(raw_position_data, setting.location_ds_rate)
raw_position_data = get_avg_speed(raw_position_data, int(setting.location_ds_rate*0.2))

#%% save data
raw_position_data.to_hdf(output.raw_position_data, 'raw_position_data', mode='w')
# raw_position_data=pd.read_hdf(output.raw_position_data)

#%% bin the position data over trials

processed_position_data = pd.DataFrame() # make dataframe for processed position data
processed_position_data = vr_spatial_data.bin_speed_over_trials(raw_position_data,processed_position_data)
processed_position_data = vr_spatial_data.bin_data_trial_by_trial(raw_position_data,processed_position_data)
processed_position_data = vr_spatial_data.calculate_total_trial_numbers(raw_position_data, processed_position_data)


#%% Analyze stops
processed_position_data = vr_stop_analysis.calculate_stops(raw_position_data, processed_position_data, 0.01, setting.location_ds_rate)
processed_position_data = vr_stop_analysis.calculate_average_stops(raw_position_data,processed_position_data)
processed_position_data = vr_stop_analysis.find_first_stop_in_series(processed_position_data)
processed_position_data = vr_stop_analysis.find_rewarded_positions(raw_position_data,processed_position_data)

processed_position_data["new_trial_indices"] = raw_position_data["new_trial_indices"]

#%% plotting position data
PostSorting.vr_make_plots.plot_stops_on_track(raw_position_data, processed_position_data, output.stop_raster)
PostSorting.vr_make_plots.plot_stop_histogram(raw_position_data, processed_position_data, output.stop_histogram)
PostSorting.vr_make_plots.plot_speed_histogram(processed_position_data, output.speed_histogram)

#%% save data
processed_position_data.to_hdf(output.processed_position_data,'processed_position_data', mode='w')



#%%
