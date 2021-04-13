import numpy as np
import os
import pandas as pd
import open_ephys_IO
import PostSorting.parameters
import math
import gc
from scipy import stats
import PostSorting.vr_stop_analysis
import PostSorting.vr_make_plots
import PostSorting.vr_cued
import setting
from file_utility import load_openephys_file

from scipy import signal

def downsample(x,downsample_factor):
    """downsample data
    
    Arguments:
        x {np.narray} -- data to be downsample
        downsample_ratio {int} -- downsampling factor
    """
    minx = x.min()
    maxx = x.max()
    pad = downsample_factor*20
    xpad = np.pad(x,pad,'edge') # padding the signal to avoid filter artifact
    xs = signal.decimate(xpad,downsample_factor,ftype='fir')
    xs[xs<minx] = minx #remove the filtering artifact at the sharp transition
    xs[xs>maxx] = maxx
    return xs[20:-20] #remove the padding

def correct_for_restart(location):
    cummulative_minimums = np.minimum.accumulate(location)
    cummulative_maximums = np.maximum.accumulate(location)

    local_min_median = np.median(cummulative_minimums)
    local_max_median = np.median(cummulative_maximums)

    #location [location >local_max_median] = local_max_median
    location [location <local_min_median] = local_min_median # deals with if the VR is switched off during recording - location value drops to 0 - min is usually 0.56 approx
    return location


def get_raw_location(recording_folder, output_path, movement_ch_suffix = setting.movement_ch_suffix):
    """
    Loads raw location continuous channel from ACD1.continuous
    # input: spatial dataframe, path to local recording folder, initialised parameters
    # output: raw location as numpy array
    """
    print('Extracting raw location...')

    location = load_openephys_file(recording_folder + '/*' + movement_ch_suffix)

    location=correct_for_restart(location)
    PostSorting.vr_make_plots.plot_movement_channel(location, output_path)
    return np.asarray(location, dtype=np.float16)


def calculate_track_location(position_data, recorded_location, track_length):
    # recorded_location = get_raw_location(recording_folder, output_path) # get raw location from DAQ pin
    print('Converting raw location input to cm...')
    recorded_startpoint = min(recorded_location)
    recorded_endpoint = max(recorded_location)
    recorded_track_length = recorded_endpoint - recorded_startpoint
    distance_unit = recorded_track_length/track_length  # Obtain distance unit (cm) by dividing recorded track length to actual track length
    location_in_cm = (recorded_location - recorded_startpoint) / distance_unit
    position_data['x_position_cm'] = np.asarray(location_in_cm, dtype=np.float16) # fill in dataframe

    return position_data


def smooth_position(position_data, loc_sampling_rate, lowpass=10):
    '''
    Smooth the data
    The position output from the Arduino is discrete. Many factors contribute to this incl. 
    DAC resolution from arduino, position calculation in Blender etc. Occationally, random noise in signal
    will make a datapoint fall in another bin. If it happens to have a spike in that time point, then
    the dwell time of the animal in that location is very low, leading to a exceedingly high firing rate
    need to use smoothing to remove this effect
    Note: this function should be called after trial starting point has been detected 
    (otherwise the teleport point will also be smoothed out) 
    '''
    b,a = signal.butter(5,lowpass/(loc_sampling_rate/2))
    position_data['x_position_cm'] = signal.filtfilt(b,a, position_data['x_position_cm'])

    return position_data

# calculate time from start of recording in seconds for each sampling point
def calculate_time(position_data, sampling_rate):
    print('Calculating time...')
    position_data['time_seconds'] = position_data['trial_number'].index/sampling_rate # convert sampling rate to time (seconds) by dividing by 30
    return position_data


# for each sampling point, calculates time from last sample point
def calculate_instant_dwell_time(position_data, sampling_rate):
    print('Calculating dwell time...')
    position_data['dwell_time_ms'] = 1/sampling_rate #the old way of diff will introduce nan at first number, which will affect dwell time for position later
    return position_data


def check_for_trial_restarts(trial_indices, loc_sampling_rate, min_time):
    #TODO: vectorizing this function should imporove performance
    skip = min_time*loc_sampling_rate
    new_trial_indices=[]
    for icount,i in enumerate(range(len(trial_indices)-1)):
        index_difference = trial_indices[icount] - trial_indices[icount+1]
        if index_difference > - skip: #ignore those points, teleporting to the start of track
            continue
        else:
            index = trial_indices[icount]
            new_trial_indices = np.append(new_trial_indices,index)
    return new_trial_indices


def get_new_trial_indices(position_data, loc_sampling_freq = setting.location_ds_rate):
    location_diff = position_data['x_position_cm'].diff()  # Get the raw location from the movement channel
    trial_indices = np.where(location_diff < -20)[0]# return indices where is new trial
    trial_indices = check_for_trial_restarts(trial_indices, loc_sampling_freq, 0.5)# check if trial_indices values are within 1500 of eachother, if so, delete
    new_trial_indices=np.hstack((0,trial_indices,len(location_diff))) # add start and end indicies so fills in whole arrays
    assert np.all(np.diff(trial_indices)> loc_sampling_freq), f"Some trials are too short. trial_indices={trial_indices}"
    return new_trial_indices


def fill_in_trial_array(new_trial_indices,trials):
    trial_count = 0
    for icount,i in enumerate(range(len(new_trial_indices)-1)):
        new_trial_index = int(new_trial_indices[icount])
        next_trial_index = int(new_trial_indices[icount+1])
        trials[new_trial_index:next_trial_index] = trial_count
        trial_count += 1
    return trials


# calculates trial number from location
def calculate_trial_numbers(position_data, output_path):
    print('Calculating trial numbers...')
    trials = np.zeros((position_data.shape[0]))
    new_trial_indices = get_new_trial_indices(position_data)
    trials = fill_in_trial_array(new_trial_indices,trials)
    PostSorting.vr_make_plots.plot_trials(trials, output_path)

    position_data['trial_number'] = np.asarray(trials, dtype=np.uint16)
    position_data['new_trial_indices'] = pd.Series(new_trial_indices)
    print('This mouse did ', int(max(trials)), ' trials')
    gc.collect()
    return position_data


# two continuous channels represent trial type
def load_first_trial_channel(recording_folder, first_trial_channel_suffix = setting.first_trial_channel_suffix):
    first = []
    # file_path = search4File(recording_folder + '/*' + first_trial_channel_suffix) #todo this should bw in params, it is 100 for me, 105 for Tizzy (I don't have _0)
    # trial_first = OpenEphys.loadContinuousFast(file_path)['data']
    trial_first=load_openephys_file(recording_folder + '/*' + first_trial_channel_suffix)
    first.append(trial_first)
    return np.asarray(first, dtype=np.uint8).ravel()


# two continuous channels represent trial type
def load_second_trial_channel(recording_folder, second_trial_channel = setting.second_trial_channel_suffix):
    second = []
    # file_path = search4File(recording_folder + '/*' + second_trial_channel) #todo this should bw in params, it is 100 for me, 105 for Tizzy (I don't have _0)
    # trial_second = OpenEphys.loadContinuousFast(file_path)['data']
    trial_second = load_openephys_file(recording_folder + '/*' + second_trial_channel)
    second.append(trial_second)
    return np.asarray(second, dtype=np.uint8).ravel()


def calculate_trial_types(position_data, first_ch, second_ch, output_path):
    print('Loading trial types from continuous...')
    PostSorting.vr_make_plots.plot_trial_channels(first_ch, second_ch, output_path)
    trial_type = np.zeros_like(second_ch); 
    trial_type[:]=np.nan
    
    new_trial_indices = position_data['new_trial_indices'].values
    new_trial_indices = new_trial_indices[~np.isnan(new_trial_indices)]

    print('Calculating trial type...')
    for icount,i in enumerate(range(len(new_trial_indices)-1)):
        new_trial_index = int(new_trial_indices[icount])
        next_trial_index = int(new_trial_indices[icount+1])
        second = stats.mode(second_ch[new_trial_index:next_trial_index])[0]
        first = stats.mode(first_ch[new_trial_index:next_trial_index])[0]
        if second < 2 and first < 2: # if beaconed
            trial_type[new_trial_index:next_trial_index] = int(0)
        if second > 2 and first < 2: # if non beaconed
            trial_type[new_trial_index:next_trial_index] = int(1)
        if second > 2 and first > 2: # if probe
            trial_type[new_trial_index:next_trial_index] = int(2)
    position_data['trial_type'] = np.asarray(trial_type, dtype=np.uint8)
    return (position_data,first_ch,second_ch)

def calculate_instant_velocity(position_data, output_path, sampling_rate, lowpass=False):
    print('Calculating velocity...')
    location = np.array(position_data['x_position_cm'], dtype=np.float32)

    sampling_points_per200ms = int(sampling_rate/5)
    end_of_loc_to_subtr = location[:-sampling_points_per200ms]# Rearrange arrays in a way that they just need to be subtracted from each other
    beginning_of_loc_to_subtr = location[:sampling_points_per200ms]# Rearrange arrays in a way that they just need to be subtracted from each other
    location_to_subtract_from = np.append(beginning_of_loc_to_subtr, end_of_loc_to_subtr)
    velocity = location - location_to_subtract_from

    # use new trial indices to fix velocity around teleports
    new_trial_indices = np.unique(position_data["new_trial_indices"][~np.isnan(position_data["new_trial_indices"])])
    for new_trial_indice in new_trial_indices:
        if new_trial_indice>sampling_rate/5: # ignores first trial index
            velocity[int(new_trial_indice-sampling_rate/5):int(new_trial_indice+sampling_rate/5)] =np.nan

    #now interpolate where these nan values are
    ok = ~np.isnan(velocity)
    xp = ok.ravel().nonzero()[0]
    fp = velocity[~np.isnan(velocity)]
    x  = np.isnan(velocity).ravel().nonzero()[0]
    velocity[np.isnan(velocity)] = np.interp(x, xp, fp)
    velocity = velocity*5

    # remove negative velocity
    velocity[velocity<0] = 0

    # do a low pass filter
    if lowpass:
        b, a = signal.butter(5,5/(sampling_rate/2))
        velocity = signal.filtfilt(b,a,velocity)

    position_data['velocity'] = velocity
    PostSorting.vr_make_plots.plot_velocity(velocity, output_path)

    return position_data

def running_mean(a, n):
    '''
    Calculates moving average

    input
        a : array,  to calculate averages on
        n : integer, number of points that is used for one average calculation

    output
        array, contains rolling average values (each value is the average of the previous n values)
    '''
    cumsum = np.cumsum(np.insert(a,0,0), dtype=float)
    return np.append(a[0:n-1], ((cumsum[n:] - cumsum[:-n]) / n))

def moving_sum(array, window):
    ret = np.cumsum(array, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window:]

def get_rolling_sum(array_in, window):
    if window > (len(array_in) / 3) - 1:
        print('Window is too big, plot cannot be made.')
    inner_part_result = moving_sum(array_in, window)
    edges = np.append(array_in[-2 * window:], array_in[: 2 * window])
    edges_result = moving_sum(edges, window)
    end = edges_result[window:math.floor(len(edges_result)/2)]
    beginning = edges_result[math.floor(len(edges_result)/2):-window]
    array_out = np.hstack((beginning, inner_part_result, end))
    return array_out

def get_avg_speed_200ms(position_data, output_path, loc_sampling_rate):
    print('Calculating average speed...')
    velocity = np.array(position_data['velocity'])  # Get the raw location from the movement channel
    sampling_points_per200ms = int(loc_sampling_rate/5)
    position_data['speed_per200ms'] = running_mean(velocity, sampling_points_per200ms)  # Calculate average speed at each point by averaging instant velocities
    PostSorting.vr_make_plots.plot_running_mean_velocity(position_data['speed_per200ms'], output_path)
    return position_data

def downsampled_position_data(raw_position_data, sampling_rate = setting.sampling_rate, down_sampled_rate = setting.location_ds_rate):
    position_data = pd.DataFrame()
    downsample_factor = int(sampling_rate/ down_sampled_rate)
    position_data["x_position_cm"] = raw_position_data["x_position_cm"][::downsample_factor]
    position_data["time_seconds"] =  raw_position_data["time_seconds"][::downsample_factor]
    position_data["speed_per200ms"] = raw_position_data["speed_per200ms"][::downsample_factor]
    position_data["trial_number"] = raw_position_data["trial_number"][::downsample_factor]
    position_data["trial_type"] = raw_position_data["trial_type"][::downsample_factor]

    return position_data

def syncronise_position_data(recording_folder, output_path, track_length):
    raw_position_data = pd.DataFrame()
    raw_position_data = calculate_track_location(raw_position_data, recording_folder, output_path, track_length)
    raw_position_data = calculate_trial_numbers(raw_position_data, output_path)
    raw_position_data = calculate_trial_types(raw_position_data, recording_folder, output_path)
    raw_position_data = calculate_time(raw_position_data)
    raw_position_data = calculate_instant_dwell_time(raw_position_data)
    raw_position_data = calculate_instant_velocity(raw_position_data, output_path)
    raw_position_data = get_avg_speed_200ms(raw_position_data, output_path)
    position_data = downsampled_position_data(raw_position_data)

    gc.collect()
    return raw_position_data, position_data
