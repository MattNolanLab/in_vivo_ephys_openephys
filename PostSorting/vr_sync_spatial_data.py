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
import OpenEphys
import setting
import scipy.signal as signal
from DataframeHelper import *
import scipy.signal as signal
from file_utility import search4File

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
    location[location <0.55] = 0.56 # deals with if the VR is switched off during recording - location value drops to 0 - min is usually 0.56 approx
    return location


""""
Loads raw location continuous channel from ACD1.continuous
# input: spatial dataframe, path to local recording folder, initialised parameters
# output: raw location as numpy array
"""

def get_raw_location(recording_folder, movement_channel = setting.movement_ch_suffix):
    print('Extracting raw location...')
    file_path = search4File(recording_folder + '/*' + movement_channel)
    if os.path.exists(file_path):
        location = OpenEphys.loadContinuousFast(file_path)['data']
        location=correct_for_restart(location)
        return np.asarray(location, dtype=np.float16).ravel()

    else:
        print('Movement data was not found.')
        return None

    # PostSorting.vr_make_plots.plot_movement_channel(location, prm)


'''
Normalizes recorded values from location channel to metric (cm)

input
    prm : object, parameters
    raw_data : array, electrophysiology and movement data file, contains recorded location values

output
    normalized_location_metric : array, contains normalized location values (in cm)

The standardization is computed by finding the minimum and maximum recorded location values (min, max), and then
subtracting min from max, to get the recorded track length (recorded_length). Then, this recorded_length is devided by
the real track length to get the distance unit that will be used for the conversion.

From every recorded location point, the recorded_startpoint (min) value is subtracted to make the first location = 0.
The recorded_startpoint may not be 0 due to the sampling rate. (Please note that this is not the beginning of the
ephys recording in time, but the smallest value recorded by the rotary encoder.) This value is then divided by the
distance unit calculated in the previous step to convert the rotary encoder values to metric.
'''

def calculate_track_location(position_data, recorded_location):
    # recorded_location = get_raw_location(recording_folder, setting.movement_ch) # get raw location from DAQ pin
    print('Converting raw location input to cm...')
    recorded_startpoint = min(recorded_location)
    recorded_endpoint = max(recorded_location)
    recorded_track_length = recorded_endpoint - recorded_startpoint
    distance_unit = recorded_track_length/setting.track_length  # Obtain distance unit (cm) by dividing recorded track length to actual track length
    location_in_cm = (recorded_location - recorded_startpoint) / distance_unit
    position_data['x_position_cm'] = np.asarray(location_in_cm, dtype=np.float16) # fill in dataframe
    return position_data


# calculate time from start of recording in seconds for each sampling point
def calculate_time(position_data, sampling_rate):
    print('Calculating time...')
    position_data=addCol2dataframe(position_data, {'time_seconds': position_data['trial_number'].index/sampling_rate})
    # position_data['time_seconds'] = position_data['trial_number'].index/30000 # convert sampling rate to time (seconds) by dividing by 30
    return position_data


# for each sampling point, calculates time from last sample point
def calculate_instant_dwell_time(position_data):
    print('Calculating dwell time...')
    position_data['dwell_time_ms'] = position_data['time_seconds'].diff() # [row] - [row-1]
    return position_data



"""
# calculates trial number from location
def calculate_trial_numbers(position_data, prm):
    print('Calculating trial numbers...')
    location= np.array(position_data['x_position_cm'])  # Get the raw location from the movement channel
    trials = np.zeros((len(location)))

    new_trial_index = 0
    trial_num = 1
    for i in range(len(location)):
        if i > 0 and (location[i-1]-location[i]) > 40 and (i-new_trial_index) > 1500:
            trial_num += 1
            new_trial_index = i
        trials[i] = trial_num

    position_data['trial_number'] = np.asarray(trials, dtype=np.uint8)
    print('This mouse did ', int(max(trials)), ' trials')
    PostSorting.vr_make_plots.plot_trials(trials, prm)
    return position_data
"""

def check_for_trial_restarts(trial_indices, skip):
    new_trial_indices=[]
    for icount,i in enumerate(range(len(trial_indices)-1)):
        index_difference = trial_indices[icount] - trial_indices[icount+1]
        if index_difference > - skip: #ignore those points
            continue
        else:
            index = trial_indices[icount+1] #only include when there is a complete trial
            new_trial_indices = np.append(new_trial_indices,index)
    return new_trial_indices


def get_new_trial_indices(position_data, skip):
    location_diff = position_data['x_position_cm'].diff()  # Get the raw location from the movement channel
    trial_indices = np.where(location_diff < -40)[0]# return indices where is new trial
    trial_indices = check_for_trial_restarts(trial_indices,skip)# check if trial_indices values are within 15000 of eachother, if so, delete
    new_trial_indices=np.hstack((0,trial_indices,len(location_diff))) # add start and end indicies so fills in whole arrays
    return new_trial_indices


def fill_in_trial_array(new_trial_indices,trials):
    trial_count = 1
    for icount,i in enumerate(range(len(new_trial_indices)-1)):
        new_trial_index = int(new_trial_indices[icount])
        next_trial_index = int(new_trial_indices[icount+1])
        trials[new_trial_index:next_trial_index] = trial_count
        trial_count += 1
    return trials


# calculates trial number from location
def calculate_trial_numbers(position_data,skip = 15000):
    print('Calculating trial numbers...')
    trials = np.zeros((position_data.shape[0]))
    new_trial_indices = get_new_trial_indices(position_data, skip)
    trials = fill_in_trial_array(new_trial_indices,trials)

    position_data['trial_number'] = np.asarray(trials, dtype=np.uint16)
    position_data['new_trial_indices'] = pd.Series(new_trial_indices)
    print('This mouse did ', int(max(trials)), ' trials')
    gc.collect()
    return position_data


# two continuous channels represent trial type
def load_first_trial_channel(recording_folder, first_trial_channel_suffix = setting.first_trial_channel_suffix):
    first = []
    file_path = search4File(recording_folder + '/*' + first_trial_channel_suffix) #todo this should bw in params, it is 100 for me, 105 for Tizzy (I don't have _0)
    trial_first = OpenEphys.loadContinuousFast(file_path)['data']
    first.append(trial_first)
    return np.asarray(first, dtype=np.uint8).ravel()


# two continuous channels represent trial type
def load_second_trial_channel(recording_folder, second_trial_channel = setting.second_trial_channel_suffix):
    second = []
    file_path = search4File(recording_folder + '/*' + second_trial_channel) #todo this should bw in params, it is 100 for me, 105 for Tizzy (I don't have _0)
    trial_second = OpenEphys.loadContinuousFast(file_path)['data']
    second.append(trial_second)
    return np.asarray(second, dtype=np.uint8).ravel()


def calculate_trial_types(position_data, first_ch, second_ch):
    print('Loading trial types from continuous...')

    trial_type = np.zeros((second_ch.shape[0]));trial_type[:]=np.nan
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
            trial_type[new_trial_index:next_trial_index] = int(2)
        if second > 2 and first > 2: # if probe
            trial_type[new_trial_index:next_trial_index] = int(1)
    position_data['trial_type'] = np.asarray(trial_type, dtype=np.uint8)
    return (position_data,first_ch,second_ch)


'''
Corrects for the very small negative values that are calculated as velocity when the mouse 'teleports' back
to the beginning of the track - from the end of the track to 0.

input
    prm : obejct, parameters
    velocity : numpy array, instant velocity

output
    velocity : array, instant velocity without teleport artefacts

It finds the velocity values that are smaller than -track_length+max_velocity, and adds track_length to them. These
values will be around the beginning of the track after the mouse finished the previous trial and jumped back to the
beginning.

After the first iteration, it finds the values that are <-10 (it is highly unlikely for a mouse to have that speed), it
replaces them with the previous location value.

An alternative solution may be to nto analyze this data.

'''

def fix_teleport(velocity, track_length = setting.track_length):
    max_velocity = max(velocity)
    # If the mouse goes from the end of the track to the beginning, the velocity would be a negative value
    # if velocity< (-1)*track_length + max_velocity, then track_length is added to the value
    too_small_indices = np.where(velocity < (-track_length + max_velocity))
    too_small_values = np.take(velocity, too_small_indices)
    to_insert = too_small_values + track_length
    np.put(velocity, too_small_indices, to_insert)  # replace small values with new correct value
    # if velocity is <-10 (due to the teleportation), the previous velocity value will be used
    small_velocity = np.where(velocity < -10)  # find where speed is < 10
    small_velocity = np.asanyarray(small_velocity)
    previous_velocity_index = small_velocity - 1  # find indices right before those in previous line
    previous_velocity = np.take(velocity, previous_velocity_index)
    np.put(velocity, small_velocity, previous_velocity)  # replace small speed values with previous value
    return velocity


'''
Calculates instant velocity for every sampling point

input
    prm : object, parameters
    location : numpy array, location values (metric)
    sampling_points_per200ms : number of sampling points in 200ms signal

output
    velocity : instant velocity

calls
    fix_teleport : function to fix issues arising from the fact that when the mouse restarts the trial, it is
    teleported back to the beginning of the track, and the velocity will be a very small negative value.

The location array is duplicated, and shifted in a way that it can be subtracted from the original to avoid loops.
(The shifted array is like : first 200ms data + first 200ms data again, rest of data without last 200ms, this is
subtracted from the original location array.)
'''

def calculate_instant_velocity(position_data, sampling_rate):
    print('Calculating velocity...')
    location = np.array(position_data['x_position_cm']) # Get the raw location from the movement channel
    sampling_points_per200ms = int(sampling_rate/5)
    end_of_loc_to_subtr = location[:-sampling_points_per200ms]# Rearrange arrays in a way that they just need to be subtracted from each other
    beginning_of_loc_to_subtr = location[:sampling_points_per200ms]# Rearrange arrays in a way that they just need to be subtracted from each other
    location_to_subtract_from = np.append(beginning_of_loc_to_subtr, end_of_loc_to_subtr)
    velocity = location - location_to_subtract_from
    velocity = fix_teleport(velocity)
    position_data['velocity'] = velocity/0.2 #unit: cm/s
    return position_data


'''
Calculates moving average

input
    a : array,  to calculate averages on
    n : integer, number of points that is used for one average calculation

output
    array, contains rolling average values (each value is the average of the previous n values)
'''


def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n:] / n

'''
Calculates moving average

input
    a : array,  to calculate averages on
    n : integer, number of points that is used for one average calculation

output
    array, contains rolling average values (each value is the average of the previous n values)
'''

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

def get_rolling_average(array_in, window):
    """Get the rolling average of with a specified moving window
    
    Arguments:
        array_in {np.narray} -- input array
        window {int} -- size of the moving window to get average
    
    Returns:
        np.narray -- moving average of the input array
    """
    if window > (len(array_in) / 3) - 1:
        print('Window is too big, plot cannot be made.')
    inner_part_result = moving_sum(array_in, window)
    edges = np.append(array_in[-2 * window:], array_in[: 2 * window])
    edges_result = moving_sum(edges, window)
    end = edges_result[window:math.floor(len(edges_result)/2)]
    beginning = edges_result[math.floor(len(edges_result)/2):-window]
    array_out = np.hstack((beginning, inner_part_result, end))/window 
    return array_out



def get_avg_speed(position_data, average_window):
    """Calculate the average speed over the window specified in the argument
    
    Arguments:
        position_data {pd.DataFrame} -- position data
        average_window {int} -- length of data to take average of 
    
    Returns:
        pd.DataFrame -- dataframe with average speed added
    """
    
    print('Calculating average speed...')
    velocity = np.array(position_data['velocity'])  # Get the raw location from the movement channel
    position_data['speed_per200ms'] = get_rolling_average(velocity, average_window)# Calculate average speed at each point by averaging instant velocities
    return position_data


def drop_columns_from_dataframe(position_data):
    #position_data = position_data.drop(['velocity'], axis=1)
    return position_data


def syncronise_position_data(recording_folder, prm):
    raw_position_data = pd.DataFrame()
    raw_position_data = calculate_track_location(raw_position_data, recording_folder, prm)
    raw_position_data = calculate_trial_numbers(raw_position_data, prm)
    raw_position_data = calculate_trial_types(raw_position_data, recording_folder, prm)
    raw_position_data = calculate_time(raw_position_data)
    raw_position_data = calculate_instant_dwell_time(raw_position_data)
    raw_position_data = calculate_instant_velocity(raw_position_data, prm)
    raw_position_data = get_avg_speed_200ms(raw_position_data, prm)
    #raw_position_data = drop_columns_from_dataframe(raw_position_data)

    if prm.cue_conditioned_goal:
        raw_position_data = PostSorting.vr_cued.add_goal_location(recording_folder, raw_position_data, prm)
        raw_position_data = PostSorting.vr_cued.offset_location_by_goal(raw_position_data)

    gc.collect()
    return raw_position_data
