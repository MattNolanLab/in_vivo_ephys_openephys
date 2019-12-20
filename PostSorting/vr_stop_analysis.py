import numpy as np
import os
import pandas as pd
import math
import gc
from tqdm import tqdm
import setting
from DataframeHelper import *

def check_stop_threshold(recording_directory):
    parameters_path = recording_directory + '/parameters.txt'
    try:
        param_file_reader = open(parameters_path, 'r')
        parameters = param_file_reader.readlines()
        parameters = list([x.strip() for x in parameters])
        threshold = parameters[2]

    except Exception as ex:
        print('There is a problem with the parameter file.')
        print(ex)
    return np.float(threshold)


def keep_first_from_close_series(array, threshold):
    num_delete = 1
    while num_delete > 0:
        diff = np.ediff1d(array, to_begin= threshold + 1)
        to_delete = np.where(diff <= threshold)
        num_delete = len(to_delete[0])

        if num_delete > 0:
            array = np.delete(array, to_delete)
    return array


def get_beginning_of_track_positions(raw_position_data, skip):
    location = np.array(raw_position_data['x_position_cm']) # Get the raw location from the movement channel
    position = 0
    beginning_of_track = (location >= position) & (location <= position + 4)
    track_beginnings = findTransitWithHyst(beginning_of_track,skip)

    # beginning_plus_one = beginning_of_track + 1
    # track_beginnings = np.setdiff1d(beginning_of_track, beginning_plus_one)

    # track_beginnings = keep_first_from_close_series(track_beginnings, hysterisis)
    return track_beginnings


def remove_extra_stops(min_distance, stops):
    to_remove = []
    for stop in range(len(stops) - 1):
        current_stop = stops[stop]
        next_stop = stops[stop + 1]
        if 0 <= (next_stop - current_stop) <= min_distance:
            to_remove.append(stop+1)

    filtered_stops = np.asanyarray(stops)
    np.delete(filtered_stops, to_remove)
    return filtered_stops


def get_stop_times(raw_position_data, stop_threshold):
    stops = np.array([])
    speed = raw_position_data['speed_per200ms']
    threshold = stop_threshold
    low_speed = np.where(speed < threshold)
    low_speed = np.asanyarray(low_speed)
    low_speed_plus_one = low_speed + 1
    intersect = np.intersect1d(low_speed, low_speed_plus_one)
    stops = np.setdiff1d(low_speed, intersect)

    stops = remove_extra_stops(5, stops)
    return stops


def get_stops_on_trials_find_stops(raw_position_data, processed_position_data, all_stops, track_beginnings):
    print('extracting stops...')
    stop_locations = []
    stop_trials = []
    stop_trial_types = []
    location = np.array(raw_position_data['x_position_cm'].tolist())
    trial_type = np.array(raw_position_data['trial_type'].tolist())
    number_of_trials = raw_position_data.trial_number.max() # total number of trials
    all_stops = np.asanyarray(all_stops) #stop indices
    track_beginnings = np.asanyarray(track_beginnings)
    try:
        for trial in range(1,int(number_of_trials)-1):
            # Find the stop location in each trial
            beginning = track_beginnings[trial]
            end = track_beginnings[trial + 1]
            # all_stops = np.asanyarray(all_stops)
            stops_on_trial = all_stops[(beginning <= all_stops) & (all_stops <= end)]

            if len(stops_on_trial) > 0:
                stops = location[stops_on_trial]
                trial_types = trial_type[stops_on_trial]

                stop_locations=np.append(stop_locations,stops)
                stop_trial_types=np.append(stop_trial_types,trial_types)
                stop_trials=np.append(stop_trials,np.repeat(trial, len(stops)))
    except IndexError:
        print('indexerror')

    print('stops extracted')

    data = {"stop_location_cm": stop_locations,
         "stop_trial_number": stop_trials, 
         "stop_trial_type": stop_trial_types}
    
    #FIXME: stop_location is shorter than the total location, so there will be nan when the column is accessed
    processed_position_data = addCol2dataframe(processed_position_data, data)
    return processed_position_data


def calculate_stops(raw_position_data,processed_position_data, stop_threshold, detect_hysterisis):
    all_stops = get_stop_times(raw_position_data,stop_threshold)
    track_beginnings = get_beginning_of_track_positions(raw_position_data,detect_hysterisis)
    processed_position_data = get_stops_on_trials_find_stops(raw_position_data, processed_position_data, all_stops, track_beginnings)
    return processed_position_data


def calculate_stop_data_from_parameters(raw_position_data, processed_position_data, recording_directory):
    stop_threshold = check_stop_threshold(recording_directory)
    stop_locations, stop_trials, stop_trial_types = calculate_stops(raw_position_data, processed_position_data, stop_threshold)
    processed_position_data['stop_location_cm'] = pd.Series(stop_locations)
    processed_position_data['stop_trial_number'] = pd.Series(stop_trials)
    processed_position_data['stop_trial_type'] = pd.Series(stop_trial_types)
    return processed_position_data


def find_first_stop_in_series(processed_position_data):
    #FIXME stop_location_cm contains nan due to different column length
    stop_difference = np.diff(processed_position_data['stop_location_cm'].dropna().values)
    first_in_series_indices = np.where(stop_difference > 1)[0]
    print('Finding first stops in series')
    processed_position_data['first_series_location_cm'] = pd.Series(processed_position_data.stop_location_cm[first_in_series_indices].values)
    processed_position_data['first_series_trial_number'] = pd.Series(processed_position_data.stop_trial_number[first_in_series_indices].values)
    processed_position_data['first_series_trial_type'] = pd.Series(processed_position_data.stop_trial_type[first_in_series_indices].values)
    return processed_position_data


def take_first_reward_on_trial(rewarded_stop_locations,rewarded_trials):
    locations=[]
    trials=[]
    for tcount, trial in enumerate(np.unique(rewarded_trials)):
        trial_locations = rewarded_stop_locations[rewarded_trials == trial]
        if len(trial_locations) ==1:
            locations = np.append(locations,trial_locations)
            trials = np.append(trials,trial)
        if len(trial_locations) >1:
            locations = np.append(locations,trial_locations[0])
            trials = np.append(trials,trial)
    return np.array(locations), np.array(trials)


def find_rewarded_positions(raw_position_data,processed_position_data,reward_start=setting.reward_start, reward_stop=setting.reward_end):
    #FIXME Remove this anonying call to dropna()
    stop_locations = processed_position_data['first_series_location_cm'].dropna().values
    stop_trials = processed_position_data['first_series_trial_number'].dropna().values
    rewarded_stop_locations = stop_locations[(stop_locations >= reward_start) & (stop_locations < reward_stop)]
    rewarded_trials = stop_trials[(stop_locations >= reward_start) & (stop_locations < reward_stop)]

    locations, trials = take_first_reward_on_trial(rewarded_stop_locations, rewarded_trials)
    processed_position_data['rewarded_stop_locations'] = pd.Series(locations)
    processed_position_data['rewarded_trials'] = pd.Series(trials)
    return processed_position_data


def find_rewarded_positions_test(raw_position_data,processed_position_data):
    stop_locations = np.array(processed_position_data['stop_location_cm'])
    stop_trials = np.array(processed_position_data['stop_trial_number'])
    rewarded_stop_locations=[]
    rewarded_trials=[]
    for tcount, trial in enumerate(np.unique(stop_trials)):
            trial_locations = np.take(stop_locations, np.where(stop_trials == trial)[0])
            if len(trial_locations) > 0:
                for count in trial_locations:
                    if count >= 90 and count <= 110:
                        rewarded_stop_locations= np.append(rewarded_stop_locations, count)
                        rewarded_trials=np.append(rewarded_trials, trial)
                        break
    processed_position_data['rewarded_stop_locations'] = pd.Series(rewarded_stop_locations)
    processed_position_data['rewarded_trials'] = pd.Series(rewarded_trials)
    return processed_position_data


def get_bin_size(spatial_data):
    #bin_size_cm = 1
    track_length = spatial_data.x_position_cm.max()
    start_of_track = spatial_data.x_position_cm.min()
    #number_of_bins = (track_length - start_of_track)/bin_size_cm
    number_of_bins = 200
    bin_size_cm = (track_length - start_of_track)/number_of_bins
    bins = np.arange(start_of_track,track_length, 200)
    return bin_size_cm,number_of_bins, bins


def calculate_average_stops(raw_position_data,processed_position_data):
    stop_locations = processed_position_data.stop_location_cm.dropna().values
    bin_size_cm,number_of_bins, bins = get_bin_size(raw_position_data)
    number_of_trials = raw_position_data.trial_number.max() # total number of trials
    stops_in_bins = np.zeros((len(range(int(number_of_bins)))))
    for loc in range(int(number_of_bins)-1):
        stops_in_bin = len(stop_locations[np.where(np.logical_and(stop_locations > (loc), stop_locations <= (loc+1)))])/number_of_trials
        stops_in_bins[loc] = stops_in_bin

    processed_position_data['average_stops'] = pd.Series(stops_in_bins)
    processed_position_data['position_bins'] = pd.Series(range(int(number_of_bins)))
    return processed_position_data


def process_stops(raw_position_data,processed_position_data):
    processed_position_data = calculate_stops(raw_position_data, processed_position_data, 10.7)
    processed_position_data = calculate_average_stops(raw_position_data,processed_position_data)
    gc.collect()
    processed_position_data = find_first_stop_in_series(processed_position_data)
    processed_position_data = find_rewarded_positions(raw_position_data,processed_position_data)
    return processed_position_data


