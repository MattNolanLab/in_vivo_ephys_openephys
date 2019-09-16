import numpy as np
import os
import pandas as pd
import math
import gc


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


def get_stop_locations(raw_position_data, stop_threshold):
    stops = np.array([])
    speed = np.array(raw_position_data['speed_per200ms'].tolist())
    locations = np.array(raw_position_data['x_position_cm'].tolist())
    trials = np.array(raw_position_data['trial_number'].tolist())
    types = np.array(raw_position_data['trial_type'].tolist())

    threshold = stop_threshold
    stop_locs = np.take(locations, np.where(speed < threshold)[0])
    stop_trials = np.take(trials, np.where(speed < threshold)[0])
    stop_types = np.take(types, np.where(speed < threshold)[0])
    
    stops = remove_extra_stops(5, stop_locs)
    #stops = np.hstack((stop_locs, stop_trials, stop_types))
    return stop_locs, stop_trials, stop_types
    
    
def calculate_stops(raw_position_data,processed_position_data, threshold):
    stop_locs, stop_trials, stop_types = get_stop_locations(raw_position_data,threshold)
    return stop_locs, stop_trials, stop_types
    
    
def calculate_stop_data_from_parameters(raw_position_data, processed_position_data, recording_directory):
    stop_threshold = check_stop_threshold(recording_directory)
    stop_locations, stop_trials, stop_trial_types = calculate_stops(raw_position_data, processed_position_data, stop_threshold)
    processed_position_data['stop_location_cm'] = pd.Series(stop_locations)
    processed_position_data['stop_trial_number'] = pd.Series(stop_trials)
    processed_position_data['stop_trial_type'] = pd.Series(stop_trial_types)
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
    track_length = spatial_data.x_position_cm.max()
    start_of_track = spatial_data.x_position_cm.min()
    number_of_bins = 200
    bin_size_cm = (track_length - start_of_track)/number_of_bins
    bins = np.arange(start_of_track,track_length, 200)
    return bin_size_cm,number_of_bins, bins


def calculate_average_stops(raw_position_data,processed_position_data):
    stop_locations = processed_position_data.stop_location_cm.values
    stop_locations = stop_locations[~np.isnan(stop_locations)] #need to deal with
    bin_size_cm,number_of_bins, bins = get_bin_size(raw_position_data)
    number_of_trials = raw_position_data.trial_number.max() # total number of trials
    stops_in_bins = np.zeros((len(range(int(number_of_bins)))))
    for loc in range(int(number_of_bins)-1):
        stops_in_bin = len(stop_locations[np.where(np.logical_and(stop_locations > (loc), stop_locations <= (loc+1)))])/number_of_trials
        stops_in_bins[loc] = stops_in_bin

    processed_position_data['average_stops'] = pd.Series(stops_in_bins)
    processed_position_data['position_bins'] = pd.Series(range(int(number_of_bins)))
    return processed_position_data




def process_stops(raw_position_data,processed_position_data, prm,recording_directory):
    processed_position_data = calculate_stop_data_from_parameters(raw_position_data, processed_position_data, recording_directory)
    processed_position_data = calculate_average_stops(raw_position_data,processed_position_data)
    gc.collect()
    processed_position_data = find_rewarded_positions_test(raw_position_data,processed_position_data)
    return processed_position_data

