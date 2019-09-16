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


def get_stop_times(raw_position_data, stop_threshold):
    stops = np.array([])
    speed = np.array(raw_position_data['speed_per200ms'].tolist())
    locations = np.array(raw_position_data['x_position_cm'].tolist())

    threshold = stop_threshold
    stops = np.take(locations[np.where(speed < threshold)])

    stops = remove_extra_stops(5, stops)
    return stops
    
    
def calculate_stops(raw_position_data,processed_position_data, threshold):
    all_stops = get_stop_locations(raw_position_data,threshold)
    processed_position_data = get_stops_on_trials_find_stops(raw_position_data, processed_position_data, all_stops, track_beginnings)
    return processed_position_data
    
    
def calculate_stop_data_from_parameters(raw_position_data, processed_position_data, recording_directory):
    stop_threshold = check_stop_threshold(recording_directory)
    stop_locations, stop_trials, stop_trial_types = calculate_stops(raw_position_data, processed_position_data, stop_threshold)
    processed_position_data['stop_location_cm'] = pd.Series(stop_locations)
    processed_position_data['stop_trial_number'] = pd.Series(stop_trials)
    processed_position_data['stop_trial_type'] = pd.Series(stop_trial_types)
    return processed_position_data

    
