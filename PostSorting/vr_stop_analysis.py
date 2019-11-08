import numpy as np
import os
import pandas as pd
import math
import gc
import PostSorting.parameters

def keep_first_from_close_series(array, threshold):
    num_delete = 1
    while num_delete > 0:
        diff = np.ediff1d(array, to_begin= threshold + 1)
        to_delete = np.where(diff <= threshold)
        num_delete = len(to_delete[0])

        if num_delete > 0:
            array = np.delete(array, to_delete)
    return array


def get_beginning_of_track_positions(raw_position_data):
    #location = np.array(raw_position_data["x_position_cm"]) # Get the raw location from the movement channel
    #position = 0
    #beginning_of_track = np.where((location >= position) & (location <= position + 4))
    #beginning_of_track = np.asanyarray(beginning_of_track)
    #beginning_plus_one = beginning_of_track + 1
    #beginning_plus_one = np.asanyarray(beginning_plus_one)
    #track_beginnings = np.setdiff1d(beginning_of_track, beginning_plus_one)

    #track_beginnings = keep_first_from_close_series(track_beginnings, 30000)
    #return track_beginnings

    #return track_beginnings

    # track beginnings is returned as the start of a new trial surely?
    # so why aren't we using new_trial_indices from raw?
    new_trial_indices = raw_position_data["new_trial_indices"][~np.isnan(raw_position_data["new_trial_indices"])]
    return new_trial_indices


def remove_extra_stops(min_distance, stops):
    to_remove = []
    for stop in range(len(stops) - 1):
        current_stop = stops[stop]
        next_stop = stops[stop + 1]
        if 0 <= (next_stop - current_stop) <= min_distance:
            to_remove.append(stop+1)

    filtered_stops = np.asanyarray(stops)
    filtered_stops = np.delete(filtered_stops, to_remove)
    return filtered_stops


def get_stop_times(raw_position_data, stop_threshold):
    stops = np.array([])
    speed = np.array(raw_position_data["speed_per200ms"])

    threshold = stop_threshold
    low_speed = np.where(speed < threshold)
    low_speed = np.asanyarray(low_speed)
    low_speed_plus_one = low_speed + 1
    intersect = np.intersect1d(low_speed, low_speed_plus_one)
    stops = np.setdiff1d(low_speed, intersect)

    stops = remove_extra_stops(10, stops)
    return stops


def get_stops_on_trials_find_stops(raw_position_data, processed_position_data, all_stops, track_beginnings):
    print('extracting stops...')
    stop_locations = []
    stop_trials = []
    stop_trial_types = []
    location = np.array(raw_position_data["x_position_cm"])
    trial_type = np.array(raw_position_data["trial_type"])
    number_of_trials = raw_position_data.trial_number.max() # total number of trials
    all_stops = np.asanyarray(all_stops)
    track_beginnings = np.asanyarray(track_beginnings)

    for trial in range(0,int(number_of_trials)):
        beginning = track_beginnings[trial]
        if trial == int(number_of_trials)-1: #if last trial
            end = len(location)-1 # this returns the last index
        else:
            end = track_beginnings[trial + 1] # end of trial index

        stops_on_trial_indices = (np.where((beginning <= all_stops) & (all_stops <= end)))

        stops_on_trial = np.take(all_stops, stops_on_trial_indices)
        if len(stops_on_trial) > 0:
            stops = np.take(location, stops_on_trial)
            trial_types = np.take(trial_type, stops_on_trial)

            stop_locations=np.append(stop_locations,stops[0])
            stop_trial_types=np.append(stop_trial_types,trial_types[0])
            stop_trials=np.append(stop_trials,np.repeat(trial+1, len(stops[0])))

    print('stops extracted')

    df1 = pd.DataFrame({"stop_location_cm": pd.Series(stop_locations), "stop_trial_number": pd.Series(stop_trials), "stop_trial_type": pd.Series(stop_trial_types)})
    processed_position_data = pd.concat([processed_position_data, df1], axis=1)
    return processed_position_data


def calculate_stops(raw_position_data,processed_position_data, prm):
    #all_stops = get_stop_times(raw_position_data, prm.get_stop_threshold())
    #track_beginnings = get_beginning_of_track_positions(raw_position_data)
    #processed_position_data = get_stops_on_trials_find_stops(raw_position_data, processed_position_data, all_stops, track_beginnings)

    processed_position_data = get_stops_from_binned_speed(processed_position_data, prm)

    return processed_position_data

def get_stops_from_binned_speed(processed_position_data, prm):
    stop_threshold = prm.get_stop_threshold()
    cue_conditioned = prm.get_cue_conditioned_goal()

    n_beaconed_trials = int(processed_position_data.beaconed_total_trial_number[0])
    n_nonbeaconed_trials = int(processed_position_data.nonbeaconed_total_trial_number[0])
    n_probe_trials = int(processed_position_data.probe_total_trial_number[0])

    n_total = n_beaconed_trials + n_nonbeaconed_trials + n_probe_trials

    speed_trials_binned = list(processed_position_data.speed_trials_binned[:n_total])
    speed_trial_numbers = list(processed_position_data.speed_trial_numbers[:n_total])
    speed_trial_types = list(processed_position_data.speed_trial_types[:n_total])

    stop_location_cm = []
    stop_trial_number = []
    stop_trial_types = []

    last_was_stop = False
    for i in range(len(speed_trials_binned)):
        bin_counter = 0.5
        for speed_in_bin in speed_trials_binned[i]:
            if speed_in_bin<stop_threshold and last_was_stop is False:
                if cue_conditioned:
                    goal_location = processed_position_data.goal_location[i]
                    stop_location_cm.append(bin_counter-goal_location)
                else:
                    stop_location_cm.append(bin_counter)
                stop_trial_number.append(speed_trial_numbers[i])
                stop_trial_types.append(speed_trial_types[i])
                last_was_stop = True
            elif speed_in_bin>stop_threshold and last_was_stop is True:
                last_was_stop = False
            bin_counter+=1

    print('stops extracted')

    df1 = pd.DataFrame({"stop_location_cm": pd.Series(stop_location_cm),
                        "stop_trial_number": pd.Series(stop_trial_number),
                        "stop_trial_type": pd.Series(stop_trial_types)})
    processed_position_data = pd.concat([processed_position_data, df1], axis=1)

    return processed_position_data

def calculate_stop_data_from_parameters(raw_position_data, processed_position_data, recording_directory, prm):
    processed_position_data = calculate_stops(raw_position_data, processed_position_data, prm)
    return processed_position_data


def find_first_stop_in_series(processed_position_data):
    #stop_difference = np.array(processed_position_data['stop_location_cm'].diff())
    #first_in_series_indices = np.where(stop_difference > 1)[0]
    #print('Finding first stops in series')
    #processed_position_data['first_series_location_cm'] = pd.Series(processed_position_data.stop_location_cm[first_in_series_indices].values)
    #processed_position_data['first_series_trial_number'] = pd.Series(processed_position_data.stop_trial_number[first_in_series_indices].values)
    #processed_position_data['first_series_trial_type'] = pd.Series(processed_position_data.stop_trial_type[first_in_series_indices].values)
    #return processed_position_data

    trial_numbers = np.array([])
    trial_stops = np.array([])
    trial_types = np.array([])

    unique_trial_numbers = np.unique(np.array(processed_position_data['stop_trial_number']))
    unique_trial_numbers = unique_trial_numbers[~np.isnan(unique_trial_numbers)]  # remove nans

    for trial_number in unique_trial_numbers:
        stops = np.array(processed_position_data['stop_location_cm'])[
            np.array(processed_position_data['stop_trial_number']) == trial_number]
        trial_type = np.array(processed_position_data['stop_trial_type'])[
            np.array(processed_position_data['stop_trial_number']) == trial_number][0]
        first_trial_stop = min(stops)

        trial_numbers = np.append(trial_numbers, trial_number)
        trial_stops = np.append(trial_stops, first_trial_stop)
        trial_types = np.append(trial_types, trial_type)

    processed_position_data['first_series_location_cm'] = pd.Series(trial_stops)
    processed_position_data['first_series_trial_number'] = pd.Series(trial_numbers)
    processed_position_data['first_series_trial_type'] = pd.Series(trial_types)
    return processed_position_data


def take_first_reward_on_trial(rewarded_stop_locations,rewarded_trials):
    locations=[]
    trials=[]
    for tcount, trial in enumerate(np.unique(rewarded_trials)):
        trial_locations = np.take(rewarded_stop_locations, np.where(rewarded_trials == trial)[0])
        if len(trial_locations) ==1:
            locations = np.append(locations,trial_locations)
            trials = np.append(trials,trial)
        if len(trial_locations) >1:
            locations = np.append(locations,trial_locations[0])
            trials = np.append(trials,trial)
    return np.array(locations), np.array(trials)


def find_rewarded_positions(raw_position_data,processed_position_data):
    stop_locations = np.array(processed_position_data['first_series_location_cm'])
    stop_trials = np.array(processed_position_data['first_series_trial_number'])
    rewarded_stop_locations = np.take(stop_locations, np.where(np.logical_and(stop_locations >= 88, stop_locations < 110))[0])
    rewarded_trials = np.take(stop_trials, np.where(np.logical_and(stop_locations >= 88, stop_locations < 110))[0])

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


def get_bin_size(spatial_data, prm):
    bin_size_cm = 1
    track_length = prm.get_track_length()
    start_of_track = 0
    #number_of_bins = (track_length - start_of_track)/bin_size_cm
    number_of_bins = int(track_length/bin_size_cm)
    #bin_size_cm = (track_length - start_of_track)/number_of_bins
    bins = np.arange(start_of_track,track_length, number_of_bins)
    return bin_size_cm, number_of_bins, bins


def calculate_average_stops(raw_position_data, processed_position_data, prm):
    stop_locations = processed_position_data.stop_location_cm.values
    stop_locations = stop_locations[~np.isnan(stop_locations)] #need to deal with
    bin_size_cm, number_of_bins, bins = get_bin_size(raw_position_data, prm)
    number_of_trials = raw_position_data.trial_number.max() # total number of trials
    stops_in_bins = np.zeros((len(range(int(number_of_bins)))))

    for loc in range(int(number_of_bins)-1):
        stops_in_bin = len(stop_locations[np.where(np.logical_and(stop_locations > (loc), stop_locations <= (loc+1)))])/number_of_trials
        stops_in_bins[loc] = stops_in_bin

    processed_position_data['average_stops'] = pd.Series(stops_in_bins)
    processed_position_data['position_bins'] = pd.Series(range(int(number_of_bins)))
    return processed_position_data


def process_stops(raw_position_data,processed_position_data, prm, recording_directory):
    processed_position_data = calculate_stop_data_from_parameters(raw_position_data, processed_position_data, recording_directory, prm)
    processed_position_data = calculate_average_stops(raw_position_data,processed_position_data, prm)
    gc.collect()
    processed_position_data = find_first_stop_in_series(processed_position_data)
    processed_position_data = find_rewarded_positions(raw_position_data,processed_position_data)
    return processed_position_data


