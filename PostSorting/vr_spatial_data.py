import numpy as np
import os
import pandas as pd
import open_ephys_IO
import PostSorting.parameters
import itertools

prm = PostSorting.parameters.Parameters()


def get_raw_location(recording_folder):
    print('I am extracting raw location...')
    location = []
    file_path = recording_folder + '/' + prm.get_movement_channel()
    if os.path.exists(file_path):
        location = open_ephys_IO.get_data_continuous(prm, file_path)
    else:
        print('Movement data was not found.')
    return location


def calculate_track_location(position_data, recording_folder):

    recorded_location = get_raw_location(recording_folder)

    print('I am converting raw location input to cm...')
    recorded_startpoint = np.amin(recorded_location)
    recorded_endpoint = np.amax(recorded_location)
    recorded_track_length = recorded_endpoint - recorded_startpoint
    distance_unit = recorded_track_length/prm.get_track_length()  # Obtain distance unit (cm) by dividing recorded track length to actual track length
    location_in_cm = (recorded_location - recorded_startpoint) / distance_unit

    position_data['position_cm'] = location_in_cm # put in dataframe
    return position_data


def calculate_time(position_data):
    print('I am calculating time...')
    position_data['time_ms'] = position_data['position_cm'].index/30 # convert sampling rate to time by dividing by 30
    return position_data


def calculate_dwell_time(position_data):
    print('I am calculating dwell time...')
    position_data['dwell_time_ms'] = position_data['time_ms'].diff()
    return position_data


def calculate_trial_numbers(position_data):
    print('I am calculating trial numbers...')

    location_diff = position_data['position_cm'].diff()  # Get the raw location from the movement channel
    trials = np.zeros((len(location_diff)))

    new_trial_indices = np.where(location_diff < -150) # return indices where is new trial
    new_trial_indices = list(itertools.chain.from_iterable(new_trial_indices)) # needed to convert tuple to list
    unique_trials = np.arange(1, len(new_trial_indices), 1)

    for icount,i in enumerate(unique_trials):
        trial_start_indices = new_trial_indices[icount]
        next_trial_indices = new_trial_indices[icount+1]
        trials[trial_start_indices:next_trial_indices] = i

    position_data['trial_number'] = trials
    return position_data


def load_trial_types_from_continuous(recording_folder):

    first=[]
    file_path = recording_folder + '/' + prm.get_first_trial_channel() #todo this should bw in params, it is 100 for me, 105 for Tizzy (I don't have _0)
    trial_first = open_ephys_IO.get_data_continuous(prm, file_path)
    first.append(trial_first)
    first = np.asarray(first)

    second=[]
    file_path = recording_folder + '/' + prm.get_second_trial_channel() #todo this should bw in params, it is 100 for me, 105 for Tizzy (I don't have _0)
    trial_second = open_ephys_IO.get_data_continuous(prm, file_path)
    second.append(trial_second)
    second = np.asarray(second)

    return first, second


def calculate_trial_types(position_data, recording_folder):

    print('I am calculating trial types from continuous...')
    first, second = load_trial_types_from_continuous(recording_folder)

    trial_type = np.zeros((first.shape[1]));trial_type[:]=np.nan
    for point,p in enumerate(trial_type):
        if second[0,point] < 2: # if beaconed
            trial_type[point] = 0
        if second[0,point] > 2: # if non beaconed
            trial_type[point] = 1

    position_data['trial_type'] = trial_type
    return position_data


def calculate_instant_velocity(position_data):
    print('I am calculating velocity...')

    location = position_data.position_cm  # Get the raw location from the movement channel

    sampling_points_per200ms = int(prm.get_sampling_rate()/5)

    end_of_loc_to_subtr = location[:-sampling_points_per200ms]# Rearrange arrays in a way that they just need to be subtracted from each other
    beginning_of_loc_to_subtr = location[:sampling_points_per200ms]# Rearrange arrays in a way that they just need to be subtracted from each other

    location_to_subtract_from = np.append(beginning_of_loc_to_subtr, end_of_loc_to_subtr)
    velocity = location - location_to_subtract_from

    position_data['velocity'] = velocity

    return position_data


def calculate_stops(position_data):
    print('I am calculating stops...')
    speed = position_data.speed
    stops = np.zeros((len(speed)))

    for i in range(len(speed)):
        if i > prm.get_stop_threshold():
            stops[i] = 1
        else:
            stops[i] = 0

    position_data['stops'] = stops

    return position_data


def calculate_stop_times(position_data):
    stop_times = position_data.loc[position_data['stops'] == 1, 'time_ms']
    position_data['stop_times'] = stop_times
    return position_data



def filter_stops(position_data):
    print('I am filtering stops...')
    stops = position_data.stops
    position_data['filtered_stops'] = stops
    return position_data



def process_position_data(recording_folder):

    # make data frame
    position_data = pd.DataFrame(columns=['time_ms', 'dwell_time_ms', 'position_cm', 'velocity', 'speed', 'trial_number', 'trial_type', 'stops', 'filtered_stops', 'stop_times'])

    position_data = calculate_track_location(position_data, recording_folder)

    position_data = calculate_time(position_data)

    position_data = calculate_dwell_time(position_data)

    #position_data = calculate_instant_velocity(position_data)

    position_data = calculate_trial_numbers(position_data)

    position_data = calculate_trial_types(position_data, recording_folder)

    #position_data = calculate_stops(position_data)

    #position_data = filter_stops(position_data)

    #position_data = calculate_stop_times(position_data)

    return position_data


#  for testing
def main():
    print('-------------------------------------------------------------')

    params = PostSorting.parameters.Parameters()

    recording_folder = 'C:/Users/s1466507/Documents/Ephys/test_overall_analysis/M5_2018-03-06_15-34-44_of'

    vr_spatial_data = process_position_data(recording_folder)


if __name__ == '__main__':
    main()
