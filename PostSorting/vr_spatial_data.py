import numpy as np
import pandas as pd
import PostSorting.parameters
import gc
import PostSorting.vr_stop_analysis
import PostSorting.vr_stop_analysis_new
import PostSorting.vr_make_plots
from scipy import stats
from tqdm import tqdm
from DataframeHelper import *

def calculate_total_trial_numbers(raw_position_data,processed_position_data):
    print('calculating total trial numbers for trial types')
    trial_numbers = np.array(raw_position_data['trial_number'])
    trial_type = np.array(raw_position_data['trial_type'])
    trial_data=np.transpose(np.vstack((trial_numbers, trial_type)))
    beaconed_trials = np.delete(trial_data, np.where(trial_data[:,1]>0),0)
    unique_beaconed_trials = np.unique(beaconed_trials[:,0])
    nonbeaconed_trials = np.delete(trial_data, np.where(trial_data[:,1]!=1),0)
    unique_nonbeaconed_trials = np.unique(nonbeaconed_trials[1:,0])
    probe_trials = np.delete(trial_data, np.where(trial_data[:,1]!=2),0)
    unique_probe_trials = np.unique(probe_trials[1:,0])

    processed_position_data.at[0,'beaconed_total_trial_number'] = len(unique_beaconed_trials)
    processed_position_data.at[0,'nonbeaconed_total_trial_number'] = len(unique_nonbeaconed_trials)
    processed_position_data.at[0,'probe_total_trial_number'] = len(unique_probe_trials)
    return processed_position_data


def find_dwell_time_in_bin_by_speed(dwell_time_per_sample, speed_ms, idx):
    time_in_bin = dwell_time_per_sample[idx]
    speed_in_bin = speed_ms[idx]
    time_in_bin_moving = np.sum(time_in_bin[np.where(speed_in_bin >= 1.5)])
    time_in_bin_stationary = np.sum(time_in_bin[np.where(speed_in_bin < 1.5)])
    return time_in_bin,time_in_bin_moving, time_in_bin_stationary

def find_dwell_time_in_bin(dwell_time_per_sample, idx):
    time_in_bin = dwell_time_per_sample[idx]
    return time_in_bin

def find_time_in_bin(time_per_sample, idx):
    time_in_bin = time_per_sample[idx]
    return time_in_bin


def find_speed_in_bin(speed_ms, idx):
    speed_in_bin = (np.nanmean(speed_ms[idx]))
    return speed_in_bin


def find_trial_type_in_bin(trial_types, locations, loc):
    trial_type_in_bin = stats.mode(trial_types[np.where(np.logical_and(locations > loc, locations <= (loc+1)))])[0]
    return trial_type_in_bin


"""
calculates speed for each location bin (0-200cm) across all trials

inputs:
    position_data : pandas dataframe containing position information for mouse during that session
    
outputs:
    position_data : with additional column added for processed data
"""

def bin_data_trial_by_trial(raw_position_data,processed_position_data,number_of_bins = 200):
    print('calculate binned data per trial...')
    number_of_trials = raw_position_data.trial_number.max() # total number of trials
    trials = np.array(raw_position_data['trial_number'])
    trial_types = np.array(raw_position_data['trial_type'])
    locations = np.array(raw_position_data['x_position_cm'])
    dwell_time_per_sample = np.array(raw_position_data['dwell_time_ms'])
    time_per_sample = np.array(raw_position_data['time_seconds'])

    bin_count =[]
    trial_number_in_bin = []
    trial_type_in_bin =[]
    binned_time_ms_per_trial =[]
    binned_apsolute_elapsed_time = []

    for t in tqdm(range(1,int(number_of_trials))):
    # for t in tqdm(range(1,int(10))):
        trial_locations = locations[np.where(trials == t)]
        trial_type = int(stats.mode(trial_types[np.where(trials == t)])[0])
        for loc in range(int(number_of_bins)):
            idx = getBinnedIdx(trial_locations,loc)
            time_in_bin = find_dwell_time_in_bin(dwell_time_per_sample, idx)
            apsolute_elapsed_time_in_bin = find_time_in_bin(time_per_sample, idx)

            trial_number_in_bin.append(int(t))
            bin_count.append(int(loc))
            trial_type_in_bin.append(int(trial_type))
            binned_time_ms_per_trial.append(np.float16(time_in_bin.sum()))
            binned_apsolute_elapsed_time.append(np.float16(apsolute_elapsed_time_in_bin))

    d = {'binned_time_ms_per_trial':binned_time_ms_per_trial,
        'trial_type_in_bin': trial_type_in_bin,
        'trial_number_in_bin': trial_number_in_bin,
        'binned_apsolute_elapsed_time': binned_apsolute_elapsed_time
    }
    processed_position_data = addCol2dataframe(processed_position_data, d)
    
    return processed_position_data


def getBinnedIdx(locations,loc):
    idx = np.where(np.logical_and(locations > loc, locations <= (loc+1)))
    return idx

def bin_data_over_trials(raw_position_data, processed_position_data, number_of_bins = 200):
    print('Calculating binned data over trials...')
    binned_data = pd.DataFrame(columns=['dwell_time_ms', 'dwell_time_ms_moving', 'dwell_time_ms_stationary', 'speed_in_bin'])
    number_of_trials = raw_position_data.trial_number.max() # total number of trials
    locations = np.array(raw_position_data['x_position_cm'])
    dwell_time_per_sample = np.array(raw_position_data['dwell_time_ms'])
    speed_ms = np.array(raw_position_data['speed_per200ms'])

    dwell_time_ms = []
    dwell_time_ms_moving =[]
    dwell_time_ms_stationary =[]
    speed_in_bin = []

    for loc in tqdm(range(int(number_of_bins))):
        idx = getBinnedIdx(locations, loc)
        time_in_bin,time_in_bin_moving, time_in_bin_stationary = find_dwell_time_in_bin_by_speed(dwell_time_per_sample, speed_ms, idx)
        speed_in_bin = find_speed_in_bin(speed_ms, idx)

        dwell_time_ms = np.float16(time_in_bin.sum())/number_of_trials
        dwell_time_ms_moving = np.float16(time_in_bin_moving)/number_of_trials
        dwell_time_ms_stationary = np.float16(time_in_bin_stationary)/number_of_trials
        speed_in_bin = np.float16(speed_in_bin)

        # binned_data = binned_data.append({"dwell_time_ms":  np.float16(sum(time_in_bin))/number_of_trials, "dwell_time_ms_moving":  np.float16(time_in_bin_moving)/number_of_trials, "dwell_time_ms_stationary":  np.float16(time_in_bin_stationary)/number_of_trials, "speed_in_bin": np.float16(speed_in_bin)}, ignore_index=True)

    d = {
        'binned_time_ms': dwell_time_ms,
        'binned_time_moving_ms': dwell_time_ms_moving ,
        'binned_time_stationary_ms': dwell_time_ms_stationary,
        'binned_speed_ms': speed_in_bin 
    }
    processed_position_data = addCol2dataframe(processed_position_data, d)

    return processed_position_data

def bin_speed_over_trials(raw_position_data,processed_position_data):
    print('Calculating binned data over trials...')
    binned_data = pd.DataFrame(columns=['speed_in_bin'])
    bin_size_cm,number_of_bins,bins = PostSorting.vr_stop_analysis.get_bin_size(raw_position_data)
    locations = np.array(raw_position_data['x_position_cm'])
    speed_ms = np.array(raw_position_data['speed_per200ms'])

    for loc in tqdm(range(int(number_of_bins))):
        idx = getBinnedIdx(locations,loc)
        speed_in_bin = find_speed_in_bin(speed_ms, idx)
        binned_data = binned_data.append({"speed_in_bin": np.float16(speed_in_bin)}, ignore_index=True)

    processed_position_data['binned_speed_ms'] = binned_data['speed_in_bin']
    return processed_position_data


def drop_columns_from_dataframe(raw_position_data):
    raw_position_data.drop(['dwell_time_seconds'], axis='columns', inplace=True, errors='ignore')
    #raw_position_data.drop(['velocity'], axis='columns', inplace=True, errors='ignore')
    return raw_position_data


def process_position(raw_position_data, prm, recording_to_process):
    processed_position_data = pd.DataFrame() # make dataframe for processed position data
    #processed_position_data = bin_data_over_trials_by_speed(raw_position_data,processed_position_data)
    processed_position_data = bin_speed_over_trials(raw_position_data,processed_position_data)
    processed_position_data = bin_data_trial_by_trial(raw_position_data,processed_position_data)
    processed_position_data = calculate_total_trial_numbers(raw_position_data, processed_position_data)
    processed_position_data = PostSorting.vr_stop_analysis.process_stops(raw_position_data,processed_position_data, prm)
    gc.collect()
    prm.set_total_length_sampling_points(raw_position_data.time_seconds.values[-1])  # seconds
    processed_position_data["new_trial_indices"] = raw_position_data["new_trial_indices"]
    #raw_position_data = drop_columns_from_dataframe(raw_position_data)

    print('-------------------------------------------------------------')
    print('position data processed')
    print('-------------------------------------------------------------')
    return raw_position_data, processed_position_data


#  for testing
def main():
    print('-------------------------------------------------------------')

    params = PostSorting.parameters.Parameters()

    recording_folder = 'C:/Users/s1466507/Documents/Ephys/test_overall_analysis/M5_2018-03-06_15-34-44_of'

    vr_spatial_data = process_position(recording_folder)


if __name__ == '__main__':
    main()
