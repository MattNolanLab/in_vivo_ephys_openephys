import numpy as np
import os
import pandas as pd
import math
import gc
import PostSorting.parameters
import settings

def get_stops_from_binned_speed(processed_position_data, stop_threshold):
    stop_location_cm = []

    for index, trial_row in processed_position_data.iterrows():
        trial_row = trial_row.to_frame().T.reset_index(drop=True)
        speeds_binned = np.array(trial_row["speeds_binned"].iloc[0])
        position_bin_centres = np.array(trial_row["position_bin_centres"].iloc[0])
        stop_locations_on_trial = position_bin_centres[speeds_binned < stop_threshold]
        stop_location_cm.append(stop_locations_on_trial)

    processed_position_data["stop_location_cm"] = stop_location_cm
    return processed_position_data


def calculate_average_stops(processed_position_data):
    average_stop_location_cm = []

    for index, trial_row in processed_position_data.iterrows():
        trial_row = trial_row.to_frame().T.reset_index(drop=True)
        stop_location_cm = np.array(trial_row["stop_location_cm"].iloc[0])
        avg_stop_location = np.nanmean(stop_location_cm)
        average_stop_location_cm.append(avg_stop_location)

    processed_position_data["average_stop_location_cm"] = average_stop_location_cm
    return processed_position_data

def calculate_first_stops(processed_position_data):
    first_stop_location_cm = []

    for index, trial_row in processed_position_data.iterrows():
        trial_row = trial_row.to_frame().T.reset_index(drop=True)
        stop_location_cm = np.array(trial_row["stop_location_cm"].iloc[0])
        if len(stop_location_cm)==0:
            first_stop_location = np.nan
        else:
            first_stop_location = stop_location_cm[0]
        first_stop_location_cm.append(first_stop_location)

    processed_position_data["first_stop_location_cm"] = first_stop_location_cm
    return processed_position_data

def calculate_rewarded_stops(processed_position_data):
    reward_stop_location_cm = []

    for index, trial_row in processed_position_data.iterrows():
        trial_row = trial_row.to_frame().T.reset_index(drop=True)
        stop_location_cm = np.array(trial_row["stop_location_cm"].iloc[0])

        if ('reward_loc' in processed_position_data.columns and len(processed_position_data.reward_loc.unique())>1):
            reward_start = int(processed_position_data.at[index, 'reward_loc'])
            reward_end = reward_start + 22
            print('reward_start:',reward_start)
            reward_locations = stop_location_cm[(stop_location_cm > reward_start) & (stop_location_cm < reward_end)]
            print(reward_locations)
        else:
            reward_locations = stop_location_cm[(stop_location_cm > settings.reward_start) & (stop_location_cm < settings.reward_end)]
        
        

        if len(reward_locations)==0:
            reward_locations = []
        reward_stop_location_cm.append(list(reward_locations))

    processed_position_data["reward_stop_location_cm"] = reward_stop_location_cm
    return processed_position_data

def calculate_rewarded_trials(processed_position_data):
    rewarded_trials = []
    for index, trial_row in processed_position_data.iterrows():
        trial_row = trial_row.to_frame().T.reset_index(drop=True)
        reward_stop_location_cm = np.array(trial_row["reward_stop_location_cm"].iloc[0])
        if len(reward_stop_location_cm)==0:
            rewarded = False
        else:
            rewarded = True
        rewarded_trials.append(rewarded)

    print(f'There are totally {np.sum(rewarded_trials)} rewarded trials')
    processed_position_data["rewarded"] = rewarded_trials
    return processed_position_data

def process_stops(processed_position_data,stop_threshold):
    processed_position_data = get_stops_from_binned_speed(processed_position_data, stop_threshold)
    processed_position_data = calculate_average_stops(processed_position_data)
    processed_position_data = calculate_first_stops(processed_position_data)
    processed_position_data = calculate_rewarded_stops(processed_position_data)
    processed_position_data = calculate_rewarded_trials(processed_position_data)
    return processed_position_data


def read_blender_log(log_path):
    # Read log file saved by blender
    headers = ['time','position','speed','speed_over_gain','reward_is_received','reward_is_failed','lick_detected',
        'tone_played','track_idx','tot_trials','gain_mod','rz_start','rz_stop', 'sync_pulse']
    df_blender = pd.read_csv(log_path,skiprows=4,sep=';',names=headers)
    return df_blender