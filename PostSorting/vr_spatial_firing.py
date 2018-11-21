import PostSorting.parameters
import numpy as np
import pandas as pd

prm = PostSorting.parameters.Parameters()


def add_columns_to_dataframe(spike_data):
    spike_data["x_position_cm"] = ""
    spike_data["y_position_cm"] = ""
    spike_data["trial_number"] = ""
    spike_data["trial_type"] = ""
    spike_data["speed_per200ms"] = ""
    spike_data["beaconed_position_cm"] = ""
    spike_data["beaconed_trial_number"] = ""
    spike_data["nonbeaconed_position_cm"] = ""
    spike_data["nonbeaconed_trial_number"] = ""
    spike_data["probe_position_cm"] = ""
    spike_data["probe_trial_number"] = ""
    spike_data["avg_spike_per_bin_b"] = ""
    spike_data["avg_spike_per_bin_nb"] = ""
    spike_data["avg_spike_per_bin_p"] = ""
    spike_data["normalised_b_spike_number"] = ""
    spike_data["normalised_nb_spike_number"] = ""
    spike_data["normalised_p_spike_number"] = ""

    return spike_data


def add_speed(spike_data, spatial_data_speed):
    for cluster_index in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster_index] - 1
        cluster_firing_indices = spike_data.firing_times[cluster_index]
        spike_data.speed_per200ms[cluster_index] = spatial_data_speed[cluster_firing_indices]
    return spike_data


def add_position_x(spike_data, spatial_data_x):
    for cluster_index in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster_index] - 1
        cluster_firing_indices = spike_data.firing_times[cluster_index]
        spike_data.x_position_cm[cluster_index] = spatial_data_x[cluster_firing_indices]
    return spike_data


def add_trial_number(spike_data, spatial_data_trial_number):
    for cluster_index in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster_index] - 1
        cluster_firing_indices = spike_data.firing_times[cluster_index]
        spike_data.trial_number[cluster_index] = spatial_data_trial_number[cluster_firing_indices].values.astype(np.uint16)
    return spike_data


def add_trial_type(spike_data, spatial_data_trial_type):
    for cluster_index in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster_index] - 1
        cluster_firing_indices = spike_data.firing_times[cluster_index]
        spike_data.trial_type[cluster_index] = spatial_data_trial_type[cluster_firing_indices].values.astype(np.uint8)
    return spike_data


def find_firing_location_indices(spike_data, spatial_data):
    print('I am extracting firing locations for each cluster...')
    spike_data = add_speed(spike_data, spatial_data.speed_per200ms)
    spike_data = add_position_x(spike_data, spatial_data.x_position_cm)
    spike_data = add_trial_number(spike_data, spatial_data.trial_number)
    spike_data = add_trial_type(spike_data, spatial_data.trial_type)
    return spike_data


def split_dataframe_columns_by_speed(cluster_index,spike_data,spike_data_movement,spike_data_stationary,above_threshold_indices,below_threshold_indices):
    spike_data_movement.trial_number[cluster_index] = spike_data.trial_number[above_threshold_indices].values.astype(np.uint16)
    spike_data_movement.x_position_cm[cluster_index] = spike_data.x_position_cm[above_threshold_indices].values.astype(np.float16)
    spike_data_movement.trial_type[cluster_index] = spike_data.trial_type[above_threshold_indices].values.astype(np.uint16)

    spike_data_stationary.trial_number[cluster_index] = spike_data.trial_number[below_threshold_indices].values.astype(np.uint16)
    spike_data_stationary.x_position_cm[cluster_index] = spike_data.x_position_cm[below_threshold_indices].values.astype(np.float16)
    spike_data_stationary.trial_type[cluster_index] = spike_data.trial_type[below_threshold_indices].values.astype(np.uint16)
    return spike_data_movement,spike_data_stationary


def split_spatial_firing_by_speed(spike_data):
    spike_data_movement = spike_data.copy()
    spike_data_stationary = spike_data.copy()
    movement_threshold=2.5
    for cluster_index in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster_index] - 1
        above_threshold_indices = np.where(spike_data.speed_per200ms[cluster_index] > movement_threshold)[0]
        below_threshold_indices = np.where(spike_data.speed_per200ms[cluster_index] < movement_threshold)[0]
        spike_data_movement, spike_data_stationary = split_dataframe_columns_by_speed(spike_data,spike_data_movement,spike_data_stationary,above_threshold_indices,below_threshold_indices)
    return spike_data, spike_data_movement, spike_data_stationary


def split_spatial_firing_by_trial_type(spike_data):
    print('I am splitting firing locations by trial type...')
    for cluster_index in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster_index] - 1
        cluster_df = spike_data.loc[[cluster_index]] # dataframe for that cluster
        trials = np.array(cluster_df['trial_number'].tolist())
        locations = np.array(cluster_df['x_position_cm'].tolist())
        trial_type = np.array(cluster_df['trial_type'].tolist())
        # index out of range error in following line
        try:
            beaconed_locations = np.take(locations, np.where(trial_type == 0)[1]) #split location and trial number
            nonbeaconed_locations = np.take(locations,np.where(trial_type == 1)[1])
            probe_locations = np.take(locations, np.where(trial_type == 2)[1])
            beaconed_trials = np.take(trials, np.where(trial_type == 0)[1])
            nonbeaconed_trials = np.take(trials, np.where(trial_type == 1)[1])
            probe_trials = np.take(trials, np.where(trial_type == 2)[1])

            spike_data.at[cluster_index, 'beaconed_position_cm'] = list(beaconed_locations)
            spike_data.at[cluster_index, 'beaconed_trial_number'] = list(beaconed_trials)
            spike_data.at[cluster_index, 'nonbeaconed_position_cm'] = list(nonbeaconed_locations)
            spike_data.at[cluster_index, 'nonbeaconed_trial_number'] = list(nonbeaconed_trials)
            spike_data.at[cluster_index, 'probe_position_cm'] = list(probe_locations)
            spike_data.at[cluster_index, 'probe_trial_number'] = list(probe_trials)
        except IndexError:
            continue
    return spike_data


def split_spatial_firing_by_trial_type_test(spike_data):
    print('I am splitting firing locations by trial type...')
    for cluster_index in range(len(spike_data)):
        spatial_firing_data = pd.DataFrame() # make dataframe
        cluster_index = spike_data.cluster_id.values[cluster_index] - 1

        spatial_firing_data['trial_number'] = spike_data.at[cluster_index, 'trial_number']
        spatial_firing_data['x_position_cm'] = spike_data.at[cluster_index, 'x_position_cm']
        spatial_firing_data['trial_type'] = spike_data.at[cluster_index, 'trial_type']

        spike_data.at[cluster_index,'beaconed_trial_number'] = list(np.where(spatial_firing_data['trial_type'] == 0, spatial_firing_data['trial_number']))
        spike_data.at[cluster_index,'nonbeaconed_trial_number'] = list(np.where(spatial_firing_data['trial_type'] == 1, spatial_firing_data['trial_number']))
        spike_data.at[cluster_index,'probe_trial_number'] = list(np.where(spatial_firing_data['trial_type'] == 2, spatial_firing_data['trial_number']))
        spike_data.at[cluster_index,'beaconed_position_cm'] = list(np.where(spatial_firing_data['trial_type'] == 0, spatial_firing_data['x_position_cm']))
        spike_data.at[cluster_index,'nonbeaconed_position_cm'] = list(np.where(spatial_firing_data['trial_type'] == 1, spatial_firing_data['x_position_cm']))
        spike_data.at[cluster_index,'probe_position_cm'] = list(np.where(spatial_firing_data['trial_type'] == 2, spatial_firing_data['x_position_cm']))
    return spike_data



def process_spatial_firing(spike_data, spatial_data):
    spike_data = add_columns_to_dataframe(spike_data)
    spike_data = find_firing_location_indices(spike_data, spatial_data)
    spike_data, spike_data_movement, spike_data_stationary = split_spatial_firing_by_speed(spike_data)
    spike_data = split_spatial_firing_by_trial_type(spike_data)
    spike_data_movement = split_spatial_firing_by_trial_type(spike_data_movement)
    spike_data_stationary = split_spatial_firing_by_trial_type(spike_data_stationary)
    #spike_data = split_spatial_firing_by_trial_type_test(spike_data)
    return spike_data, spike_data_movement, spike_data_stationary
