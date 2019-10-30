import numpy as np
import pandas as pd
import PostSorting.parameters
import matplotlib.pyplot as plt

def add_goal_locations(raw_position_data, processed_position_data, prm):
    # gets goal location from raw and places it in processed_position for all, beaconed and non_beaconed

    goal_location = []
    goal_location_trial_numbers = []

    goal_location_beaconed = []
    goal_location_beaconed_trial_number = []

    goal_location_non_beaconed = []
    goal_location_non_beaconed_trial_number = []

    for trial_number in range(1, max(raw_position_data["trial_number"] + 1)):
        trial_type = np.array(raw_position_data['trial_type'][np.array(raw_position_data['trial_number']) == trial_number])[0]
        trial_goal_position_cm = np.array(raw_position_data['goal_location_cm'][np.array(raw_position_data['trial_number']) == trial_number])[0]

        goal_location.append(trial_goal_position_cm)
        goal_location_trial_numbers.append(trial_number)

        if trial_type == 0:
            goal_location_beaconed.append(trial_goal_position_cm)
            goal_location_beaconed_trial_number.append(trial_number)
        elif trial_type == 1:
            goal_location_non_beaconed.append(trial_goal_position_cm)
            goal_location_non_beaconed_trial_number.append(trial_number)

    processed_position_data['goal_location'] = pd.Series(goal_location)
    processed_position_data['goal_location_trial_numbers'] = pd.Series(goal_location_trial_numbers)

    # trial type specifics speed bins
    processed_position_data['goal_location_beaconed'] = pd.Series(goal_location_beaconed)
    processed_position_data['goal_location_beaconed_trial_number'] = pd.Series(goal_location_beaconed_trial_number)
    processed_position_data['goal_location_non_beaconed'] = pd.Series(goal_location_non_beaconed)
    processed_position_data['goal_location_non_beaconed_trial_number'] = pd.Series(goal_location_non_beaconed_trial_number)

    return processed_position_data



def calculate_binned_speed(raw_position_data,processed_position_data, prm):
    numbers_of_bins = get_number_of_bins(prm)
    bin_size_cm = get_bin_size(prm, numbers_of_bins)

    speed_trials_binned = []
    speed_trial_numbers = []

    speed_trials_beaconed = []
    speed_trials_beaconed_trial_number = []

    speed_trials_non_beaconed = []
    speed_trials_non_beaconed_trial_number = []

    for trial_number in range(1, max(raw_position_data["trial_number"]+1)):
        trial_type = np.array(raw_position_data['trial_type'][np.array(raw_position_data['trial_number']) == trial_number])[0]

        trial_x_position_cm = np.array(raw_position_data['x_position_cm'][np.array(raw_position_data['trial_number']) == trial_number])
        trial_speeds = np.array(raw_position_data['speed_per200ms'][np.array(raw_position_data['trial_number']) == trial_number])

        min_pos = min(trial_x_position_cm)
        bins = np.arange(min_pos, min_pos + prm.track_length, bin_size_cm)

        bin_means = (np.histogram(trial_x_position_cm, bins, weights = trial_speeds)[0] /
                     np.histogram(trial_x_position_cm, bins)[0])

        plt.plot(bin_means)
        plt.close()

        speed_trials_binned.append(bin_means)
        speed_trial_numbers.append(trial_number)

        if trial_type == 0:
            speed_trials_beaconed.append(bin_means)
            speed_trials_beaconed_trial_number.append(trial_number)
        elif trial_type == 1:
            speed_trials_non_beaconed.append(bin_means)
            speed_trials_non_beaconed_trial_number.append(trial_number)


    processed_position_data['speed_trials_binned'] = pd.Series(speed_trials_binned)
    processed_position_data['speed_trial_numbers'] = pd.Series(speed_trial_numbers)

    # trial type specifics speed bins
    processed_position_data['speed_trials_beaconed'] = pd.Series(speed_trials_beaconed)
    processed_position_data['speed_trials_beaconed_trial_number'] = pd.Series(speed_trials_beaconed_trial_number)
    processed_position_data['speed_trials_non_beaconed'] = pd.Series(speed_trials_non_beaconed)
    processed_position_data['speed_trials_non_beaconed_trial_number'] = pd.Series(speed_trials_non_beaconed_trial_number)

    return processed_position_data

def get_bin_size(prm, numbers_of_bins):
    bin_size = prm.track_length/numbers_of_bins
    return bin_size

def get_number_of_bins(prm):
    # bin number is equal to the track length, such that theres one bin per cm
    number_of_bins = prm.track_length
    return number_of_bins

def process_speed(raw_position_data,processed_position_data, prm, recording_directory):
    processed_position_data = add_goal_locations(raw_position_data, processed_position_data, prm)
    processed_position_data = calculate_binned_speed(raw_position_data,processed_position_data, prm)
    return processed_position_data

def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    params = PostSorting.parameters.Parameters()
    params.stop_threshold = 4.9
    params.cue_conditioned_goal = True
    params.track_length = 300

    raw_position_data = pd.read_pickle(
        r'C:\Users\44756\Desktop\test_recordings_waveform_matching\m2_d29\DataFrames\raw_position_data.pkl')  # m4
    processed_position_data = pd.read_pickle(
        r'C:\Users\44756\Desktop\test_recordings_waveform_matching\m2_d29\DataFrames\processed_position_data.pkl')  # m4
    spatial_firing = pd.read_pickle(
        r'C:\Users\44756\Desktop\test_recordings_waveform_matching\m2_d29\DataFrames\spatial_firing.pkl')  # m4

    recording_folder = r'C:\Users\44756\Desktop\test_recordings_waveform_matching\m2_d29\DataFrames'
    print('Processing ' + str(recording_folder))

    process_speed(raw_position_data, processed_position_data, params, recording_folder)


if __name__ == '__main__':
    main()