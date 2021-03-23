import numpy as np
import pandas as pd
import PostSorting.parameters
import matplotlib.pyplot as plt


def calculate_binned_time(raw_position_data,processed_position_data, prm):
    numbers_of_bins = get_number_of_bins(prm)
    bin_size_cm = get_bin_size(prm, numbers_of_bins)

    time_trials_binned = []
    time_trial_numbers = []
    time_trialtypes = []

    for trial_number in range(1, max(raw_position_data["trial_number"]+1)):
        trial_type = np.array(raw_position_data['trial_type'][np.array(raw_position_data['trial_number']) == trial_number])[0]

        trial_x_position_cm = np.array(raw_position_data['x_position_cm'][np.array(raw_position_data['trial_number']) == trial_number])
        trial_times = np.array(raw_position_data['dwell_time_ms'][np.array(raw_position_data['trial_number']) == trial_number])

        bins = np.arange(0, prm.get_track_length(), bin_size_cm)
        bin_times = np.histogram(trial_x_position_cm, bins, weights=trial_times)[0]

        time_trials_binned.append(bin_times)
        time_trial_numbers.append(trial_number)
        time_trialtypes.append(trial_type)

    processed_position_data['time_trials_binned'] = pd.Series(time_trials_binned)
    processed_position_data['time_trial_numbers'] = pd.Series(time_trial_numbers)
    processed_position_data['time_trial_types'] = pd.Series(time_trialtypes)

    return processed_position_data

def get_bin_size(prm, numbers_of_bins):
    bin_size = prm.track_length/numbers_of_bins
    return bin_size

def get_number_of_bins(prm):
    # bin number is equal to the track length, such that theres one bin per cm
    number_of_bins = prm.track_length
    return number_of_bins

def process_time(raw_position_data,processed_position_data, prm):
    processed_position_data = calculate_binned_time(raw_position_data,processed_position_data, prm)
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

    process_time(raw_position_data, processed_position_data, params, recording_folder)


if __name__ == '__main__':
    main()