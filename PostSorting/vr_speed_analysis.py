import numpy as np
import pandas as pd
import PostSorting.parameters
import matplotlib.pyplot as plt


def calculate_binned_speed(raw_position_data,processed_position_data, prm):
    numbers_of_bins = get_number_of_bins(prm)
    bin_size_cm = get_bin_size(prm, numbers_of_bins)

    speed_trials_binned = []
    speed_trial_numbers = []
    speed_trialtypes = []

    speed_trials_beaconed = []
    speed_trials_beaconed_trial_number = []

    speed_trials_non_beaconed = []
    speed_trials_non_beaconed_trial_number = []

    speed_trials_probe = []
    speed_trials_probe_trial_number = []

    for trial_number in range(1, max(raw_position_data["trial_number"]+1)):
        trial_type = np.array(raw_position_data['trial_type'][np.array(raw_position_data['trial_number']) == trial_number])[0]

        trial_x_position_cm = np.array(raw_position_data['x_position_cm'][np.array(raw_position_data['trial_number']) == trial_number])
        trial_speeds = np.array(raw_position_data['speed_per200ms'][np.array(raw_position_data['trial_number']) == trial_number])

        bins = np.arange(0, prm.get_track_length(), bin_size_cm)

        bin_means = (np.histogram(trial_x_position_cm, bins, weights = trial_speeds)[0] /
                     np.histogram(trial_x_position_cm, bins)[0])

        bin_means[np.abs(bin_means)>1000] = np.nan
        #print(np.shape(bin_means))

        position_bins = np.histogram(trial_x_position_cm, bins)[1]

        speed_trials_binned.append(bin_means)
        speed_trial_numbers.append(trial_number)
        speed_trialtypes.append(trial_type)

        if trial_type == 0:
            speed_trials_beaconed.append(bin_means)
            speed_trials_beaconed_trial_number.append(trial_number)
        elif trial_type == 1:
            speed_trials_non_beaconed.append(bin_means)
            speed_trials_non_beaconed_trial_number.append(trial_number)
        elif trial_type == 2:
            speed_trials_probe.append(bin_means)
            speed_trials_probe_trial_number.append(trial_number)

    position_bins = pd.DataFrame({"position_bins": position_bins})
    speed_trials_binned = pd.DataFrame({"speed_trials_binned": speed_trials_binned})
    speed_trial_numbers = pd.DataFrame({"speed_trial_numbers": speed_trial_numbers})
    speed_trial_types = pd.DataFrame({"speed_trial_types": speed_trialtypes})
    speed_trials_beaconed = pd.DataFrame({"speed_trials_beaconed": speed_trials_beaconed})
    speed_trials_beaconed_trial_number = pd.DataFrame({"speed_trials_beaconed_trial_number": speed_trials_beaconed_trial_number})
    speed_trials_non_beaconed = pd.DataFrame({"speed_trials_non_beaconed": speed_trials_non_beaconed})
    speed_trials_non_beaconed_trial_number = pd.DataFrame({"speed_trials_non_beaconed_trial_number": speed_trials_non_beaconed_trial_number})
    speed_trials_probe = pd.DataFrame({"speed_trials_probe": speed_trials_probe})
    speed_trials_probe_trial_number = pd.DataFrame({"speed_trials_probe_trial_number": speed_trials_probe_trial_number})

    processed_position_data = pd.concat([processed_position_data, position_bins], axis=1)
    processed_position_data = pd.concat([processed_position_data, speed_trials_binned], axis=1)
    processed_position_data = pd.concat([processed_position_data, speed_trial_numbers], axis=1)
    processed_position_data = pd.concat([processed_position_data, speed_trial_types], axis=1)
    processed_position_data = pd.concat([processed_position_data, speed_trials_beaconed], axis=1)
    processed_position_data = pd.concat([processed_position_data, speed_trials_beaconed_trial_number], axis=1)
    processed_position_data = pd.concat([processed_position_data, speed_trials_non_beaconed], axis=1)
    processed_position_data = pd.concat([processed_position_data, speed_trials_non_beaconed_trial_number], axis=1)
    processed_position_data = pd.concat([processed_position_data, speed_trials_probe], axis=1)
    processed_position_data = pd.concat([processed_position_data, speed_trials_probe_trial_number], axis=1)

    return processed_position_data

def get_bin_size(prm, numbers_of_bins):
    bin_size = prm.track_length/numbers_of_bins
    return bin_size

def get_number_of_bins(prm):
    # bin number is equal to the track length, such that theres one bin per cm
    number_of_bins = prm.track_length
    return number_of_bins

def process_speed(raw_position_data,processed_position_data, prm):
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