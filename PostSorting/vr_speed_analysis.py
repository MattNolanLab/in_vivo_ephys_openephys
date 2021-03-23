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

    for trial_number in range(1, max(raw_position_data["trial_number"]+1)):
        trial_type = np.array(raw_position_data['trial_type'][np.array(raw_position_data['trial_number']) == trial_number])[0]

        trial_x_position_cm = np.array(raw_position_data['x_position_cm'][np.array(raw_position_data['trial_number']) == trial_number])
        trial_speeds = np.array(raw_position_data['speed_per200ms'][np.array(raw_position_data['trial_number']) == trial_number])

        bins = np.arange(0, prm.get_track_length(), bin_size_cm)

        bin_means = (np.histogram(trial_x_position_cm, bins, weights = trial_speeds)[0] /
                     np.histogram(trial_x_position_cm, bins)[0])

        bin_means[np.abs(bin_means)>1000] = np.nan
        position_bins = np.histogram(trial_x_position_cm, bins)[1]

        speed_trials_binned.append(bin_means)
        speed_trial_numbers.append(trial_number)
        speed_trialtypes.append(trial_type)

    position_bins = pd.DataFrame({"position_bins": position_bins})
    speed_trials_binned = pd.DataFrame({"speed_trials_binned": speed_trials_binned})
    speed_trial_numbers = pd.DataFrame({"speed_trial_numbers": speed_trial_numbers})
    speed_trial_types = pd.DataFrame({"speed_trial_types": speed_trialtypes})

    processed_position_data = pd.concat([processed_position_data, position_bins], axis=1)
    processed_position_data = pd.concat([processed_position_data, speed_trials_binned], axis=1)
    processed_position_data = pd.concat([processed_position_data, speed_trial_numbers], axis=1)
    processed_position_data = pd.concat([processed_position_data, speed_trial_types], axis=1)

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


if __name__ == '__main__':
    main()