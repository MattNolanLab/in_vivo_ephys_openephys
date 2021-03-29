import numpy as np
import pandas as pd
import PostSorting.parameters
from scipy import stats
import matplotlib.pyplot as plt


def calculate_binned_speed(raw_position_data, processed_position_data, prm):
    numbers_of_bins = get_number_of_bins(prm)
    bin_size_cm = get_bin_size(prm, numbers_of_bins)

    speeds_binned = []
    trial_numbers = []
    trial_types = []
    position_bin_centres = []

    for trial_number in range(1, max(raw_position_data["trial_number"]+1)):
        trial_type = int(stats.mode(np.array(raw_position_data['trial_type'][np.array(raw_position_data['trial_number']) == trial_number]), axis=None)[0])

        trial_x_position_cm = np.array(raw_position_data['x_position_cm'][np.array(raw_position_data['trial_number']) == trial_number])
        trial_speeds = np.array(raw_position_data['speed_per200ms'][np.array(raw_position_data['trial_number']) == trial_number])

        bins = np.arange(0, prm.get_track_length(), bin_size_cm)
        bin_centres = 0.5*(bins[1:]+bins[:-1])

        # this calculates the average speed within the bin i.e. all speeds in bin summated and then divided by the number of datapoints within the bin
        bin_means = (np.histogram(trial_x_position_cm, bins, weights = trial_speeds)[0] /
                     np.histogram(trial_x_position_cm, bins)[0])
        bin_means[np.abs(bin_means)>1000] = np.nan

        speeds_binned.append(bin_means)
        trial_numbers.append(trial_number)
        trial_types.append(trial_type)
        position_bin_centres.append(bin_centres)

    processed_position_data["speeds_binned"] = speeds_binned
    processed_position_data["trial_number"] = trial_numbers
    processed_position_data["trial_type"] = trial_types
    processed_position_data["position_bin_centres"] = position_bin_centres
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