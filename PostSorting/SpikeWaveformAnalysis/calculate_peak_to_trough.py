import glob
import matplotlib.pylab as plt
import numpy as np
import os
import open_ephys_IO
import pandas as pd
from scipy.signal import butter, lfilter, hilbert, decimate


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def bandpass_filter(data, low=300, high=6000, fs=30000):
    if len(data.shape) == 1:
        filtered_data = butter_bandpass_filter(data, lowcut=low, highcut=high, fs=fs, order=2)

    else:
        filtered_data = np.zeros((data.shape[0], data.shape[1]))
        for channel in range(data.shape[0]):
            filtered_data[channel] = butter_bandpass_filter(data[channel, :], lowcut=low, highcut=high, fs=fs, order=2)
    return filtered_data


def remove_outlier_waveforms(all_waveforms):
    # remove snippets that have data points > 3 standard dev away from mean
    mean = all_waveforms.mean(axis=1)
    sd = all_waveforms.std(axis=1)
    distance_from_mean = all_waveforms.T - mean
    max_deviations = 3
    outliers = np.sum(distance_from_mean > max_deviations * sd, axis=1) > 0
    return all_waveforms[:, ~outliers]


def add_trough_to_peak_to_df(spatial_firing):
    peak_to_trough = []
    snippet_peak_position = []
    snippet_trough_position = []
    for index, cell in spatial_firing.iterrows():
        primary_channel = cell.primary_channel - 1
        all_waveforms_with_noise = cell.random_snippets[primary_channel]
        all_waveforms = remove_outlier_waveforms(all_waveforms_with_noise)
        mean_waveform = all_waveforms.mean(axis=1)
        peak = np.argmax(np.absolute(mean_waveform))
        if peak < len(mean_waveform):
            trough = np.argmax(mean_waveform[peak:]) + peak
        else:
            trough = np.argmin(mean_waveform)
        snippet_peak_position.append(peak)
        snippet_trough_position.append(trough)
        peak_to_trough.append(np.abs(peak-trough))

    spatial_firing['peak_to_trough'] = peak_to_trough
    spatial_firing['snippet_peak_position'] = snippet_peak_position
    spatial_firing['snippet_trough_position'] = snippet_trough_position
    return spatial_firing


def visualize_peak_to_trough_detection(spatial_firing):
    for index, cell in spatial_firing.iterrows():
        primary_channel = cell.primary_channel - 1
        all_waveforms_with_noise = cell.random_snippets[primary_channel]
        all_waveforms = remove_outlier_waveforms(all_waveforms_with_noise)
        mean_waveform = all_waveforms.mean(axis=1)
        # plt.plot(all_waveforms_with_noise, color='grey', alpha=0.6)
        plt.plot(all_waveforms, color='skyblue', alpha=0.8)
        plt.plot(mean_waveform, linewidth=3, color='navy')
        plt.axvline(cell.snippet_peak_position, color='red')
        plt.axvline(cell.snippet_trough_position, color='red')
        plt.show()


def sort_folder_names(list_of_names):
    list_of_names.sort(key=lambda x: int(x.split('CH')[1].split('.')[0]))
    return list_of_names


def load_all_channels(path):
    """
    Function to laod all channels in folder.
    """
    sorted_list_of_folders = sort_folder_names(glob.glob(path + '/*CH*continuous'))
    all_channels = False
    is_loaded = False
    is_first = True
    channel_count = 0
    for file_path in sorted_list_of_folders:
        if os.path.exists(file_path):
            channel_data = open_ephys_IO.get_data_continuous(file_path).astype(np.int16)
            if is_first:
                all_channels = np.zeros((len(list(glob.glob(path + '/*CH*continuous'))), channel_data.size), np.int16)
                is_first = False
                is_loaded = True
            all_channels[channel_count, :] = channel_data
            channel_count += 1
    return all_channels, is_loaded


def add_filtered_big_snippets_to_data(recording_folder_path, spatial_firing):
    # reproduce MS pipeline and take out a bigger snippet
    raw_ephys_data, is_loaded = load_all_channels(recording_folder_path)
    for index, cell in spatial_firing.iterrows():
        primary_channel = cell.primary_channel - 1
        raw_data = raw_ephys_data[primary_channel, :]
        filtered_data = bandpass_filter(raw_data, low=300, high=6000)  # filtered in spike range TODO FIX THIS

    return spatial_firing


def analyse_waveform_shapes(recording_folder_path):
    """
    Loads theta phase angles and finds the angle that corresponds to each spike. This function currently saves a new
    spatial_firing data frame with the theta phase angle added to it.
    """
    print('Calculate peak to trough distance for each cell.')
    spatial_firing_path = recording_folder_path + 'MountainSort/DataFrames/spatial_firing.pkl'
    if os.path.exists(spatial_firing_path):
        spatial_firing = pd.read_pickle(spatial_firing_path)
        # spatial_firing = add_filtered_big_snippets_to_data(recording_folder_path, spatial_firing)
        spatial_firing = add_trough_to_peak_to_df(spatial_firing)
        visualize_peak_to_trough_detection(spatial_firing)
        spatial_firing.to_pickle(recording_folder_path + 'MountainSort/DataFrames/spatial_firing.pkl')

    else:
        print('There is no spatial firing data for this recording: ' + recording_folder_path)
        return False


def main():
    # there are 2 grid cells in this recording and one of them (#7) looks theta modulated
    # recording_folder_path = '/mnt/datastore/Klara/Open_field_opto_tagging_p038/M13_2018-05-14_09-37-33_of/'
    # recording_folder_path = '/mnt/datastore/Klara/CA1_to_deep_MEC_in_vivo/M10_2021-12-10_08-37-27_of/'
    recording_folder_path = '/mnt/datastore/Klara/CA1_to_deep_MEC_in_vivo_extras/PSAM/M10_2021-11-26_16-07-10_of/'
    analyse_waveform_shapes(recording_folder_path)


if __name__ == '__main__':
    main()