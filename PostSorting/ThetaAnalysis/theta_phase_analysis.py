import glob
import os
import open_ephys_IO
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, hilbert


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


def sort_folder_names(list_of_names):
    list_of_names.sort(key=lambda x: int(x.split('CH')[1].split('.')[0]))
    return list_of_names


def bandpass_filter(data, low=5, high=9, fs=30000):
    if len(data.shape) == 1:
        filtered_data = butter_bandpass_filter(data, lowcut=low, highcut=high, fs=fs, order=2)

    else:
        filtered_data = np.zeros((data.shape[0], data.shape[1]))
        for channel in range(data.shape[0]):
            filtered_data[channel] = butter_bandpass_filter(data[channel, :], lowcut=low, highcut=high, fs=fs, order=2)
    return filtered_data


def load_all_channels(path, just_load_one=False):
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
            if just_load_one:
                return np.array(channel_data), True
            if is_first:
                all_channels = np.zeros((len(list(glob.glob(path + '/*CH*continuous'))), channel_data.size), np.int16)
                is_first = False
                is_loaded = True
            all_channels[channel_count, :] = channel_data
            channel_count += 1
    return all_channels, is_loaded


def plot_results(channel_data, filtered_data, angle):
    # this is not executed but useful for testing and debugging so I will leave them here for now
    plt.plot(channel_data[:200000], color='grey', label='raw voltage')
    plt.plot(filtered_data[:200000], color='skyblue', label='theta filtered')
    plt.legend()
    plt.show()
    plt.cla()

    plt.cla()
    plt.plot(channel_data[200000:400000], color='grey', label='raw voltage')
    plt.plot(filtered_data[200000:400000], color='skyblue', label='theta filtered')
    plt.xlabel('Time (sampling points)')
    plt.ylabel('Voltage (mV)')
    # plt.plot(hilbert_transformed[:200000], color='red', label='hilbert transformed', alpha=0.5)
    plt.legend()
    plt.show()

    plt.cla()
    plt.plot(channel_data[200000:210000], color='grey', label='raw voltage')
    plt.plot(filtered_data[200000:210000], color='skyblue', label='theta filtered')
    plt.plot(angle[200000:210000] * 10, color='red', label='angle')
    plt.xlabel('Time (sampling points)')
    plt.ylabel('Voltage (mV)')
    # plt.plot(hilbert_transformed[:200000], color='red', label='hilbert transformed', alpha=0.5)
    plt.legend()
    plt.show()


def calculate_and_save_theta_phase_angles(recording_folder_path):
    print('I will calculate theta phase angles and save them.')
    all_channels, is_loaded = load_all_channels(recording_folder_path, just_load_one=False)
    for channel in range(all_channels.shape[0]):
        channel_data = all_channels[channel, :]

        filtered_data = bandpass_filter(channel_data, low=5, high=9)  # filtered in theta range
        analytic_signal = hilbert(filtered_data)  # hilbert transform https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html
        angle = np.angle(analytic_signal)  # this is the theta angle (radians)
        np.save(recording_folder_path + 'channel_angle_' + str(channel) + '.npy', angle)  # save array with angles


def add_down_sampled_angle_to_position_df(recording_folder_path, number_of_channels=16):
    position_df_path = recording_folder_path + 'MountainSort/DataFrames/position.pkl'
    if os.path.exists(position_df_path):
        position_data = pd.read_pickle(position_df_path)
    else:
        print('There is no position data for this recoriding: ' + recording_folder_path)
        return False
    for channel in range(number_of_channels):
        ch_angle = np.load(recording_folder_path + 'channel_angle_' + str(channel) + '.npy')
        # downsample and add to df




def analyse_theta_modulation(recording_folder_path):
    # calculate_and_save_theta_phase_angles(recording_folder_path)
    add_down_sampled_angle_to_position_df(recording_folder_path)
    # down sample and add to position df (make another position df)
    # add to spatial firing df (corresponding spike times) / just make example for now




def main():
    recording_folder_path = '/mnt/datastore/Klara/Open_field_opto_tagging_p038/M13_2018-05-14_09-37-33_of/'
    analyse_theta_modulation(recording_folder_path)


if __name__ == '__main__':
    main()