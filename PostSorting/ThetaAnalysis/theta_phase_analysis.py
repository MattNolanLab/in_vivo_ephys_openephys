import glob
import os
import open_ephys_IO
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, hilbert, resample


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


def down_sample_ephys_data(ephys_data, position_data):
    # indices = np.round(np.linspace(0, len(ephys_data) - 1, len(position_data))).astype(int)
    # ephys_downsampled = ephys_data[indices]
    ephys_downsampled = resample(ephys_data, len(position_data))
    return ephys_downsampled


def up_sample_position_data(position_data, upsample_factor):
    position_data = position_data.reset_index()
    position_data.index = range(0, upsample_factor * len(position_data), upsample_factor)
    position_data_with_nans = position_data.reindex(index=range(upsample_factor * len(position_data)))
    interpolated = position_data_with_nans.interpolate()
    # todo test this!!!
    # plt.plot(interpolated.position_x[:100], interpolated.position_y[:100])
    # plt.show()
    return interpolated


def add_down_sampled_angle_to_position_df(recording_folder_path, number_of_channels=16, upsample_factor=4):
    print('Add downsampled angle to upsampled position data and save (position_theta.pkl)')
    position_df_path = recording_folder_path + 'MountainSort/DataFrames/position.pkl'
    if os.path.exists(position_df_path):
        position_data = pd.read_pickle(position_df_path)
        position_data_theta = up_sample_position_data(position_data, upsample_factor=upsample_factor)

    else:
        print('There is no position data for this recording: ' + recording_folder_path)
        return False
    for channel in range(number_of_channels):
        ch_angle = np.load(recording_folder_path + 'channel_angle_' + str(channel) + '.npy')
        ch_angle_downsampled = down_sample_ephys_data(ch_angle, position_data_theta)
        position_data_theta['theta_angle_' + str(channel)] = ch_angle_downsampled

    position_data_theta.to_pickle(recording_folder_path + 'MountainSort/DataFrames/position_theta.pkl')


def add_theta_phase_to_firing_data(recording_folder_path):
    """
    Loads theta phase angles and finds the angle that corresponds to each spike. This function currently saves a new
    spatial_firing data frame with the theta phase angle added to it.
    """
    print('Find theta phase for each action potential.')
    spatial_firing_path = recording_folder_path + 'MountainSort/DataFrames/spatial_firing.pkl'
    if os.path.exists(spatial_firing_path):
        spatial_firing = pd.read_pickle(spatial_firing_path)
        spatial_firing_theta = spatial_firing

    else:
        print('There is no spatial firing data for this recording: ' + recording_folder_path)
        return False
    angles_at_spikes = []
    for index, cell in spatial_firing.iterrows():
        # load theta angle for primary channel (the ch where the cell had the highest amplitude
        primary_channel = ((cell.tetrode - 1) * 4 + cell.primary_channel) - 1  # numbering is from 1 in this df
        corresponding_theta_angle = np.load(recording_folder_path + 'channel_angle_' + str(primary_channel) + '.npy')
        phase_angles_at_spikes = corresponding_theta_angle[cell.firing_times.astype(int)]
        angles_at_spikes.append(phase_angles_at_spikes)
        # plt.hist(phase_angles_at_spikes)
    spatial_firing_theta['theta_phase_angle'] = angles_at_spikes
    spatial_firing_theta.to_pickle(recording_folder_path + 'MountainSort/DataFrames/spatial_firing_theta.pkl')


def analyse_theta_modulation(recording_folder_path):
    # calculate_and_save_theta_phase_angles(recording_folder_path)
    # add_down_sampled_angle_to_position_df(recording_folder_path, upsample_factor=4)  # upsample position data (120 Hz)
    add_theta_phase_to_firing_data(recording_folder_path)  # find theta phase for each spike
    # make separate df for example cell


def main():
    recording_folder_path = '/mnt/datastore/Klara/Open_field_opto_tagging_p038/M13_2018-05-14_09-37-33_of/'
    analyse_theta_modulation(recording_folder_path)


if __name__ == '__main__':
    main()