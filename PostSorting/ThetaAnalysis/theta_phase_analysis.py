import glob
import os
import open_ephys_IO
import matplotlib.pylab as plt
import numpy as np
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


def analyse_theta_modulation(recording_folder_path):
    all_channels, is_loaded = load_all_channels(recording_folder_path, just_load_one=False)
    # file_path = recording_folder_path + '100_CH1.continuous'  # test
    # channel_data = open_ephys_IO.get_data_continuous(file_path).astype(np.int16)  # this is the raw voltage data
    for channel in range(all_channels.shape[0]):
        channel_data = all_channels[channel, :]

        filtered_data = bandpass_filter(channel_data, low=5, high=9)  # filtered in theta range
        analytic_signal = hilbert(filtered_data)  # hilbert transform https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html
        angle = np.angle(analytic_signal)  # this is the theta angle (radians)
        np.save(recording_folder_path + 'channel_angle_' + str(channel) + '.npy', angle)  # save array with angles
    # instantaneous_phase = np.unwrap(angle)
    # down sample
    # save



def main():
    recording_folder_path = '/mnt/datastore/Klara/Open_field_opto_tagging_p038/M13_2018-05-14_09-37-33_of/'
    analyse_theta_modulation(recording_folder_path)


if __name__ == '__main__':
    main()