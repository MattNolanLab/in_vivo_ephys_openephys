import glob
import sys
import shutil
import threading
import os
import numpy as np
import pandas as pd
import OverallAnalysis.false_positives
import data_frame_utility
import plot_utility

import matplotlib.pylab as plt


server_path = '//cmvm.datastore.ed.ac.uk/cmvm/sbms/groups/mnolan_NolanLab/ActiveProjects/Klara/Open_field_opto_tagging_p038/'
server_test_file = '//cmvm.datastore.ed.ac.uk/cmvm/sbms/groups/mnolan_NolanLab/ActiveProjects/Klara/test_analysis/M5_2018-03-05_13-30-30_of/parameters.txt'


def add_combined_id_to_df(df_all_mice):
    animal_ids = [session_id.split('_')[0] for session_id in df_all_mice.session_id.values]
    dates = [session_id.split('_')[1] for session_id in df_all_mice.session_id.values]
    tetrode = df_all_mice.tetrode.values
    cluster = df_all_mice.cluster_id.values

    combined_ids = []
    for cell in range(len(df_all_mice)):
        id = animal_ids[cell] +  '-' + dates[cell] + '-Tetrode-' + str(tetrode[cell]) + '-Cluster-' + str(cluster[cell])
        combined_ids.append(id)
    df_all_mice['false_positive_id'] = combined_ids
    return df_all_mice


def get_field_data():
    local_path = '/Users/s1466507/Documents/Ephys/recordings/all_mice_df.pkl'
    path_to_data = 'C:/Users/s1466507/Documents/Ephys/recordings/'
    save_output_path = 'C:/Users/s1466507/Documents/Ephys/overall_figures/'
    false_positives_path = path_to_data + 'false_positives_all.txt'
    df_all_mice = pd.read_pickle(local_path)
    list_of_false_positives = OverallAnalysis.false_positives.get_list_of_false_positives(false_positives_path)
    df_all_mice = add_combined_id_to_df(df_all_mice)
    df_all_mice['false_positive'] = df_all_mice['false_positive_id'].isin(list_of_false_positives)

    good_cluster = df_all_mice.false_positive == False


def format_bar_chart(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlabel('Head direction [degrees]')
    ax.set_ylabel('Head-direction preference')
    return ax


def add_mean_and_std_to_field_df(field_data, number_of_bins=20):
    fields_means = []
    fields_stdevs = []
    real_data_hz_all_fields = []
    time_spent_in_bins_all = []
    field_histograms_hz_all = []
    for index, field in field_data.iterrows():
        field_histograms = field['shuffled_data']
        field_spikes_hd = field['hd_in_field_spikes']  # real hd when the cell fired
        field_session_hd = field['hd_in_field_session']  # hd from the whole session in field
        time_spent_in_bins = np.histogram(field_session_hd, bins=number_of_bins)[0]
        time_spent_in_bins_all.append(time_spent_in_bins)
        field_histograms_hz = field_histograms * 30 / time_spent_in_bins  # sampling rate is 30Hz for movement data
        field_histograms_hz_all.append(field_histograms_hz)
        mean_shuffled = np.mean(field_histograms_hz, axis=0)
        fields_means.append(mean_shuffled)
        std_shuffled = np.std(field_histograms_hz, axis=0)
        fields_stdevs.append(std_shuffled)

        real_data_hz = np.histogram(field_spikes_hd, bins=20)[0] * 30 / time_spent_in_bins
        real_data_hz_all_fields.append(real_data_hz)
    field_data['shuffled_means'] = fields_means
    field_data['shuffled_std'] = fields_stdevs
    field_data['hd_histogram_real_data'] = real_data_hz_all_fields
    field_data['time_spent_in_bins'] = time_spent_in_bins_all
    field_data['field_histograms_hz'] = field_histograms_hz_all
    return field_data


def test_if_real_hd_differs_from_shuffled(field_data):
    real_and_shuffled_data_differ_bin = []
    number_of_diff_bins = []
    for index, field in field_data.iterrows():
        diff_field = np.abs(field.shuffled_means - field.hd_histogram_real_data) > field.shuffled_std * 2
        number_of_diffs = diff_field.sum()
        real_and_shuffled_data_differ_bin.append(diff_field)
        number_of_diff_bins.append(number_of_diffs)
    field_data['real_and_shuffled_data_differ_bin'] = real_and_shuffled_data_differ_bin
    field_data['number_of_different_bins'] = number_of_diff_bins
    return field_data


def plot_bar_chart_for_fields(field_data, path):
    for index, field in field_data.iterrows():
        mean = field['shuffled_means']
        std = field['shuffled_std']
        field_spikes_hd = field['hd_in_field_spikes']
        time_spent_in_bins = field['time_spent_in_bins']
        field_histograms_hz = field['field_histograms_hz']
        x_pos = np.arange(field_histograms_hz.shape[1])
        fig, ax = plt.subplots()
        ax = format_bar_chart(ax)
        ax.bar(x_pos, mean, yerr=std*2, align='center', alpha=0.7, color='black', ecolor='grey', capsize=10)
        x_labels = ["0", "", "", "", "", "90", "", "", "", "", "180", "", "", "", "", "270", "", "", "", ""]
        plt.xticks(x_pos, x_labels)
        real_data_hz = np.histogram(field_spikes_hd, bins=20)[0] * 30 / time_spent_in_bins
        plt.scatter(x_pos, real_data_hz, marker='o', color='red', s=40)
        plt.savefig(path + 'shuffle_analysis/' + str(field['cluster_id']) + '_field_' + str(index) + '_SD')


def analyze_shuffled_data(field_data, save_path, number_of_bins=20):
    field_data = add_mean_and_std_to_field_df(field_data, number_of_bins)
    field_data = test_if_real_hd_differs_from_shuffled(field_data)
    plot_bar_chart_for_fields(field_data, save_path)
    return field_data


def get_random_indices_for_shuffle(field, number_of_times_to_shuffle):
    number_of_spikes_in_field = field['number_of_spikes_in_field']
    time_spent_in_field = field['time_spent_in_field']
    shuffle_indices = np.random.randint(0, time_spent_in_field, size=(number_of_times_to_shuffle, number_of_spikes_in_field))
    return shuffle_indices


def shuffle_field_data(field_data, number_of_times_to_shuffle, path, number_of_bins):
    if os.path.exists(path + 'shuffle_analysis') is True:
        shutil.rmtree(path + 'shuffle_analysis')
    os.makedirs(path + 'shuffle_analysis')
    field_histograms_all = []
    for index, field in field_data.iterrows():
        print('I will shuffle data in the fields.')
        field_histograms = np.zeros((number_of_times_to_shuffle, number_of_bins))
        shuffle_indices = get_random_indices_for_shuffle(field, number_of_times_to_shuffle)
        for shuffle in range(number_of_times_to_shuffle):
            shuffled_hd = field['hd_in_field_session'][shuffle_indices[shuffle]]
            hist, bin_edges = np.histogram(shuffled_hd, bins=number_of_bins, range=(0, 6.28))  # from 0 to 2pi
            field_histograms[shuffle, :] = hist
        field_histograms_all.append(field_histograms)
    print(path)
    field_data['shuffled_data'] = field_histograms_all
    return field_data


def process_recordings():
    if os.path.exists(server_test_file):
        print('I see the server.')
    for recording_folder in glob.glob(server_path + '*'):
        os.path.isdir(recording_folder)
        spike_data_frame_path = recording_folder + '/MountainSort/DataFrames/spatial_firing.pkl'
        position_data_frame_path = recording_folder + '/MountainSort/DataFrames/position.pkl'
        if os.path.exists(spike_data_frame_path):
            print('I found a firing data frame.')
            spatial_firing = pd.read_pickle(spike_data_frame_path)
            position_data = pd.read_pickle(position_data_frame_path)
            field_df = data_frame_utility.get_field_data_frame(spatial_firing, position_data)
            field_df = shuffle_field_data(field_df, 1000, recording_folder + '/MountainSort/', number_of_bins=20)


def local_data_test():
    # local_path = '/Users/s1466507/Documents/Ephys/recordings/M12_2018-04-10_14-22-14_of/MountainSort/'
    local_path = '/Users/s1466507/Documents/Ephys/recordings/M5_2018-03-06_15-34-44_of/MountainSort/'
    # local_path = '/Users/s1466507/Documents/Ephys/recordings/M13_2018-05-01_11-23-01_of/MountainSort/'
    # local_path = '/Users/s1466507/Documents/Ephys/recordings/M14_2018-05-16_11-29-05_of/MountainSort/'

    spatial_firing = pd.read_pickle(local_path + '/DataFrames/spatial_firing.pkl')
    position_data = pd.read_pickle(local_path + '/DataFrames/position.pkl')

    field_df = data_frame_utility.get_field_data_frame(spatial_firing, position_data)
    field_df = shuffle_field_data(field_df, 1000, local_path, number_of_bins=20)
    field_df = analyze_shuffled_data(field_df, local_path, number_of_bins=20)
    # save new field df so that another script can read them all and combine to look at the distribution of rejects

def main():
    sys.setrecursionlimit(100000)
    threading.stack_size(200000000)
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    #process_recordings()
    local_data_test()


if __name__ == '__main__':
    main()
