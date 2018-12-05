import glob
import os
import numpy as np
import pandas as pd
import OverallAnalysis.false_positives
import data_frame_utility

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

    # read data in in a way that it is converter to field df
    # use tmp data for now from a selected cell / field


def plot_histogram_of_hd_in_field(field, index, path):
    plt.cla()
    plt.hist(field['hd_in_field_spikes'], color='red')
    plt.savefig(path + 'shuffle_analysis/hd_cluster_' + str(field['cluster_id']) + '_field_' + str(index))
    plt.close()


def plot_shuffled_histogram(shuffled_hd, field, index, shuffle, path):
    plt.hist(shuffled_hd, color='navy')
    plt.savefig(path + 'shuffle_analysis/hd_cluster_' + str(field['cluster_id']) + '_field_' + str(index) + '_shuffle' + str(shuffle))
    plt.close()


def plot_bar_chart_for_field(field_histograms, field_spikes_hd, field_session_hd, number_of_bins, path):
    time_spent_in_bins = np.histogram(field_session_hd, bins=number_of_bins)[0]
    field_histograms_normalized = field_histograms / time_spent_in_bins
    mean = np.mean(field_histograms_normalized, axis=0)
    # mean_normalized = mean / time_spent_in_bins
    std = np.std(field_histograms_normalized, axis=0)
    number_of_events_in_bins = np.sum(field_histograms_normalized, axis=0)
    sem = std / np.sqrt(number_of_events_in_bins)
    x_pos = np.arange(field_histograms_normalized.shape[1])
    fig, ax = plt.subplots()
    plt.ylim(0, 1)
    ax.bar(x_pos, mean, yerr=std*2, align='center', alpha=0.7, color='black', capsize=10)
    # ax.bar(x_pos, mean, yerr=sem, align='center', alpha=0.5, color='black', capsize=10)
    real_data = np.histogram(field_spikes_hd, bins=20)[0] / time_spent_in_bins
    plt.scatter(x_pos, real_data, marker='o', color='red', s=400)
    #ax.bar(x_pos, real_data, align='center', alpha=0.5, color='lime')

    ax.set_ylabel('Head-direction preference')
    plt.show()


def shuffle_field_data(field_data, number_of_times_to_shuffle, path):
    if os.path.exists(path + 'shuffle_analysis') is False:
        os.makedirs(path + 'shuffle_analysis')
    number_of_bins = 20

    for index, field in field_data.iterrows():
        print('I will shuffle data in the fields.')
        field_histograms = np.zeros((number_of_times_to_shuffle, number_of_bins))

        #plot_histogram_of_hd_in_field(field, index, path)
        number_of_spikes_in_field = field['number_of_spikes_in_field']
        time_spent_in_field = field['time_spent_in_field']
        shuffle_indices = np.random.randint(0, time_spent_in_field, size=(number_of_times_to_shuffle, number_of_spikes_in_field))

        for shuffle in range(number_of_times_to_shuffle):
            shuffled_hd = field['hd_in_field_session'][shuffle_indices[shuffle]]
            #plot_shuffled_histogram(shuffled_hd, field, index, shuffle, path)
            hist, bin_edges = np.histogram(shuffled_hd, bins=number_of_bins, range=(0, 6.28))  # from 0 to 2pi
            field_histograms[shuffle, :] = hist
        plot_bar_chart_for_field(field_histograms, field['hd_in_field_spikes'], field['hd_in_field_session'], number_of_bins, path)
        print(field_histograms)
    print(path)
    return


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
            shuffle_field_data(field_df, 20, recording_folder + '/MountainSort/')


def local_data_test():
    local_path = '/Users/s1466507/Documents/Ephys/recordings/M12_2018-04-10_14-22-14_of/MountainSort/'

    spatial_firing = pd.read_pickle(local_path + '/DataFrames/spatial_firing.pkl')
    position_data = pd.read_pickle(local_path + '/DataFrames/position.pkl')

    field_df = data_frame_utility.get_field_data_frame(spatial_firing, position_data)
    shuffle_field_data(field_df, 1000, local_path)


def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    #process_recordings()
    local_data_test()





if __name__ == '__main__':
    main()
