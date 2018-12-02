import os
import numpy as np
import pandas as pd
import OverallAnalysis.false_positives
import data_frame_utility

import matplotlib.pylab as plt


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


def shuffle_field_data(field_data, number_of_times_to_shuffle, local_path):
    if os.path.exists(local_path + 'shuffle_analysis') is False:
        os.makedirs(local_path + 'shuffle_analysis')

    for index, field in field_data.iterrows():
        print(field)
        number_of_spikes_in_field = field['number_of_spikes_in_field']
        time_spent_in_field = field['time_spent_in_field']
        shuffle_indices = np.random.randint(0, time_spent_in_field, size=(number_of_times_to_shuffle, number_of_spikes_in_field))
        for shuffle in range(number_of_times_to_shuffle):
            shuffled_hd = field['hd_in_field_session'][shuffle_indices[shuffle]]
            plt.hist(shuffled_hd)
            plt.savefig(local_path + 'shuffle_analysis/shuffled_hd_cluster' + str(field['cluster_id']) + '_field_' + str(index) + '_shuffle' + str(shuffle))
            plt.close()

    return field_data


def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    # get_field_data()
    local_path = '/Users/s1466507/Documents/Ephys/recordings/M5_2018-03-06_15-34-44_of/MountainSort/'
    spatial_firing = pd.read_pickle(local_path + '/DataFrames/spatial_firing.pkl')
    position_data = pd.read_pickle(local_path + '/DataFrames/position.pkl')

    field_df = data_frame_utility.get_field_data_frame(spatial_firing, position_data)
    shuffled_data = shuffle_field_data(field_df, 20, local_path)

    #field_spike_times = spike_data.spike_times_in_fields[5][0]
    #times_in_session_fields = spike_data.times_in_session_fields[5][0]
    pass







if __name__ == '__main__':
    main()
