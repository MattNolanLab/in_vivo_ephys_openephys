import glob
import matplotlib.pylab as plt
import os
import pandas as pd
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects.packages import importr


server_path = '//ardbeg.mvm.ed.ac.uk/nolanlab/Klara/Open_field_opto_tagging_p038/'
analysis_path = '/Users/s1466507/Dropbox/Edinburgh/grid_fields/analysis/'


# load field data from server - must include hd in fields
def load_data_frame_field_data(output_path):
    if os.path.exists(output_path):
        field_data = pd.read_pickle(output_path)
        return field_data
    else:
        field_data_combined = pd.DataFrame()
        for recording_folder in glob.glob(server_path + '*'):
            os.path.isdir(recording_folder)
            data_frame_path = recording_folder + '/MountainSort/DataFrames/shuffled_fields.pkl'
            if os.path.exists(data_frame_path):
                print('I found a field data frame.')
                field_data = pd.read_pickle(data_frame_path)
                if 'field_id' in field_data:
                    field_data_to_combine = field_data[['session_id', 'cluster_id', 'field_id', 'indices_rate_map',
                                             'spike_times', 'number_of_spikes_in_field', 'position_x_spikes',
                                             'position_y_spikes', 'hd_in_field_spikes', 'hd_hist_spikes',
                                             'times_session', 'time_spent_in_field', 'position_x_session',
                                             'position_y_session', 'hd_in_field_session', 'hd_hist_session',
                                             'hd_histogram_real_data', 'time_spent_in_bins', 'field_histograms_hz']].copy()

                    field_data_combined = field_data_combined.append(field_data_to_combine)
                    print(field_data_combined.head())
        field_data_combined.to_pickle(output_path)
        return field_data_combined


# select accepted fields based on list of fields that were correctly identified by field detector
def tag_accepted_fields(field_data, accepted_fields):
    unique_id = field_data.session_id + '_' + field_data.cluster_id.apply(str) + '_' + (field_data.field_id + 1).apply(str)
    field_data['unique_id'] = unique_id
    unique_id = accepted_fields['Session ID'] + '_' + accepted_fields['Cell'].apply(str) + '_' + accepted_fields['field'].apply(str)
    accepted_fields['unique_id'] = unique_id
    field_data['accepted_field'] = field_data.unique_id.isin(accepted_fields.unique_id)
    return field_data


# run 2 sample watson test and put it in df
def run_two_sample_watson_test(hd_cluster, hd_session):
    circular = importr("circular")
    watson_two_test = circular.watson_two_test
    hd_cluster = ro.FloatVector(hd_cluster)
    hd_session = ro.FloatVector(hd_session)
    stat = watson_two_test(hd_cluster, hd_session)
    return stat[0][0]  # this is the part of the return r object that is the stat


# call R to tun two sample watson test on HD from firing field when the cell fired vs HD when the mouse was in the field
def compare_hd_when_the_cell_fired_to_heading(field_data):
    two_watson_stats = []
    for index, field in field_data.iterrows():
        print('analyzing ' + field.unique_id)
        hd_cluster = field.hd_in_field_spikes
        hd_session = field.hd_in_field_session
        two_watson_stat = run_two_sample_watson_test(hd_cluster, hd_session)
        two_watson_stats.append(two_watson_stat)
    field_data['watson_two_stat'] = two_watson_stats
    return field_data


def plot_histogram_of_watson_stat(field_data):
    plt.hist(field_data.watson_two_stat, bins=30)
    plt.axvline(x=0.385, linewidth=5, color='red')
    pass


def main():
    field_data = load_data_frame_field_data(analysis_path + 'all_mice_fields_watson_test.pkl')   # for two-sample watson analysis
    accepted_fields = pd.read_excel(analysis_path + 'list_of_accepted_fields.xlsx')
    field_data = tag_accepted_fields(field_data, accepted_fields)
    field_data = compare_hd_when_the_cell_fired_to_heading(field_data)
    plot_histogram_of_watson_stat(field_data)



if __name__ == '__main__':
    main()