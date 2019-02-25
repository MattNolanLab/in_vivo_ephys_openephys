from collections import deque
import glob
import math
import matplotlib.pylab as plt
import numpy as np
import os
import pandas as pd
import plot_utility
import PostSorting.open_field_head_direction

# compare head-direction preference of firing fields of grid cells and conjunctive cells

server_path = '//ardbeg.mvm.ed.ac.uk/nolanlab/Klara/Open_field_opto_tagging_p038/'
analysis_path = '/Users/s1466507/Dropbox/Edinburgh/grid_fields/analysis/compare_grid_and_conjunctive_fields/'
sampling_rate = 30000


# load data frame and save it
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
    unique_cell_id = field_data.session_id + '_' + field_data.cluster_id.apply(str)
    field_data['unique_id'] = unique_id
    field_data['unique_cell_id'] = unique_cell_id
    unique_id = accepted_fields['Session ID'] + '_' + accepted_fields['Cell'].apply(str) + '_' + accepted_fields['field'].apply(str)
    accepted_fields['unique_id'] = unique_id
    field_data['accepted_field'] = field_data.unique_id.isin(accepted_fields.unique_id)
    return field_data


# todo: replace this with python implementation
def read_cell_type_from_accepted_clusters(field_data, accepted_fields):
    accepted_fields_to_merge = accepted_fields[['unique_id', 'cell type', 'grid score', 'hd score']]
    field_data_merged = pd.merge(field_data, accepted_fields_to_merge, on='unique_id')
    return field_data_merged


def get_angle_of_population_mean_vector(hd_hist):
    angles = np.linspace(-179, 180, 360)
    angles_rad = angles*np.pi/180
    dy = np.sin(angles_rad)
    dx = np.cos(angles_rad)
    totx = sum(dx * hd_hist)/sum(hd_hist)
    toty = sum(dy * hd_hist)/sum(hd_hist)
    # r = np.sqrt(totx*totx + toty*toty)
    # population_mean_vector_angle = np.arctan(toty / totx)
    population_mean_vector_angle = math.degrees(math.atan2(toty, totx)) * (-1)
    population_mean_vector_angle += 180
    return population_mean_vector_angle


# combine hd from all fields and calculate angle (:=alpha) between population mean vector for cell and 0 (use hd score code)
def calculate_population_mean_vector_angle(field_data):
    list_of_cells = field_data.unique_cell_id.unique()
    angles_to_rotate_by = []
    hd_from_all_fields_clusters = []
    for cell in list_of_cells:
        cell_fields = list(field_data.unique_cell_id == cell)
        number_of_fields = len(field_data[cell_fields])
        hd_from_all_fields = field_data.hd_in_field_spikes[cell_fields]
        hd_from_all_fields_session = list(field_data.hd_in_field_session[cell_fields])
        hd_from_all_fields_cluster = [item for sublist in hd_from_all_fields for item in sublist]
        hd_from_all_fields_session = [item for sublist in hd_from_all_fields_session for item in sublist]
        hd_histogram_session = PostSorting.open_field_head_direction.get_hd_histogram(hd_from_all_fields_session) / sampling_rate
        hd_histogram_cluster = PostSorting.open_field_head_direction.get_hd_histogram(hd_from_all_fields_cluster)
        hd_histogram_cluster = hd_histogram_cluster / hd_histogram_session
        angle_to_rotate_by = get_angle_of_population_mean_vector(hd_histogram_cluster)
        angles_to_rotate_by.extend([angle_to_rotate_by] * number_of_fields)
        hd_from_all_fields_clusters.extend([hd_histogram_cluster] * number_of_fields)

    field_data['population_mean_vector_angle'] = angles_to_rotate_by
    field_data['hd_hist_from_all_fields'] = hd_from_all_fields_clusters
    return field_data


# rotate combined distribution by angle and save in df for each field for each cell (each field will have the cell hist)
def rotate_by_population_mean_vector(field_data):
    rotated_histograms = []
    for index, field in field_data.iterrows():
        histogram_to_rotate = deque(field.hd_hist_from_all_fields)
        angle_to_rotate_by = field.population_mean_vector_angle
        histogram_to_rotate.rotate(int(round(angle_to_rotate_by)))  # rotates in place
        rotated_histograms.append(histogram_to_rotate)
    field_data['hd_hist_from_all_fields_rotated'] = rotated_histograms
    return field_data


# combine all distributions for each cell type into plot
def plot_rotated_histograms_for_cell_type(field_data, cell_type='grid'):
    print('analyze ' + cell_type + ' cells')
    list_of_cells = field_data.unique_cell_id.unique()
    histograms = []
    for cell in list_of_cells:
        cell_type_filter = field_data['cell type'] == cell_type  # filter for cell type
        fields_of_cell = field_data.unique_cell_id == cell  # filter for cell
        if not field_data[fields_of_cell & cell_type_filter].empty:
            histogram = field_data[fields_of_cell & cell_type_filter].hd_hist_from_all_fields_rotated.iloc[0]
            histograms.append(histogram)

    plt.cla()
    hd_polar_fig = plt.figure()
    ax = hd_polar_fig.add_subplot(1, 1, 1)
    for histogram in histograms:
        theta = np.linspace(0, 2 * np.pi, 361)
        ax = plt.subplot(1, 1, 1, polar=True)
        ax = plot_utility.style_polar_plot(ax)
        ax.plot(theta[:-1], histogram, color='gray', linewidth=2, alpha=70)
    # combine them to make one polar plot
    average_histogram = np.average(histograms, axis=0)
    theta = np.linspace(0, 2 * np.pi, 361)
    ax.plot(theta[:-1], average_histogram, color='red', linewidth=10)
    plt.savefig(analysis_path + 'rotated_hd_histograms_' + cell_type + '.png', dpi=300, bbox_inches="tight")
    plt.close()


def main():
    field_data = load_data_frame_field_data(analysis_path + 'all_mice_fields_grid_vs_conjunctive_fields.pkl')   # for two-sample watson analysis
    accepted_fields = pd.read_excel(analysis_path + 'list_of_accepted_fields.xlsx')
    field_data = tag_accepted_fields(field_data, accepted_fields)
    field_data = read_cell_type_from_accepted_clusters(field_data, accepted_fields)
    field_data = calculate_population_mean_vector_angle(field_data)
    field_data = rotate_by_population_mean_vector(field_data)
    plot_rotated_histograms_for_cell_type(field_data, cell_type='grid')
    plot_rotated_histograms_for_cell_type(field_data, cell_type='conjunctive')
    plot_rotated_histograms_for_cell_type(field_data, cell_type='na')
    plot_rotated_histograms_for_cell_type(field_data, cell_type='hd')


if __name__ == '__main__':
    main()

# combine result for all cells