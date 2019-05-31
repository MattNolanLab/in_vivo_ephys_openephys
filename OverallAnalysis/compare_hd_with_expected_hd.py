import glob
import matplotlib.pylab as plt
import math_utility
import math
import numpy as np
import os
import OverallAnalysis.folder_path_settings
import pandas as pd
import PostSorting.open_field_head_direction
import PostSorting.open_field_make_plots
import plot_utility
from rpy2 import robjects as robj
from scipy.stats import circstd
from rpy2.robjects import pandas2ri
import scipy.stats
import seaborn
import PostSorting.compare_first_and_second_half
import PostSorting.open_field_head_direction
import PostSorting.parameters

prm = PostSorting.parameters.Parameters()
prm.set_sorter_name('MountainSort')


local_path = OverallAnalysis.folder_path_settings.get_local_path() + '/compare_hd_with_expected_hd/'
server_path_mouse = OverallAnalysis.folder_path_settings.get_server_path_mouse()
server_path_rat = OverallAnalysis.folder_path_settings.get_server_path_rat()


def load_field_data(output_path, server_path, spike_sorter, animal):
    if os.path.exists(output_path):
        field_data = pd.read_pickle(output_path)
        return field_data
    field_data_combined = pd.DataFrame()
    for recording_folder in glob.glob(server_path + '*'):
        os.path.isdir(recording_folder)
        data_frame_path = recording_folder + spike_sorter + '/DataFrames/shuffled_fields.pkl'
        spatial_firing_path = recording_folder + spike_sorter + '/DataFrames/spatial_firing.pkl'
        position_path = recording_folder + spike_sorter + '/DataFrames/position.pkl'
        if os.path.exists(data_frame_path):
            print('I found a field data frame.')
            field_data = pd.read_pickle(data_frame_path)
            spatial_firing = pd.read_pickle(spatial_firing_path)
            position = pd.read_pickle(position_path)
            prm.set_file_path(recording_folder)
            # spatial_firing = PostSorting.compare_first_and_second_half.analyse_first_and_second_halves(prm, position, spatial_firing)
            if 'shuffled_data' in field_data:
                field_data_to_combine = field_data[['session_id', 'cluster_id', 'field_id', 'position_x_spikes',
                                                    'position_y_spikes', 'position_x_session', 'position_y_session',
                                                    'field_histograms_hz', 'indices_rate_map', 'hd_in_field_spikes',
                                                    'hd_in_field_session', 'spike_times', 'times_session',
                                                    'time_spent_in_field', 'number_of_spikes_in_field']].copy()
                field_data_to_combine['normalized_hd_hist'] = field_data.hd_hist_spikes / field_data.hd_hist_session
                if 'hd_score' in field_data:
                    field_data_to_combine['hd_score'] = field_data.hd_score
                if 'grid_score' in field_data:
                    field_data_to_combine['grid_score'] = field_data.grid_score
                rate_maps = []
                length_recording = []
                position_xs = []
                position_ys = []
                synced_times = []
                hds = []

                for cluster in range(len(field_data.cluster_id)):
                    rate_map = spatial_firing[field_data.cluster_id.iloc[cluster] == spatial_firing.cluster_id].firing_maps
                    rate_maps.append(rate_map)
                    length_of_recording = (position.synced_time.max() - position.synced_time.min())
                    length_recording.append(length_of_recording)
                    position_xs.append(position.position_x_pixels)
                    position_ys.append(position.position_y_pixels)
                    synced_times.append(position.synced_time)
                    hds.append(position.hd)

                field_data_to_combine['rate_map'] = rate_maps
                field_data_to_combine['recording_length'] = length_recording
                field_data_to_combine['position_x_pixels'] = position_xs
                field_data_to_combine['position_y_pixels'] = position_ys
                field_data_to_combine['synced_time'] = synced_times
                field_data_to_combine['hd'] = hds
                field_data_combined = field_data_combined.append(field_data_to_combine)
                print(field_data_combined.head())
    field_data_combined.to_pickle(output_path)
    return field_data_combined


# get head-direction hist from bins of field
def get_hd_in_field_spikes(rate_map_indices, spatial_data, prm):
    hd_in_field_hist = np.zeros((len(rate_map_indices), 360))
    for index, bin_in_field in enumerate(rate_map_indices):
        inside_bin = PostSorting.open_field_head_direction.get_indices_for_bin(bin_in_field, spatial_data, prm)
        hd = inside_bin.hd.values + 180
        hd_hist = np.histogram(hd, bins=360, range=(0, 360))[0]
        # hd_hist = PostSorting.open_field_head_direction.get_hd_histogram(hd)
        hd_in_field_hist[index] = hd_hist
    return hd_in_field_hist


def get_rate_map_values_for_bins(rate_map_indices, rate_map):
    rates = np.zeros((len(rate_map_indices), 1))
    for index, bin_in_field in enumerate(rate_map_indices):
        rate = rate_map[bin_in_field[0], bin_in_field[1]]
        rates[index] = rate
    return rates


def process_data(animal):
    if animal == 'mouse':
        output_path = local_path + 'mouse_data.pkl'
        server_path = server_path_mouse
        spike_sorter = '/MountainSort'
        prm.set_pixel_ratio(440)
    else:
        server_path = server_path_rat
        spike_sorter = ''
        output_path = local_path + 'rat_data.pkl'
        prm.set_pixel_ratio(1)
    field_data = load_field_data(output_path, server_path, spike_sorter, animal)
    for index, field in field_data.iterrows():
        spatial_data_field = pd.DataFrame()
        spatial_data_field['x'] = field.position_x_pixels
        spatial_data_field['y'] = field.position_y_pixels
        spatial_data_field['hd'] = field.hd
        spatial_data_field['synced_time'] = field.synced_time
        rate_map_indices = field.indices_rate_map
        hd_in_field_histograms = get_hd_in_field_spikes(rate_map_indices, spatial_data_field, prm)
        rates_for_bins = get_rate_map_values_for_bins(rate_map_indices, field.rate_map.iloc[0])
        weighed_hists = hd_in_field_histograms * rates_for_bins
        weighed_hist_sum = np.sum(weighed_hists, axis=0)
        hd_dist_session = (field.hd_in_field_session * 180 / np.pi)
        hd_hist_session = np.histogram(hd_dist_session, bins=360, range=(0, 360))[0]
        hd_hist_normed_estimate = np.nan_to_num(weighed_hist_sum / hd_hist_session)
        estimate_smooth = PostSorting.open_field_head_direction.get_rolling_sum(hd_hist_normed_estimate, window=23)
        estimate_smooth /= 23

        hd_spikes_real_hist = PostSorting.open_field_head_direction.get_hd_histogram(field.hd_in_field_spikes)
        hd_session_real_hist = PostSorting.open_field_head_direction.get_hd_histogram(field.hd_in_field_session)
        norm_hist_real = np.nan_to_num(hd_spikes_real_hist / hd_session_real_hist)
        plt.cla()
        # plt.scatter(range(0, 360), hd_hist_normed_estimate, color='red', alpha=0.3, normed=True)
        # plt.plot(hd_hist_normed_estimate, color='red')
        max_est = max(hd_hist_normed_estimate)
        max_real = max(norm_hist_real)
        scale = max_real / max_est
        fig, ax = plt.subplots()
        ax.plot(norm_hist_real, color='red', alpha=0.4, label='real', linewidth=5)
        ax.plot(estimate_smooth * scale, color='black', alpha=0.4, label='estimate', linewidth=5)
        plt.title('hd ' + str(round(field.hd_score, 1)) + ' grid ' + str(round(field.grid_score, 1)))
        legend = plt.legend()
        legend.get_frame().set_facecolor('none')
        plt.savefig(local_path + animal + field.session_id + str(field.cluster_id) + str(field.field_id) + 'estimated_hd_rate_vs_real.png')


def main():
    process_data('mouse')
    # process_data('rat')


if __name__ == '__main__':
    main()