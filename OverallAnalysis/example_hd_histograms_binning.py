import numpy as np
import OverallAnalysis.folder_path_settings
import pandas as pd
import PostSorting.open_field_head_direction
import PostSorting.open_field_make_plots


local_path = OverallAnalysis.folder_path_settings.get_local_path() + '/example_hd_histograms/'
server_path_mouse = OverallAnalysis.folder_path_settings.get_server_path_mouse()
server_path_rat = OverallAnalysis.folder_path_settings.get_server_path_rat()
server_path_simulated = OverallAnalysis.folder_path_settings.get_server_path_simulated()


def plot_example_hd_histograms():
    position = pd.read_pickle(local_path + 'position.pkl')
    hd_pos = np.array(position.hd)
    hd_pos = (hd_pos + 180) * np.pi / 180

    spatial_firing = pd.read_pickle(local_path + 'spatial_firing.pkl')
    hd = np.array(spatial_firing.hd.iloc[0])
    hd = (hd + 180) * np.pi / 180
    hd_spike_histogram_23 = PostSorting.open_field_head_direction.get_hd_histogram(hd, window_size=23)
    hd_spike_histogram_10 = PostSorting.open_field_head_direction.get_hd_histogram(hd, window_size=10)
    hd_spike_histogram_20 = PostSorting.open_field_head_direction.get_hd_histogram(hd, window_size=20)
    hd_spike_histogram_30 = PostSorting.open_field_head_direction.get_hd_histogram(hd, window_size=30)
    hd_spike_histogram_40 = PostSorting.open_field_head_direction.get_hd_histogram(hd, window_size=40)

    hd_spike_histogram_23_pos = PostSorting.open_field_head_direction.get_hd_histogram(hd_pos, window_size=23)
    hd_spike_histogram_10_pos = PostSorting.open_field_head_direction.get_hd_histogram(hd_pos, window_size=10)
    hd_spike_histogram_20_pos = PostSorting.open_field_head_direction.get_hd_histogram(hd_pos, window_size=20)
    hd_spike_histogram_30_pos = PostSorting.open_field_head_direction.get_hd_histogram(hd_pos, window_size=30)
    hd_spike_histogram_40_pos = PostSorting.open_field_head_direction.get_hd_histogram(hd_pos, window_size=40)

    hd_spike_histogram_10_norm = hd_spike_histogram_10 / hd_spike_histogram_10_pos
    hd_spike_histogram_20_norm = hd_spike_histogram_20 / hd_spike_histogram_20_pos
    hd_spike_histogram_30_norm = hd_spike_histogram_30 / hd_spike_histogram_30_pos
    hd_spike_histogram_40_norm = hd_spike_histogram_40 / hd_spike_histogram_40_pos

    PostSorting.open_field_make_plots.plot_polar_hd_hist(hd_spike_histogram_10_norm, hd_spike_histogram_10_pos, 10, local_path, color1='navy', color2='lime', title='')
    PostSorting.open_field_make_plots.plot_polar_hd_hist(hd_spike_histogram_20_norm, hd_spike_histogram_20_pos, 20, local_path, color1='navy',
                                                         color2='lime', title='')
    PostSorting.open_field_make_plots.plot_polar_hd_hist(hd_spike_histogram_30_norm, hd_spike_histogram_30_pos, 30, local_path, color1='navy',
                                                         color2='lime', title='')
    PostSorting.open_field_make_plots.plot_polar_hd_hist(hd_spike_histogram_40_norm, hd_spike_histogram_40_pos, 40, local_path, color1='navy',
                                                         color2='lime', title='')


def main():
    plot_example_hd_histograms()


if __name__ == '__main__':
    main()
