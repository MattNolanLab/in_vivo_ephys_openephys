import numpy as np
import OverallAnalysis.folder_path_settings
import pandas as pd
import PostSorting.open_field_head_direction


local_path = OverallAnalysis.folder_path_settings.get_local_path() + '/example_hd_histograms/'
server_path_mouse = OverallAnalysis.folder_path_settings.get_server_path_mouse()
server_path_rat = OverallAnalysis.folder_path_settings.get_server_path_rat()
server_path_simulated = OverallAnalysis.folder_path_settings.get_server_path_simulated()


def plot_example_hd_histograms():
    spatial_firing = pd.read_pickle(local_path + 'spatial_firing.pkl')
    hd = spatial_firing.hd.iloc[0]
    hd = (hd + 180) * np.pi / 180
    hd_spike_histogram_23 = PostSorting.open_field_head_direction.get_hd_histogram(hd, window_size=23)
    hd_spike_histogram_10 = PostSorting.open_field_head_direction.get_hd_histogram(hd, window_size=10)
    hd_spike_histogram_20 = PostSorting.open_field_head_direction.get_hd_histogram(hd, window_size=20)
    hd_spike_histogram_30 = PostSorting.open_field_head_direction.get_hd_histogram(hd, window_size=30)
    hd_spike_histogram_40 = PostSorting.open_field_head_direction.get_hd_histogram(hd, window_size=40)




def main():
    plot_example_hd_histograms()


if __name__ == '__main__':
    main()
