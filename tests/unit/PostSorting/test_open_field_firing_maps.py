import numpy as np
from PostSorting.open_field_firing_maps import *
import pandas as pd

def test_calculate_firing_rate_for_cluster_parallel():
    cluster = 0
    smooth = 22.0
    firing_data_spatial = pd.read_pickle("/mnt/datastore/Harry/MouseOF/test_recordings/M5_2018-03-06_15-34-44_of/MountainSort/DataFrames/spatial_firing.pkl")
    spatial = pd.read_pickle("/mnt/datastore/Harry/MouseOF/test_recordings/M5_2018-03-06_15-34-44_of/MountainSort/DataFrames/position.pkl")
    positions_x = spatial.position_x_pixels.values
    positions_y = spatial.position_y_pixels.values
    number_of_bins_x = 41
    number_of_bins_y = 41
    bin_size_pixels = 11
    min_dwell = 3.0
    min_dwell_distance_pixels = 22
    dt_position_ms = spatial.synced_time.diff().mean()*1000

    firing_rate_map_old = calculate_firing_rate_for_cluster_parallel(cluster, smooth,
                                                                     firing_data_spatial,
                                                                     positions_x, positions_y,
                                                                     number_of_bins_x, number_of_bins_y,
                                                                     bin_size_pixels, min_dwell,
                                                                     min_dwell_distance_pixels,
                                                                     dt_position_ms)

    firing_rate_map_new = calculate_firing_rate_for_cluster_parallel(cluster, smooth,
                                                                     firing_data_spatial,
                                                                     positions_x, positions_y,
                                                                     number_of_bins_x, number_of_bins_y,
                                                                     bin_size_pixels, min_dwell,
                                                                     min_dwell_distance_pixels,
                                                                     dt_position_ms)

    assert np.allclose(firing_rate_map_old, firing_rate_map_new, rtol=1e-05, atol=1e-08)

def main():
    test_calculate_firing_rate_for_cluster_parallel()

if __name__ == '__main__':
    main()