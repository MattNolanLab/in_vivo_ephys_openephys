import numpy as np
from scipy.stats.stats import pearsonr
import PostSorting.open_field_firing_maps


def make_same_sized_rate_maps(trajectory_1, trajectory_2, spatial_firing_1, spatial_firing_2, prm):
    whole_trajectory = trajectory_1.append(trajectory_2)
    min_dwell, min_dwell_distance_cm = PostSorting.open_field_firing_maps.get_dwell(whole_trajectory, prm)
    bin_size_cm = PostSorting.open_field_firing_maps.get_bin_size(prm)
    number_of_bins_x, number_of_bins_y = PostSorting.open_field_firing_maps.get_number_of_bins(whole_trajectory, prm)
    position_heat_map_first = PostSorting.open_field_firing_maps.get_position_heatmap_fixed_bins(trajectory_1, number_of_bins_x, number_of_bins_y, bin_size_cm, min_dwell_distance_cm, min_dwell)
    position_heat_map_second = PostSorting.open_field_firing_maps.get_position_heatmap_fixed_bins(trajectory_2, number_of_bins_x, number_of_bins_y, bin_size_cm, min_dwell_distance_cm, min_dwell)
    print('made trajectory heatmaps')
    # get bin positions based on trajectory1 + trajectory 2

    # bin everything accordingly
    # see if specifying number of bins for rate map function is good enough
    # return correlation value between two halves


def calculate_spatial_correlation_between_rate_maps(first, second, position_first, position_second, prm):
    """
    This function accepts two sets of data (so for example first and second halves of the recording), makes a rate map
    for both halves in a way that the rate maps correspond, and correlates these rate maps to obtain a spatial correlation
    score.

    first : spatial firing data frame containing data for rate map # 1
    second: spatial firing data frame with data for rate map # 2

    position_first: position df for first rate map
    position_second: position df for second rate map
    """
    rate_map_first, rate_map_second = make_same_sized_rate_maps(position_first, position_second, first, second, prm)
    # possibly need to remove nans here and maybe count how many there are and return that number as well
    pearson_r, p = pearsonr(rate_map_first.flatten(), rate_map_second.flatten())
    return pearson_r



def main():
    trajectory_1 = []
    trajectory_2 = []
    spatial_firing_1 = []
    spatial_firing_2 = []
    make_same_sized_rate_maps(trajectory_1, trajectory_2, spatial_firing_1, spatial_firing_2)


if __name__ == '__main__':
    main()