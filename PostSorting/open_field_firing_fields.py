import numpy as np
import pandas as pd
import PostSorting.open_field_head_direction

# import matplotlib.pylab as plt


# return indices of neighbors of bin considering borders
def find_neighbors(bin_to_test, max_x, max_y):
    x = bin_to_test[0]
    y = bin_to_test[1]

    neighbors = [[x, y+1], [x, y-1], [x+1, y], [x-1, y]]

    if x == max_x:
        neighbors = [[x, y+1], [x, y-1], [x-1, y]]
    if y == max_y:
        neighbors = [[x, y-1], [x+1, y], [x-1, y]]
    if x == max_x and y == max_y:
        neighbors = [[x, y-1], [x-1, y]]
    if x == 0:
        neighbors = [[x, y+1], [x, y-1], [x+1, y]]
    if y == 0:
        neighbors = [[x, y+1], [x+1, y], [x-1, y]]
    if x == 0 and y == 0:
        neighbors = [[x, y+1], [x+1, y]]

    if x == max_x and y == 0:
        neighbors = [[x, y+1], [x-1, y]]

    if y == max_y and x == 0:
        neighbors = [[x, y-1], [x+1, y]]

    return neighbors


# return the masked rate map and change the neighbor's indices to 1 if they are above threshold
def find_neighborhood(masked_rate_map, rate_map, firing_rate_of_max):
    changed = False
    threshold = firing_rate_of_max * 20 / 100

    firing_field_bins = np.array(np.where(masked_rate_map > 0))
    firing_field_bins = firing_field_bins.T

    for bin_to_test in firing_field_bins:
        masked_rate_map[bin_to_test[0], bin_to_test[1]] = 2
        neighbors = find_neighbors(bin_to_test, max_x=(masked_rate_map.shape[0]-1), max_y=(masked_rate_map.shape[1]-1))
        for neighbor in neighbors:
            if masked_rate_map[neighbor[0], neighbor[1]] == 2:
                continue

            firing_rate = rate_map[neighbor[0], neighbor[1]]
            if firing_rate >= threshold:
                masked_rate_map[neighbor[0], neighbor[1]] = 1
                changed = True

    return masked_rate_map, changed


# check if the detected field is big enough to be a firing field
def test_if_field_is_big_enough(field_indices):
    number_of_pixels = len(field_indices)
    if number_of_pixels > 45:
        return True
    return False


# this is to avoid identifying the whole rate map as a field
def test_if_field_is_small_enough(field_indices, rate_map):
    number_of_pixels_in_field = len(field_indices)
    number_of_pixels_on_map = len(rate_map.flatten())
    if number_of_pixels_in_field > number_of_pixels_on_map / 2:
        return False
    else:
        return True


# test if the firing rate of the detected local maximum is higher than average + std firing
def test_if_highest_bin_is_high_enough(rate_map, highest_rate_bin):
    flat_rate_map = rate_map.flatten()
    rate_map_without_removed_fields = np.take(flat_rate_map, np.where(flat_rate_map >= 0))
    average_rate = np.mean(rate_map_without_removed_fields)
    std_rate = np.std(rate_map)

    firing_rate_of_highest_bin = rate_map[highest_rate_bin[0], highest_rate_bin[1]]

    if firing_rate_of_highest_bin > average_rate + std_rate:
        return True
    else:
        return False


# find indices for an individual firing field
def find_current_maxima_indices(rate_map):
    highest_rate_bin = np.unravel_index(rate_map.argmax(), rate_map.shape)
    found_new = test_if_highest_bin_is_high_enough(rate_map, highest_rate_bin)
    max_fr = rate_map[highest_rate_bin]
    if found_new is False:
        return None, found_new, None

    # plt.imshow(rate_map)
    # plt.scatter(highest_rate_bin[1], highest_rate_bin[0], marker='o', s=500, color='yellow')
    masked_rate_map = np.full((rate_map.shape[0], rate_map.shape[1]), 0)
    masked_rate_map[highest_rate_bin] = 1
    changed = True
    while changed:
        masked_rate_map, changed = find_neighborhood(masked_rate_map, rate_map, rate_map[highest_rate_bin])

    field_indices = np.array(np.where(masked_rate_map > 0)).T
    found_new = test_if_field_is_big_enough(field_indices)
    if found_new is False:
        return None, found_new, None
    found_new = test_if_field_is_small_enough(field_indices, rate_map)
    if found_new is False:
        field_indices = None

    return field_indices, found_new, max_fr


# mark indices of firing fields that are already found (so we don't find them again)
def remove_indices_from_rate_map(rate_map, indices):
    for index in indices:
        rate_map[index[0], index[1]] = -10
    return rate_map


# find firing fields and add them to spatial firing data frame
def analyze_firing_fields(spatial_firing, spatial_data, prm):
    print('I will identify individual firing fields if possible.')
    firing_fields = []
    max_firing_rates = []
    for cluster in range(len(spatial_firing)):
        cluster = spatial_firing.cluster_id.values[cluster] - 1
        firing_fields_cluster = []
        max_firing_rates_cluster = []
        rate_map = spatial_firing.firing_maps[cluster].copy()
        found_new = True
        while found_new:
            # plt.show()
            field_indices, found_new, max_firing_rate = find_current_maxima_indices(rate_map)
            if found_new:
                firing_fields_cluster.append(field_indices)
                max_firing_rates_cluster.append(max_firing_rate)
                rate_map = remove_indices_from_rate_map(rate_map, field_indices)
        # plt.clf()
        firing_fields.append(firing_fields_cluster)
        max_firing_rates.append(max_firing_rates_cluster)

    spatial_firing['firing_fields'] = firing_fields
    spatial_firing['field_max_firing_rate'] = max_firing_rates
    spatial_firing = analyze_hd_in_firing_fields(spatial_firing, spatial_data, prm)
    return spatial_firing


def analyze_hd_in_firing_fields(spatial_firing, spatial_data, prm):
    print('I will analyze head-direction in the detected firing fields.')
    hd_session_all = []
    hd_cluster_all = []
    field_p_all = []
    field_stat_all = []
    max_firing_rates_all = []
    preferred_hd_all = []
    hd_score_all = []

    for cluster in range(len(spatial_firing)):
        cluster = spatial_firing.cluster_id.values[cluster] - 1
        number_of_firing_fields = len(spatial_firing.firing_fields[cluster])
        firing_fields_cluster = spatial_firing.firing_fields[cluster]
        hd_session = []
        hd_cluster = []
        field_p = []
        field_stat = []
        max_firing_rate = []
        preferred_hd = []
        hd_score = []

        if number_of_firing_fields > 0:
            for field_id, field in enumerate(firing_fields_cluster):
                hd_in_field_session = PostSorting.open_field_head_direction.get_hd_in_firing_rate_bins_for_session(spatial_data, field, prm)
                hd_in_field_cluster = PostSorting.open_field_head_direction.get_hd_in_firing_rate_bins_for_cluster(spatial_firing, field, cluster, prm)
                p, stat = PostSorting.open_field_head_direction.compare_hd_distributions_in_cluster_to_session(hd_in_field_session, hd_in_field_cluster)
                hd_hist_session = PostSorting.open_field_head_direction.get_hd_histogram(hd_in_field_session)
                hd_hist_session /= prm.get_sampling_rate()
                hd_hist_cluster = PostSorting.open_field_head_direction.get_hd_histogram(hd_in_field_cluster)
                hd_hist_cluster = np.divide(hd_hist_cluster, hd_hist_session, out=np.zeros_like(hd_hist_cluster), where=hd_hist_session != 0)
                max_firing_rate_cluster = np.max(hd_hist_cluster.flatten())
                hd_score_cluster = PostSorting.open_field_head_direction.get_hd_score_for_cluster(hd_hist_cluster)
                preferred_direction = np.where(hd_hist_cluster == max_firing_rate_cluster)
                hd_session.append(list(hd_hist_session))
                hd_cluster.append(list(hd_hist_cluster))
                field_p.append(p)
                field_stat.append(stat)
                max_firing_rate.append(max_firing_rate_cluster/1000)
                preferred_hd.append(preferred_direction[0])
                hd_score.append(hd_score_cluster)
        else:
            hd_session.append([None])
            hd_cluster.append([None])
            field_p.append([None])
            field_stat.append([None])
            max_firing_rate.append(None)
            preferred_hd.append(None)
            hd_score.append(None)
        hd_session_all.append(hd_session)
        hd_cluster_all.append(hd_cluster)
        field_p_all.append(field_p)
        field_stat_all.append(field_stat)
        max_firing_rates_all.append(max_firing_rate)
        preferred_hd_all.append(preferred_hd)
        hd_score_all.append(hd_score)

    spatial_firing['firing_fields_hd_session'] = hd_session_all
    spatial_firing['firing_fields_hd_cluster'] = hd_cluster_all
    spatial_firing['field_hd_p'] = field_p_all
    spatial_firing['field_stat'] = field_stat_all
    spatial_firing['field_hd_max_rate'] = max_firing_rates_all
    spatial_firing['field_preferred_hd'] = preferred_hd_all
    spatial_firing['field_hd_score'] = hd_score_all
    return spatial_firing


def main():
    firing_rate_maps = np.load('C:/Users/s1466507/Documents/Ephys/test_overall_analysis/M5_2018-03-06_15-34-44_of/M5_2018-03-06_15-34-44_of.npy')
    cluster_id = np.arange(len(firing_rate_maps))
    spatial_firing = pd.DataFrame(cluster_id)
    spatial_firing['firing_maps'] = list(firing_rate_maps)
    spatial_firing = analyze_firing_fields(spatial_firing)

if __name__ == '__main__':
    main()