import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import math


def get_position_heatmap(spatial_data):
    position = spatial_data[['position_x', 'position_y']]

    min_position_x = min(position.position_x)
    max_position_x = max(position.position_x)
    number_of_bins_x = int(math.ceil((max_position_x - min_position_x) / 2))

    min_position_y = min(position.position_y)
    max_position_y = max(position.position_x)
    number_of_bins_y = int(math.ceil((max_position_y - min_position_y) / 2))

    x_cut = pd.cut(position.position_x, np.linspace(min_position_x, max_position_x, number_of_bins_x), right=False)
    y_cut = pd.cut(position.position_y, np.linspace(min_position_y, max_position_y, number_of_bins_y), right=False)
    heatmap = np.nan_to_num(position.groupby([x_cut, y_cut]).count().position_x.values)
    heatmap = np.reshape(heatmap, (number_of_bins_x - 1, number_of_bins_y - 1))
    #plt.imshow(heatmap, cmap='jet', interpolation='nearest')
    #plt.show()
    return heatmap


def get_spike_heatmaps(firing_data_spatial):
    spatial_firing_maps = pd.DataFrame(columns=['firing_map'])
    for cluster in range(len(firing_data_spatial)):
        cluster_firings = pd.DataFrame({'position_x': firing_data_spatial.position_x[cluster], 'position_y': firing_data_spatial.position_y[cluster]})
        firing_heatmap = get_position_heatmap(cluster_firings)
        spatial_firing_maps = spatial_firing_maps.append({
            "firing_map": firing_heatmap
        }, ignore_index=True)
    firing_data_spatial['firing_maps'] = spatial_firing_maps.firing_map
    return firing_data_spatial


def make_firing_field_maps(spatial_data, firing_data_spatial, prm):
    position_heat_map = get_position_heatmap(spatial_data)
    spike_heat_maps = get_spike_heatmaps(firing_data_spatial)
    # plt.imshow(spike_heat_maps.firing_maps[5] / position_heat_map, cmap='jet', interpolation='lanczos')

    return position_heat_map