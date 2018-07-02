import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import math


def get_position_heatmap(spatial_data, prm):
    position = spatial_data[['position_x', 'position_y']]

    min_position_x = min(position.position_x)
    max_position_x = max(position.position_x)
    number_of_bins_x = int(math.ceil((max_position_x - min_position_x) / 2))

    min_position_y = min(position.position_y)
    max_position_y = max(position.position_x)
    number_of_bins_y = int(math.ceil((max_position_y - min_position_y) / 2))

    x_cut = pd.cut(position.position_x, np.linspace(min_position_x, max_position_x, number_of_bins_x), right=False)
    y_cut = pd.cut(position.position_y, np.linspace(min_position_y, max_position_y, number_of_bins_y), right=False)
    dwell_time = np.nan_to_num(position.groupby([x_cut, y_cut]).count().position_x.values)
    dwell_time = np.reshape(dwell_time, (number_of_bins_x - 1, number_of_bins_y - 1))
    plt.imshow(dwell_time, cmap='jet', interpolation='nearest')
    plt.show()
    return dwell_time


def make_firing_field_maps(spatial_data, firing_data_spatial, prm):
    position_heat_map = get_position_heatmap(spatial_data, prm)
    return position_heat_map