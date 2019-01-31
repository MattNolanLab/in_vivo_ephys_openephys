from mat4py import loadmat
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import PostSorting.open_field_spatial_data


def get_position_data_frame(matlab_data):
    position_data = pd.DataFrame()
    position_data['time_seconds'] = matlab_data['t']
    position_data.time_seconds = position_data.time_seconds.sum()
    position_data['x_left_cleaned'] = matlab_data['x1']
    position_data.x_left_cleaned = position_data.x_left_cleaned.sum()
    position_data['x_right_cleaned'] = matlab_data['x2']
    position_data.x_right_cleaned = position_data.x_right_cleaned.sum()
    position_data['y_left_cleaned'] = matlab_data['y1']
    position_data.y_left_cleaned = position_data.y_left_cleaned.sum()
    position_data['y_right_cleaned'] = matlab_data['y2']
    position_data.y_right_cleaned = position_data.y_right_cleaned.sum()

    position_data = PostSorting.open_field_spatial_data.calculate_position(position_data)  # get central position and interpolate missing data
    position_data = PostSorting.open_field_spatial_data.calculate_head_direction(position_data)  # use coord from the two beads to get hd and interpolate
    position_data = PostSorting.open_field_spatial_data.shift_to_start_from_zero_at_bottom_left(position_data)
    position_data = PostSorting.open_field_spatial_data.calculate_central_speed(position_data)
    position_of_mouse = position_data[['time_seconds', 'position_x', 'position_x_pixels', 'position_y', 'position_y_pixels', 'hd', 'speed']].copy()
    # plt.plot(position_data.position_x, position_data.position_y) # this is to plot the trajectory. it looks weird
    return position_of_mouse


def main():
    matlab_data = loadmat('//ardbeg.mvm.ed.ac.uk/nolanlab/Klara/grid_field_analysis/moser_data/Sargolini/11084-03020501_t2c1.mat')
    # todo: iterate over all data and load them for analysis - get session ID so cells can be added to session
    position_data = get_position_data_frame(matlab_data)

    # get spike df     # spike_times = data['ts']
    ''' this should have for each cell in the session - the session should be identified by the session id in the position data (?)
    "session_id": session_id,
    "cluster_id":  int(cluster),
    "tetrode": tetrode,
    "primary_channel": ch,
    "firing_times": cluster_firings,
    "firing_times_opto": cluster_firings_opto
    '''

if __name__ == '__main__':
    main()