import glob
from mat4py import loadmat
import matplotlib.pylab as plt
import numpy as np
import os
import pandas as pd
import PostSorting.open_field_spatial_data


# this is necessary, because several datasets are missing tracking information from the second LED
def check_if_all_columns_exist(matlab_data):
    if len(matlab_data['post']) == 0:
        print('The position data is missing the timestamp values.')
        return False
    if len(matlab_data['posx']) == 0:
        print('The position data is missing the x1 coordinates.')
        return False
    if len(matlab_data['posx2']) == 0:
        print('The position data is missing the x2 coordinates.')
        return False
    if len(matlab_data['posy']) == 0:
        print('The position data is missing the y1 coordinates.')
        return False
    if len(matlab_data['posy2']) == 0:
        print('The position data is missing the y2 coordinates.')
        return False
    return True


def get_position_data_frame(matlab_data):
    all_columns_exist = check_if_all_columns_exist(matlab_data)
    if all_columns_exist:
        position_data = pd.DataFrame()
        position_data['time_seconds'] = matlab_data['post']
        position_data.time_seconds = position_data.time_seconds.sum()
        position_data['x_left_cleaned'] = matlab_data['posx']
        position_data.x_left_cleaned = position_data.x_left_cleaned.sum()
        position_data['x_right_cleaned'] = matlab_data['posx2']
        position_data.x_right_cleaned = position_data.x_right_cleaned.sum()
        position_data['y_left_cleaned'] = matlab_data['posy']
        position_data.y_left_cleaned = position_data.y_left_cleaned.sum()
        position_data['y_right_cleaned'] = matlab_data['posy2']
        position_data.y_right_cleaned = position_data.y_right_cleaned.sum()

        position_data = PostSorting.open_field_spatial_data.calculate_position(position_data)  # get central position and interpolate missing data
        position_data = PostSorting.open_field_spatial_data.calculate_head_direction(position_data)  # use coord from the two beads to get hd and interpolate
        position_data = PostSorting.open_field_spatial_data.shift_to_start_from_zero_at_bottom_left(position_data)
        position_data = PostSorting.open_field_spatial_data.calculate_central_speed(position_data)
        position_of_mouse = position_data[['time_seconds', 'position_x', 'position_y', 'hd', 'speed']].copy()
        # plt.plot(position_data.position_x, position_data.position_y) # this is to plot the trajectory. it looks weird
        return position_of_mouse
    else:
        return False


# search for all cells in the session where the position data was found correctly
def get_firing_data(folder_to_search_in, session_id, firing_data):
    for name in glob.glob(folder_to_search_in + '/*' + session_id + '*'):
        if os.path.exists(name):
            if 'EEG' not in name and 'EGF' not in name and 'POS' not in name and 'md5' not in name:
                print('I found this cell' + name)
                firing_times = pd.DataFrame()
                firing_times['times'] = loadmat(name)['cellTS']

                # get firing data from this cell (firing times)
    return firing_data


def process_data(folder_to_search_in):
    for name in glob.glob(folder_to_search_in + '/*.mat'):
        if os.path.exists(name):
            if 'POS' in name:
                print('I found this:' + name)
                position_data_matlab = loadmat(name)
                position_data = get_position_data_frame(position_data_matlab)
                if position_data is not False:
                    # print(position_data.head())
                    # example file name: 10073-17010302_POS.mat - ratID-sessionID_POS.mat
                    session_id = name.split('\\')[-1].split('.')[0].split('-')[1].split('_')[0]
                    print('Session ID = ' + session_id)
                    firing_data = pd.DataFrame()
                    firing_data = get_firing_data(folder_to_search_in, session_id, firing_data)






def main():
    process_data('//ardbeg.mvm.ed.ac.uk/nolanlab/Klara/grid_field_analysis/moser_data/Sargolini/all_data')
    # matlab_data = loadmat('//ardbeg.mvm.ed.ac.uk/nolanlab/Klara/grid_field_analysis/moser_data/Sargolini/11084-03020501_t2c1.mat')
    # todo: iterate over all data and load them for analysis - get session ID so cells can be added to session. iterate in all_data

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