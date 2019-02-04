import file_utility
import glob
from mat4py import loadmat
import matplotlib.pylab as plt
import numpy as np
import os
import pandas as pd
import PostSorting.open_field_firing_maps
import PostSorting.open_field_head_direction
import PostSorting.open_field_spatial_data
import PostSorting.parameters

prm = PostSorting.parameters.Parameters()


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


# calculate the sampling rate of the position data (camera) based on the intervals in the array
def calculate_position_sampling_rate(position_data):
    times = position_data.time_seconds
    interval = times[1] - times[0]
    sampling_rate = 1 / interval
    return sampling_rate


# search for all cells in the session where the position data was found correctly
def get_firing_data(folder_to_search_in, session_id, firing_data):
    firing_times_all_cells = []
    session_ids_all = []
    cell_names_all = []
    cluster_id_all = []
    cell_counter = 1
    for name in glob.glob(folder_to_search_in + '/*' + session_id + '*'):
        if os.path.exists(name) and os.path.isdir(name) is False:
                if 'EEG' not in name and 'EGF' not in name and 'POS' not in name and 'md5' not in name:
                    cell_id = name.split('\\')[-1].split('_')[-1].split('.')[0]
                    print('I found this cell:' + name)
                    firing_times = pd.DataFrame()
                    firing_times['times'] = loadmat(name)['cellTS']
                    firing_times['times'] = firing_times['times'].sum()
                    firing_times_all_cells.append(firing_times.times.values)
                    cell_names_all.append(cell_id)
                    session_ids_all.append(session_id)
                    cluster_id_all.append(cell_counter)
                    cell_counter += 1
    firing_data['session_id'] = session_ids_all
    firing_data['cell_id'] = cell_names_all
    firing_data['cluster_id'] = cluster_id_all
    firing_data['firing_times'] = firing_times_all_cells
    return firing_data


# get corresponding position data for firing events
def get_spatial_data_for_firing_events(firing_data, position_data, sampling_rate_position_data):
    spike_position_x_all = []
    spike_position_y_all = []
    spike_hd_all = []
    spike_speed_all = []
    for index, cell in firing_data.iterrows():
        firing_times = cell.firing_times.round(2)  # turn this into position indices based on sampling rate
        corresponding_indices_in_position_data = np.round(firing_times / (1 / sampling_rate_position_data))
        spike_x = position_data.position_x[corresponding_indices_in_position_data]
        spike_y = position_data.position_y[corresponding_indices_in_position_data]
        spike_hd = position_data.hd[corresponding_indices_in_position_data]
        spike_speed = position_data.speed[corresponding_indices_in_position_data]
        spike_position_x_all.append(spike_x)
        spike_position_y_all.append(spike_y)
        spike_hd_all.append(spike_hd)
        spike_speed_all.append(spike_speed)
    firing_data['position_x'] = spike_position_x_all
    firing_data['position_y'] = spike_position_y_all
    firing_data['hd'] = spike_hd_all
    firing_data['speed'] = spike_speed_all
    return firing_data


# load firing data and get corresponding spatial data
def fill_firing_data_frame(position_data, firing_data, name, folder_to_search_in, session_id):
    sampling_rate_of_position_data = calculate_position_sampling_rate(position_data)
    # example file name: 10073-17010302_POS.mat - ratID-sessionID_POS.mat
    session_id = name.split('\\')[-1].split('.')[0].split('-')[1].split('_')[0]
    print('Session ID = ' + session_id)
    firing_data_session = pd.DataFrame()
    firing_data_session = get_firing_data(folder_to_search_in, session_id, firing_data_session)
    firing_data = firing_data.append(firing_data_session)
    # get corresponding position and HD data for spike data frame
    firing_data = get_spatial_data_for_firing_events(firing_data, position_data, sampling_rate_of_position_data)
    return firing_data


# make folder for output and set parameter object to point at it
def create_folder_structure(file_path, session_id, rat_id, prm):
    main_folder = file_path.split('\\')[:-1][0]
    main_recording_session_folder = main_folder + '/' + session_id + '-' + rat_id
    prm.set_file_path(main_recording_session_folder)
    if os.path.isdir(main_recording_session_folder) is False:
        os.makedirs(main_recording_session_folder)
        print('I made this folder: ' + main_recording_session_folder)


def process_data(folder_to_search_in):
    prm.set_sampling_rate(48000)  # this is according to Sarolini et al. (2006)
    prm.set_sorter_name('Manual')
    # prm.set_is_stable(True)  # todo: this needs to be removed - R analysis won't run for now
    firing_data = pd.DataFrame()
    for name in glob.glob(folder_to_search_in + '/*.mat'):
        if os.path.exists(name):
            if 'POS' in name:
                print('I found this:' + name)
                position_data_matlab = loadmat(name)
                position_data = get_position_data_frame(position_data_matlab)
                session_id = name.split('\\')[-1].split('.')[0].split('-')[1].split('_')[0]
                rat_id = name.split('\\')[-1].split('.')[0].split('-')[0]
                if position_data is not False:
                    create_folder_structure(name, session_id, rat_id, prm)
                    firing_data = fill_firing_data_frame(position_data, firing_data, name, folder_to_search_in, session_id)
                    hd_histogram, spatial_firing = PostSorting.open_field_head_direction.process_hd_data(firing_data, position_data, prm)
                    position_heat_map, spatial_firing = PostSorting.open_field_firing_maps.make_firing_field_maps(position_data, firing_data, prm)
                    # TODO : call individual functions separately, figure out what to do with the dwell time
                    pass
                    #  # spatial_firing = PostSorting.open_field_grid_cells.process_grid_data(spatial_firing)
                    # spatial_firing = PostSorting.open_field_firing_fields.analyze_firing_fields(spatial_firing, position_data, prm)
                    # save data frames
                    # make plots


    print('Processing finished.')


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