import csv
import glob
import numpy as np
import os
import pandas as pd
import math_utility


import PostSorting.parameters

''' The name of bonsai output files is not standardised in all experiments, so this function checks all csv
files in the recording folder and reads the first line. Our bonsai output files start with the date and 'T'
for example: 2017-11-21T
One recording folder has no more than one bonsai file, so this is sufficient for identification.
'''


def find_bonsai_file(recording_folder):
    path_to_bonsai_file = ''
    is_found = False
    for name in glob.glob(recording_folder + '/*.csv'):
        if os.path.exists(name):
            with open(name, newline='') as file:
                try:
                    reader = csv.reader(file)
                    row1 = next(reader)
                    if "T" not in row1[0]:
                        continue
                    else:
                        if len(row1[0].split('T')[0]) == 10:
                            path_to_bonsai_file = name
                            is_found = True
                except Exception as ex:
                    print('Could not read csv file:')
                    print(name)
                    print(ex)

    return path_to_bonsai_file, is_found

''' Read raw position data and sync LED intensity from Bonsai file amd convert time to seconds
'''


def convert_time_to_seconds(position_data):
    position_data['hours'], position_data['minutes'], position_data['seconds'] = position_data['time'].str.split(':', 2).str
    position_data['hours'] = position_data['hours'].astype(int)
    position_data['minutes'] = position_data['minutes'].astype(int)
    position_data['seconds'] = position_data['seconds'].astype(float)
    position_data['time_seconds'] = position_data['hours'] * 3600 + position_data['minutes']*60 + position_data['seconds']
    position_data['time_seconds'] = position_data['time_seconds'] - position_data['time_seconds'][0]
    return position_data


def read_position(path_to_bonsai_file):
    position_data = pd.read_csv(path_to_bonsai_file, sep=' ', header=None)
    if len(list(position_data)) > 6:
        position_data = position_data.drop([6], axis=1)  # remove column of NaNs due to extra space at end of lines
    position_data['date'], position_data['time'] = position_data[0].str.split('T', 1).str

    position_data['time'], position_data['str_to_remove'] = position_data['time'].str.split('+', 1).str
    position_data = position_data.drop([0, 'str_to_remove'], axis=1)  # remove first column that got split into date and time

    position_data.columns = ['x_left', 'y_left', 'x_right', 'y_right', 'syncLED', 'date', 'time']
    position_data = convert_time_to_seconds(position_data)
    return position_data


def calculate_speed(position_data):
    elapsed_time = position_data['time_seconds'].diff()
    distance_travelled = np.sqrt(position_data['x_left'].diff().pow(2) + position_data['y_left'].diff().pow(2))
    position_data['speed_left'] = distance_travelled / elapsed_time
    distance_travelled = np.sqrt(position_data['x_right'].diff().pow(2) + position_data['y_right'].diff().pow(2))
    position_data['speed_right'] = distance_travelled / elapsed_time
    return position_data


def calculate_central_speed(position_data):
    elapsed_time = position_data['time_seconds'].diff()
    distance_travelled = np.sqrt(position_data['position_x'].diff().pow(2) + position_data['position_y'].diff().pow(2))
    position_data['speed'] = distance_travelled / elapsed_time
    return position_data


def remove_jumps(position_data, prm):
    max_speed = 1  # m/s, anything above this is not realistic
    pixel_ratio = prm.get_pixel_ratio()
    max_speed_pixels = max_speed * pixel_ratio
    speed_exceeded_left = position_data['speed_left'] > max_speed_pixels
    position_data['x_left_without_jumps'] = position_data.x_left[speed_exceeded_left == False]
    position_data['y_left_without_jumps'] = position_data.y_left[speed_exceeded_left == False]

    speed_exceeded_right = position_data['speed_right'] > max_speed_pixels
    position_data['x_right_without_jumps'] = position_data.x_right[speed_exceeded_right == False]
    position_data['y_right_without_jumps'] = position_data.y_right[speed_exceeded_right == False]

    remains_left = (len(position_data) - speed_exceeded_left.sum())/len(position_data)*100
    remains_right = (len(position_data) - speed_exceeded_right.sum())/len(position_data)*100
    print('{} % of right side tracking data, and {} % of left side'
          ' remains after removing the ones exceeding speed limit.'.format(remains_right, remains_left))
    return position_data


def get_distance_of_beads(position_data):
    distance_between_beads = np.sqrt((position_data['x_left'] - position_data['x_right']).pow(2) + (position_data['y_left'] - position_data['y_right']).pow(2))
    return distance_between_beads


def remove_points_where_beads_are_far_apart(position_data):
    minimum_distance = 40
    distance_between_beads = get_distance_of_beads(position_data)
    distance_exceeded = distance_between_beads > minimum_distance
    position_data['x_left_cleaned'] = position_data.x_left_without_jumps[distance_exceeded == False]
    position_data['x_right_cleaned'] = position_data.x_right_without_jumps[distance_exceeded == False]
    position_data['y_left_cleaned'] = position_data.y_left_without_jumps[distance_exceeded == False]
    position_data['y_right_cleaned'] = position_data.y_right_without_jumps[distance_exceeded == False]
    return position_data


def curate_position(position_data, prm):
    position_data = remove_jumps(position_data, prm)
    position_data = remove_points_where_beads_are_far_apart(position_data)
    return position_data


def calculate_position(position_data):
    position_data['position_x_tmp'] = (position_data['x_left_cleaned'] + position_data['x_right_cleaned']) / 2
    position_data['position_y_tmp'] = (position_data['y_left_cleaned'] + position_data['y_right_cleaned']) / 2

    position_data['position_x'] = position_data['position_x_tmp'].interpolate()  # interpolate missing data
    position_data['position_y'] = position_data['position_y_tmp'].interpolate()
    return position_data


def calculate_head_direction(position):
    position['head_dir_tmp'] = np.degrees(np.arctan((position['y_left_cleaned'] + position['y_right_cleaned']) / (position['x_left_cleaned'] + position['x_right_cleaned'])))
    rho, hd = math_utility.cart2pol(position['x_right_cleaned'] - position['x_left_cleaned'], position['y_right_cleaned'] - position['y_left_cleaned'])
    position['hd'] = np.degrees(hd)
    position['hd'] = position['hd'].interpolate()  # interpolate missing data
    return position


def convert_to_cm(position_data, params):
    pixel_ratio = params.get_pixel_ratio()
    position_data['position_x_pixels'] = position_data.position_x
    position_data['position_y_pixels'] = position_data.position_y
    position_data['position_x'] = position_data.position_x / pixel_ratio * 100
    position_data['position_y'] = position_data.position_y / pixel_ratio * 100
    return position_data


def shift_to_start_from_zero_at_bottom_left(position_data):
    # this is copied from MATLAB script, 0.0001 is here to 'avoid bin zero in first point'
    position_data['position_x'] = position_data.position_x - min(position_data.position_x) #+ 0.0001
    position_data['position_y'] = position_data.position_y - min(position_data.position_y) #+ 0.0001
    return position_data


def process_position_data(recording_folder, params):
    path_to_bonsai_file, is_found = find_bonsai_file(recording_folder)
    position_data = read_position(path_to_bonsai_file)  # raw position data from bonsai output
    position_data = calculate_speed(position_data)
    position_data = curate_position(position_data, params)  # remove jumps from data, and when the beads are far apart
    position_data = calculate_position(position_data)  # get central position and interpolate missing data
    position_data = calculate_head_direction(position_data)  # use coord from the two beads to get hd and interpolate
    position_data = calculate_central_speed(position_data)
    position_data = shift_to_start_from_zero_at_bottom_left(position_data)
    position_data = convert_to_cm(position_data, params)
    position_of_mouse = position_data[['time_seconds', 'position_x', 'position_x_pixels', 'position_y', 'position_y_pixels', 'hd', 'syncLED', 'speed']].copy()
    return position_of_mouse


#  this is here for testing
def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    params = PostSorting.parameters.Parameters()
    params.set_pixel_ratio(440)

    recording_folder = 'C:/Users/s1466507/Documents/Ephys/test_overall_analysis/M5_2018-03-06_15-34-44_of'
    # recording_folder = 'C:/Users/s1466507/Documents/Ephys/test_overall_analysis/M0_2017-11-21_15-52-53'
    position_of_mouse = process_position_data(recording_folder, params)


if __name__ == '__main__':
    main()