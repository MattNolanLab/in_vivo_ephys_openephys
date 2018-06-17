import csv
import glob
import os
import pandas as pd

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
    position_data['time_seconds'] = position_data['hours'] * 3600 + position_data['minutes']*60 + position_data['seconds']
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


def remove_jumps(position_data, prm):
    max_speed = 1  # m/s, anything above this is not realistic
    return position_data


def curate_position(position_data, prm):
    position_data = remove_jumps(position_data, prm)
    return position_data


def calculate_position(position_data, prm):
    position_of_mouse = ''

    # calculate central position (left_x+right_x)/2
    # remove positions that are more than 15cm away from previous position
    return position_of_mouse


def calculate_head_direction(position_data):
    head_direction_of_mouse = []
    # calculate head-direction based on the tracked balls
    return head_direction_of_mouse\



def process_position_data(recording_folder, params):
    path_to_bonsai_file, is_found = find_bonsai_file(recording_folder)
    position_data = read_position(path_to_bonsai_file)  # raw position data from bonsai output
    position_data_curated = curate_position(position_data, params)
    position_of_mouse = calculate_position(position_data_curated, params)
    head_direction_of_mosue = calculate_head_direction(position_data_curated)
    # put these in session dataframe


def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    params = PostSorting.parameters.Parameters()
    params.set_pixel_ratio(440)

    recording_folder = 'C:/Users/s1466507/Documents/Ephys/test_overall_analysis/M5_2018-03-06_15-34-44_of'
    # recording_folder = 'C:/Users/s1466507/Documents/Ephys/test_overall_analysis/M0_2017-11-21_15-52-53'
    process_position_data(recording_folder, params)




if __name__ == '__main__':
    main()