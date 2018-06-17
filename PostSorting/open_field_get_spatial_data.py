import csv
import glob
import os

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

    return path_to_bonsai_file, is_found


def read_position(path_to_bonsai_file):
    position_data = []
    # read csv file and return
    # time and xy coordinates for the two beads tracked
    # could be dataframe with time as index 'position_data'
    return position_data


def calculate_position(position_data):
    position_of_mouse = []
    # calculate center of two tracked beads to get the position of the mouse, handle when only one point is available
    return position_of_mouse


def calculate_head_direction(position_data):
    head_direction_of_mouse = []
    # calculate head-direction based on the tracked balls
    return  head_direction_of_mouse


def process_position_data(recording_folder):
    path_to_bonsai_file, is_found = find_bonsai_file(recording_folder)
    position_data = read_position(recording_folder + path_to_bonsai_file)
    position_of_mouse = calculate_position(position_data)
    head_direction_of_mosue = calculate_head_direction(position_data)
    # put these in session dataframe


def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    recording_folder = 'C:/Users/s1466507/Documents/Ephys/test_overall_analysis/M5_2018-03-06_15-34-44_of'
    process_position_data(recording_folder)


if __name__ == '__main__':
    main()