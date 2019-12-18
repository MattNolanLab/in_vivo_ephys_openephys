import math_utility
import numpy as np
import pandas as pd


def calculate_heading_direction(position_x, position_y, pad_first_value=True):

    '''
    Calculate heading direction of animal based on the central position of the tracking markers.
    Method from:
    https://doi.org/10.1016/j.brainres.2014.10.053

    input : position_x and position_y of the animal (arrays)
            pad_first_value - if True, the first value will be repeated so the output array's shape is the
            same as the input
    output : heading direction of animal
    based on the vector from consecutive samples
    '''

    delta_x = np.diff(position_x)
    delta_y = np.diff(position_y)
    heading_direction = np.arctan(delta_y / delta_x)
    rho, heading_direction = math_utility.cart2pol(delta_x, delta_y)

    heading_direction_deg = np.degrees(heading_direction)
    if pad_first_value:
        heading_direction_deg = np.insert(heading_direction_deg, 0, heading_direction_deg[0])

    return heading_direction_deg


def add_heading_direction_to_position_data_frame(position):
    x = position.position_x
    y = position.position_y
    heading_direction = calculate_heading_direction(x, y, pad_first_value=True)
    position['heading_direction'] = heading_direction
    return position


def add_heading_direction_to_spatial_firing_data_frame(spatial_firing, position):
    if 'heading_direction' not in position:
        position = add_heading_direction_to_position_data_frame(position)

    # add corresponding MD for each cluster


def main():
    x = [0, 1, 2, 2, 1]
    y = [0, 1, 1, 0, 1]
    heading_direction_deg = calculate_heading_direction(x, y)

    path = 'C:/Users/s1466507/Documents/Ephys/recordings/M5_2018-03-06_15-34-44_of/MountainSort/DataFrames/'
    position_path = path + 'position.pkl'
    position = pd.read_pickle(position_path)
    spatial_firing_path = path + 'spatial_firing.pkl'
    spatial_firing = pd.read_pickle(spatial_firing_path)
    position = add_heading_direction_to_position_data_frame(position)



if __name__ == '__main__':
    main()