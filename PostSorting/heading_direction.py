import numpy as np


def calculate_heading_direction(position_x, position_y):

    '''
    Calculate heading direction of animal based on the central position of the tracking markers.
    Method from:
    https://doi.org/10.1016/j.brainres.2014.10.053
    '''

    delta_x = np.diff(position_x)
    delta_y = np.diff(position_y)
    heading_direction = np.arctan(delta_x / delta_y)
    heading_direction_deg = np.degrees(heading_direction)
    return heading_direction_deg


def add_heading_direction_to_data_frame(spatial_firing):
    pass


def main():
    x = [0, 1, 2, 2, 1]
    y = [0, 1, 1, 0, 1]
    calculate_heading_direction(x, y)


if __name__ == '__main__':
    main()