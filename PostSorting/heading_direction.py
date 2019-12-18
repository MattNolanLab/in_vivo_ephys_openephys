import numpy as np


def calculate_heading_direction(position_x, position_y, pad_fist_value=True):

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
    heading_direction_deg = np.degrees(heading_direction)
    if pad_fist_value:
        heading_direction_deg = np.insert(heading_direction_deg, 0, heading_direction_deg[0])

    return heading_direction_deg


def add_heading_direction_to_data_frame(spatial_firing):
    pass


def main():
    x = [0, 1, 2, 2, 1]
    y = [0, 1, 1, 0, 1]
    calculate_heading_direction(x, y)


if __name__ == '__main__':
    main()