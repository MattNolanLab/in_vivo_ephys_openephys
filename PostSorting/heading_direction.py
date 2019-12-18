import numpy as np


def calculate_heading_direction(position_x, position_y):
    delta_x = np.diff(position_x)
    delta_y = np.diff(position_y)
    heading_direction = np.arctan(delta_x / delta_y)



def add_heading_direction_to_data_frame(spatial_firing):
    pass


def main():
    x = [0, 1, 2, 3, 5, 5]
    y = [5, 6, 6, 1, 2, 3]
    calculate_heading_direction(x, y)


if __name__ == '__main__':
    main()