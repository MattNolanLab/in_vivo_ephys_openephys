def find_bonsai_file(recording_folder):
    bonsai_file_name = ''
    # find csv bonsai output based on name or content and return the name of the folder
    return bonsai_file_name


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
    bonsai_file_name = find_bonsai_file(recording_folder)
    position_data = read_position(recording_folder + bonsai_file_name)
    position_of_mouse = calculate_position(position_data)
    head_direction_of_mosue = calculate_head_direction(position_data)
    # put these in session dataframe