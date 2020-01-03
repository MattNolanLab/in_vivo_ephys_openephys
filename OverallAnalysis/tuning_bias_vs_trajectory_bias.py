'''
I will quantify how similar the trajectory hd distribution is to a uniform distribution (1 sample watson test)
 and then correlate the results of this to the number of significant bins from the distributive shuffled analysis
'''

import numpy as np
import OverallAnalysis.folder_path_settings
import OverallAnalysis.shuffle_field_analysis_all_animals
import pandas as pd

analysis_path = OverallAnalysis.folder_path_settings.get_local_path() + '/tuning_bias_vs_trajectory_bias/'
local_path_to_shuffled_field_data_mice = analysis_path + 'shuffled_field_data_all_mice.pkl'
local_path_to_shuffled_field_data_rats = analysis_path + 'shuffled_field_data_all_rats.pkl'


def process_data(animal):
    if animal == 'mouse':
        local_path_to_field_data = local_path_to_shuffled_field_data_mice
        accepted_fields = pd.read_excel(analysis_path + 'list_of_accepted_fields.xlsx')
        shuffled_field_data = pd.read_pickle(local_path_to_field_data)
        shuffled_field_data = OverallAnalysis.shuffle_field_analysis_all_animals.tag_accepted_fields_mouse(shuffled_field_data, accepted_fields)

    else:
        local_path_to_field_data = local_path_to_shuffled_field_data_rats
        accepted_fields = pd.read_excel(analysis_path + 'included_fields_detector2_sargolini.xlsx')
        shuffled_field_data = pd.read_pickle(local_path_to_field_data)
        shuffled_field_data = OverallAnalysis.shuffle_field_analysis_all_animals.tag_accepted_fields_rat(shuffled_field_data, accepted_fields)

    grid = shuffled_field_data.grid_score >= 0.4
    hd = shuffled_field_data.hd_score >= 0.5
    grid_cells = np.logical_and(grid, np.logical_not(hd))

    # add trajectory hd vs uniform results to df


def main():
    process_data()



if __name__ == '__main__':
    main()


