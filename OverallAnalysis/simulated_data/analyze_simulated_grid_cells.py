import glob
import os
import pandas as pd
import PostSorting.make_plots
import PostSorting.open_field_make_plots
import PostSorting.open_field_firing_fields
import PostSorting.open_field_head_direction
import PostSorting.open_field_grid_cells
import PostSorting.open_field_spatial_data
import PostSorting.parameters
import OverallAnalysis.grid_analysis_other_labs

import matplotlib.pylab as plt

analysis_path = '//ardbeg.mvm.ed.ac.uk/nolanlab/Klara/grid_field_analysis/simulated_data/'

prm = PostSorting.parameters.Parameters()


# load data frames and reorganize to be similar to real data to make it easier to rerun analyses
def organize_data():
    spatial_data_path = analysis_path + 'seed_spatial_data'
    spatial_data = pd.read_pickle(spatial_data_path)
    position_data = pd.DataFrame()
    position_data['synced_time'] = spatial_data.synced_time.iloc[0]
    position_data['position_x'] = spatial_data.position_x.iloc[0]
    position_data['position_y'] = spatial_data.position_y.iloc[0]
    for name in glob.glob(analysis_path + '*'):
        if os.path.exists(name) and os.path.isdir(name) is False and name != spatial_data_path:
            if not os.path.isdir(name + '_simulated'):
                cell = pd.read_pickle(name)
                os.mkdir(name + '_simulated')
                position_data.to_pickle(name + '_simulated/position.pkl')
                cell.to_pickle(name + '_simulated/spatial_firing.pkl')


def get_rate_maps(position_data, firing_data):
    position_heat_map, spatial_firing = OverallAnalysis.grid_analysis_other_labs.firing_maps.make_firing_field_maps(position_data, firing_data, prm)
    return position_heat_map, spatial_firing


def process_data():
    organize_data()
    for name in glob.glob(analysis_path + '*'):
        if os.path.isdir(name):
            if os.path.exists(name + '/position.pkl'):
                position = pd.read_pickle(name + '/position.pkl')
                # process position data - add hd etc
                spatial_firing = pd.read_pickle(name + '/spatial_firing.pkl')
                hd_histogram, spatial_firing = PostSorting.open_field_head_direction.process_hd_data(spatial_firing, position, prm)

                position_heat_map, spatial_firing = get_rate_maps(position, spatial_firing)
                spatial_firing = PostSorting.open_field_grid_cells.process_grid_data(spatial_firing)
                spatial_firing = PostSorting.open_field_firing_fields.analyze_firing_fields(spatial_firing, position, prm)
                # save_data_frames(spatial_firing, position_data)
                # make_plots(position_data, spatial_firing, position_heat_map, hd_histogram, prm)



def main():
    process_data()


if __name__ == '__main__':
    main()