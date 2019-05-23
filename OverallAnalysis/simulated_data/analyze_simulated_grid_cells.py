import glob
import os
import pandas as pd

analysis_path = '//ardbeg.mvm.ed.ac.uk/nolanlab/Klara/grid_field_analysis/simulated_data/'


def process_data():
    spatial_data_path = analysis_path + 'seed_spatial_data'
    spatial_data = pd.read_pickle(spatial_data_path)
    for name in glob.glob(analysis_path + '*'):
        if os.path.exists(name) and os.path.isdir(name) is False and name != spatial_data_path:
            cell = pd.read_pickle(name)
            os.mkdir(name)
            spatial_data.to_pickle(name + 'position.pkl')
            cell.to_pickle(name + 'spatial_firing.pkl')


def main():
    process_data()


if __name__ == '__main__':
    main()