#  This script is to perform overall analysis on multiple days recorded from a group of animals
import OverallAnalysis.organize_cluster_data
import pandas as pd

path_to_data = 'C:/Users/s1466507/Desktop/data_all.csv'


def run_analyses():
    spike_data_frame = pd.read_csv(path_to_data)  # reads csv, puts it in df
    print(spike_data_frame.head())
    pass


def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    run_analyses()


if __name__ == '__main__':
    main()