#  This script is to perform overall analysis on multiple days recorded from a group of animals
import OverallAnalysis.organize_cluster_data
import pandas as pd
import scipy.io

path_to_data = 'C:/Users/s1466507/Documents/Ephys/test_overall_analysis/'


def get_snippets(filename):
    snippets = scipy.io.loadmat(path_to_data + '/' + filename + 'Firings0.mat')
    return snippets


def run_analyses():
    spike_data_frame = pd.read_csv(path_to_data + 'data_all.csv')  # reads csv, puts it in df
    print(spike_data_frame.head())
    #snippets = get_snippets(path_to_data + )

    pass


def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    run_analyses()


if __name__ == '__main__':
    main()