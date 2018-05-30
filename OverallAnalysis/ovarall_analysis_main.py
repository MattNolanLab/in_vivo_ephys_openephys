#  This script is to perform overall analysis on multiple days recorded from a group of animals
import OverallAnalysis.organize_cluster_data
import pandas as pd
from pandas.tools.plotting import parallel_coordinates
import scipy.io
import h5py
import matplotlib.pylab as plt
import numpy as np

path_to_data = 'C:/Users/s1466507/Documents/Ephys/test_overall_analysis/'
save_output_path = 'C:/Users/s1466507/Documents/Ephys/overall_figures/'
false_positives_path = path_to_data + 'false_positives.txt'


def describe_dataset(spike_data_frame):
    good_cluster = spike_data_frame['goodcluster'] == 1
    number_of_good_cluters = spike_data_frame[good_cluster].count()
    print('Number of good clusters is:')
    print(number_of_good_cluters.id)

def tag_false_positives(spike_df):
    pass



def get_snippets(filename):
    path = path_to_data + '/' + filename + 'Firings0.mat'
    with h5py.File(path, 'r') as snippets:
        snippets.keys()


    snippets2 = h5py.File(path,'r')
    data = snippets.get('data/variable1')
    data = np.array(data) # For converting to numpy array

    return snippets


def plot_good_cells_per_day(spike_data_frame):
    for name, group in spike_data_frame.groupby('animal'):
        # print(name)
        by_day = group.groupby('day').goodcluster.agg('sum')
        # print(by_day)
        plt.style.use('ggplot')
        plt.xlabel('Days', fontsize=14)
        plt.ylabel('Number of good clusters', fontsize=14)
        by_day.plot(xlim=(-2, 16), ylim=(0, 20), linewidth=6)
        plt.savefig('C:/Users/s1466507/Documents/Ephys/overall_figures/good_cells_per_day' + name + '.png')
    plt.savefig('C:/Users/s1466507/Documents/Ephys/overall_figures/good_cells_per_day.png')


def some_examples(spike_data_frame):
    spike_data_frame = pd.read_csv(path_to_data + 'data_all.csv')  # reads csv, puts it in df
    good_cluster = spike_data_frame['goodcluster'] == 1
    light_responsive = spike_data_frame['lightscoreP'] <= 0.05

    good_light_responsive = spike_data_frame[good_cluster & light_responsive]

    number_of_good_cluters = spike_data_frame[good_cluster].count()
    print('Number of good clusters is:')
    print(number_of_good_cluters.id)

    print('Number of good clusters per animal:')
    print(spike_data_frame.groupby('animal').goodcluster.sum())

    print('Number of clusters per animal:')
    print(spike_data_frame.groupby('animal').id.agg(['count']))

    print('Number of days per animal:')
    print(spike_data_frame.groupby('animal').day.nunique())

    print('Number of responses per animal:')
    print(spike_data_frame[light_responsive].groupby('animal').day.nunique())

    print('Avg firing freq per animal:')
    print(spike_data_frame[good_cluster].groupby('animal').avgFR.agg(['mean', 'median', 'count', 'min', 'max']))
    firing_freq = spike_data_frame[good_cluster].groupby('animal').avgFR.agg(['mean', 'median', 'count', 'min', 'max'])
    print(firing_freq.head())

    print(spike_data_frame[good_cluster].groupby(['animal', 'day']).avgFR.agg(['mean', 'median', 'count', 'min', 'max']))

    print(spike_data_frame[good_cluster].groupby(['animal', 'day']).goodcluster.agg(['count']))
    good_clusters_per_day = spike_data_frame[good_cluster].groupby(['animal', 'day']).goodcluster.agg(['count'])


def plot_firing_rate_hist(spike_data_frame):
    plt.style.use('ggplot')
    # plt.hist(spike_data_frame.avgFR)
    good_cluster = spike_data_frame['goodcluster'] == 1
    plt.hist(spike_data_frame[good_cluster].avgFR, bins=100)
    plt.show()


def run_analyses():
    spike_data_frame = pd.read_csv(path_to_data + 'data_all.csv')  # reads csv, puts it in df
    good_cluster = spike_data_frame['goodcluster'] == 1
    light_responsive = spike_data_frame['lightscoreP'] <= 0.05

    #describe_dataset(spike_data_frame)
    #plot_firing_rate_hist(spike_data_frame)
    #plot_good_cells_per_day(spike_data_frame)











   #  print(good_light_responsive[["id","cluster","animal", "goodcluster", "lightscoreP"]])


    #snippets = get_snippets('M0_2017-11-21_15-52-53/')  # I will use the folder name as an ID here once it's added to the spreadsheet

    pass


def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    run_analyses()


if __name__ == '__main__':
    main()