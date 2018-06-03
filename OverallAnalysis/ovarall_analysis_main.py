#  This script is to perform overall analysis on multiple days recorded from a group of animals
import OverallAnalysis.false_positives
import OverallAnalysis.plot_histograms
import OverallAnalysis.describe_dataset
import OverallAnalysis.read_snippet_data
import pandas as pd
from pandas.tools.plotting import parallel_coordinates
import scipy.io

import matplotlib.pylab as plt
import numpy as np
import os
import OverallAnalysis.plot_histograms

path_to_data = 'C:/Users/s1466507/Documents/Ephys/test_overall_analysis/'
save_output_path = 'C:/Users/s1466507/Documents/Ephys/overall_figures/'
false_positives_path = path_to_data + 'false_positives.txt'


def run_analyses():
    spike_data_frame = pd.read_csv(path_to_data + 'data_all.csv')  # reads csv, puts it in df
    accepted_clusters = OverallAnalysis.false_positives.get_accepted_clusters(spike_data_frame, false_positives_path)

    OverallAnalysis.read_snippet_data.analyze_snippets(spike_data_frame, path_to_data, save_output_path)

    OverallAnalysis.describe_dataset.describe_dataset(accepted_clusters)
    OverallAnalysis.describe_dataset.plot_good_cells_per_day(accepted_clusters)

    OverallAnalysis.plot_histograms.plot_firing_rate_hist(accepted_clusters, save_output_path)
    OverallAnalysis.plot_histograms.plot_grid_score_hist(accepted_clusters, save_output_path)
    OverallAnalysis.plot_histograms.plot_max_fr_spatial(accepted_clusters, save_output_path)
    OverallAnalysis.plot_histograms.plot_max_fr_head_dir(accepted_clusters, save_output_path)

    excitatory = accepted_clusters['maxFRspatial'] <= 10
    excitatory_cells = accepted_clusters[excitatory]
    OverallAnalysis.plot_histograms.plot_firing_rate_hist(excitatory_cells, save_output_path + 'excitatory_')
    OverallAnalysis.plot_histograms.plot_grid_score_hist(excitatory_cells, save_output_path + 'excitatory_')
    OverallAnalysis.plot_histograms.plot_max_fr_spatial_excitatory(excitatory_cells, save_output_path + 'excitatory_')
    OverallAnalysis.plot_histograms.plot_max_fr_head_dir_excitatory(excitatory_cells, save_output_path + 'excitatory_')

    OverallAnalysis.plot_histograms.plot_grid_score_vs_firing_rate(accepted_clusters, save_output_path)


   #  print(good_light_responsive[["id","cluster","animal", "goodcluster", "lightscoreP"]])


    #snippets = get_snippets('M0_2017-11-21_15-52-53/')  # I will use the folder name as an ID here once it's added to the spreadsheet



def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    run_analyses()


if __name__ == '__main__':
    main()