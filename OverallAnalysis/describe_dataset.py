import matplotlib.pylab as plt


def some_examples(spike_data_frame):
    good_cluster = spike_data_frame['goodcluster'] == 1
    light_responsive = spike_data_frame['lightscoreP'] <= 0.05

    good_light_responsive = spike_data_frame[good_cluster & light_responsive]

    print('Number of responses per animal:')
    print(spike_data_frame[light_responsive].groupby('animal').day.nunique())

    print('Avg firing freq per animal:')
    print(spike_data_frame[good_cluster].groupby('animal').avgFR.agg(['mean', 'median', 'count', 'min', 'max']))
    firing_freq = spike_data_frame[good_cluster].groupby('animal').avgFR.agg(['mean', 'median', 'count', 'min', 'max'])
    print(firing_freq.head())

    print(spike_data_frame[good_cluster].groupby(['animal', 'day']).avgFR.agg(['mean', 'median', 'count', 'min', 'max']))

    print(spike_data_frame[good_cluster].groupby(['animal', 'day']).goodcluster.agg(['count']))
    good_clusters_per_day = spike_data_frame[good_cluster].groupby(['animal', 'day']).goodcluster.agg(['count'])


def describe_dataset(spike_data_frame):
    good_cluster = spike_data_frame['goodcluster'] == 1
    number_of_good_clusters = spike_data_frame[good_cluster].count()
    print('Number of good clusters is:')
    print(number_of_good_clusters.id)

    print('Number of good clusters per animal:')
    print(spike_data_frame.groupby('animal').goodcluster.sum())

    print('Number of days per animal:')
    print(spike_data_frame.groupby('animal').day.nunique())


    print('Number of grid cells:')
    grid_cell = spike_data_frame['gridscore'] >= 0.3
    print(spike_data_frame[grid_cell].groupby('animal').day.nunique())
    print(spike_data_frame[grid_cell].fig_name_id)

    print('Number of hd cells:')
    hd_cell = spike_data_frame['r_HD'] >= 0.4
    print(spike_data_frame[hd_cell].groupby('animal').day.nunique())
    print(spike_data_frame[hd_cell].fig_name_id)


    print('Number of light responsive cells (low frequency):')
    light_responsive = spike_data_frame['lightscoreP'] <= 0.05
    print('Number of responses per animal:')
    print(spike_data_frame[light_responsive].groupby('animal').day.nunique())
    print(spike_data_frame[light_responsive].fig_name_id)


    print('Number of light responsive cells (high_frequency 100Hz):')
    light_responsive_high_100 = spike_data_frame['lightscore_p3'] <= 0.05
    print(spike_data_frame[light_responsive_high_100].groupby('animal').day.nunique())
    print(spike_data_frame[light_responsive_high_100].fig_name_id)

    print('Number of light responsive cells (high_frequency 200Hz):')
    light_responsive_high_200 = spike_data_frame['lightscore_p4'] <= 0.05
    print(spike_data_frame[light_responsive_high_200].groupby('animal').day.nunique())
    print(spike_data_frame[light_responsive_high_200].fig_name_id)


def plot_good_cells_per_day(spike_data_frame):
    for name, group in spike_data_frame.groupby('animal'):
        by_day = group.groupby('day').goodcluster.agg('sum')
        plt.xlabel('Days', fontsize=14)
        plt.ylabel('Number of good clusters', fontsize=14)
        by_day.plot(xlim=(-2, 16), ylim=(0, 20), linewidth=6)
        plt.savefig('C:/Users/s1466507/Documents/Ephys/overall_figures/good_cells_per_day/good_cells_per_day_' + name + '.png')