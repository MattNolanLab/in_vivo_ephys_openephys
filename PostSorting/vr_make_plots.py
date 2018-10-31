import os
import matplotlib.pylab as plt
import plot_utility
import numpy as np
import PostSorting.vr_stop_analysis
import math
import matplotlib.image as mpimg
import pandas as pd

# plot the raw movement channel to check all is good
def plot_movement_channel(location, prm):
    plt.plot(location)
    plt.savefig(prm.get_local_recording_folder_path() + '/Figures/movement' + '.png')
    plt.close()

# plot the trials to check all is good
def plot_trials(trials, prm):
    plt.plot(trials)
    plt.savefig(prm.get_local_recording_folder_path() + '/Figures/trials' + '.png')
    plt.close()

# plot the raw trial channels to check all is good
def plot_trial_channels(trial1, trial2, prm):
    plt.plot(trial1[0,:])
    plt.savefig(prm.get_local_recording_folder_path() + '/Figures/trial_type1' + '.png')
    plt.close()
    plt.plot(trial2[0,:])
    plt.savefig(prm.get_local_recording_folder_path() + '/Figures/trial_type2' + '.png')
    plt.close()


def split_stop_data_by_trial_type(spatial_data):
    locations,trials,trial_type = PostSorting.vr_stop_analysis.load_stop_data(spatial_data)
    stop_data=np.transpose(np.vstack((locations, trials, trial_type)))
    beaconed = np.delete(stop_data, np.where(stop_data[:,2]>0),0)
    nonbeaconed = np.delete(stop_data, np.where(stop_data[:,2]!=1),0)
    probe = np.delete(stop_data, np.where(stop_data[:,2]!=2),0)
    return beaconed, nonbeaconed, probe


def plot_stops_on_track(spatial_data, prm):
    print('I am plotting stop rasta...')
    save_path = prm.get_local_recording_folder_path() + '/Figures/behaviour'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    stops_on_track = plt.figure(figsize=(6,6))
    ax = stops_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

    beaconed,nonbeaconed,probe = split_stop_data_by_trial_type(spatial_data)

    ax.plot(beaconed[:,0], beaconed[:,1], 'o', color='0.5', markersize=2)
    ax.plot(nonbeaconed[:,0], nonbeaconed[:,1], 'o', color='blue', markersize=2)
    ax.plot(probe[:,0], probe[:,1], 'o', color='red', markersize=2)
    #ax.plot(spatial_data.first_series_location_cm, spatial_data.first_series_trial_number, 'o', color='Black', markersize=4)
    #ax.plot(spatial_data.rewarded_stop_locations, spatial_data.rewarded_trials, '>', color='Red', markersize=4)
    plt.ylabel('Stops on trials', fontsize=12, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
    #plt.xlim(min(spatial_data.position_bins),max(spatial_data.position_bins))
    plt.xlim(0,200)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plot_utility.style_track_plot(ax, 200)
    x_max = max(spatial_data.trial_number)+0.5
    plot_utility.style_vr_plot(ax, x_max)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.12, right = 0.87, top = 0.92)
    plt.savefig(prm.get_local_recording_folder_path() + '/Figures/behaviour/stop_raster' + '.png', dpi=200)
    plt.close()


def plot_stop_histogram(spatial_data, prm):
    print('plotting stop histogram...')
    save_path = prm.get_local_recording_folder_path() + '/Figures/behaviour'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    stop_histogram = plt.figure(figsize=(6,4))
    ax = stop_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax.plot(spatial_data.position_bins,spatial_data.average_stops, '-', color='Black')
    plt.ylabel('Stops (cm/s)', fontsize=12, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
    plt.xlim(0,200)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plot_utility.style_track_plot(ax, 200)
    x_max = max(spatial_data.average_stops)+0.5
    plot_utility.style_vr_plot(ax, x_max)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.12, right = 0.87, top = 0.92)
    plt.savefig(prm.get_local_recording_folder_path() + '/Figures/behaviour/stop_histogram' + '.png', dpi=200)
    plt.close()


def plot_speed_histogram(spatial_data, prm):
    print('plotting speed histogram...')
    save_path = prm.get_local_recording_folder_path() + '/Figures/behaviour'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    speed_histogram = plt.figure(figsize=(6,4))
    ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax.plot(spatial_data.position_bins,spatial_data.binned_speed_ms, '-', color='Black')
    plt.ylabel('Speed (cm/s)', fontsize=12, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
    plt.xlim(0,200)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plot_utility.style_track_plot(ax, 200)
    x_max = max(spatial_data.binned_speed_ms)+0.5
    plot_utility.style_vr_plot(ax, x_max)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.12, right = 0.87, top = 0.92)

    plt.savefig(prm.get_local_recording_folder_path() + '/Figures/behaviour/speed_histogram' + '.png', dpi=200)
    plt.close()


def plot_combined_behaviour(spatial_data, prm):
    print('making combined behaviour plot...')
    save_path = prm.get_local_recording_folder_path() + '/Figures/behaviour'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    combined = plt.figure(figsize=(6,9))
    ax = combined.add_subplot(3, 1, 1)  # specify (nrows, ncols, axnum)

    beaconed,nonbeaconed,probe = split_stop_data_by_trial_type(spatial_data)

    ax.plot(beaconed[:,0], beaconed[:,1], 'o', color='0.5', markersize=2)
    ax.plot(nonbeaconed[:,0], nonbeaconed[:,1], 'o', color='blue', markersize=2)
    ax.plot(probe[:,0], probe[:,1], 'o', color='red', markersize=2)
    #ax.plot(spatial_data.first_series_location_cm, spatial_data.first_series_trial_number, 'o', color='Black', markersize=4)
    #ax.plot(spatial_data.rewarded_stop_locations, spatial_data.rewarded_trials, '>', color='Red', markersize=4)
    plt.ylabel('Stops on trials', fontsize=12, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
    #plt.xlim(min(spatial_data.position_bins),max(spatial_data.position_bins))
    plt.xlim(0,200)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plot_utility.style_track_plot(ax, 200)
    x_max = max(spatial_data.trial_number)+0.5
    plot_utility.style_vr_plot(ax, x_max)

    ax = combined.add_subplot(3, 1, 2)  # specify (nrows, ncols, axnum)
    ax.plot(spatial_data.position_bins,spatial_data.average_stops, '-', color='Black')
    plt.ylabel('Stops (cm/s)', fontsize=12, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
    plt.xlim(0,200)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plot_utility.style_track_plot(ax, 200)
    x_max = max(spatial_data.average_stops)+0.5
    plot_utility.style_vr_plot(ax, x_max)

    ax = combined.add_subplot(3, 1, 3)  # specify (nrows, ncols, axnum)
    ax.plot(spatial_data.position_bins,spatial_data.binned_speed_ms, '-', color='Black')
    plt.ylabel('Speed (cm/s)', fontsize=12, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
    plt.xlim(0,200)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plot_utility.style_track_plot(ax, 200)
    x_max = max(spatial_data.binned_speed_ms)+0.5
    plot_utility.style_vr_plot(ax, x_max)
    plt.subplots_adjust(hspace = .5, wspace = .35,  bottom = 0.06, left = 0.12, right = 0.87, top = 0.92)

    plt.savefig(prm.get_local_recording_folder_path() + '/Figures/behaviour/combined_behaviour' + '.png', dpi=200)
    plt.close()


def plot_spikes_on_track(spike_data,spatial_data, prm):
    print('plotting spike rastas...')
    save_path = prm.get_local_recording_folder_path() + '/Figures/spike_trajectories'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster_index in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster_index] - 1
        spikes_on_track = plt.figure(figsize=(6,8))
        ax = spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

        cluster_firing_indices = spike_data.firing_times[cluster_index]
        ax.plot(spatial_data.x_position_cm[cluster_firing_indices], spatial_data.trial_number[cluster_firing_indices], '|', color='Black', markersize=5)
        ax.plot(spike_data.loc[cluster_index].nonbeaconed_position_cm, spike_data.loc[cluster_index].nonbeaconed_trial_number, '|', color='Red', markersize=5)
        ax.plot(spike_data.loc[cluster_index].probe_position_cm, spike_data.loc[cluster_index].probe_trial_number, '|', color='Blue', markersize=5)

        plt.ylabel('Spikes on trials', fontsize=12, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
        plt.xlim(0,200)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        plot_utility.style_track_plot(ax, 200)
        x_max = max(spatial_data.trial_number[cluster_firing_indices])+0.5
        plot_utility.style_vr_plot(ax, x_max)

        plt.savefig(prm.get_local_recording_folder_path() + '/Figures/spike_trajectories/' + spike_data.session_id[cluster_index] + 'track_firing_Cluster_' + str(cluster_index +1) + '.png', dpi=200)
        plt.close()


def plot_firing_rate_maps(spike_data, prm):
    print('I am plotting firing rate maps...')
    save_path = prm.get_local_recording_folder_path() + '/Figures/spike_rate'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster_index in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster_index] - 1
        avg_spikes_on_track = plt.figure()

        bins=range(200)

        unsmooth_b = np.array(spike_data.at[cluster_index, 'avg_spike_per_bin_b'])
        unsmooth_nb = np.array(spike_data.at[cluster_index, 'avg_spike_per_bin_nb'])
        unsmooth_p = np.array(spike_data.at[cluster_index, 'avg_spike_per_bin_p'])

        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(bins, unsmooth_b, '-', color='Black')
        try:
            ax.plot(bins, unsmooth_nb, '-', color='Red')
            ax.plot(bins, unsmooth_p, '-', color='Blue')
        except ValueError:
            continue
        ax.locator_params(axis = 'x', nbins=3)
        ax.set_xticklabels(['0', '100', '200'])
        plt.ylabel('Spike rate (hz)', fontsize=14, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=14, labelpad = 10)
        plt.xlim(0,200)
        x_max = max(spike_data.avg_spike_per_bin_nb[cluster_index])+5
        plot_utility.style_vr_plot(ax, x_max)
        plot_utility.style_track_plot(ax, 200)

        plt.savefig(prm.get_local_recording_folder_path() + '/Figures/spike_rate/' + spike_data.session_id[cluster_index] + 'rate_map_Cluster_' + str(cluster_index +1) + '.png', dpi=200)
        plt.close()


def plot_combined_spike_raster_and_rate(spike_data, spatial_data, prm):
    print('plotting combined spike rastas and spike rate...')
    save_path = prm.get_local_recording_folder_path() + '/Figures/combined_spike_plots'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster_index in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster_index] - 1
        spikes_on_track = plt.figure(figsize=(6,10))

        ax = spikes_on_track.add_subplot(2, 1, 1)  # specify (nrows, ncols, axnum)
        cluster_firing_indices = spike_data.firing_times[cluster_index]
        ax.plot(spatial_data.x_position_cm[cluster_firing_indices], spatial_data.trial_number[cluster_firing_indices], '|', color='Black', markersize=5)
        ax.plot(spike_data.loc[cluster_index].nonbeaconed_position_cm, spike_data.loc[cluster_index].nonbeaconed_trial_number, '|', color='Red', markersize=5)
        ax.plot(spike_data.loc[cluster_index].probe_position_cm, spike_data.loc[cluster_index].probe_trial_number, '|', color='Blue', markersize=5)
        plt.ylabel('Spikes on trials', fontsize=12, labelpad = 10)
        plt.xlim(0,200)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plot_utility.style_track_plot(ax, 200)
        x_max = max(spatial_data.trial_number[cluster_firing_indices])+0.5
        plot_utility.style_vr_plot(ax, x_max)

        ax = spikes_on_track.add_subplot(2, 1, 2)  # specify (nrows, ncols, axnum)
        bins=range(200)
        unsmooth_b = np.array(spike_data.at[cluster_index, 'avg_spike_per_bin_b'])
        unsmooth_nb = np.array(spike_data.at[cluster_index, 'avg_spike_per_bin_nb'])
        unsmooth_p = np.array(spike_data.at[cluster_index, 'avg_spike_per_bin_p'])
        ax.plot(bins, unsmooth_b, '-', color='Black')
        try:
            ax.plot(bins, unsmooth_nb, '-', color='Red')
            ax.plot(bins, unsmooth_p, '-', color='Blue')
        except ValueError:
            continue
        ax.locator_params(axis = 'x', nbins=3)
        ax.set_xticklabels(['0', '100', '200'])
        plt.ylabel('Spike rate (hz)', fontsize=14, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=14, labelpad = 10)
        plt.xlim(0,200)
        x_max = max(spike_data.avg_spike_per_bin_nb[cluster_index])+5
        plot_utility.style_vr_plot(ax, x_max)
        plot_utility.style_track_plot(ax, 200)
        plt.subplots_adjust(hspace = .5, wspace = .35,  bottom = 0.06, left = 0.12, right = 0.87, top = 0.92)

        plt.savefig(prm.get_local_recording_folder_path() + '/Figures/combined_spike_plots/' + spike_data.session_id[cluster_index] + 'track_firing_Cluster_' + str(cluster_index +1) + '.png', dpi=200)
        plt.close()




def make_combined_figure(prm, spatial_firing):
    print('I will make the combined images now.')
    save_path = prm.get_output_path() + '/Figures/combined'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    plt.close('all')
    figures_path = prm.get_output_path() + '/Figures/'
    for cluster in range(len(spatial_firing)):
        cluster = spatial_firing.cluster_id.values[cluster] - 1

        stop_raster_path = figures_path + 'behaviour/stop_raster.png'
        stop_avg_path = figures_path + 'behaviour/stop_histogram.png'
        speed_avg_path = figures_path + 'behaviour/speed_histogram.png'

        spike_raster_path = figures_path + 'firing_scatters/' + spatial_firing.session_id[spatial_firing] + 'track_firing_Cluster_' + str(cluster +1) + '.png'
        rate_map_path = figures_path + 'rate_maps/' + spatial_firing.session_id[spatial_firing] + 'rate_map_Cluster_' + str(cluster +1) + '.png'
        spike_histogram_path = figures_path + 'firing_properties/' + spatial_firing.session_id[cluster] + '_' + str(cluster + 1) + '_spike_histogram.png'
        autocorrelogram_10_path = figures_path + 'firing_properties/' + spatial_firing.session_id[cluster] + '_' + str(cluster + 1) + '_autocorrelogram_10ms.png'
        autocorrelogram_250_path = figures_path + 'firing_properties/' + spatial_firing.session_id[cluster] + '_' + str(cluster + 1) + '_autocorrelogram_250ms.png'
        waveforms_path = figures_path + 'firing_properties/' + spatial_firing.session_id[cluster] + '_' + str(cluster + 1) + '_waveforms.png'

        number_of_firing_fields = 0
        if 'firing_fields' in spatial_firing:
            number_of_firing_fields = len(spatial_firing.firing_fields[cluster])
        number_of_rows = math.ceil((number_of_firing_fields + 1)/6) + 2

        grid = plt.GridSpec(number_of_rows, 6, wspace=0.2, hspace=0.2)
        if os.path.exists(waveforms_path):
            waveforms = mpimg.imread(waveforms_path)
            waveforms_plot = plt.subplot(grid[0, 0])
            waveforms_plot.axis('off')
            waveforms_plot.imshow(waveforms)
        if os.path.exists(spike_histogram_path):
            spike_hist = mpimg.imread(spike_histogram_path)
            spike_hist_plot = plt.subplot(grid[0, 3])
            spike_hist_plot.axis('off')
            spike_hist_plot.imshow(spike_hist)
        if os.path.exists(autocorrelogram_10_path):
            autocorrelogram_10 = mpimg.imread(autocorrelogram_10_path)
            autocorrelogram_10_plot = plt.subplot(grid[0, 1])
            autocorrelogram_10_plot.axis('off')
            autocorrelogram_10_plot.imshow(autocorrelogram_10)
        if os.path.exists(autocorrelogram_250_path):
            autocorrelogram_250 = mpimg.imread(autocorrelogram_250_path)
            autocorrelogram_250_plot = plt.subplot(grid[0, 2])
            autocorrelogram_250_plot.axis('off')
            autocorrelogram_250_plot.imshow(autocorrelogram_250)
        if os.path.exists(stop_raster_path):
            stop_raster = mpimg.imread(stop_raster_path)
            stop_raster_plot = plt.subplot(grid[1, 0])
            stop_raster_plot.axis('off')
            stop_raster_plot.imshow(stop_raster)
        if os.path.exists(stop_avg_path):
            stop_avg = mpimg.imread(stop_avg_path)
            stop_avg_plot = plt.subplot(grid[1, 1])
            stop_avg_plot.axis('off')
            stop_avg_plot.imshow(stop_avg)
        if os.path.exists(speed_avg_path):
            speed_avg = mpimg.imread(speed_avg_path)
            speed_avg_plot = plt.subplot(grid[1, 2])
            speed_avg_plot.axis('off')
            speed_avg_plot.imshow(speed_avg)
        if os.path.exists(spike_raster_path):
            spike_raster = mpimg.imread(spike_raster_path)
            spike_raster_plot = plt.subplot(grid[1, 3])
            spike_raster_plot.axis('off')
            spike_raster_plot.imshow(spike_raster)
        if os.path.exists(rate_map_path):
            rate_map = mpimg.imread(rate_map_path)
            rate_map_plot = plt.subplot(grid[1, 4])
            rate_map_plot.axis('off')
            rate_map_plot.imshow(rate_map)

        plt.savefig(save_path + '/' + spatial_firing.session_id[cluster] + '_' + str(cluster + 1) + '.png', dpi=1000)
        plt.close()



def main():
    prm = PostSorting.parameters.Parameters()
    prm.set_sampling_rate(30000)
    recording_folder = '/Users/sarahtennant/Work/Analysis/Opto_data/PVCre1/M1_D27_2018-10-05_11-17-55' # test recording
    prm.set_local_recording_folder_path('C:/Users/sarahtennant/Work/Analysis/Opto_data/PVCre1/M1_D27_2018-10-05_11-17-55/DataFrames')
    prm.set_output_path('C:/Users/sarahtennant/Work/Analysis/Opto_data/PVCre1/M1_D27_2018-10-05_11-17-55/')
    spatial_firing = pd.read_pickle(prm.get_local_recording_folder_path() + '/spatial_firing.pkl')
    spatial_data = pd.read_pickle(prm.get_local_recording_folder_path() + '/position.pkl')
    make_combined_figure(prm, spatial_firing)

if __name__ == '__main__':
    main()
