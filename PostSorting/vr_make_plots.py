import os
import matplotlib.pylab as plt
import plot_utility
import numpy as np
import PostSorting.vr_stop_analysis

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

        plt.savefig(prm.get_local_recording_folder_path() + '/Figures/spike_trajectories/track_firing_Cluster_' + str(cluster_index +1) + '.png', dpi=200)
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

        plt.savefig(prm.get_local_recording_folder_path() + '/Figures/spike_rate/rate_map_Cluster_' + str(cluster_index +1) + '.png', dpi=200)
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

        plt.savefig(prm.get_local_recording_folder_path() + '/Figures/combined_spike_plots/track_firing_Cluster_' + str(cluster_index +1) + '.png', dpi=200)
        plt.close()
