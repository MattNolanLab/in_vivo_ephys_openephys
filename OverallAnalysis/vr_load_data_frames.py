import os
import glob
import pandas as pd
import PostSorting.vr_ramp_cell_test
import OverallAnalysis.vr_make_individual_plots

prm = PostSorting.parameters.Parameters()

def make_plots(recording_folder,spike_data, processed_position_data):
    OverallAnalysis.vr_make_individual_plots.plot_spikes_on_track(recording_folder,spike_data, processed_position_data, prm, prefix='_movement')
    #PostSorting.vr_make_plots.plot_stop_histogram(raw_position_data, processed_position_data, prm)
    #PostSorting.vr_make_plots.plot_speed_histogram(processed_position_data, prm)
    #PostSorting.vr_make_plots.plot_combined_behaviour(raw_position_data, processed_position_data, prm)
    #PostSorting.make_plots.plot_waveforms(spike_data, prm)
    #PostSorting.make_plots.plot_spike_histogram(spike_data, prm)
    #PostSorting.make_plots.plot_autocorrelograms(spike_data, prm)
    #PostSorting.vr_make_plots.plot_spikes_on_track(spike_data,raw_position_data, processed_position_data, prm, prefix='_all')
    OverallAnalysis.vr_make_individual_plots.plot_firing_rate_maps(recording_folder, spike_data, prefix='_movement')
    OverallAnalysis.vr_make_individual_plots.plot_combined_spike_raster_and_rate(recording_folder, spike_data, processed_position_data, prefix='_movement')
    #PostSorting.vr_make_plots.make_combined_figure(prm, spike_data, prefix='_all')
    #PostSorting.vr_make_plots.plot_spike_rate_vs_speed(spike_data, processed_position_data, prm)
    return


def process_a_dir(recording_folder):
    recording_folder = '/Users/sarahtennant/Work/Analysis/Opto_data/PVCre1/M1_D31_2018-11-01_12-28-25' # test recording
    local_output_path = '/Users/sarahtennant/Work/Analysis/Opto_data/PVCre1/M1_D31_2018-11-01_12-28-25/Dataframes/spatial_firing.pkl'
    spike_data_frame_path = recording_folder + '/DataFrames/spatial_firing.pkl'
    spatial_data_frame_path = recording_folder + '/DataFrames/position.pkl'

    if os.path.exists(recording_folder):
        print('I found the test file.')

    if os.path.exists(local_output_path):
        print('I found the output folder.')

    os.path.isdir(recording_folder)
    if os.path.exists(spike_data_frame_path):
        print('I found a firing data frame.')
        spike_data = pd.read_pickle(spike_data_frame_path)
        '''
        'session_id' 'cluster_id' 'tetrode' 'primary_channel' 'firing_times'
         'firing_times_opto' 'number_of_spikes' 'mean_firing_rate' 'isolation'
         'noise_overlap' 'peak_snr' 'peak_amp' 'random_snippets' 'x_position_cm'
         'trial_number' 'trial_type' 'normalised_b_spike_number' 'normalised_nb_spike_number' 'normalised_p_spike_number'
    
        '''

    if os.path.exists(spike_data_frame_path):
        print('I found a spatial data frame.')
        processed_position_data = pd.read_pickle(spatial_data_frame_path)
        '''
        'binned_time_ms' 'binned_time_moving_ms' 'binned_time_stationary_ms' 'binned_speed_ms' 'beaconed_total_trial_number'
         'nonbeaconed_total_trial_number' 'probe_total_trial_number' 'stop_location_cm' 'stop_trial_number'
         'stop_trial_type' 'rewarded_stop_locations' 'rewarded_trials' 'average_stops' 'position_bins'
    
        '''
    return spike_data, processed_position_data



#  this is here for testing
def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    recording_folder = '/Users/sarahtennant/Work/Analysis/Opto_data/PVCre1/M1_D31_2018-11-01_12-28-25' # test recording
    print('Processing ' + str(recording_folder))

    spike_data, processed_position_data = process_a_dir(recording_folder)
    #spike_data = PostSorting.vr_ramp_cell_test.analyse_ramp_firing(prm,spike_data)
    make_plots(recording_folder,spike_data, processed_position_data)



if __name__ == '__main__':
    main()
