import numpy as npimport osimport pandas as pdimport open_ephys_IOimport PostSorting.parametersimport itertoolsimport mathimport matplotlib.pylab as pltprm = PostSorting.parameters.Parameters()def get_raw_location(recording_folder):    print('I am extracting raw location...')    file_path = recording_folder + '/' + prm.get_movement_channel()    if os.path.exists(file_path):        location = open_ephys_IO.get_data_continuous(prm, file_path)    else:        print('Movement data was not found.')    location[location <0.55] = 0.56 # deals with if the VR is switched off during recording - location value drops to 0 - min is usually 0.56 approx    plt.plot(location)    plt.savefig(prm.get_local_recording_folder_path() + '/Figures/movement' + '.png')    plt.close()    return np.asarray(location)def calculate_track_location(position_data, recording_folder):    recorded_location = get_raw_location(recording_folder) # get raw location from DAQ pin    print('I am converting raw location input to cm...')    recorded_startpoint = min(recorded_location)    recorded_endpoint = max(recorded_location)    recorded_track_length = recorded_endpoint - recorded_startpoint    distance_unit = recorded_track_length/prm.get_track_length()  # Obtain distance unit (cm) by dividing recorded track length to actual track length    location_in_cm = (recorded_location - recorded_startpoint) / distance_unit    position_data['x_position_cm'] = location_in_cm # fill in dataframe    return position_datadef calculate_time(position_data):    print('I am calculating time...')    position_data['time_ms'] = position_data['x_position_cm'].index/30000 # convert sampling rate to time (seconds) by dividing by 30    return position_datadef calculate_instant_dwell_time(position_data):    print('I am calculating dwell time...')    position_data['dwell_time_ms'] = position_data['time_ms'].diff() # [row] - [row-1]    return position_datadef calculate_binned_dwell_time(position_data):    print('I am calculating binned dwell time...')    dwell_rate_map = pd.DataFrame(columns=['trial_number','bin_count', 'dwell_time_ms'])    bin_size_cm,number_of_bins = get_bin_size(position_data)    number_of_trials = position_data.trial_number.max() # total number of trials    trials = np.array(position_data['trial_number'].tolist())    locations = np.array(position_data['x_position_cm'].tolist())    dwell_time_per_sample = np.array(position_data['dwell_time_ms'].tolist())  # Get the raw location from the movement channel    for t in range(1,int(number_of_trials)):        trial_locations = np.take(locations, np.where(trials == t)[0])        for loc in range(int(number_of_bins)):            time_in_bin = sum(dwell_time_per_sample[np.where(np.logical_and(trial_locations > loc, trial_locations <= (loc+1)))])            dwell_rate_map = dwell_rate_map.append({"trial_number": int(t), "bin_count": int(loc),  "dwell_time_ms":  (time_in_bin)}, ignore_index=True)    position_data['binned_time_ms'] = dwell_rate_map['dwell_time_ms']    return position_datadef calculate_trial_numbers(position_data):    print('I am calculating trial numbers...')    location_diff = position_data['x_position_cm'].diff()  # Get the raw location from the movement channel    trials = np.zeros((len(location_diff)))    new_trial_indices = np.where(location_diff < -150) # return indices where is new trial    new_trial_indices = list(itertools.chain.from_iterable(new_trial_indices)) # needed to convert tuple to list    unique_trials = np.arange(1, len(new_trial_indices), 1)    for icount,i in enumerate(unique_trials):        trial_start_indices = new_trial_indices[icount]        next_trial_indices = new_trial_indices[icount+1]        trials[trial_start_indices:next_trial_indices] = i    position_data['trial_number'] = trials    print('This mouse did ', int(max(trials)), ' trials')    return position_datadef load_trial_types_from_continuous(recording_folder):    first = []    file_path = recording_folder + '/' + prm.get_first_trial_channel() #todo this should bw in params, it is 100 for me, 105 for Tizzy (I don't have _0)    trial_first = open_ephys_IO.get_data_continuous(prm, file_path)    first.append(trial_first)    first=np.asarray(first)    second = []    file_path = recording_folder + '/' + prm.get_second_trial_channel() #todo this should bw in params, it is 100 for me, 105 for Tizzy (I don't have _0)    trial_second = open_ephys_IO.get_data_continuous(prm, file_path)    second.append(trial_second)    second = np.asarray(second)    return first,seconddef calculate_trial_types(position_data, recording_folder):    print('I am calculating trial types from continuous...')    first, second = load_trial_types_from_continuous(recording_folder)    trial_type = np.zeros((first.shape[1]));trial_type[:]=np.nan    for point,p in enumerate(trial_type):        if second[0,point] < 2 and first[0,point] < 2: # if beaconed            trial_type[point] = 0        if second[0,point] > 2 and first[0,point] < 2: # if beaconed            trial_type[point] = 1        if second[0,point] > 2 and first[0,point] > 2: # if non beaconed            trial_type[point] = 2    position_data['trial_type'] = trial_type    return position_datadef calculate_instant_velocity(position_data):    print('I am calculating velocity...')    location = np.array(position_data['x_position_cm']) # Get the raw location from the movement channel    sampling_points_per200ms = int(prm.get_sampling_rate()/5)    end_of_loc_to_subtr = location[:-sampling_points_per200ms]# Rearrange arrays in a way that they just need to be subtracted from each other    beginning_of_loc_to_subtr = location[:sampling_points_per200ms]# Rearrange arrays in a way that they just need to be subtracted from each other    location_to_subtract_from = np.append(beginning_of_loc_to_subtr, end_of_loc_to_subtr)    velocity = location - location_to_subtract_from    position_data['velocity'] = velocity    return position_datadef moving_sum(array, window):    ret = np.cumsum(array, dtype=float)    ret[window:] = ret[window:] - ret[:-window]    return ret[window:]def get_rolling_sum(array_in, window):    if window > (len(array_in) / 3) - 1:        print('Window is too big, plot cannot be made.')    inner_part_result = moving_sum(array_in, window)    edges = np.append(array_in[-2 * window:], array_in[: 2 * window])    edges_result = moving_sum(edges, window)    end = edges_result[window:math.floor(len(edges_result)/2)]    beginning = edges_result[math.floor(len(edges_result)/2):-window]    array_out = np.hstack((beginning, inner_part_result, end))    return array_outdef get_avg_speed_200ms(position_data):    print('Calculating average speed...')    velocity = np.array(position_data['velocity'])  # Get the raw location from the movement channel    sampling_points_per200ms = int(prm.get_sampling_rate()/5)    position_data['speed_per200ms'] = get_rolling_sum(velocity, sampling_points_per200ms)# Calculate average speed at each point by averaging instant velocities    return position_datadef calculate_stops(position_data):    speed = np.array(position_data['speed_per200ms'].tolist())    print('I am finding stops')    threshold = prm.get_stop_threshold()    stop_indices = np.where(speed < threshold)[1]    position_data['stop_location_cm'] = position_data.x_position_cm[stop_indices].values    position_data['stop_trial_number'] = position_data.trial_number[stop_indices].values    position_data['stop_trial_type'] = position_data.trial_type[stop_indices].values    return position_datadef find_first_stop_in_series(position_data):    stop_difference = np.array(position_data['stop_location_cm'].diff().tolist())    first_in_series_indices = np.where(stop_difference > 1)[1]    print('I am finding first stops in series')    position_data['first_series_location_cm'] = position_data.stop_location_cm[first_in_series_indices].values    position_data['first_series_trial_number'] = position_data.stop_trial_number[first_in_series_indices].values    position_data['first_series_trial_type'] = position_data.stop_trial_type[first_in_series_indices].values    return position_datadef find_rewarded_positions(position_data):    stop_locations = np.array(position_data['first_series_location_cm'].tolist())    stop_trials = np.array(position_data['first_series_trial_number'].tolist())    rewarded_stop_locations = np.delete(stop_locations, np.where(np.logical_and(stop_locations > 110, stop_locations < 90))[1])    rewarded_trials = np.delete(stop_trials, np.where(np.logical_and(stop_locations > 110, stop_locations < 90))[1])    position_data['rewarded_stop_locations'] = rewarded_stop_locations    position_data['rewarded_trials'] = rewarded_trials    return position_datadef get_bin_size(spatial_data):    bin_size_cm = 1    track_length = spatial_data.x_position_cm.max()    start_of_track = spatial_data.x_position_cm.min()    number_of_bins = (track_length - start_of_track)/bin_size_cm    return bin_size_cm,number_of_binsdef calculate_average_stops(position_data):    stop_locations = np.array(position_data['first_series_location_cm'].tolist())    #stop_trials = np.array(position_data['first_series_trial_number'])    bin_size_cm,number_of_bins = get_bin_size(position_data)    number_of_trials = position_data.trial_number.max() # total number of trials    stops_in_bins = np.zeros((len(range(int(number_of_bins)))))    for loc in range(int(number_of_bins)):        stops_in_bin = len(stop_locations[np.where(np.logical_and(stop_locations > (loc), stop_locations <= (loc+1)))])/number_of_trials        stops_in_bins[loc] = stops_in_bin    #stops_in_bins = moving_average(stops_in_bins, 1)    position_data.average_stops.iloc[range(int(number_of_bins))] = stops_in_bins    position_data.position_bins.iloc[range(int(number_of_bins))] = range(int(number_of_bins))    return position_datadef process_position_data(recording_folder):    # make data frame    position_data = pd.DataFrame(columns=['time_ms', 'dwell_time_ms', 'x_position_cm', 'velocity', 'speed_per200ms', 'trial_number', 'trial_type', 'stop_location_cm', 'stop_trial_number', 'stop_trial_type', 'first_series_position_cm', 'first_series_trial_number', 'first_series_trial_type', 'rewarded_locations', 'rewarded_trials', 'average_stops', 'sd_stops', 'position_bins', 'binned_time_ms'])    position_data = calculate_track_location(position_data, recording_folder)    position_data = calculate_time(position_data)    position_data = calculate_instant_dwell_time(position_data)    position_data = calculate_trial_numbers(position_data)    position_data = calculate_trial_types(position_data, recording_folder)    position_data = calculate_instant_velocity(position_data)    position_data = get_avg_speed_200ms(position_data)    #position_data = calculate_stops(position_data)    #position_data = find_first_stop_in_series(position_data)    #position_data = find_rewarded_positions(position_data)    #position_data = calculate_average_stops(position_data)    position_data = calculate_binned_dwell_time(position_data)    #position_data = find_first_stop_in_trial(position_data)    return position_data#  for testingdef main():    print('-------------------------------------------------------------')    params = PostSorting.parameters.Parameters()    recording_folder = 'C:/Users/s1466507/Documents/Ephys/test_overall_analysis/M5_2018-03-06_15-34-44_of'    vr_spatial_data = process_position_data(recording_folder)if __name__ == '__main__':    main()