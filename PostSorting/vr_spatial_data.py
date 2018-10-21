import numpy as npimport osimport pandas as pdimport open_ephys_IOimport PostSorting.parametersimport mathfrom scipy import statsimport PostSorting.vr_stop_analysisimport PostSorting.vr_make_plots# for testing: load behavioural datadef load_spatial_data(position_data,recording_folder):    print('Loading spatial data...')    location_in_cm = np.load(recording_folder + "/Data_test/location.npy")    position_data['x_position_cm'] = location_in_cm # fill in dataframe    trials = np.load(recording_folder + "/Data_test/trials.npy")    position_data['trial_number'] = trials # fill in dataframe    return position_data""""Corrects for if blender was restarted during the recording# input: raw location as numpy array# function: location is usually represented from 0.55-2.55 mV on the DAQ channel. If blender restarts the pin suddenly jumps to 0. This really screws up the conversion of raw location to cm so we need to remove that datapoint.  # output: corrected location as numpy array"""def correct_for_restart(location):    location[location <0.55] = 0.56 # deals with if the VR is switched off during recording - location value drops to 0 - min is usually 0.56 approx    return location""""Loads raw location continuous channel from ACD1.continuous# input: spatial dataframe, path to local recording folder, initialised parameters# output: raw location as numpy array"""def get_raw_location(position_data,recording_folder, prm):    print('Extracting raw location...')    file_path = recording_folder + '/' + prm.get_movement_channel()    if os.path.exists(file_path):        location = open_ephys_IO.get_data_continuous(prm, file_path)    else:        print('Movement data was not found.')    location=correct_for_restart(location)    position_data['raw_position'] = location # fill in dataframe    PostSorting.vr_make_plots.plot_movement_channel(location, prm)    return np.asarray(location, dtype=np.float16)'''Normalizes recorded values from location channel to metric (cm)input    prm : object, parameters    raw_data : array, electrophysiology and movement data file, contains recorded location valuesoutput    normalized_location_metric : array, contains normalized location values (in cm)The standardization is computed by finding the minimum and maximum recorded location values (min, max), and thensubtracting min from max, to get the recorded track length (recorded_length). Then, this recorded_length is devided bythe real track length to get the distance unit that will be used for the conversion.From every recorded location point, the recorded_startpoint (min) value is subtracted to make the first location = 0.The recorded_startpoint may not be 0 due to the sampling rate. (Please note that this is not the beginning of theephys recording in time, but the smallest value recorded by the rotary encoder.) This value is then divided by thedistance unit calculated in the previous step to convert the rotary encoder values to metric.'''def calculate_track_location(position_data, recording_folder, prm):    recorded_location = get_raw_location(position_data,recording_folder, prm) # get raw location from DAQ pin    PostSorting.vr_make_plots.plot_movement_channel(recorded_location, prm)    print('Converting raw location input to cm...')    recorded_startpoint = min(recorded_location)    recorded_endpoint = max(recorded_location)    recorded_track_length = recorded_endpoint - recorded_startpoint    distance_unit = recorded_track_length/prm.get_track_length()  # Obtain distance unit (cm) by dividing recorded track length to actual track length    location_in_cm = (recorded_location - recorded_startpoint) / distance_unit    position_data['x_position_cm'] = location_in_cm # fill in dataframe    np.save(recording_folder + "/Data_test/location", location_in_cm)#for testing    return position_data# calculate time from start of recording in seconds for each sampling pointdef calculate_time(position_data):    print('Calculating time...')    position_data['time_ms'] = position_data['x_position_cm'].index/30000 # convert sampling rate to time (seconds) by dividing by 30    return position_data# for each sampling point, calculates time from last sample pointdef calculate_instant_dwell_time(position_data):    print('Calculating dwell time...')    position_data['dwell_time_ms'] = position_data['time_ms'].diff() # [row] - [row-1]    return position_data# finds time animal spent in each location bin for each trialdef calculate_binned_dwell_time(position_data):    print('Calculating binned dwell time...')    dwell_rate_map = pd.DataFrame(columns=['trial_number','bin_count', 'dwell_time_ms'])    bin_size_cm,number_of_bins = PostSorting.vr_stop_analysis.get_bin_size(position_data)    number_of_trials = position_data.trial_number.max() # total number of trials    trials = np.array(position_data['trial_number'].tolist())    locations = np.array(position_data['x_position_cm'].tolist())    dwell_time_per_sample = np.array(position_data['dwell_time_ms'].tolist())  # Get the raw location from the movement channel    for t in range(1,int(number_of_trials)):        trial_locations = np.take(locations, np.where(trials == t)[0])        for loc in range(int(number_of_bins)):            time_in_bin = sum(dwell_time_per_sample[np.where(np.logical_and(trial_locations > loc, trial_locations <= (loc+1)))])            if time_in_bin == 0: # this only happens if the session is started/stopped in the middle of a trial                dwell_rate_map = dwell_rate_map.append({"trial_number": int(t), "bin_count": int(loc),  "dwell_time_ms":  float(0.001)}, ignore_index=True)            else:                dwell_rate_map = dwell_rate_map.append({"trial_number": int(t), "bin_count": int(loc),  "dwell_time_ms":  (time_in_bin)}, ignore_index=True)    position_data['binned_time_ms'] = dwell_rate_map['dwell_time_ms']    return position_data# calculates trial number from locationdef calculate_trial_numbers(position_data, prm):    print('Calculating trial numbers...')    location= np.array(position_data['x_position_cm'])  # Get the raw location from the movement channel    trials = np.zeros((len(location)))    new_trial_index = 0    trial_num = 1    for i in range(len(location)):        if i > 0 and (location[i-1]-location[i]) > 100 and (i-new_trial_index) > 1500:            trial_num += 1            new_trial_index = i        trials[i] = trial_num    position_data['trial_number'] = np.asarray(trials, dtype=np.uint8)    print('This mouse did ', int(max(trials)), ' trials')    np.save(prm.get_filepath() + "/Data_test/trials", np.asarray(trials, dtype=np.uint8))#for testing    PostSorting.vr_make_plots.plot_trials(trials, prm)    return position_data# two continuous channels represent trial typedef load_first_trial_channel(recording_folder, prm):    first = []    file_path = recording_folder + '/' + prm.get_first_trial_channel() #todo this should bw in params, it is 100 for me, 105 for Tizzy (I don't have _0)    trial_first = open_ephys_IO.get_data_continuous(prm, file_path)    first.append(trial_first)    return np.asarray(first, dtype=np.uint8)# two continuous channels represent trial typedef load_second_trial_channel(recording_folder, prm):    second = []    file_path = recording_folder + '/' + prm.get_second_trial_channel() #todo this should bw in params, it is 100 for me, 105 for Tizzy (I don't have _0)    trial_second = open_ephys_IO.get_data_continuous(prm, file_path)    second.append(trial_second)    return np.asarray(second, dtype=np.uint8)def calculate_trial_types(position_data, recording_folder, prm):    print('Loading trial types from continuous...')    first_ch = load_first_trial_channel(recording_folder, prm)    second_ch = load_second_trial_channel(recording_folder, prm)    PostSorting.vr_make_plots.plot_trial_channels(first_ch, second_ch, prm)    location_diff = position_data['x_position_cm'].diff()  # Get the raw location from the movement channel    trial_indices = np.where(location_diff < -150) [0]# return indices where is new trial    trial_type = np.zeros((second_ch.shape[1]));trial_type[:]=np.nan    new_trial_indices=np.hstack((0,trial_indices,len(trial_type)))    print('Calculating trial type...')    for icount,i in enumerate(range(len(new_trial_indices)-1)):        new_trial_index = new_trial_indices[icount]        next_trial_index = new_trial_indices[icount+1]        second = stats.mode(second_ch[0,new_trial_index:next_trial_index])[0]        first = stats.mode(first_ch[0,new_trial_index:next_trial_index])[0]        if second < 2 and first < 2: # if beaconed            trial_type[new_trial_index:next_trial_index] = int(0)        if second > 2 and first < 2: # if non beaconed            trial_type[new_trial_index:next_trial_index] = int(2)        if second > 2 and first > 2: # if probe            trial_type[new_trial_index:next_trial_index] = int(1)    position_data['trial_type'] = np.asarray(trial_type, dtype=np.uint8)    np.save(prm.get_filepath() + "/Data_test/trial_type", np.asarray(trial_type, dtype=np.uint8))#for testing    return position_datadef calculate_total_trial_numbers(position_data):    trial_numbers = np.array(position_data['trial_number'])    trial_type = np.array(position_data['trial_type'])    trial_data=np.transpose(np.vstack((trial_numbers, trial_type)))    beaconed_trials = np.delete(trial_data, np.where(trial_data[:,1]>0),0)    unique_beaconed_trials = np.unique(beaconed_trials[:,0])    nonbeaconed_trials = np.delete(trial_data, np.where(trial_data[:,1]!=1),0)    unique_nonbeaconed_trials = np.unique(nonbeaconed_trials[1:,0])    probe_trials = np.delete(trial_data, np.where(trial_data[:,1]!=2),0)    unique_probe_trials = np.unique(probe_trials[1:,0])    position_data['beaconed_total_trial_number'] = len(unique_beaconed_trials)    position_data['nonbeaconed_total_trial_number'] = len(unique_nonbeaconed_trials)    position_data['probe_total_trial_number'] = len(unique_probe_trials)    #position_data['beaconed_trial_numbers'] = unique_beaconed_trials    #position_data['nonbeaconed_trial_numbers'] = unique_nonbeaconed_trials    #position_data['probe_trial_numbers'] = unique_probe_trials    return position_data'''Corrects for the very small negative values that are calculated as velocity when the mouse 'teleports' backto the beginning of the track - from the end of the track to 0.input    prm : obejct, parameters    velocity : numpy array, instant velocityoutput    velocity : array, instant velocity without teleport artefactsIt finds the velocity values that are smaller than -track_length+max_velocity, and adds track_length to them. Thesevalues will be around the beginning of the track after the mouse finished the previous trial and jumped back to thebeginning.After the first iteration, it finds the values that are <-10 (it is highly unlikely for a mouse to have that speed), itreplaces them with the previous location value.An alternative solution may be to nto analyze this data.'''def fix_teleport(velocity):    max_velocity = max(velocity)    track_length = 200    # If the mouse goes from the end of the track to the beginning, the velocity would be a negative value    # if velocity< (-1)*track_length + max_velocity, then track_length is added to the value    too_small_indices = np.where(velocity < (-track_length + max_velocity))    too_small_values = np.take(velocity, too_small_indices)    to_insert = too_small_values + track_length    np.put(velocity, too_small_indices, to_insert)  # replace small values with new correct value    # if velocity is <-10 (due to the teleportation), the previous velocity value will be used    small_velocity = np.where(velocity < -10)  # find where speed is < 10    small_velocity = np.asanyarray(small_velocity)    previous_velocity_index = small_velocity - 1  # find indices right before those in previous line    previous_velocity = np.take(velocity, previous_velocity_index)    np.put(velocity, small_velocity, previous_velocity)  # replace small speed values with previous value    return velocity'''Calculates instant velocity for every sampling pointinput    prm : object, parameters    location : numpy array, location values (metric)    sampling_points_per200ms : number of sampling points in 200ms signaloutput    velocity : instant velocitycalls    fix_teleport : function to fix issues arising from the fact that when the mouse restarts the trial, it is    teleported back to the beginning of the track, and the velocity will be a very small negative value.The location array is duplicated, and shifted in a way that it can be subtracted from the original to avoid loops.(The shifted array is like : first 200ms data + first 200ms data again, rest of data without last 200ms, this issubtracted from the original location array.)'''def calculate_instant_velocity(position_data, prm):    print('Calculating velocity...')    location = np.array(position_data['x_position_cm']) # Get the raw location from the movement channel    sampling_points_per200ms = int(prm.get_sampling_rate()/5)    end_of_loc_to_subtr = location[:-sampling_points_per200ms]# Rearrange arrays in a way that they just need to be subtracted from each other    beginning_of_loc_to_subtr = location[:sampling_points_per200ms]# Rearrange arrays in a way that they just need to be subtracted from each other    location_to_subtract_from = np.append(beginning_of_loc_to_subtr, end_of_loc_to_subtr)    velocity = location - location_to_subtract_from    velocity = fix_teleport(velocity)    position_data['velocity'] = velocity    np.save(prm.get_filepath() + "/Data_test/", velocity) #for testing    return position_data'''Calculates moving averageinput    a : array,  to calculate averages on    n : integer, number of points that is used for one average calculationoutput    array, contains rolling average values (each value is the average of the previous n values)'''def moving_sum(array, window):    ret = np.cumsum(array, dtype=float)    ret[window:] = ret[window:] - ret[:-window]    return ret[window:]def get_rolling_sum(array_in, window):    if window > (len(array_in) / 3) - 1:        print('Window is too big, plot cannot be made.')    inner_part_result = moving_sum(array_in, window)    edges = np.append(array_in[-2 * window:], array_in[: 2 * window])    edges_result = moving_sum(edges, window)    end = edges_result[window:math.floor(len(edges_result)/2)]    beginning = edges_result[math.floor(len(edges_result)/2):-window]    array_out = np.hstack((beginning, inner_part_result, end))    return array_out'''Calculate average speed for the last 200ms at each particular sampling point, based on velocityinput    prm : object, parameters    velocity : numpy array, instant velocity values    sampling_points_per200ms : number of sampling points in 200msoutput    avg_speed : numpy array, contains average speed for each location. The first 200ms are filled with 0s.'''def get_avg_speed_200ms(position_data, prm):    print('Calculating average speed...')    velocity = np.array(position_data['velocity'])  # Get the raw location from the movement channel    sampling_points_per200ms = int(prm.get_sampling_rate()/5)    position_data['speed_per200ms'] = get_rolling_sum(velocity, sampling_points_per200ms)# Calculate average speed at each point by averaging instant velocities    return position_datadef calculate_binned_speed(position_data):    bin_size_cm,number_of_bins = PostSorting.vr_stop_analysis.get_bin_size(position_data)    number_of_trials = position_data.trial_number.max() # total number of trials    speed_ms = np.array(position_data['speed_per200ms'].tolist())    locations = np.array(position_data['x_position_cm'].tolist())    speed = []    for loc in range(int(number_of_bins)):        speed_in_bin = np.mean(speed_ms[np.where(np.logical_and(locations > loc, locations <= (loc+1)))])/number_of_trials        speed = np.append(speed,speed_in_bin)    position_data.binned_speed_ms.iloc[range(int(number_of_bins))] = speed    return position_datadef process_position_data(recording_folder, prm):    position_data = pd.DataFrame(columns=['time_ms', 'dwell_time_ms', 'x_position_cm', 'velocity', 'speed_per200ms', 'binned_speed_ms', 'trial_number', 'trial_type', 'stop_location_cm', 'stop_trial_number', 'stop_trial_type', 'rewarded_locations', 'rewarded_trials', 'average_stops', 'sd_stops', 'position_bins', 'binned_time_ms'])    if os.path.isfile(recording_folder + "/Data_test/location.npy") is False: # for testing        position_data = calculate_track_location(position_data, recording_folder, prm)        position_data = calculate_trial_numbers(position_data, prm)    else:        position_data = load_spatial_data(position_data, recording_folder) # load spatial data    position_data = calculate_trial_types(position_data, recording_folder, prm)    position_data = calculate_total_trial_numbers(position_data)    position_data = calculate_time(position_data)    position_data = calculate_instant_dwell_time(position_data)    position_data = calculate_binned_dwell_time(position_data)    position_data = calculate_instant_velocity(position_data, prm)    position_data = get_avg_speed_200ms(position_data, prm)    position_data = calculate_binned_speed(position_data)    position_data = PostSorting.vr_stop_analysis.process_stops(position_data, prm)    prm.set_total_length_sampling_points(position_data.time_ms.values[-1])  # seconds    return position_data#  for testingdef main():    print('-------------------------------------------------------------')    params = PostSorting.parameters.Parameters()    recording_folder = 'C:/Users/s1466507/Documents/Ephys/test_overall_analysis/M5_2018-03-06_15-34-44_of'    vr_spatial_data = process_position_data(recording_folder)if __name__ == '__main__':    main()