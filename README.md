# in_vivo_ephys_openephys
![master](https://github.com/MattNolanLab/in_vivo_ephys_openephys/actions/workflows/test.yml/badge.svg)

## Overview
Analysis for in vivo electrophysiology recordings saved in open ephys format. 

The current pipeline runs on a linux computer and uses MountainSort 3 to automatically sort data. The analysis is set up to minimize user interaction as much as possible, and analyses are initiated by copying files to a designated computer. Sorted clusteres are automatically curated and output plots of spatial firing properties are generated. Spatial scores (grid score, HD score, speed score) are calculated and saved in pickle (pandas) data frames.

The main script (control_sorting_analysis.py) monitors a designated folder (nolanlab/to_sort/recordings) on the computer, and calls all processing scripts if users put recordings in this folder (and added a copied.txt file as well to indicate that copying is complete).
Another option is to add a text file with a list of folders on the server that the script will copy when the 'priority' sorting folder is empty.


(1) OpenEphys continuous files are converted to mda format (in Python) both tetrode by tetrode (4 files) and all 16 channels together into one mda file. The folder structure required by MountainSort (MS) is created in this step, dead channels are removed.

(2) MountainSort (MS) is called (via a shell script written by the previous Python step) to perform spike sorting in the mda files, and saves the results in the local folder

(3) Post-processing is done. This makes plots of firing fields, light stimulation plots depending on the data, and saves the output on the lab's server based on a parameter file that's saved by the user in the original recording folder.

![image](https://user-images.githubusercontent.com/16649631/43050846-7988ff56-8e07-11e8-8f01-3b96ffd6278c.png)


## User guide for running analysis pipeline
# **High priority sorting**
1. Acquire data in Open Ephys, and save in openephys format, and upload it to the server
2. Connect to the sorting computer using an SSH connection
3. Copy the whole recording folder including paramters.txt and dead_channels.txt and any movement information to nolanlab/to_sort/recordings

### parameters.txt
This should be added to every recording folder before the analysis is done. The first row should have the session type, which is either vr or openfield. The second line should have the location on the server starting from your name, so for example:
> openfield

> Klara/Open_field_opto_tagging_p038/M3_2018-03-13_10-44-37_of

### dead_channels.txt
This should only be made if the recording contains dead channels. Each channel id for a dead channel (1-16) should be in a new line. So for example if 1 and 3 are dead, dead_channels.txt should have
> 1

> 3

4. When the folder is fully copied, **put copied.txt in the folder** (so the script knows it's ready for sorting)

Do not ever put more than 10 folders in this folder. The sorting computer has 250GB of space, which is used for temporary sorting files in addition to your files stored here. Please always check how many folders others put in there.

# **Low priority sorting**
1. Acquire data in Open Ephys, and save in openephys format, and upload it to the server
2. Create a text file with any name and in each line put the end of the server path to a folder (same format as parameters file).
3. Copy this text file to the sorting computer using an SSH connection to nolanlab/to_sort/downtime_sort
These folders will be copied to the sorting computer one by one whenever nolanlab/to_sort/recordings is empty

Your results will be uploaded to the server based on the path you gave in the parameters file.
The sorting will be logged in sorting_log.txt that will be put in your recording folder on the server if possible. If your recording crashes, the folder name will be added to crashlist.txt that is located in nolanlab/to_sort/crashlist.txt

## Post-sorting scripts and data frames 
### Open field
To facilitate efficient and reproducible analysis of electrophysiological and behavioural data we have developed a framework that uses data frames implemented in Python. Our overall philosophy is to clearly separate pre-processing of data from analyses that directly address experimental questions. Bonsai, OpenEphys, and MountainSort are used to pre-process the experimental data. For example, Mountainsort is first used for spike clustering. The outputs of this pre-processing are used to initialize data frames that are then used for subsequent analyses.

The framework use two data frames, one for analyses at the level of behavioural sessions, and one for analyses at the level of spike clusters. Data from new behavioural sessions, or clusters, are added to the frames as new rows. Outputs of new analyses are added as new columns. Because the data frames store data and analyses, for multiple sessions and clusters respectively, it's straightforward to implement new analyses over many sessions or clusters without writing complicated looping code. The framework is currently implemented only for open field sessions. 

New analysis code should be added in a way that uses the data frames. If analyses require access to raw data, then a processing step should be used to add the required data to the data frames. Results of subsequent analyses should be added into these data frames as new columns. For instance, if we implement calculating the grid score of cells, this should be a new column in the data frame that contains information on clusters.

At the end of the analysis, three data frames are saved: the position data frame, the cluster data frame, and a thrird data frame that contains the rejected 'noisy' clusters. These are saved as pkl files in each recording folder in /DataFrames on the server.

## Description of two main data frames
The 'session' data frame contains processed data describing the position and head-direction of the animal. Each row is data from one session. The columns are organized as follows:

**synced_spatial_data** 
_(this is the name of the df in the main code)_

* synced_time : arrays of time in seconds, synchronized with the ephys data
* position_x : arrays of x coordinates of position of animal in arena in cm
* position_y : y coordinate of position of animal in arena in cm
* position_x_pixels : x coordinate of position of animal in arena in pixels
* position_y_pixels : y coordinate of position of animal in arena in pixels
* hd : head-direction of animal in degrees [-180;180]
* speed : speed of animal

`synced_spatial_data.head()`
![image](https://user-images.githubusercontent.com/16649631/43079289-9a13ab22-8e84-11e8-9b57-80518fdfda63.png)

***

The 'clusters' data frame contains data for each cluster and their spatial firing. Each row is a cluster. The columns are organized as follows:

**spatial_firing**
_(this is the name of the df in the main code)_

* session_id : name of main recording folder (example: M5_2018-03-06_15-34-44_of)
* cluster_id : id of cluster within session (1 - number of clusters)
* tetrode : id of tetrode within session (1 - 4)
* primary_channel : channel where the event was detected on (1 - 4)
* firing_times : array of all firing event times that belong to cluster from the open field exploration (in sampling points)
* number_of_spikes : total number of spikes in session excluding opto tagging part
* mean_firing_rate : total number of spikes / total time exclding opto tagging data [Hz]
* firing_times_opto : array of firing events from the opto tagging part of the recording (in sampling points)
* position_x : x coordinate of position of animal in arena in cm corresponding to each firing event from the exploration
* position_y : y coordinate of position of animal in arena in cm corresponding to each firing event from the exploration
* position_x_pixels : x coordinate of position of animal in arena in pixels corresponding to each firing event from the exploration
* position_y_pixels : y coordinate of position of animal in arena in pixels corresponding to each firing event from the exploration
* hd : head-direction of animal in degrees corresponding to each firing event from the exploration [-180;180]
* firing_maps : binned data array for each cluster with firing rate maps
* hd_spike_histogram : polar histogram of HD when the cell fired. For each degree the number of events are counted and then smoothing is done on this data by adding the values up in a (23 degree) rolling window. For each degree between 0 and 360 the number of events between n-window and n+window is added up. This histogram is then divided by the histogram obtained from all the HD data from the session divided by the sampling rate.
`spike_histogram = spike_histogram_hd/(hd_histogram_session/ephys_sampling_rate)`

This is then normalized on the plot hd_hist*(max(hd_hist_cluster))/max(hd_hist)) is plotted. 

_We should discuss whether this is a good way, it does not make a lot of sense to me. It is not exactly as in the MATLAB version._
* firing_fields : lists of indices that belong to an individual firing field detected. One cluster can have multiple lists. (Indices correspond to the firing rate map.)
For example on this image, the yellow circles represent the local maximum that the algorithm found and then all the blue parts around them were taken for that particular firing field. This cluster will have four lists added to its firing fields.

* firing_fields_hd_session : head-direction histograms that correspond to firing fields (each cluster has a list) - so this data is only from when the animal was in the given field
* firing_fields_hd_cluster : head-direction histograms that correspond to firing fields when the cell fired - this data is from when the animal was in the field AND the cell fired

* field_hd_p : Kuiper p values corresponding to the head-direction histograms of each field
* field_stat : Kuiper raw statistic corresponding to the head-direction histograms of each field
* field_hd_max_rate : maximum firing rate in field
* field_preferred_hd : preferred head-direction in field
* field_hd_score : hd score in field (see hd score definition above)
* field_max_firing_rate : max firing rate in given field among rate bins

![image](https://user-images.githubusercontent.com/16649631/43839928-480eb3c2-9b17-11e8-96a4-f2da8b4de1c6.png)

* max_firing_rate : the highest among the firing rates of the bins of the rate map (Hz)

* max_firing_rate_hd : the highest among the firing rates of angles of the polar histogram (Hz)

* preferred_HD : the head-direction angle where the cell fires the most (highest rate), degrees

* hd_score : score between 0 and 1. The higher the score, the more head-direction specific the cell is.

`        dy = np.sin(angles_rad)`
        `dx = np.cos(angles_rad)`

        `totx = sum(dx * hd_hist)/sum(hd_hist)`
        `toty = sum(dy * hd_hist)/sum(hd_hist)`
        `r = np.sqrt(totx*totx + toty*toty)`
        `hd_scores.append(r)`

* hd_p : result of two-sample Kuiper test on the distribution of hd from the whole session and the distribution of hd when the cell fired. The probability of obtaining two samples this different from the same distribution.

* hd_stat : the raw test statistic from the Kuiper test described above

### Circular statistics are done in R using the circular.r package 

(documentation: https://cran.r-project.org/web/packages/circular/circular.pdf)

* watson_test_hd - stats restuls of two sample Watson test comparing distribution of HD from the whole session to the HD when the cell fired. p value ranged can be inferred from stats
https://github.com/MattNolanLab/in_vivo_ephys_openephys/blob/add_python_post_clustering/PostSorting/process_fields.r

* kuiper_cluster - one sample Kuiper test stats for HD when the cell fired

* kuiper_session - one sample Kuiper test stats for HD from the whole session

* watson_cluster - one sample Watson test stats for HD when the cell fired

* watson_session - one sample Watson test stats for HD for the whole session

`spike_data_spatial.head()`

![image](https://user-images.githubusercontent.com/16649631/43079705-b854d6c8-8e85-11e8-949a-303653e65fbf.png)

***
## post_process_recording
The following section will detail what happens in each line of the part of the code that calls analyses that initialize and fill the data frames described above.

In the PostSorting main, parameters (such as pixel ratio and sampling rate) are initialized using the parameters class:

`initialize_parameters(recording_to_process)`

The spatial data frame is initialized and filled by extracting data from the Bonsai output file:

`spatial_data = process_position_data(recording_to_process, session_type, prm)`

Light pulses are loaded into arrays. It is also determined at what time point the opto stimulation started.

`opto_on, opto_off, is_found = process_light_stimulation(recording_to_process, prm)`

The Bonsai data is trimmed to start at the same time as the electrophysiology data. Data from before the first simultaneously recorded sync pulses is discarded. (See more detail on how it is synchronized and why [here](https://dsp.stackexchange.com/questions/50063/how-can-i-synchronize-signal-from-an-led-and-ttl-pulses-better).)

`synced_spatial_data = sync_data(recording_to_process, prm, spatial_data)`

Firing times of clusters are loaded into the cluster data frame, where each cluster is one row.

`spike_data = PostSorting.load_firing_data.create_firing_data_frame(recording_to_process, session_type, prm)`

Spatial data is added to the spike data frame. For each firing time the corresponding location and head-direction is added.

`spike_data_spatial = PostSorting.open_field_spatial_firing.process_spatial_firing(spike_data, synced_spatial_data)`

Firing rate map arrays are calculated and added to the spike data frame.

`position_heat_map, spatial_firing = PostSorting.open_field_firing_maps.make_firing_field_maps(synced_spatial_data, spike_data_spatial, prm)`


### Example function to calculate speed

Calculate array 'elapsed_time' from the time column of the position data frame:

`elapsed_time = position_data['time_seconds'].diff()`

Calculate distance traveled based on x and y position coordinates using Pythagorean theorem:

`distance_travelled = np.sqrt(position_data['position_x'].diff().pow(2) + position_data['position_y'].diff().pow(2))`

Calculate speed and add it to 'position_data' data frame by dividing distance traveled by elapsed time for each data point:
`position_data['speed'] = distance_travelled / elapsed_time`

***

### VR

### parameters 

Specific parameters need to be set for the vr analysis environment. 


* **stop_threshold** this is the value in which the animals speed has to drop below for a stop to be extracted (<0.7 cm/second)
* **movement_channel** this is the pin on the DAQ which has the movement of the animal along the track
* **first_trial_channel** this is the first pin on the DAQ which has the trial type information 
* **second_trial_channel** this is the first pin on the DAQ which has the trial type information 


## Structure of dataframes

The spatial data frame contains processed data describing the position of the animal in the virtual reality. The columns are organized as follows:

**vr_spatial_data (name of the df in the main code)**

* time_ms : arrays of time in seconds, synchronized with the ephys data
* position_cm : arrays of x coordinates of position of animal in virtual track in cm, synchronized with the ephys data
* trial_number : arrays of the current trial number, synchronized with the ephys data
* trial_type : arrays of the current trial type (beaconed, non beaconed, probe), synchronized with the ephys data
* velocity : instant velocity of animal (cm/s), synchronized with the ephys data
* speed : speed of animal averaged over 200 ms (cm/s), synchronized with the ephys data
* stops : whether an animal has stopped (0/1 : no/yes), synchronized with the ephys data
* filtered_stops : stops within 1 cm of each other are removed 
* stop_times : array of times which the animal has stopped


**spike_data**
_(this is the name of the df in the main code)_

* session_id : name of main recording folder (example: M5_2018-03-06_15-34-44_of)
* cluster_id : id of cluster within session (1 - number of clusters)
* tetrode : id of tetrode within session (1 - 4)
* primary_channel : channel where the event was detected on (1 - 4)
* firing_times : array of all firing event times that belong to cluster from the vr (in sampling points)



---------------------------------------------------------------------------------------------------
## Folder structure on sorting computer / Eleanor instance
### High priority sorting folder:
> /home/nolanlab/to_sort/recordings

copy recording folders here

### Low priority sorting folder:
> /home/nolanlab/to_sort/sort_downtime

put lists of recordings to sort on server here

### Sorting files:
> /home/nolanlab/to_sort/sort_downtime/sorting_files

There is a backup of this folder in \\cmvm.datastore.ed.ac.uk\cmvm\sbms\groups\mnolan_NolanLab\ActiveProjects\Klara\sorting_files

these are parameters that MountainSort needs, they will be copied into your recordings, please don't touch this folder

### Crash list:
> /home/nolanlab/to_sort/crash_list.txt

list of recordings that failed


![image](https://user-images.githubusercontent.com/16649631/43050836-5d067eb2-8e07-11e8-8e45-80811090c03c.png)

---------------------------------------------------------------------------------------------------
## Bonsai output file

For open field recordings, Bonsai saves the position of two beads on the headstage of the animal and the intensity of an LED used to synchronize the position data in Bonsai with the electrophysiology data in OpenEphys.

the csv file saved by Bonsai contains the following information in each line:
- date of recording
- 'T'
- exact time of given line
- x position of left side bead on headstage
- y position of left side bead on headstage
- x position of right side bead on headstage
- y position of right side bead on headstage
- intensity of sync LED

example line:
2018-03-06T15:34:39.8242304+00:00 106.0175 134.4123 114.1396 148.1054 1713 
