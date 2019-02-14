install.packages('reticulate')
install.packages('lme4')
library(lme4)
library(ggplot2)

setwd("~/Work/Analysis/in_vivo_ephys_openephys/OverallAnalysis")

#specify python environment (needs to be >v.3)
Sys.setenv(RETICULATE_PYTHON = "/usr/local/Cellar/python3/3.6.1/Frameworks/Python.framework/Versions/3.6/bin/python3.6") 
#use_virtualenv("/usr/local/Cellar/python3/3.6.1/Frameworks/Python.framework/Versions/3.6/bin/python3.6") # this way also works

require("reticulate")

source_python("pickle_reader.py") # run python script which loads the dataframes
pickle_data <- read_pickle_file("/Users/sarahtennant/Work/Analysis/R_invivo_OpenEphys/spatial_firing.pkl") # path to dataframe to load

#extract vector from column & row in dataframe
cluster_firing_times <-pickle_data[10,'firing_times'] # ramp cell in this dataset
cluster_firing_locations <-pickle_data[10,'x_position_cm'] # ramp cell in this dataset
cluster_firing_trials <-pickle_data[10,'trial_number'] # ramp cell in this dataset
cluster_firing_rate_beaconed <-pickle_data[10,'avg_spike_per_bin_b']

#convert list to vector
cluster_firing_rate_beaconed = unlist(cluster_firing_rate_beaconed, use.names = FALSE)
#create location bins for lmem
location_bins = seq(1, 200, by=1)

#linear model
lmmodel = lm(location_bins ~ cluster_firing_rate_beaconed, data = pickle_data)
summary(lmmodel)

#plot some basic stuff
#qplot(location_bins, cluster_firing_rate_beaconed)
qplot(location_bins, cluster_firing_rate_beaconed) + geom_smooth(method='lm')

