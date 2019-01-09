install.packages('reticulate')

#specify python environment (needs to be >v.3)
Sys.setenv(RETICULATE_PYTHON = "/usr/local/Cellar/python3/3.6.1/Frameworks/Python.framework/Versions/3.6/bin/python3.6") 
#use_virtualenv("/usr/local/Cellar/python3/3.6.1/Frameworks/Python.framework/Versions/3.6/bin/python3.6") # this way also works

require("reticulate")

source_python("pickle_reader.py") # run python script which loads the dataframes
pickle_data <- read_pickle_file("/Users/sarahtennant/Work/Analysis/R_invivo_OpenEphys/spatial_firing.pkl") # path to dataframe to load
