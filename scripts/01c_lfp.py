'''
This script filter and downsample the data files to LFP signals
'''
#%%
import logging
import os

import numpy as np
import pandas as pd
import settings
from PreClustering.pre_process_ephys_data import (filterRecording,
                                                  get_sorting_range)
from tqdm import tqdm
from utils import SnakeIOHelper

logger = logging.getLogger(os.path.basename(__file__)+':'+__name__)
import tempfile

from utils import file_utility
from scipy import signal


#%% define input and output
# note: need to run this in the root folder of project

(sinput, soutput) = SnakeIOHelper.getSnake(locals(), 'workflow/workflow_vr.smk', [settings.debug_folder+'/processed/lfp.npy'],
    'generate_lfp')

#%% Load
data = file_utility.load_OpenEphysRecording(sinput.recording_to_sort)

#%% downsample

data_ds = signal.decimate(data, int(settings.sampling_rate/settings.lfp_fs))

#%% filter
data_ds = filterRecording(data_ds, settings.sampling_rate, hp_freq=settings.lfp_hp, lp_freq=settings.lfp_lp) # filter in LFP range

# %% Save
np.save(soutput.lfp_file, data_ds)
# %%
