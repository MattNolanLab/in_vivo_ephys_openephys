# Fit the GLM model 
#%%
import pickle
import time
import warnings

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.signal as signal
import seaborn as sns
import settings
import utils.glmneuron
import utils.GLMPostProcessor
import xarray as xr
from utils.analysis.peak_detect import (findBestRampScore2, findBreakPt,
                                  getRampScore4, makeLinearGrid)
from palettable.colorbrewer.qualitative import Paired_8 as colors
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from tqdm import tqdm
from utils import *
from utils import SnakeIOHelper
from utils.analysis.utils import *

#%%
(sinput, soutput) = SnakeIOHelper.getSnake(
    locals(),
    'workflow/workflow_vr.smk',
    [settings.debug_folder + "/processed/ramp_score.pkl"],
    "calculate_ramp_score",
)

#%% Load previous results
with open(sinput.binned_data,'rb') as f:
    data = xr.Dataset.from_dict(pickle.load(f))

if data.spiketrain.data.item(0) is None:
    # No spikes found
    data = None
    SnakeIOHelper.makeDummpyOutput(soutput)

#%%
'''
- bin the firing rate according to location
- get the average firing rate for each bin
- smooth the curve a bit to fit the inflation points
- calculate the ramp score for all combination of these inflation points
- return the maximum inflation points pair and rampscore
'''

#%% Pre-processing data
if data:
    track_length = data.track_length
    reward_loc = data.reward_loc + 0.2 # start within the reward zone
    pos_grid = data.pos_grid.data
    speed = data.speed.data
    spiketrain = data.spiketrain.data
    pos_binned = np.argmax(pos_grid,axis=1)*(track_length/settings.position_bin)
    pos_bins = np.arange(track_length, step=track_length/settings.position_bin).astype(int)
    spiketrain = gaussian_filter1d(spiketrain,2,axis=0) #smooth
    spiketrain[spiketrain<0] = 0 #avoid negative firing rate
    trial_type = data.trial_type

# %% Calculate the average firing rate at each location
if data:
    spd_idx = (speed>3)
    pos_binned_filt = pos_binned[spd_idx]
    spiketrain_filt = spiketrain[spd_idx,:]
    trial_type_filt = trial_type[spd_idx]

#%% Calculate rampscore
if data:
    pos_range = [[30,reward_loc],[reward_loc+20,track_length-30],[0,track_length]]
    ramp_type = ['outbound','homebound','all']
    trial_types_num = [0,1,2,None]
    trial_type_name=['beaconed','non-beaconed','probe','all']
    ramp_scores = []

    for p_range,rt in zip(pos_range, ramp_type):

        bps = []
        fr_smooth = []
        for tt,tt_name in zip(trial_types_num,trial_type_name): # distinguish bewteen different trial type
            if tt is not None:
                trial_type_idx = (trial_type_filt==tt)
                spiketrain_tmp = spiketrain_filt[trial_type_idx,:]
                pos_binned_tmp = pos_binned_filt[trial_type_idx]
            else:
                spiketrain_tmp = spiketrain_filt
                pos_binned_tmp = pos_binned_filt

            if spiketrain_tmp.shape[0] > 30*(1000/settings.binSize):
                #only do analysis on enough data (>30s)
                for n in tqdm(range(len(data.neuron))):
            
                    pos_bin_range = pos_bins[(pos_bins >= p_range[0]) & (pos_bins < p_range[1])]
                    score,bp,meanCurve,normCurve = findBestRampScore2(spiketrain_tmp[:,n],
                        pos_binned_tmp,
                        pos_bin_range)

                    ramp_scores.append({
                        'score':score,
                        'breakpoint': bp,
                        'cluster_id': data.neuron.data[n],
                        'fr_smooth': meanCurve,
                        'ramp_region': rt,
                        'pos_bin': pos_bin_range,
                        'trial_type': tt_name
                    })
            else:
                for n in tqdm(range(len(data.neuron.data))):
                    ramp_scores.append({
                        'score': np.nan,
                        'breakpoint': np.nan,
                        'cluster_id': data.neuron.data[n],
                        'fr_smooth': np.nan,
                        'ramp_region': rt,
                        'pos_bin': np.nan,
                        'trial_type': tt_name
                    })


    df_ramp = pd.DataFrame(ramp_scores)
    df_ramp.head()
    df_ramp['session_id'] = data.session_id
# %% Plot ramp score and breakpoints

if data:
    for rt in ramp_type:
        df = df_ramp[df_ramp.ramp_region==rt]
        fig,ax = getWrappedSubplots(5,len(data.neuron), (2,2), dpi=100)

        for n in range(len(data.neuron.data)):
            cell = df.iloc[n]
            ax[n].axvspan(cell.breakpoint[0], cell.breakpoint[1],alpha=0.2,
                color=colors.mpl_colors[0])

            sns.lineplot(x=pos_binned_filt,y=spiketrain_filt[:,n]*(1000/settings.binSize),ax=ax[n],n_boot=30)
        
            ax[n].set_title(f'Cell {cell.cluster_id}: {cell.score:.2f}')

        fig.subplots_adjust(hspace=0.5,wspace=0.5)
        fig.suptitle(f'{rt}')
        fig.savefig(soutput[f'ramp_score_plot_{rt}'])
# %% save
if data:
    df_ramp.to_pickle(soutput.ramp_score)
