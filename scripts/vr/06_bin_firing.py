'''
Bin firing and position
'''

#%%
import settings
import numpy as np
import pickle
import scipy.signal as signal
import scipy.io as sio
from PostSorting import glmneuron
import pandas as pd
import tqdm
from utils import OpenEphys
from utils import SnakeIOHelper
import shutil
import xarray as xr
from pathlib import Path
import re

#%%
(sinput, soutput) = SnakeIOHelper.getSnake(
    locals(),
    'workflow/workflow_vr.smk',
    [settings.debug_folder + "/processed/binned_data.pkl"],
    "bin_data",
)

#%%  Add positions
raw_position = pd.read_pickle(sinput.raw_position)
spatial_firing = pd.read_pickle(sinput.spatial_firing_vr)

#%% Add position
binPeriod = f"{settings.binSize}ms"
position_xr = xr.DataArray(
    raw_position.x_position_cm,
    dims=("time"),
    coords={"time": pd.TimedeltaIndex(raw_position.time_seconds.values, unit="s")},
)
position_xr = position_xr.resample(time=binPeriod).mean()  # downsample

#%% Add spikes
if len(spatial_firing.firing_times) > 0:
    spiketrains_xr = glmneuron.getSpikePopulationXR(
        spatial_firing.firing_times, spatial_firing.cluster_id, settings.sampling_rate
    )
    spiketrains_xr = spiketrains_xr.resample(time=binPeriod).sum()  # bining the spikes
    spiketrains_xr = spiketrains_xr.fillna(0)  # fill the empty bin
else:
    spiketrains_xr = None

#%% Trial type
if len(spatial_firing.firing_times) > 0:
    trial_type = glmneuron.mergeSeriesOnIndex(
        spatial_firing.trial_type, spatial_firing.firing_times
    )
    trial_type_xr = xr.DataArray(
        trial_type.values,
        dims=("time"),
        coords={"time": pd.TimedeltaIndex(trial_type.index / settings.sampling_rate, unit="s")},
    )
    trial_type_xr = glmneuron.addZeroStartTime(trial_type_xr)
    trial_type_xr = trial_type_xr.resample(time=binPeriod).last().ffill(dim="time")
else:
    trial_type_xr = None

#%% Add trial number
if len(spatial_firing.firing_times) > 0:
    trial_number = glmneuron.mergeSeriesOnIndex(
        spatial_firing.trial_number, spatial_firing.firing_times
    )
    trial_number_xr = xr.DataArray(
        trial_number.values,
        dims={"time"},
        coords={"time": pd.TimedeltaIndex(trial_number.index / settings.sampling_rate, unit="s")},
    )
    trial_number_xr = glmneuron.addZeroStartTime(trial_number_xr)
    trial_number_xr = trial_number_xr.resample(time=binPeriod).last().ffill(dim="time")
else:
    trial_number_xr = None


#%% Load reward info
if len(spatial_firing.firing_times) > 0:
    trial_number_rs = trial_number_xr.data

    # Match with the trial number
    isRewarded = np.empty_like(trial_number_rs)
    isRewarded[:] = False

    try:
        position_df = pd.read_pickle(sinput.processed_position)
        rewarded_trials = np.where(position_df.rewarded)[0]

        for rt in rewarded_trials:
            isRewarded[trial_number_rs == rt] = True

        isRewarded_xr = xr.DataArray(
            isRewarded, dims={"time"}, coords=trial_number_xr.coords
        )

    except:
        # if position dataframe no found. Consider all as not rewarded
        print('Cannot find info for rewarded trials')
        isRewarded_xr = xr.DataArray(
            isRewarded, dims={"time"}, coords=trial_number_xr.coords
        )
else:
    isRewarded_xr = None

#%% Make one-hot vector
(posgrid, posvec) = glmneuron.pos_1d_map(
    position_xr.data, settings.position_bin, settings.track_length
)

(speedgrid, speedvec, speed) = glmneuron.speed_map_1d(
    position_xr.data, settings.speed_bin, 1000 / settings.binSize, maxSpeed=50, removeWrap=True
) # because binSize is specified in ms, so need to use 1000 to divide

(accelgrid, accelvec, accel) = glmneuron.speed_map_1d(
    speed, settings.accel_bin, 1000 / settings.binSize, maxSpeed=250, minSpeed=-250, removeWrap=True
) 

#%%
posgrid_xr = xr.DataArray(
    posgrid,
    dims=("time", "pos_bins"),
    coords={"time": position_xr.time, "pos_bins": posvec},
)

speedgrid_xr = xr.DataArray(
    speedgrid,
    dims=("time", "speed_bins"),
    coords={"time": position_xr.time, "speed_bins": speedvec},
)

accelgrid_xr = xr.DataArray(
    accelgrid,
    dims=("time", "accel_bins"),
    coords={"time": position_xr.time, "accel_bins": accelvec},
)


speed_xr = xr.DataArray(speed, dims=("time"), coords={"time": position_xr.time})
accel_xr = xr.DataArray(accel, dims=("time"), coords={"time": position_xr.time})

#%% Combine into one dataset
binned_data = xr.Dataset(
    {
        "spiketrain": spiketrains_xr,
        "position": position_xr,
        "speed": speed_xr,
        "accel": accel_xr,
        "pos_grid": posgrid_xr,
        "speed_grid": speedgrid_xr,
        "accel_grid": accelgrid_xr,
        "trial_number": trial_number_xr,
        "trial_type": trial_type_xr,
        "isRewarded": isRewarded_xr,
    }
)

# remove nan
binned_data = binned_data.dropna(dim="time")

#%% add some attributes
path_parts = Path(sinput.raw_position).parts

# use Regex to extract metadata from folder name
pattern = re.compile(r"(?P<animal>.*)_D(?P<session>\d+)_(?P<date>.+)_(?P<time>.+)")
m = pattern.match(path_parts[-3])

binned_data.attrs["session_id"] = path_parts[-3]
binned_data.attrs["animal"] = m["animal"]
binned_data.attrs["session"] = int(m["session"])
binned_data.attrs["date"] = m["date"]
binned_data.attrs["time"] = m["time"]

# also store the analysis setting
varName = [x for x in dir(settings) if not x.startswith("__")]
binned_data.attrs.update({k: v for k, v in settings.__dict__.items() if k in varName})

#%% Save data
with open(soutput.bin_data, "wb") as f:
    pickle.dump(binned_data.to_dict(), f)

# %%
