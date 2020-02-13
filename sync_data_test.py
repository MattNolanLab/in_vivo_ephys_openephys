#%%
import pickle
import numpy as np 
import matplotlib.pylab as plt
import pandas as pd 
from PostSorting.open_field_sync_data import * 

#%% 
bonsai_df = pd.read_pickle('testData/bonsai.pkl')
oe_df = pd.read_pickle('testData/ephys.pkl')

#%%
def getRisingEdge(pulse,thres=None):
    pulse_on = pulse > (np.median(pulse) + 2*np.std(pulse))
    pulse_on = np.concatenate([[pulse_on[0]], pulse_on]) # make the returned array the same size as input
    return np.diff(pulse_on) > 0 

syncLED = bonsai_df.syncLED.values
bonsai_time = bonsai_df.time_seconds.values
syncLED_rising = bonsai_time[getRisingEdge(syncLED)]
syncLED_rising = syncLED_rising- syncLED_rising[0]

#%%
oe_syncPulse = oe_df.sync_pulse.values
oe_time = oe_df.time.values
oe_rising = oe_time[getRisingEdge(oe_syncPulse)]
oe_rising  = oe_syncPulse_rising - oe_syncPulse_rising[0]

#%%
def getStartingEdge(led_risingEdge, ephys_risingEdge, span2search=10):
    # Try to remove the extra pulse at the beginning
    # It will remove the rising edge one by one and compare the rising edge time with the
    # Open Ephys recording. Use the one that shows the smallest difference
  
    assert len(led_risingEdge) >= len(ephys_risingEdge)
    mse = np.zeros((span2search,))
    for i in range(span2search):
        led = led_risingEdge[i:]
        led = led -led[0]
        ephys = ephys_risingEdge[:len(led)]
        mse[i] = np.mean((led-ephys)**2)
    
    minShift = np.argmin(mse)
    return minShift

#TODO: uneven sampling of ephys data

shift = getStartingEdge(syncLED_rising,oe_rising)
syncLED_rising_adjust = syncLED_rising[shift:][0] - oe_rising[0]
ephys_rising_adjust = 
meanLatencyTime = (syncLED_rising[shift:] - ephys_risingEdge_shifted).mean()

#%%
bonsai_df.index = pd.to_timedelta(bonsai_df.time_seconds,'s')
bonsai_resampled = bonsai_df.resample('20ms').nearest()
#%%
sync_data_ephys_downsampled,downsample_rate = downsample_ephys_data(oe_df, bonsai_df)


#%%
with open('../testData/test_data.pkl','rb') as f:
    test_data = pickle.load(f)

# %%

bonsai = test_data['bonsai']
bonsai = bonsai > np.median(bonsai)
oe = test_data['oe'] 
oe = oe > np.median(oe)

# %%
plt.plot(bonsai)
plt.plot(oe)

# %%
