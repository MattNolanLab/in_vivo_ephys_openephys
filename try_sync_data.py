
#%%
from PostSorting.open_field_sync_data import *
from PostSorting.open_field_spatial_data import *
#%%
sync_data = pd.read_pickle('sync_data_ephys.pkl')

# %%
sync_pulse = sync_data.sync_pulse.values
# %%
sync_pulse_ds = sync_pulse[np.arange(0,len(sync_pulse),1000)]
# %%
spatial_data = pd.read_pickle('spatial_data.pkl')
bonsai = spatial_data['syncLED'].values
# %%
t = spatial_data.time_seconds
t2 = np.arange(len(t))/30
bonsai_rs = np.interp(t2,t,bonsai)
bonsai_rs = reduce_noise(bonsai_rs,np.median(bonsai) + 4 * np.std(bonsai))
plt.plot(bonsai_rs)
# %%
corr = np.correlate(bonsai_rs,sync_pulse_ds,'full')
plt.plot(corr)

# %%
# pos = read_position('/home/ubuntu/to_sort/recordings/M1_D32_2021-02-23_15-42-30/M1-0208-of2021-02-23T15_41_27.csv')
pos = pd.read_pickle('position_data.pkl')
timeindex = pd.TimedeltaIndex(pos.time_seconds,'s')
pos2 = pos.set_index(timeindex)
pos2 = pos2.resample(f'{1/30:.4f}S').interpolate('slinear')
pos2.syncLED.plot()


#%%
from scipy.interpolate import interp1d

def resample_pos(pos,fs=30):
    t = pos.time_seconds.values
    t2 = np.arange(0,len(t))/fs
    df = {}
    for col in pos.columns:
        f = interp1d(t,pos[col].values, fill_value='extrapolate')
        df[col] = f(t2)

    df['time_seconds'] = t2

    return pd.DataFrame(df)

pos2 = resample_pos(pos.drop(pos.columns[5:10],axis=1))
# %%
plt.plot(pos2.time_seconds,pos2.syncLED);plt.xlim([10,11])
plt.plot(pos.time_seconds,pos.syncLED,'r');plt.xlim([10,11])
# %%
pos = pd.read_pickle('position_data.pkl')
pos2 = resample_position_data(pos)
pos2.syncLED.plot()
# %%
