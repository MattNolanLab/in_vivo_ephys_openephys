
#%%
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.toolkit as st
from scipy.signal import butter,filtfilt
import numpy as np
import time


Fs = 30000
signal = np.random.rand(16,120*Fs)
recording = se.NumpyRecordingExtractor(signal,Fs)

def filterRecording(recording, sampling_freq, lp_freq=300,hp_freq=6000,order=3):
    fn = sampling_freq / 2.
    band = np.array([lp_freq, hp_freq]) / fn

    b, a = butter(order, band, btype='bandpass')

    if not (np.all(np.abs(np.roots(a)) < 1) and np.all(np.abs(np.roots(a)) < 1)):
        raise ValueError('Filter is not stable')
    
    for i in range(recording._timeseries.shape[0]):
        recording._timeseries[i,:] = filtfilt(b,a,recording._timeseries[i,:])

    return recording


# Direct filtering with scipy
start = time.time()
filterRecording(recording,Fs) #for faster operation later
print(f'\n Scipy.filtfilt lasped time: {time.time()-start} ')


# Filtering using the filter object
start = time.time()
recording_sort = st.preprocessing.bandpass_filter(recording, freq_min=300, freq_max=6000,cache=True, type='butter')
print(f'\n st.preprocessing.bandpass_filter lasped time: {time.time()-start} ')

# Filtering using the filter object (small chunk size)
start = time.time()
recording_sort = st.preprocessing.bandpass_filter(recording, freq_min=300, freq_max=6000,cache=True, type='butter',chunk_size=1000)
print(f'\n st.preprocessing.bandpass_filter (small chunk) lasped time: {time.time()-start} ')



#%%
