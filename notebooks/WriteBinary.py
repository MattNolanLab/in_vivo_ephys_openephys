#%%
# Write data to the new binary format
import OpenEphys
import setting
import numpy as np 
import json
import os
from pathlib import Path

#%%
outFolder = 'testData'
debug_folder = '/media/data2/pipeline_testing_data/M1_D31_2018-11-01_12-28-25'

#%%
def load_OpenEphysRecording4BinaryFile(folder,num_tetrodes=setting.num_tetrodes,
     data_file_prefix=setting.data_file_prefix, data_file_suffix=setting.data_file_suffix,
     dtype=float):
    signal = []
    headers = []
    for i in range(num_tetrodes*4):
        fname = folder+'/'+data_file_prefix+str(i+1)+data_file_suffix+'.continuous'
        dataFile = OpenEphys.loadContinuousFast(fname, dtype=dtype)
        x= dataFile['data']
        headers.append(dataFile['header'])

        if i==0:
            #preallocate array on first run
            signal = np.zeros((num_tetrodes*4,x.shape[0]),dtype=dtype)
        signal[i,:] = x
    return signal,headers

# %%
data,headers = load_OpenEphysRecording4BinaryFile(debug_folder,4,dtype=np.int16)

# %% Make some test data
Fs = 30000
nChan = 16
signals = np.random.randint(0,1000,(Fs*60*3,nChan),dtype='<i2')
datapath ='testData/continuous/open-ephys'
try:
    os.makedirs(datapath)
except FileExistsError:
    print('Folder exists. Skipping')

with open(datapath+'/continuous.dat','wb') as f:
    f.write(signals.tobytes())

np.save(datapath+'/timestamps.npy',np.arange(signals.shape[0],dtype='i8'))

# %%
def writeData(parentFolder,signals):
    #signals should be in time x channel format
    if signals.shape[0] < signals.shape[1]:
        signals = signals.T

    # make parent folders
    folderpath = Path(parentFolder) / 'continuous' / 'open-ephys'
    folderpath.mkdir(parents=True, exist_ok=True)

    # write data file
    with (folderpath / 'continuous.dat').open('wb') as f:
        f.write(signals.tobytes())

    # create and write timestamp
    timestamps = np.arange(signals.shape[0],dtype='i8')
    np.save(str(folderpath/'timestamps.npy'), timestamps)

writeData('testData',data)

#%% Make the structure json file
def writeStructFile(filename,headers):
    structDict = {'GUI version':'0.4.5','continuous':[], 'events':[], 'spikes':[]}

    structDict['continuous'] = [{
        "folder_name":"open-ephys/",
        "sample_rate":headers[0]['sampleRate'],
        "source_processor_name":"Demo source",
        "source_processor_id":100,
        "source_processor_sub_idx":0,
        "recorded_processor":"Demo source",
        "recorded_processor_id":100,
        "num_channels":16,
        "channels":[]
    }]

    #assemble channel data
    channels = []
    for i,h in enumerate(headers):
        channels.append({
            
            "channel_name": h['channel'],
            "description":"Demo data channel",
            "identifier":"genericdata.continuous",
            "history":"Demo source",
            "bit_volts":h['bitVolts'],
            "units":"uV",
            "source_processor_index":i,
            "recorded_processor_index":i
                
        })

    structDict['continuous'][0]['channels'] = channels

    with open(filename,'w') as f:
        f.write(json.dumps(structDict,indent=4))

writeStructFile('testData/structure.oebin',headers)
# %%
