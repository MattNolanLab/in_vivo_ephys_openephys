#%%
import OpenEphys
import numpy as np
import os
import shutil

#%% create short files for debugging purpose
basePath = 'testData/M1_D27_2018-10-26_13-10-36_of'
newPath = basePath+'_short'
shortenLength = 30000*60*10

try:
    os.mkdir(newPath)
except  FileExistsError:
    print("Folder already exist")

for f in os.scandir(basePath):
    if f.name.endswith('.continuous'):
        print(f'{f.name} truncated')
        file = OpenEphys.loadContinuousFast(f.path)
        OpenEphys.writeContinuousFile(newPath+'/'+f.name, file['header'],
            file['timestamps'], file['data'][:shortenLength], file['recordingNumber'])
    else:
        if f.is_dir():
            shutil.copytree(f.path, newPath+'/'+f.name)
        else:
            shutil.copyfile(f.path, newPath+'/'+f.name)

        print(f'{f.name} copied')



#%%
