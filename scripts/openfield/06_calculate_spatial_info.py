#%%

from PostSorting.spatial_information import *
import SnakeIOHelper 
import setting
import pandas as pd

#%% define input and output

(sinput, soutput) = SnakeIOHelper.getSnake(locals(), 'workflow/workflow_of.smk',
     [setting.debug_folder+'/processed/spatial_info.pkl'],
    'process_spatial_info')


# %% Calculate the spatial info for each cluster
spatial_firing = pd.read_pickle(sinput.spatial_firing)
position= pd.read_pickle(sinput.position)

spatial_info_df = process_spatial_info(spatial_firing, position)

spatial_info_df.to_pickle(soutput.spatial_info)

# %%
