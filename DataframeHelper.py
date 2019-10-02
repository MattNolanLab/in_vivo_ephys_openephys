# some helper functions for working with dataframes
import pandas as pd

def addCol2dataframe(dataframe, name, col):
    """Add numpy array to dataframe as column
    
    Arguments:
        dataframe {pd.DataFrame} -- the dataframe to append to
        name {str} -- name of the adde column
        col {np.narray} -- numpy array to be added as column
    
    Returns:
        np.DataFrame -- dataframe with the added column
    """
    d = pd.DataFrame({
        name: col
    })
    return pd.concat([dataframe, d], axis=1)