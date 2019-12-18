# some helper functions for working with dataframes
import pandas as pd
import numpy as np

def addCol2dataframe(dataframe, col:dict):
    """Add numpy array to dataframe as column, overwrite if column already exist
    
    Arguments:
        dataframe {pd.DataFrame} -- the dataframe to append to
        col {dict} -- keyword value pair of the column names and data
    
    Returns:
        np.DataFrame -- dataframe with the added column
    """

    #remove col if already exist
    for k in col.keys():
        if k in dataframe.columns:
            dataframe=dataframe.drop(k,1)

    d =  {}
    for k,v in col.items():
        d[k] = pd.Series(v)
    df = pd.DataFrame(d)
    return pd.concat([dataframe, df], axis=1)


def findTransitWithHyst(condition,skip,edge='rising'):
    d = np.diff(condition)
    
    #find the edges
    if edge=='rising':
        idx = np.where(d>0)[0]
    else:
        idx = np.where(d<0)[0]
    transitPt = [idx[0]]
    for i in idx[1:]:
        #only collect a point if it is at least a certain distance from the last item
        if (i-transitPt[-1])>skip:
            transitPt.append(i)
    return transitPt