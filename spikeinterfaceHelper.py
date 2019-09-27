##some helper functions to be used with spike interface
import pandas as pd
import matplotlib.pylab as plt

def sorter2dataframe(sorter):
    """Convert data in a sorter to dataframe
    
    Arguments:
        sorter {SortingExtractor} -- the sorter to extract data frome
    
    Returns:
        pandas.DataFrame -- dataframe containing the spike train, unit featuers and spike features of the sorter
    """
    clusterList = []

    for i in sorter.get_unit_ids():
        unit_property=sorter.get_unit_property_names(i)
        spike_property = sorter.get_unit_spike_feature_names(i)

        propertyDict = {}
        for p in unit_property:
            propertyDict[p] = sorter.get_unit_property(i,p)

        for p in spike_property:
            propertyDict[p] = sorter.get_unit_spike_features(i,p)

        propertyDict['spike_train'] = sorter.get_unit_spike_train(i)
        propertyDict['unit_id'] = i
        propertyDict['sampling_frequency'] = sorter.get_sampling_frequency()

        clusterList.append(propertyDict)
    
    return pd.DataFrame(clusterList)
 