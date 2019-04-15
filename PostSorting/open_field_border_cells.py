import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PostSorting.open_field_firing_fields

'''
calculates the border scores according to Solstad et al (2008)

"Putative border fields were identified first by identifying collections of neighboring pixels
 with firing rates higher than 0.3 times the maximum firing rate and covering a total area of 
 at least 200 cm2. For all experiments in square or rectangular environments, the coverage of
 a given wall of by a field was then estimated as the fraction of pixels along the wall that 
 was occupied by the field, and cM was defined as the maximum coverage of any single field 
 over any of the four walls of the environment. The mean firing distance dm was computed by 
 averaging the distance to the nearest wall over all pixels in the map belonging to some of its
 fields, weighted by the firing rate. To achieve this, the firing rate was normalized by its sum
 over all pixels belonging to some field, resembling a probability distribution. Finally, dm 
 was normalized by half of the shortest side of the environment (i.e. the largest possible 
 distance to its perimeter) so as to obtain a fraction between 0 and 1. A border score was defined 
 by comparing dm with the maximum coverage of any wall by a single field cM,   
 
 b = (cM - dm) / (cM + dm)
 
 Border scores ranged from -1 for cells with central firing fields to +1 for cells with fields that 
 perfectly line up along at least one entire wall. Intuitively, the border scores provide an idea of 
 the expansion of fields across walls rather than away from them. It should be noted that the measure 
 saturates when the width of the field approaches half the length of the environment.   
  
  ‘Border cells’ were defined as cells with border scores above 0.5. Only cells with stable border 
  fields (spatial correlation > 0.5) were included in the sample. In experiments with walls inserted 
  into the recording enclosure, the analysis was restricted to border cells with fields along a 
  single wall, i.e. cells where the border score for the preferred wall was at least twice as 
  high as the score for any of the remaining three walls."
'''

def process_border_data(spatial_firing):

    # TODO check about bottom left corner of rate maps (why its not interpolated correctly)
    # TODO raise issue of passing an argument to is_field_big_enough/small enough

    border_scores = []

    for index, cluster in spatial_firing.iterrows():
        cluster_id = cluster.cluster_id
        #print(index, "index")
        #print(cluster_id, "=cluster_id")

        firing_rate_map = cluster.firing_maps
        firing_rate_map = putative_border_fields_clip_by_firing_rate(firing_rate_map)

        fig, ax = plt.subplots()
        im = ax.imshow(firing_rate_map)
        fig.tight_layout()
        plt.show()

        firing_fields_cluster, _ = PostSorting.open_field_firing_fields.get_firing_field_data(spatial_firing, index, threshold=30)
        firing_fields_cluster = fields2map(firing_fields_cluster, firing_rate_map)
        firing_fields_cluster = clip_fields_by_size(firing_fields_cluster, bin_size_cm=2.5)
        firing_fields_cluster = put_firing_rates_back(firing_fields_cluster, firing_rate_map)


        border_score = calculate_border_score(firing_fields_cluster, bin_size_cm=2.5)

        border_scores.append(border_score)

        #plot_fields_in_cluster(firing_fields_cluster)
        plot_fields_in_cluster_border_scores(firing_fields_cluster, border_score)

    spatial_firing['border_score'] = border_scores
    return spatial_firing

def put_firing_rates_back(firing_fields_cluster, firing_rate_map):

    new = []
    for field in firing_fields_cluster:
        new.append(np.multiply(field, firing_rate_map))

    return new

def calculate_border_score(firing_fields_cluster, bin_size_cm):

    # only execute if there are firing fields to analyse
    if len(firing_fields_cluster)>0:

        normalised_distance_mat = distance_matrix(firing_fields_cluster[0], bin_size_cm)

        normalized_fields = []
        dm = []

        maxcM = 0

        for field in firing_fields_cluster:

            field_count = field.copy()
            field_count[field_count > 0] = 1

            wall1_cM = np.sum(field_count[0])/len(field_count[0])
            wall2_cM = np.sum(field_count[:,0])/len(field_count[:,0])
            wall3_cM = np.sum(field_count[:,-1])/len(field_count[:,-1])
            wall4_cM = np.sum(field_count[-1])/len(field_count[-1])

            # reassign max cM if found bigger in a different field or wall

            if wall1_cM>maxcM:
                maxcM= wall1_cM
            elif wall2_cM>maxcM:
                maxcM= wall2_cM
            elif wall3_cM > maxcM:
                maxcM = wall3_cM
            elif wall4_cM > maxcM:
                maxcM = wall4_cM

            normalized_field = field/np.sum(field)

            dm_for_field = np.multiply(normalized_field, normalised_distance_mat)  # weight by shortest distance to the perimeter
            dm_for_field = np.sum(dm_for_field)

            dm.append(dm_for_field)

        dm_all_fields = np.mean(dm)

        # final measure of mean firing distance
        dm = dm_all_fields.copy()
        cM = maxcM

        border_score = (cM - dm) / (cM + dm)

        return border_score

    else:
        border_score = np.nan

        # if no fields are found return NaN for border score (discredit these)
        return border_score


def distance_matrix(field, bin_size_cm):
    '''
    generates a matrix the same size as the rate map with elements
    corresponding to the mean shortest distance to the edge of the arena
    :param field: field rate map 2d np.array()
    :param bin_size_cm: int
    :return: distance matrix of same dimensions of field (unit cm)
    '''

    x, y = np.shape(field)

    r = np.arange(x)
    r2 = np.arange(y)

    d1 = np.minimum(r, r[::-1])
    d2 = np.minimum(r2, r2[::-1])

    distance_matrix = np.minimum.outer(d1, d2)

    distance_matrix = distance_matrix + 1
    distance_matrix = distance_matrix * bin_size_cm
    distance_matrix = distance_matrix - (bin_size_cm/2)

    distance_matrix = distance_matrix/np.max(distance_matrix) # normalise to largest distance to border

    return distance_matrix


def stack_fields(firing_fields_clusters):
    '''
    this functions stacks the firing fields back together
    :param firing_fields_clusters: masked rate maps with individual firing fields
    :return: rate map with all firing fields 0= out of field, 1 = in field
    '''
    stacked = np.sum(firing_fields_clusters, axis=0)
    return stacked


def plot_fields_in_cluster(firing_fields_cluster):
    for field in firing_fields_cluster:
        fig, ax = plt.subplots()
        im = ax.imshow(field)
        fig.tight_layout()
        plt.show()

def plot_fields_in_cluster_border_scores(firing_fields_cluster, border_score):
    for field in firing_fields_cluster:
        fig, ax = plt.subplots()
        im = ax.imshow(field)
        fig.tight_layout()

        title = "border_score: " + str(border_score)
        ax.set_title(title)
        plt.show()

def fields2map(firing_fields_cluster, firing_rate_map_template):
    '''
    :param firing_field_cluster: coordinates of firing fields for a given cluster
    :param firing_rate_map_template: example firing rate map to copy structure
    :return: rate map per field
    '''
    firing_fields = []

    for field in firing_fields_cluster:
        field_firing = firing_rate_map_template.copy() * 0

        for i in range(len(field)):
            field_firing[field[i][0]][field[i][1]] = 1

        firing_fields.append(field_firing)

    return firing_fields

def clip_fields_by_size(masked_rate_maps, bin_size_cm=2.5):
    '''
    clips the fields in the firing rate map if the neighbouring regions don't sum to 200cm2
    :param firing_rate_map: smoothened firing rate map, preclipped by max firing rate
    :return: clipped firing rate
    '''
    bin_volume_cm2 = bin_size_cm*bin_size_cm

    new_masked_rate_maps = []

    for field in masked_rate_maps:
        if np.sum(field*bin_volume_cm2)>200:   # as specified by Solstad et al (2008), only fields larger than 200cm2 are considered
            new_masked_rate_maps.append(field)

    return new_masked_rate_maps



def putative_border_fields_clip_by_firing_rate(firing_rate_map):
    '''
    clips the fields in the firing rate map if the firing rate is below 0.3x max firing rate
    :param firing_rate_map: smoothened firing rate map
    :return: firing_rate_map clipped by 0.3x max firing rate
    '''
    max_firing = np.max(firing_rate_map)
    firing_rate_map[firing_rate_map < 0.3*max_firing] = 0
    return firing_rate_map


#  this is here for testing
def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    # get_correlation_vector(np.array([[1, 1, 1, 1], [2, 2, 2, 9], [3, 3, 3, 3], [4, 4, 4, 4]]))
    # firing_rate_map_matlab = np.genfromtxt('C:/Users/s1466507/Documents/Ephys/test_overall_analysis/M5_2018-03-06_15-34-44_of/matlab_rate_map.csv', delimiter=',')
    # get_correlation_vector(firing_rate_map_matlab)

    spatial_firing = pd.read_pickle('/home/harry/Downloads/spatial_firing_.pkl')
    spatial_firing = process_border_data(spatial_firing)
    print(spatial_firing)


if __name__ == '__main__':
    main()