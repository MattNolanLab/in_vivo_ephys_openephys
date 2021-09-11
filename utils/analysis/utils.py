import glob
import pandas as pd
from tqdm import tqdm
import numpy as np 
import math
from utils.GLMPostProcessor import GLMPostProcessor
from scipy import signal
from tqdm import tqdm
from statsmodels.formula.api import ols
import statsmodels.api as sm 
import scipy
import seaborn as sns
import matplotlib.pylab as plt
from math import ceil
from IPython.core.display import display, HTML


genotype_pal = {"WT": "#3274A1", "KO": "#E1812C"}

def find_files_in_folders(folders):
    # folders is a list of glob patterns
    # find all the file matched

    if not isinstance(folders,list):
        folders = [folders]

    flist = []

    for f in folders:
        flist += glob.glob(f)

    return flist


def concat_df_in_folders(folders, add_session_id=False, session_id=None):
    # Concatenate the dataframes found in the specified folders
    # folders should be a list of path of pickled dataframe

    dfs = []

    for i,f in tqdm(enumerate(folders)):
        df = pd.read_pickle(f)

        if add_session_id and session_id is not None:
            # add session_id into the dataframe
            df['session_id']  = session_id[i]

        dfs.append(df)

    df_all = pd.concat(dfs)

    return df_all


def gaussian_kernel(kernx):
    kerny = np.exp(np.power(kernx, 2)/2 * (-1))
    return kerny

def calculate_rate_map2(spike_positions_x, spike_positions_y, positions_x, positions_y, 
    number_of_bins, arena_size = 100):
    # assume the arena is square shape
   
    firing_rate_map = np.zeros((number_of_bins, number_of_bins))
    bin_size = arena_size / number_of_bins
    min_dwell_distance_pixels = 5
    min_dwell = 3
    smooth = 5
    dt_position_ms = 1/30*1000

    for x in range(number_of_bins):
        for y in range(number_of_bins):
            px = x /number_of_bins * arena_size + (bin_size/2)
            py = y /number_of_bins* arena_size + (bin_size/2)
            # print(px,py)

            # find the distance to the closet bin
            spike_distances = np.sqrt(np.power(px - spike_positions_x, 2) + np.power(py - spike_positions_y, 2))
            spike_distances = spike_distances[~np.isnan(spike_distances)]

            # find the distance to the clost bin for all positions
            occupancy_distances = np.sqrt(np.power((px - positions_x), 2) + np.power((py - positions_y), 2))
            occupancy_distances = occupancy_distances[~np.isnan(occupancy_distances)]
            bin_occupancy = len(np.where(occupancy_distances < min_dwell_distance_pixels)[0]) #how many points in nearby area
            # depending on the distance to the current position bin, all spikes will have more or less
            # effect as determined by a gaussian kernel, the final effect is the summation of all the small
            # gaussian windows
            if bin_occupancy >= min_dwell:
                firing_rate_map[x, y] = sum(gaussian_kernel(spike_distances/smooth)) / (sum(gaussian_kernel(occupancy_distances/smooth)) * (dt_position_ms/1000))

            else:
                firing_rate_map[x, y] = 0
    #firing_rate_map = np.rot90(firing_rate_map)
    return firing_rate_map

def get_spatial_info(rate_map,occ_map):
    # return spatial information in terms of bit per spike
    # rate_map: firing rate
    # occ_map: probability of occupancy in each pin

    mean_fr = rate_map.mean()
    
    #make it a linear matrix for easier computation
    occ = occ_map.ravel()
    fr = rate_map.ravel()
    
    H1 = 0
    for i in range(len(fr)):
        if fr[i]>0.0001:
            H1 += fr[i] * np.log2(fr[i]/mean_fr)*occ[i]
    
    return H1/mean_fr

def getTuningCurve(spiketrain, pos_grid,train_model=True):
    glm = GLMPostProcessor(spiketrain, pos_grid,[0,0,1,0],0.1)
    if train_model:
        glm.run()
    result = glm.getTuningCurves(spiketrain, pos_grid, 0.1)
    return result['model_curve'], result['firing_rate']


def analyze_spatial_corr(data):
    # data: binned xarray of firing and position

    #smooth the firing rate
    gauss_win = signal.windows.gaussian(11,1) 
    gauss_win /= gauss_win.max() # make sure central part is 1

    spiketrain = data.spiketrain.data
        
    try:
        for i in range(spiketrain.shape[1]):
            spiketrain[:,i] = np.convolve(data.spiketrain.data[:,i],gauss_win,mode='same')
    except:
        # no spike detected, return immediately
        return None

    # Calcualte the corelation scores of firing map in each trial to the overall map
    # also compute the overall correlation score bewtween trials

    dfs = []

    totalTrial = int(np.max(data.trial_number.data))

    for cell_idx in range(spiketrain.shape[1]):
        #get the overall firing map
        curve_all, frmap_all = getTuningCurve(spiketrain[:,cell_idx],data.pos_grid.data)
        corr_list = []
        frmap_list = []
        trial_number_list = []
        trial_type_list =[]

        trial_nums = np.unique(data.trial_number)
        
        for trial in trial_nums: #trial number starts from 1
            trial_range = (data.trial_number>=trial) & (data.trial_number<trial+1)
            
            #Determine the trial type
            trial_type = data.trial_type.data[data.trial_number==trial][0]
            
            #get the firing map and analyze the correlation
            if np.sum(trial_range)>0:
                curve,frmap = getTuningCurve(spiketrain[trial_range,cell_idx],data.pos_grid.data[trial_range],train_model=False)
                frmap_list.append(frmap)
                corr_list.append(np.corrcoef(frmap,frmap_all)[0,1]) #compare with the all firing rate map
                trial_type_list.append(trial_type)
                trial_number_list.append(trial)
        
        # analyze the overall correlation i.e. the stability of the firing map
        # for cells with unstable firing, the cross-correlation between trials wil be low
        frmaps = np.stack(frmap_list)
        frmaps_corr = np.corrcoef(frmaps)
        
        df_tmp = pd.DataFrame({
            'frmaps':frmap_list,
            'corrs':corr_list,
            'trial_number': trial_nums
        })
        df_tmp['cluster_id'] = data.neuron.data[cell_idx]
        df_tmp['frmaps_corr'] = frmaps_corr.mean()
        
        dfs.append(df_tmp)
        
    df_corr = pd.concat(dfs)
    df_corr['session_id'] = data.session_id
    df_corr['animal'] = data.animal
    df_corr['day'] = data.session


    return df_corr

def anova_comparison(col_name, df, compare_handler= True, type=2):
    # type 2 comparison may have problem if some samples are missing from one of the sub-category
    # but usually it is more reliable as it doesn't depends on the order of the variables

    print(f'Total samples {len(df)}')    
    if compare_handler:
        model = ols(f'{col_name} ~ C(genotype)*C(handler)',data=df).fit()
    else:
        model = ols(f'{col_name} ~ C(genotype)',data=df).fit()

    anova_table = sm.stats.anova_lm(model, typ=type)
    print(anova_table)
    
    return anova_table.iloc[0]['PR(>F)']

def kruskal_compare(col_name, df):
    wt = df[col_name][df.genotype=='WT']
    ko = df[col_name][df.genotype=='KO']
    cmp = scipy.stats.kruskal(wt,ko)
    print(cmp)
    return cmp.pvalue

def formatLabel(label):
    return label.replace('_', ' ').capitalize()

def addPvalue(ax, x1,x2, y, pvalue, h=2, axis_padding=0.15, col='k'):
    
    if pvalue>0.05:
        plabel='ns'
    else:
        plabel = f'p={pvalue:.2e}'
    
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    ax.text((x1+x2)*.5, y+h+0.01, plabel, ha='center', va='bottom', color=col)
    
    #addjust the margin
    ax.set_ylim([0, y+h+axis_padding])

def filter_cell_percentile(col_name, df2score, percentile=95):
    thres = df2score.groupby('animal')[col_name].transform(lambda x: np.percentile(x.dropna(), percentile))

    thres2 = df2score.groupby('animal')[col_name].apply(lambda x: np.percentile(x.dropna(), percentile))
    print('Threshold')
    print(thres2)

    dfcompare = df2score[df2score[col_name]>thres]
    return dfcompare

def plotComparison_subject(col_name,dfcompare, ax, hide_legend=True,plot_scatter=False):
    # plot the comparison plot, filtering score by percentile first

    dfcompare = dfcompare.sort_values(['animal','genotype'])
    print(dfcompare.groupby('animal')[col_name].agg(['mean','median','count']))
    # KO is blue, WT is orange, KO is on  left
    ax = sns.boxplot(x='animal',hue='genotype',dodge=False,y=col_name,data=dfcompare, palette=genotype_pal, ax=ax)
    if plot_scatter:
        ax = sns.stripplot(x='animal',hue='genotype', alpha=0.5,y=col_name,data=dfcompare,color='gray', ax=ax)
    ax.set(ylabel = formatLabel(col_name))
    if hide_legend:
        ax.get_legend().remove()

    return ax

def plotComparison_simple(col_name, dfcompare, ax, axis_padding=0.1, p_y_pos=None, hide_legend=True, plot_scatter=False):
    dfcompare = dfcompare.sort_values(['animal','genotype'])
    p_value = kruskal_compare(col_name,dfcompare)
    ax = sns.boxplot(x='genotype',y=col_name,data=dfcompare, palette=genotype_pal, ax=ax)
    if plot_scatter:
        ax = sns.stripplot(x='genotype',y=col_name,data=dfcompare,color='k', ax=ax)
    ax.set(ylabel = formatLabel(col_name))

    if p_y_pos is not None:
        pypos = p_y_pos
    else:
        pypos = dfcompare[col_name].max()+0.05
    addPvalue(ax, 0, 1, pypos, p_value, h=0.1, axis_padding=axis_padding)
    return ax

def getCumFreq(col_name,max_bin, df):
    # get cummulative distribution
    bins = np.arange(0, max_bin,0.01)
    freq = np.histogram(df[col_name].values,bins)[0]
    cumdist = np.cumsum(freq)/np.sum(freq)
    df2 =  pd.DataFrame({'cumdist':cumdist, col_name:bins[:-1]})
    df2['genotype'] = df['genotype'].iloc[0]
    df2['handler'] = df['handler'].iloc[0]
    return df2

def analyzeCumFreq(col_name, dfcompare):
    max_bin = dfcompare[col_name].max()
    df_cumfreq = dfcompare.groupby('animal').apply(lambda x: getCumFreq(col_name,max_bin,x)).reset_index()
    df_cumfreq = df_cumfreq.drop(columns=['level_1'])
    return df_cumfreq

def plotCumFreq(df_cumFreq,col_name, xlabel=None):
    g = sns.relplot(y='cumdist', x = col_name, col='handler',hue='genotype',style='animal',kind='line', palette=genotype_pal,data=df_cumFreq)
    g._legend.remove()
    g.fig.legend(bbox_to_anchor=(1.1, 1.2))
    if xlabel is not None:
        g.set_axis_labels(xlabel,'Cummulative\n Probability')
    else:
        g.set_axis_labels(formatLabel(col_name),'Cummulative\n Probability')
    return g


def power_analysis(dfcompare, col_name, repeat = 1000, alpha=0.05, start_size =3, end_size= 10, verbose=False):
    # perform poweranalysis on the data by random sampling
    
    df_mean = dfcompare.groupby(['animal','genotype']).mean()
    df_groupmean = df_mean.groupby('genotype').mean()
    df_std = df_mean.groupby('genotype').std()

    #parameter for power analysis
    diff = df_groupmean.loc['KO'] - df_groupmean.loc['WT']

    p_analysis = []

    for size in tqdm(range(start_size, end_size)):
        npsig = 0

        for _ in range(repeat):

            # randomly sample from population, and do the statistical test
            wt = np.random.normal(0, df_std.loc['WT'][col_name],size)
            ko = np.random.normal(diff[col_name], df_std.loc['KO'][col_name], size)

            cmp = scipy.stats.kruskal(wt,ko)

            if cmp.pvalue < alpha:
                npsig +=1

        p_analysis.append( {'size':size, 'power':npsig/repeat})

    df_p = pd.DataFrame(p_analysis)
    
    if verbose:
        print('Mean')
        print(diff)

        print('Std')
        print(df_std)
                                       
    return df_p
                                       
    
def show_spike_trajectory(sel_row):
    # load spike trajectories from the processed folder
    
    if sel_row.animal[0] == 'T':
        img_path = f'/mnt/datastore/Teris/FragileX/data/OpenField/{sel_row.session_id}/processed/figures/spike_trajectories/{sel_row.session_id}_track_firing_Cluster_{sel_row.cluster_id}.png'
    else:
        img_path = f'/mnt/datastore/Junji/Data/2021cohort1/vr/{sel_row.session_id}/processed/figures/spike_trajectories/{sel_row.session_id}_track_firing_Cluster_{sel_row.cluster_id}.png'
    
    plt.figure(figsize=(8,8))
    img = plt.imread(img_path)
    plt.imshow(img)
    plt.axis('off');


# H0 = 0;
# H1 = 0;
# SumRate = 0;
# SumOcc = 0;
# for j = 1:length(frmap(:,1))
#   for k = 1:length(frmap(1,:))
#     if ((posmap(j, k) > 0) & (frmap(j, k) >= 0))
#       SumRate = SumRate + frmap(j, k);
#       SumOcc = SumOcc + 1;
#     end
#   end
# end

# for j = 1:length(frmap(:,1))
#   for k = 1:length(frmap(1,:))
#         if (frmap(j, k) > 0.0001)
#             H1 = H1 + -frmap(j, k) * (log(frmap(j, k)) / log(2)) / SumOcc;
#         end
#     end
# end

# MeanRate = SumRate / SumOcc;
# H0 = -MeanRate * (log(MeanRate) / log(2));
# info = (H0 - H1) / MeanRate;


def saveDebug(d,filename):
    # save the dictionary as pickle for debugging

    with open(filename,'wb') as f:
        pickle.dump(d,f)

def take_first_reward_on_trial(rewarded_stop_locations,rewarded_trials,rewarded_sample_idx):
    unique_rewarded_trial = np.unique(rewarded_trials)
    locations=np.zeros((len(unique_rewarded_trial)))
    trials=np.zeros_like(locations)
    sample_idx = np.zeros_like(locations)

    for tcount, trial in enumerate(unique_rewarded_trial):
        idx = np.where(rewarded_trials == trial)[0]
        trial_locations = rewarded_stop_locations[idx]
        trial_stop_idx = rewarded_sample_idx[idx]

        if len(trial_locations) ==1:
            locations[tcount] = trial_locations
            trials[tcount] = trial
            sample_idx[tcount] = idx
        if len(trial_locations) >1:
            locations[tcount]  = trial_locations[0]
            trials[tcount] = trial
            sample_idx[tcount] = idx
    return np.array(locations), np.array(trials)


def getWrappedSubplots(ncol,total_n,figsize,**kwargs):
    """Create wrapped subplots

    Args:
        ncol (int): number of columns 
        total_n (int): total number of subplots
        figsize ((width,height)): tuple of figsize in each plot

    Returns:
        [(fig,axes)]: wrapped subplots axes handle
    """
    #Automatically create a wrapped subplots axis
    nrow = ceil(total_n/ncol)

    fig,ax = plt.subplots(nrow, ncol, figsize = (figsize[0]*ncol, figsize[1]*nrow), **kwargs)

    return fig, ax.ravel()


def show_combined_of_plot(sel_row, figsize=(8,8)):
    # load spike trajectories from the processed folder
    if sel_row.animal[0] == 'T':
        img_path = f'/mnt/datastore/Teris/FragileX/data/OpenField/{sel_row.session_id}/processed/figures/combined/{sel_row.session_id}_{sel_row.cluster_id}.png'
    else:
        animal_number =  sel_row.animal[2] 
        img_path = f'/mnt/datastore/Junji/Data/2021cohort1/openfield/m{animal_number}/{sel_row.session_id}/processed/figures/combined/{sel_row.session_id}_{sel_row.cluster_id}.png'
    
    print(img_path)
    plt.figure(figsize=figsize)
    img = plt.imread(img_path)
    plt.imshow(img)
    plt.axis('off');
    
    
def show_combined_vr_plot(sel_row, figsize=(8,8)):
    # load spike trajectories from the processed folder
    if sel_row.animal[0] == 'T':
        img_path = f'/mnt/datastore/Teris/FragileX/data/VR/{sel_row.session_id}/processed/figures/combined/{sel_row.session_id}_{sel_row.cluster_id}.png'
    else:
        animal_number =  sel_row.animal[2] 
        img_path = f'/mnt/datastore/Junji/Data/2021cohort1/vr/{sel_row.session_id}/processed/figures/combined/{sel_row.session_id}_{sel_row.cluster_id}.png'
    
    print(img_path)
    plt.figure(figsize=figsize)
    img = plt.imread(img_path)
    plt.imshow(img)
    plt.axis('off');
    
def show_ramp_plot(sel_row, figsize=(8,8)):

    region = sel_row.ramp_region
    # load spike trajectories from the processed folder
    if sel_row.animal[0] == 'T':
        img_path = f'/mnt/datastore/Teris/FragileX/data/VR/{sel_row.session_id}/processed/ramp_score/ramp_score_plot_{region}.png'
    else:
        animal_number =  sel_row.animal[2] 
        img_path = f'/mnt/datastore/Junji/Data/2021cohort1/vr/{sel_row.session_id}/processed/ramp_score/ramp_score_plot_{region}.png'
    
    print(img_path)
#     print_link(img_path)

    plt.figure(figsize=figsize,dpi=200)
    img = plt.imread(img_path)
    plt.imshow(img)
    plt.axis('off');
    
    
def print_link(path):
    display(HTML(f"""<a href="{path}">{path}</a>"""))

    
def analyze_trial_corr(df):
    # here we should have the df for a particular recording
    df = df.sort_values('trial_number')
    
    # get the correlation of the first 2 trials as the baseline
    baseline_corr = np.percentile(df.corrs,5) #take the 5 and 95 percentile to compare
    
    # Compare with the max
    max_corr = np.percentile(df.corrs,95)
    
    mean_corr = np.mean(df.corrs)
    
    return pd.Series({'corr_change':(max_corr-baseline_corr), 'corr_change_ratio':(max_corr-baseline_corr)/baseline_corr, 'corr_mean':mean_corr})
