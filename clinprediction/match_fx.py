# Written by Alice S. Tang and Chi-Yo Tsai
# Last updated 3/5/2023
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import seaborn as sns
import sys
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression, LinearRegression

def match_unique(treated_df, non_treated_df, matched = pd.DataFrame(), ratio = 1):
    """
    treated_df, non_treated_df, matched, matches = match_unique(treated_df, non_treated_df, matched, ratio = 1)
    description: function meant to be used recurvively. unique matches are stored in the 'matched' dataframe passed in, 
        and non-unique matches are returned in treated_df and non_treated_df to be utilized for matching again.
    inputs:
        treated_df: dataframe for case patients. rows are patients, columns are everything you want to match on.
        non_treated_df: dataframe for control patients. rows are patients, columns are everything you want to match on.
        matched: (default = pd.DataFrame()). Dataframe to store unique matches. 
            Pass in, and it will be spit out with 'matches' appended to it.
        matches: dataframe of all unique matches this round. 
        ratio: ratio for matches. 
    outputs: 
        treated_df, non_treated_df: inputs with unique successful matches removed. to be used iteratively in this function again.
        matched: appends matches to input dataframe
        matches: all unique matches done this round.
    """
    if len(treated_df) == 0: 
        print('No matching is needed. size of cohort left is 0')
        matches = pd.DataFrame()
        return treated_df, non_treated_df, matched, matches
    
    if len(treated_df)*ratio > len(non_treated_df):
        print('WARNING: unique matches are not possible at this ratio.')
     
    treated_x = treated_df.values
    non_treated_x = non_treated_df.values
            
    nbrs = NearestNeighbors(n_neighbors = ratio, algorithm='ball_tree').fit(non_treated_x)
    distances, indices = nbrs.kneighbors(treated_x)
    
    ix_name = non_treated_df.index.name
    non_treated_pts = non_treated_df.index[indices.flatten()].values.reshape(indices.shape)
    matches = pd.DataFrame(index = treated_df.index, data = non_treated_pts, 
                           columns = ['match{}'.format(i) for i in range(ratio)])
    matches = pd.wide_to_long(matches.reset_index(), stubnames = 'match', i = ix_name, j = 'match_num').sort_index().reset_index()

    _, ix = np.unique(matches.match, return_index=True)
    res = np.zeros_like(matches.match, dtype=bool)
    res[ix] = True
    
    matches['res'] = res
    all_unique_match_pts = matches.groupby(ix_name).res.sum()
    all_unique_match_pts = all_unique_match_pts[all_unique_match_pts == ratio].index # case with 'ratio' unique matches
    matches = matches[matches.person_id.isin((all_unique_match_pts))]
    matched = matched.append(matches)

    treated_df = treated_df.drop(matches.person_id.unique())
    non_treated_df = non_treated_df.drop(matches.match.unique())
    return treated_df, non_treated_df, matched, matches

def get_matching_pairs(treated_df, non_treated_df, scale = False, ratio = 1, replacement = False, max_bin_size = 1000):
    """
    matched = get_matching_pairs(treated_df, non_treated_df, scaler=True, ratio = 1, replacement = False)
    description: function that returns matched rows in non_treated_df for each of treated_df.
    
    inputs:
        treated_df: dataframe for case patients. rows are patients, columns are everything you want to match on.
        non_treated_df: dataframe for control patients. rows are patients, columns are everything you want to match on.
        scale: (default = False) whether to use StandardScaler() to scale the columns.
        ratio: (default = 1) ratio for matches.
        replacement: (default = False) whether to match with replacement in non_treated_df. False means you want unique matches.
    outputs: 
        matched: subset of non_treated_df that are matches of inputs. 
            the 'index_case' column indicates the index of who it is matched to in treated_df.
    """
    if len(treated_df)*ratio > len(non_treated_df):
        print('WARNING: unique matches are not possible at this ratio. returning all control patients...')
        return non_treated_df
    
    treated_x = treated_df.values
    non_treated_x = non_treated_df.values
    ix_name = non_treated_df.index.name
    non_treated_df_save = non_treated_df.copy()
    treated_df_save = treated_df.copy()
    
    if scale:
        scaler = StandardScaler()
        scaler.fit(treated_x)
        treated_x = scaler.transform(treated_x)
        non_treated_x = scaler.transform(non_treated_x)
        treated_df = pd.DataFrame(treated_x, index = treated_df.index, columns = treated_df.columns)
        non_treated_df = pd.DataFrame(non_treated_x, index = non_treated_df.index, columns = non_treated_df.columns)

    if replacement:
        nbrs = NearestNeighbors(n_neighbors = ratio, algorithm='ball_tree').fit(non_treated_x)
        distances, indices = nbrs.kneighbors(treated_x)
        matched = non_treated_df_save.iloc[indices.flatten()]
        matched['index_case'] = np.repeat(treated_df.index.values,ratio)
        return matched
    else:
        matched = pd.DataFrame()
        iter_ix = 0
        print('Matching...')

        progress_tot = len(treated_df)
        pbar = tqdm(total = progress_tot)
        try: 
            while len(treated_df) > 0:
                if iter_ix > progress_tot+1: break;
                if len(treated_df) == 0: break;
                if iter_ix < 5:
                    treated_df, non_treated_df, matched, matches = match_unique(treated_df, non_treated_df, matched, ratio = ratio)
                else: 
                    # compute distance on remaining
                    treated_x = treated_df.values
                    non_treated_x = non_treated_df.values
                    
                    len_remaining = len(treated_x)
                    bin_size = min(max_bin_size, len_remaining)
                    nbrs = NearestNeighbors(n_neighbors = bin_size*ratio, algorithm='ball_tree').fit(non_treated_x) # len(non_treated_x)

                    used = set()
                    rem_matched = np.empty(shape=(bin_size, ratio), dtype = int)
                    indices = nbrs.kneighbors(treated_x[0: bin_size, :], return_distance = False) # get all matches ranked

                    # get unique matches
                    for r_idx, r in enumerate(indices):
                        count = 0
                        for c in r:
                            if c not in used:
                                used.add(c)
                                rem_matched[r_idx][count] = non_treated_df.index[c]
                                count += 1
                                if count == ratio:
                                    break
                        assert(count == ratio)
                                

                    matches = pd.DataFrame(index = treated_df.index[:bin_size], data = rem_matched, columns = ['match{}'.format(i) for i in range(ratio)])
                    matches = pd.wide_to_long(matches.reset_index(), stubnames = 'match', i = ix_name, j = 'match_num').sort_index().reset_index()
                    matches['res'] = True
                    matched = matched.append(matches)
                    treated_df = treated_df.drop(matches.person_id.unique())
                    non_treated_df = non_treated_df.drop(matches.match.unique())

                iter_ix+=1;

                pbar.update(len(matches[ix_name].unique()))
            pbar.close()
        except: 
            pbar.close()
            raise (sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2]) # error_type, error_instance, traceback = sys.exc_info()
        matched = non_treated_df_save.reset_index().merge(matched[[ix_name,'match_num','match']].rename({ix_name:'index_case','match':'pid'},axis=1), 
                         left_on = 'person_id', right_on = 'pid', how = 'inner').set_index('person_id')
        return matched

def match_patients_onedf(allpts, options, filter_col = '',return_split = True):
    allpts[match_col] = allpts[match_col].astype(bool)

    cohortpts = allpts[allpts[match_col]]
    controlpts = allpts[~allpts[match_col]]

    dxgroup = options['dxgroup']; comparison = options['comparison']; 
    ratio = options['ratio'];
    odir_main = options['odir_main']; 
    cols_match_cat = options['match_params']['cols_match_cat']; 
    cols_match_num = options['match_params']['cols_match_num'];
    if 'cols_match_vis' in options['match_params']: 
        cols_match_num = cols_match_num + options['match_params']['cols_match_vis']
    scale_features = options['match_params']['match_scale_features'];
    w_replacement = options['match_params']['match_with_replacement'];
    w_pscore = options['match_params']['match_with_pscore']

    return match_patients(cohortpts, controlpts, cols_match_cat, cols_match_num, ratio, 
            scale_features, w_replacement, return_split = return_split, pscore = w_pscore, odir = odir_main)

    pass

def match_patients(cohortpts, controlpts, dxgroup = '', cols_match_cat = [], cols_match_num = [], ratio = 1, 
                   scale = False, replacement = False, return_split = True, pscore = False, th = .05, odir = None):
    """
    cohortpts, controlpts, matched = match_patients(cohortpts, controlpts, dxgroup, cols_match_cat = [], cols_match_num = [], ratio = 1, 
                   scaler = False, replacement = False, return_split = True)
    OR returns cohortpts.append(controlpts) with dxgroup column if return_split = False.
    
    inputs:
        cohortpts: table of patients with columns: person_id, cols_match_num and cols_match_cat
        controlpts: table of patients for controls, with columns: person_id, cols_match_num and cols_match_cat
        dxgroup: (default = '') column of cohortpts to subset on. 'AD','PSP','svPPA','bvFTD', for example. 
        cols_match_cat (default = []): list of categorical columns to match on.
        cols_match_num (default = []): list of numerical column to match on. datetime columns are ok, will be converted to ordinal value. 
        scale: (default = False) whether to use StandardScaler() to scale the columns.
        ratio: (default = 1) ratio for matches.
        replacement: (default = False) whether to match with replacement in non_treated_df. False means you want unique matches.
        return_split: (default = True) if true, will return 3 objects (cohortpts_out, controlpts_out, matched)
            if false, will return a single object with a column indicating group membership.
        pscore: (default = False) whether to compute propensity score for matching, or match using all features in nearest neighbor algorithm. 
        th: (default = .05) th is the threshold to maintain columns (removes sparse columns).
        odir: (default = None). If not none, will save propensity score figures in this directory.
    outputs: 
        cohortpts, controlpts
        matched: subset of non_treated_df that are matches of inputs. 
            the 'index_case' column indicates the index of who it is matched to in treated_df.
    """
    if dxgroup == '': dxgroup = 'cohort'
    if (len(cols_match_cat) == 0) and (len(cols_match_num) == 0): 
        raise Exception('No covariates indicated.')
    
    if dxgroup in cohortpts.columns: 
        cohort_to_match = cohortpts[cohortpts[dxgroup] == 1]
    else: cohort_to_match = cohortpts

    cohort_to_match =  cohort_to_match.set_index('person_id')[cols_match_cat+cols_match_num]
    background_pts = controlpts.set_index('person_id')[cols_match_cat+cols_match_num]
    cohort_to_match['cohort'] = 1
    background_pts['cohort'] = 0
    all_pts = cohort_to_match.append(background_pts)

    all_pts = pd.concat((all_pts[cols_match_cat].astype('category').apply(lambda x: x.cat.codes, axis=0),
                          all_pts[cols_match_num+['cohort']]), axis=1)

    date_cols = all_pts.select_dtypes(include=[np.datetime64]).columns

    for col in date_cols:
        all_pts[col] = all_pts[col].apply(lambda x: np.nan if pd.isna(x) else np.around(x.toordinal(), -3))
        all_pts[col] = all_pts[col].fillna(all_pts[col].min())
        all_pts[col] -= all_pts[col].min()
        # Ordinal value of Earliest Datetime 0001-01-01 00:00:00 is 1. of Latemost Datetime 9999-12-31 23:59:59 is 3652059. 
    all_pts = all_pts.fillna(0)
    
    if pscore:
        X = all_pts.drop('cohort',axis=1)
        X = RobustScaler().fit_transform(X)
        lr = LogisticRegression().fit(X, all_pts['cohort'])
        all_pts['pscore'] = lr.predict_proba(X)[:,1]
        all_pts = all_pts[['cohort','pscore']]

    all_pts = all_pts.groupby('cohort')
    
    treated_df = all_pts.get_group(1).drop('cohort',axis=1)
    non_treated_df = all_pts.get_group(0).drop('cohort',axis=1)
    
    matched = get_matching_pairs(treated_df, non_treated_df, scale = scale, ratio = ratio, replacement = replacement)
    
    cohortpts_out = cohortpts[cohortpts[dxgroup]==1].copy()
    controlpts_out = controlpts.merge(matched[['index_case']].reset_index(), on ='person_id', how = 'inner').copy()

    if pscore: 
        cohortpts_out = cohortpts_out.merge(treated_df['pscore'].reset_index(), on = 'person_id', how = 'left')
        controlpts_out = controlpts_out.merge(non_treated_df['pscore'].reset_index(), on = 'person_id', how = 'left')
        sns.kdeplot(cohortpts_out.pscore, label = 'case', common_norm = True, bw_adjust=.2)
        sns.kdeplot(controlpts_out.pscore, label = 'control', common_norm = True, bw_adjust=.2)
        plt.legend(); 
        if odir is not None: plt.savefig(odir + 'pscore_plot.png', bbox_inches = 'tight', format = 'png')
        else: plt.show();

    cohortpts_out['index_case'] = cohortpts_out['person_id']
    cohortpts_out[dxgroup] = 1
    controlpts_out[dxgroup] = 0
    
    for col in date_cols:
        cohortpts_out[col] = cohortpts_out[col].fillna(cohortpts_out[col].min()).dt.strftime('%Y%M%d')
        controlpts_out[col] = controlpts_out[col].fillna(controlpts_out[col].min()).dt.strftime('%Y%M%d')
        
    if return_split:
        return cohortpts_out, controlpts_out, matched
    else:
        return cohortpts_out.append(controlpts_out), matched