import pandas as pd
import os
import numpy as np
import joblib
from tableone import TableOne
from datetime import date

def age_visit_timefilt(cohortpts, controlpts, all_visits, timefilt):
    print('get age, number previous visits, and duration in EMR')
    cohortpts['date_age'] = cohortpts['mindatept_age'] + timefilt/365
    controlpts['date_age'] = controlpts['mindatept_age'] + timefilt/365

    cohortpts['date_oi'] = cohortpts['mindatept'] + np.timedelta64(timefilt, 'D')
    controlpts['date_oi'] = controlpts['mindatept'] + np.timedelta64(timefilt, 'D')

    cohortpts['yrs_in_ehr'] = ((cohortpts['date_oi'] - cohortpts['min_date']) / np.timedelta64(1,'Y')).round(1).fillna(0)
    cohortpts.loc[cohortpts['yrs_in_ehr'] < 0, 'yrs_in_ehr'] = 0
    controlpts['yrs_in_ehr'] = ((controlpts['date_oi'] - controlpts['min_date']) / np.timedelta64(1,'Y')).round(1).fillna(0)
    controlpts.loc[controlpts['yrs_in_ehr'] < 0, 'yrs_in_ehr'] = 0

    visitstemp = all_visits.merge(cohortpts[['person_id','date_oi']], on = 'person_id', how = 'inner', validate='m:1')
    visitstemp = visitstemp[visitstemp.visit_start_date <= visitstemp.date_oi].groupby('person_id')['visit_start_date'].nunique().rename('n_prev_visits').reset_index()
    if 'n_prev_visits' in cohortpts: cohortpts.drop('n_prev_visits', axis=1, inplace = True)
    cohortpts = cohortpts.merge(visitstemp, on = 'person_id', how = 'left', validate = '1:1')
    cohortpts['n_prev_visits'] = cohortpts['n_prev_visits'].fillna(0)

    visitstemp = all_visits.merge(controlpts[['person_id','date_oi']], on = 'person_id', how = 'inner', validate='m:1')
    visitstemp = visitstemp[visitstemp.visit_start_date <= visitstemp.date_oi].groupby('person_id')['visit_start_date'].nunique().rename('n_prev_visits').reset_index()
    if 'n_prev_visits' in controlpts: controlpts.drop('n_prev_visits', axis=1, inplace = True)
    controlpts = controlpts.merge(visitstemp, on = 'person_id', how = 'left', validate = '1:1')
    controlpts['n_prev_visits'] = controlpts['n_prev_visits'].fillna(0)
    
    return cohortpts, controlpts
    
def read_in_patients(options):
    """
    cohortpts, controlpts, all_visits = read_in_patients(options)
    inputs: options, a dictionary with fields: 
                input_dir (input directory)
    outputs: cohortpts, controlpts, all_visits
    """
    pdir = options['input_dir']
    ref_year = date.today().year;
    
    cohortpts = pd.read_csv(pdir + 'cohort.csv')
    controlpts = pd.read_csv(pdir + 'control.csv')
    
    cohortdemo = pd.read_csv(pdir + 'cohort_demographic.csv', index_col = 0)
    cohortpts = cohortdemo.merge(cohortpts[['person_id','mindatept']])

    cohortpts['Sex'] = cohortpts['Sex'].replace({r"\*":"", r"(^$|Deleted|Unspecified|Not Applicable)":"Unknown"}, regex = True).fillna('Unknown')
    cohortpts['RaceEthnicity'] = cohortpts['RaceEthnicity'].replace({r"(\*|/Declined)":"", 
                                r"(^$|Deleted|Unspecified|Not Applicable)":"Unknown", 
                                r"(Multi-Race/Ethnicity)": "Other"}, regex = True).fillna('Unknown')

    controlpts['Sex'] = controlpts['Sex'].replace({r"\*":"", r"(^$|Deleted|Unspecified|Not Applicable)":"Unknown"}, 
                                            regex = True).fillna('Unknown')
    controlpts['RaceEthnicity'] = controlpts['RaceEthnicity'].replace({r"(\*|/Declined)":"", 
                               r"(^$|Deleted|Unspecified|Not Applicable)":"Unknown", 
                                r"(Multi-Race/Ethnicity)": "Other"}, regex = True).fillna('Unknown')
    
    # Processing times and ages
    for x in ['mindatept','min_date','max_date','birth_datetime']: 
        cohortpts[x] = pd.to_datetime(cohortpts[x]); controlpts[x] = pd.to_datetime(controlpts[x]);
    controlpts['mindatept'] = controlpts['max_date'] - np.timedelta64(1, 'Y')

    for x in ['mindatept', 'min_date','max_date']: 
        cohortpts[x+'_age'] = ((cohortpts[x] - cohortpts['birth_datetime'])/ np.timedelta64(1, 'Y')).round(1)
        controlpts[x+'_age'] = ((controlpts[x] - controlpts['birth_datetime'])/ np.timedelta64(1, 'Y')).round(1)
        controlpts.loc[controlpts[x+'_age'].isna(),x+'_age'] = \
            ((controlpts[x] - pd.to_datetime('19310101'))/ np.timedelta64(1, 'Y')).round(1) # assume 1931 birthdate if get NAN
        cohortpts.loc[cohortpts[x+'_age'] <=0,x+'_age'] = 0
        controlpts.loc[controlpts[x+'_age'] <=0,x+'_age'] = 0

    assert cohortpts.person_id.value_counts().max()==1
    assert controlpts.person_id.value_counts().max()==1

    # read in visits and append
    print('reading in visit information, and adding to patients...')
    visits = pd.read_csv(pdir + '/visits.csv', index_col = 0)
    visits_c = pd.read_csv(pdir + '/controlvisits.csv', index_col = 0)

    visits['visit_start_date'] = pd.to_datetime(visits['visit_start_date'])
    visits_c['visit_start_date'] = pd.to_datetime(visits_c['visit_start_date'])
    all_visits = visits.append(visits_c)
    
    return cohortpts, controlpts, all_visits

def filter_pts(cohortpts, controlpts):
    # max date < min date
    print('remove if max_date < min_date')
    cohortmaskout = (cohortpts['max_date'] - cohortpts['min_date']).dt.days < 0
    controlmaskout = (controlpts['max_date'] - controlpts['min_date']).dt.days < 0
    print('\tremove {} cases, {} controls'.format(cohortmaskout.sum(), controlmaskout.sum()))
    cohortpts = cohortpts[~cohortmaskout]
    controlpts = controlpts[~controlmaskout]

    print('remove if mindatept before min_date')
    cohortmaskout = (cohortpts['mindatept'] < cohortpts['min_date'])
    controlmaskout = (controlpts['mindatept'] < controlpts['min_date'])
    print('\tremove {} cases'.format(cohortmaskout.sum()))
    print('\tremove {} controls'.format(controlmaskout.sum()))
    cohortpts = cohortpts[~cohortmaskout]
    controlpts = controlpts[~controlmaskout]
    
    return cohortpts, controlpts

### DISPLAYING AND PLOTTING FUNCTIONS
def display_table(pts, groupby, options, splitby = None):
    cols_match_cat = options['match_params']['cols_match_cat']; 
    cols_match_num = options['match_params']['cols_match_num'];
    if splitby is not None: 
        for xx in pts[splitby].unique():
            print(splitby, ':', xx)
            mytable = TableOne(pts[pts[splitby]==xx], columns = cols_match_num + cols_match_cat, 
                               groupby=groupby, categorical = cols_match_cat, smd = True, pval = True);
            display(mytable)
    else: 
        mytable = TableOne(pts, columns = cols_match_num + cols_match_cat, 
                           groupby=groupby, categorical = cols_match_cat, smd = True, 
                           pval = True);
        display(mytable)
        
def display_train_test_split(cohortpts, controlpts, person_id_train, person_id_test, options):
    temp = cohortpts.append(controlpts).set_index('person_id').reindex(np.concatenate((person_id_train,person_id_test)))
    temp.loc[person_id_train,'train']=1; temp.loc[person_id_test,'train']=0;
    display_table(temp, 'train', options, splitby = 'AD')
    
