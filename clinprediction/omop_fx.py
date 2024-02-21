import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import seaborn as sns
import time
import joblib
import matplotlib.pyplot as plt

class OMOPData():
    def __init__(self, omopdir):
        self.omopdir = omopdir
        self.tables = list()
        self.concepts = self.read_in_concepts(omopdir)
        self.concept_ancestor, self.concept_relationship = self.read_in_concept_relations(omopdir)
        
        self.patient_dir = None
        self.conditions = None
        self.drugs = None
        self.measures = None
        # self.measures_abnormal = None - 
        
    def read_in_concepts(self, cdir):
        concepts = pd.read_csv(cdir + 'concept.csv', index_col = 0, low_memory = False)
        concepts = concepts[['concept_id','concept_name','domain_id']].drop_duplicates()
        concepts.concept_name = concepts.concept_name.str.lower()
        return concepts
    
    def read_in_concept_relations(self, cdir):
        concept_relation = pd.read_csv(cdir + 'concept_relationship.csv', index_col = 0)
        concept_ancestor = pd.read_csv(cdir + 'concept_ancestor.csv', index_col = 0)
        return concept_ancestor, concept_relation
        
    def read_in_omop_csv(self, directory, read_in_controls = False):
        self.patient_dir = directory
        print('reading in data...')
        conditions = pd.read_csv(directory + 'conditions.csv', low_memory = False)
        measures = pd.read_csv(directory + 'measures.csv', low_memory = False)
        drugs = pd.read_csv(directory + 'drugs.csv', low_memory = False)
        print('case sizes: conditions: {}, drugs: {}, measures: {}'\
              .format(conditions.shape, drugs.shape, measures.shape))
        print('n_casept: '+ str(len(set(conditions.person_id).union(set(drugs.person_id).union(measures.person_id)))))
        
        if read_in_controls:
            conditions['iscontrol'] = 0; measures['iscontrol'] = 0; 
            drugs['iscontrol'] = 0;
            
            print('reading in control data...')
            conditions_c = pd.read_csv(directory + 'controlconditions.csv', low_memory = False)
            measures_c = pd.read_csv(directory + 'controlmeasures.csv', low_memory = False)
            drugs_c = pd.read_csv(directory + 'controldrugs.csv', low_memory = False)
            conditions_c['iscontrol'] = 1; measures_c['iscontrol'] = 1; 
            drugs_c['iscontrol'] = 1;
            print('controlfile sizes: conditions: {}, drugs: {}, measures: {}'\
              .format(conditions_c.shape, drugs_c.shape, measures_c.shape))
            print('n_controlfilept: ' + str(len(set(conditions_c.person_id).union(set(drugs_c.person_id).union(measures_c.person_id)))))
            
            conditions = conditions.append(conditions_c).reset_index()
            measures = measures.append(measures_c).reset_index()
            drugs = drugs.append(drugs_c).reset_index()

        self.conditions = conditions
        self.drugs = drugs
        self.measures = measures
        print('full size of data')
        self.size_of_data()
        return;
        
    def size_of_data(self):
        print('conditions: {}, drugs: {}'.format(self.conditions.shape, self.drugs.shape))
        print('measures: {}'.format(self.measures.shape))
        return;
    
    def size_of_data_col(self, col):
        print('conditions:\n{}'.format(self.conditions[col].value_counts()))
        print('drugs:\n{}'.format(self.drugs[col].value_counts()))
        print('measures\n{}'.format(self.measures[col].value_counts()))
        return;

    def load_compressed(self, pdir):
        self.patient_dir = pdir
        infile = self.patient_dir+'omop_cache.joblib'
        print('reading in {}'.format(infile))
        omopin = joblib.load(infile)
        self.conditions = omopin['conditions']
        self.drugs = omopin['drugs']
        self.measures = omopin['measures']
        del omopin

    def save_compressed(self):
        self.conditions = self.conditions[['person_id','condition_concept_id','condition_start_date',
                     'condition_concept_id_value', 'iscontrol']]
        self.drugs = self.drugs[['person_id','drug_concept_id','drug_exposure_start_date','iscontrol']]
        self.measures = self.measures[['person_id','measurement_date','measurement_concept_id', 'value_as_number',
                          'range_low','range_high','measurement_source_value','iscontrol']]
        
        outfile = self.patient_dir+'omop_cache.joblib'
        print('saving into {}'.format(outfile))
        joblib.dump({'conditions': self.conditions, 
                    'drugs': self.drugs,
                    'measures': self.measures}
                    ,outfile)

def process_abnormal_measures(measures, binary_map):   
    measures['Abnormal_omop'] = measures['value_as_number'] <= measures['range_low']
    measures['below_range_omop'] = measures['Abnormal_omop'].copy().astype(int)
    measures['Abnormal_omop'] = measures['Abnormal_omop']|(measures['value_as_number'] >= measures['range_high'])
    measures['above_range_omop'] = (measures['value_as_number'] >= measures['range_high']).astype(int)
    
    measures_all = measures[~measures['measurement_date'].isna()]
    if binary_map is not None: 
        measures_all['value_binary'] = measures_all['value_source_value'].map(binary_map)
        measures_ab = measures_all[(measures_all['Abnormal_omop']==1)|(measures_all['value_binary']==1)]
    else: 
        measures_ab = measures_all[(measures_all['Abnormal_omop']==1)]
    
    return measures_ab

def pivot_omop(allptomop, pts, feats = None):
    if(len(pts) < 100000): 
        allptomop_pivot = pd.pivot_table(allptomop[allptomop.person_id.isin(pts)], 
                                           index = 'person_id', columns = 'concept_id', 
                   values = 'domain_id', aggfunc = lambda x: 1, fill_value = 0)
    else: 
        allptomop_pivot_list = list()
        for i in np.arange(np.ceil(len(pts)/100000)): 
            allptomop_pivot_temp = pd.pivot_table(\
                             allptomop[allptomop.person_id.isin(\
                                    pts[ 100000*int(i) : int(np.minimum(len(pts),100000*(i+1))) ])], 
                             index = 'person_id', columns = 'concept_id', 
                             values = 'domain_id', aggfunc = lambda x: 1, fill_value = 0)
            allptomop_pivot_list.append(allptomop_pivot_temp)
        allptomop_pivot = pd.concat(allptomop_pivot_list).fillna(0)
        del allptomop_pivot_list
        
    if feats is not None:
        feat_pivot = allptomop_pivot.columns
        feat_missing = np.setdiff1d(feats, feat_pivot)
        allptomop_pivot[feat_missing] = 0
        allptomop_pivot = allptomop_pivot[feats]
        
    return allptomop_pivot

def filter_omopdata_by_time(omopdata, pts, timefilt = 0, use_inputs_index0 = False):
    conditions, drugs, measures = omopdata.conditions, omopdata.drugs, omopdata.measures 

    conditions = conditions[conditions.person_id.isin(pts.person_id) & (conditions.condition_concept_id != 0)]
    drugs = drugs[drugs.person_id.isin(pts.person_id) & (drugs.drug_concept_id != 0)]
    measures = measures[measures.person_id.isin(pts.person_id) & (measures.measurement_concept_id != 0)]
                    
    (conditions, drugs, measures) = \
        filter_omop_by_time(conditions, drugs, measures, pts, 
                    timefilt = timefilt, use_inputs_index0 = use_inputs_index0)
    
    return (conditions, drugs, measures)

def filter_omop_by_time(conditions, drugs, measures, pts, 
                        timefilt = 0, use_inputs_index0 = False):
    """
    (conditions, drugs, measures) = 
                        filter_omop_by_time(conditions, drugs, measures, 
                        pts, timefilt = 0, use_inputs_index0 = False)
    Inputs: conditions - conditions table. date column: condition_start_date
            drugs - drug table. date column: drug_exposure_start_date
            measures - measures table. date column: measurement_date
            pts - list of patients, of interest is the column: mindatept (as index 0)
                        inner join in the list of patients. 
            timefilt - maximum difference between concept date minus index 0. 
                        negative numbers indicate 'before index 0', positive numbers indicate after.
                        (default = 0)
            use_inputs_index_0 (boolean. default = 0)
                    if true, then use the input tables to determine latest date for each patient as index 0
    Outputs: a tuple with filtered (conditions, drugs, measures) 
    """
    # Rename start date. WARNING: COLUMN CAN CHANGE
    if 'start_date' not in conditions.columns: conditions['start_date'] = pd.to_datetime(conditions['condition_start_date'])
    if 'start_date' not in drugs.columns: drugs['start_date'] = pd.to_datetime(drugs['drug_exposure_start_date'])
    if 'start_date' not in measures.columns: measures['start_date'] =  pd.to_datetime(measures['measurement_date'])

    emr_dict = {'conditions': conditions, 'drugs': drugs, 'measures':measures}
    if use_inputs_index0:
        if 'max_date' in pts.columns: 
            pts['max_date'] = pd.to_datetime(pts['max_date'])
            person_minmax = pd.concat(
                    [conditions[['person_id','start_date']].groupby('person_id')['start_date'].agg(['min','max']),
                    drugs[['person_id','start_date']].groupby('person_id')['start_date'].agg(['min','max']),
                    measures[['person_id','start_date']].groupby('person_id')['start_date'].agg(['min','max']),
                    pts[['person_id','max_date']].groupby('person_id')['max_date'].agg(['min','max'])]
            )
        else: 
            person_minmax = pd.concat(
                    [conditions[['person_id','start_date']].groupby('person_id')['start_date'].agg(['min','max']),
                    drugs[['person_id','start_date']].groupby('person_id')['start_date'].agg(['min','max']),
                    measures[['person_id','start_date']].groupby('person_id')['start_date'].agg(['min','max'])]
            )
        pts = person_minmax.groupby('person_id').agg({'min':'min','max':'max'})
        pts = pts.rename({'max':'mindatept'},axis=1).reset_index()
        
    for x in emr_dict.keys(): # for each conditions, drugs, measures
        if 'mindatept' in emr_dict[x].columns: emr_dict[x] = emr_dict[x].drop('mindatept',axis=1)
        emr_dict[x] = emr_dict[x].merge(pts[['person_id','mindatept']], on = 'person_id', how = 'inner')
        emr_dict[x]['mindatept'] = pd.to_datetime(emr_dict[x]['mindatept'], errors = 'coerce')
        emr_dict[x] = emr_dict[x][~emr_dict[x].mindatept.isna()]
        emr_dict[x]['mindatept'] = pd.to_datetime(emr_dict[x]['mindatept'])
        emr_dict[x]['datediff'] = emr_dict[x]['start_date'] - emr_dict[x]['mindatept']

    conditions, drugs, measures = emr_dict.values()

    if timefilt is not None: 
        conditions = conditions[conditions['datediff'].dt.days <= timefilt]
        drugs = drugs[drugs['datediff'].dt.days <= timefilt]
        measures = measures[measures['datediff'].dt.days <= timefilt]
        
    return (conditions, drugs, measures)

