import pyodbc
import pandas as pd
import numpy as np
import os

## DATABASE RELATED FUNCTIONS

# Code for extracting OMOP concept ID information
def conceptToName(conceptIdList):
    conceptlist = conceptIdList[~conceptIdList.isna()]
    if conceptlist.dtypes == float:
        conceptlist = conceptlist.astype(int)
    conceptlist = conceptlist.unique().astype(str)
    conceptlist = conceptlist[((conceptlist!='None') & (conceptlist!='nan'))]
    
    sqlstring = """SELECT * 
                FROM concept
                WHERE concept_id in ('{}')
            """.format("','".join(conceptlist))
    return pd.read_sql_query(sqlstring,con)

def displayConceptNamesInDF(dataframe):
    for col in dataframe.columns[dataframe.columns.str.endswith('concept_id')]:
        print(col)
        display(conceptToName(dataframe[col]))
        
def conceptToNameMap(conceptIdList): # you can generally use this. 
    return conceptToName(conceptIdList)[['concept_id','concept_name']]\
        .set_index('concept_id').to_dict()['concept_name']

def conceptToXMap(conceptIdList, Xname): # X includes domain_id, vocabulary_id, concept_class_id
    return conceptToName(conceptIdList)[['concept_id',Xname]]\
        .set_index('concept_id').to_dict()[Xname]


# IDENTIFY RELATED AND CHILDREN/GRANDCHILDREN OMOP CONCEPTS
def sqlstring_rd_condition(query):
    sqlstring = """DECLARE @dz VARCHAR(100)
            SET @dz = '%{}%'

            (SELECT DISTINCT cr.concept_id_1, cc.concept_name, cc.vocabulary_id, cc.concept_code 
                                   ,cr.concept_id_2, cc2.concept_name as concept_name2, 
                                   cc2.vocabulary_id as vocabulary_id2, cc2.concept_code as concept_code2 
                                   ,cr.relationship_id 
                            FROM [OMOP_DEID].[omop].[concept_relationship] cr 
                            LEFT JOIN [OMOP_DEID].[omop].[concept] cc 
                              on cr.concept_id_1 = cc.concept_id 
                            LEFT JOIN [OMOP_DEID].[omop].[concept] cc2 
                              on cr.concept_id_2 = cc2.concept_id 
                            inner join omop_deid.omop.condition_occurrence CO
                                on cr.concept_id_2 = CO.condition_concept_id
                            WHERE cc.concept_name like @dz and cr.relationship_id IN ('Mapped from','Subsumes','Concept same_as to')
            )
            UNION
            (SELECT ca.ancestor_concept_id as concept_id_1, cc.concept_name, cc.vocabulary_id, cc.concept_code 
                         ,ca.descendant_concept_id as concept_id_2, cc2.concept_name as concept_name2, 
                         cc2.vocabulary_id as vocabulary_id2, cc2.concept_code as concept_code2 
                         ,CONCAT('ancestor:', ca.min_levels_of_separation,'-',ca.max_levels_of_separation) as relationship_id
                    FROM [OMOP_DEID].[omop].[concept_ancestor] ca 
                    LEFT JOIN [OMOP_DEID].[omop].[concept] cc 
                        on ca.ancestor_concept_id = cc.concept_id 
                    LEFT JOIN [OMOP_DEID].[omop].[concept] cc2 
                        on ca.descendant_concept_id = cc2.concept_id 
                    INNER JOIN (select DISTINCT condition_concept_id FROM [OMOP_DEID].[omop].[condition_occurrence]) CO 
                        on ca.descendant_concept_id = CO.condition_concept_id 
                     WHERE cc.concept_name like @dz 
            )
            """.format(query)
    return sqlstring

def get_string_condition_related_descendants(query_string):
    temp = pd.read_sql_query(sqlstring_rd_condition(query_string), con)
    tempsame = pd.concat((temp[['concept_id_1','concept_name','vocabulary_id','concept_code']], 
           temp[['concept_id_1','concept_name','vocabulary_id','concept_code']].rename({'concept_id_1':'concept_id_2',
                'concept_name':'concept_name2', 'vocabulary_id':'vocabulary_id2','concept_code':'concept_code2'}, axis=1))
          ,axis=1)
    tempsame['relationship_id'] = 'same'
    temp = temp.append(tempsame)
    temp = temp.groupby(['concept_id_1','concept_id_2']).last().reset_index().sort_values('concept_name')
    return temp

def get_ATC_reldesc(qstring):
    print('Inserting temporary table')
    curr = con.cursor()
    sql = """
            IF OBJECT_ID(N'tempdb..#ccrelation') IS NOT NULL
            BEGIN
            DROP TABLE #ccrelation
            END

            SELECT ca.ancestor_concept_id as concept_id_1, cc.concept_name, cc.vocabulary_id, cc.concept_code 
                         ,ca.descendant_concept_id as concept_id_2, cc2.concept_name as concept_name2, 
                         cc2.vocabulary_id as vocabulary_id2, cc2.concept_code as concept_code2 
                         ,CONCAT('ancestor:', ca.min_levels_of_separation,'-',ca.max_levels_of_separation) as relationship_id
                    INTO #ccrelation
                    FROM [OMOP_DEID].[omop].[concept_ancestor] ca 
                    LEFT JOIN [OMOP_DEID].[omop].[concept] cc 
                        on ca.ancestor_concept_id = cc.concept_id 
                    LEFT JOIN [OMOP_DEID].[omop].[concept] cc2 
                        on ca.descendant_concept_id = cc2.concept_id 
                    INNER JOIN (select DISTINCT drug_concept_id FROM [OMOP_DEID].[omop].[drug_exposure]) DE 
                        on ca.descendant_concept_id = DE.drug_concept_id 
                    WHERE cc.concept_code like '{}'
            """.format(qstring)
    curr.execute(sql)
    
    print('Performing ultimate mappings')
    sqlstring = """
            (
            SELECT  cci.concept_id_1 as sc, cci.concept_name as scname, cci.concept_code as sccode, cci.vocabulary_id as scvocab, 
                    cci.concept_id_2 as mapped, cci.concept_name2 as mapped_name, cci.vocabulary_id2 as mapped_vocab,
                    cci.concept_code2 as mapped_code, cci.relationship_id as map_relation,
                    ca.descendant_concept_id as ultconcept, cc3.concept_name as ult_name, cc3.vocabulary_id as ult_vocab, 
                    cc3.concept_code as ult_code, CONCAT('ancestor:', ca.min_levels_of_separation,'-',ca.max_levels_of_separation) as ult_relation
            FROM #ccrelation cci
            LEFT JOIN [OMOP_DEID].[omop].[concept_ancestor] ca 
                ON ca.ancestor_concept_id = cci.concept_id_2
            LEFT JOIN [OMOP_DEID].[omop].[concept] cc3
                ON ca.descendant_concept_id = cc3.concept_id 
            INNER JOIN (SELECT DISTINCT drug_concept_id FROM [OMOP_DEID].[omop].[drug_exposure]) DE
                ON ca.descendant_concept_id = DE.drug_concept_id 
            )
            UNION
            (
            SELECT DISTINCT cci.concept_id_1 as sc, cci.concept_name as scname, cci.concept_code as sccode, cci.vocabulary_id as scvocab, 
                        cci.concept_id_2 as mapped, cci.concept_name2 as mapped_name, cci.vocabulary_id2 as mapped_vocab,
                        cci.concept_code2 as mapped_code, cci.relationship_id as map_relation,
                        ccr2.concept_id_2 as ultconcept, cc3.concept_name as ult_name, cc3.vocabulary_id as ult_vocab,
                        cc3.concept_code as ult_code, ccr2.relationship_id as ult_relation
            FROM #ccrelation as cci
            LEFT JOIN [OMOP_DEID].[omop].[concept_relationship] ccr2
                ON cci.concept_id_2 = ccr2.concept_id_1
            LEFT JOIN [OMOP_DEID].[omop].[concept] cc3
                ON ccr2.concept_id_2 = cc3.concept_id
            INNER JOIN (SELECT DISTINCT drug_concept_id FROM [OMOP_DEID].[omop].[drug_exposure]) DE
                ON DE.drug_concept_id = cc3.concept_id
            )
            """

    return pd.read_sql_query(sqlstring, con)

def get_ATC_related_descendants(query_string):
    temp = get_ATC_reldesc(query_string)
    temp = temp.groupby(['mapped','ultconcept']).last().reset_index().sort_values('sccode')
    return temp

def get_icd10_reldesc(qstring):
    print('Inserting temporary table')
    curr = con.cursor()
    sql = """
            IF OBJECT_ID(N'tempdb..##ccrelation') IS NOT NULL
            BEGIN
            DROP TABLE ##ccrelation
            END

            SELECT cr.concept_id_1, cc.concept_name, cc.vocabulary_id, cc.concept_code ,cr.concept_id_2, cc2.concept_name as concept_name2, 
                 cc2.vocabulary_id as vocabulary_id2, cc2.concept_code as concept_code2,cr.relationship_id
            INTO ##ccrelation
            FROM [OMOP_DEID].[omop].[concept_relationship] cr
            LEFT JOIN [OMOP_DEID].[omop].[concept] cc
                ON cr.concept_id_1 = cc.concept_id
            LEFT JOIN [OMOP_DEID].omop.[concept] cc2
                ON cr.concept_id_2 = cc2.concept_id
            WHERE(cc.concept_code LIKE '{}' AND cc.vocabulary_id LIKE '%ICD10CM%')
                AND cr.relationship_id IN ('Subsumes','Is a', 'Maps to')""".format(qstring)
    curr.execute(sql)
    
    print('Performing ultimate mappings')
    sqlstring = """
            (
            SELECT  cci.concept_id_1 as sc, cci.concept_name as scname, cci.concept_code as sccode, cci.vocabulary_id as scvocab, 
                    cci.concept_id_2 as mapped, cci.concept_name2 as mapped_name, cci.vocabulary_id2 as mapped_vocab,
                    cci.concept_code2 as mapped_code, cci.relationship_id as map_relation,
                    ca.descendant_concept_id as ultconcept, cc3.concept_name as ult_name, cc3.vocabulary_id as ult_vocab, 
                    cc3.concept_code as ult_code, CONCAT('ancestor:', ca.min_levels_of_separation,'-',ca.max_levels_of_separation) as ult_relation
            FROM ##ccrelation cci
            LEFT JOIN [OMOP_DEID].[omop].[concept_ancestor] ca 
                ON ca.ancestor_concept_id = cci.concept_id_2
            LEFT JOIN [OMOP_DEID].[omop].[concept] cc3
                ON ca.descendant_concept_id = cc3.concept_id 
            INNER JOIN (SELECT DISTINCT condition_concept_id FROM [OMOP_DEID].[omop].[condition_occurrence]) CO 
                ON ca.descendant_concept_id = CO.condition_concept_id 
            )
            UNION
            (
            SELECT DISTINCT cci.concept_id_1 as sc, cci.concept_name as scname, cci.concept_code as sccode, cci.vocabulary_id as scvocab, 
                        cci.concept_id_2 as mapped, cci.concept_name2 as mapped_name, cci.vocabulary_id2 as mapped_vocab,
                        cci.concept_code2 as mapped_code, cci.relationship_id as map_relation,
                        ccr2.concept_id_2 as ultconcept, cc3.concept_name as ult_name, cc3.vocabulary_id as ult_vocab,
                        cc3.concept_code as ult_code, ccr2.relationship_id as ult_relation
            FROM ##ccrelation as cci
            LEFT JOIN [OMOP_DEID].[omop].[concept_relationship] ccr2
                ON cci.concept_id_2 = ccr2.concept_id_1
            LEFT JOIN [OMOP_DEID].[omop].[concept] cc3
                ON ccr2.concept_id_2 = cc3.concept_id
            INNER JOIN (SELECT DISTINCT condition_concept_id FROM [OMOP_DEID].[omop].[condition_occurrence]) CO 
                ON CO.condition_concept_id = cc3.concept_id
            WHERE ccr2.relationship_id IN ('Subsumes','Maps to','Mapped from')
            )"""

    return pd.read_sql_query(sqlstring, con)

def get_icd10_related_descendants(query_string):
    temp = get_icd10_reldesc(query_string)
    tempsame = pd.concat((temp[['sc', 'scname', 'sccode', 'scvocab', 'mapped', 'mapped_name',
                               'mapped_vocab', 'mapped_code', 'map_relation']], 
                          temp[['mapped', 'mapped_name', 'mapped_vocab', 'mapped_code']].rename({'mapped':'ultconcept',
                               'mapped_name':'ult_name', 'mapped_vocab':'ult_vocab','mapped_code':'ult_code'}, axis=1))
                          ,axis=1)
    tempsame['ult_relation'] = 'same'
    temp = temp.append(tempsame)
    temp = temp.groupby(['mapped','ultconcept']).last().reset_index().sort_values('sccode')
    return temp

def ICD10_code_to_chapter(let):
    if let == 'nan':
        return 'NaN';
    elif let[0] == 'A' or let[0] == 'B':
        return 'A00–B99';
    elif let[0] == 'C' or (let[0] == 'D' and int(let[1])>=0 and int(let[1])<5):
        return 'C00–D48';
    elif let[0] == 'D' and int(let[1])>=5 and int(let[1])<9:
        return 'D50–D89';
    elif let[0] == 'E':
        return 'E00–E90';
    elif let[0] == 'H' and int(let[1])>=0 and int(let[1])<6:
        return 'H00–H59';
    elif let[0] == 'H' and int(let[1])>=6 and int(let[1])<=9:
        return 'H60–H95';
    elif let[0] == 'K':
        return 'K00–K93';
    elif let[0] == 'P':
        return 'P00–P96';
    elif let[0] == 'S' or let[0] == 'T':
        return 'S00–T98';
    elif let[0] in ['V','W','X','Y']:
        return 'V01–Y98';
    elif let[0] in ['F', 'G','I', 'J', 'L', 'M', 'N', 'O','Q','R','Z','U']:
        return '{}00–{}99'.format(let[0], let[0]);
    else:
        return let;
    
def ICDchapter_to_name(chp):
    if chp == 'nan': return 'NaN';
    elif chp == 'A00–B99': return 'Certain infectious and parasitic diseases';
    elif chp == 'C00–D48': return 'Neoplasms';
    elif chp == 'D50–D89': return 'Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism';
    elif chp == 'E00–E90': return 'Endocrine, nutritional and metabolic diseases';
    elif chp == 'F00–F99': return 'Mental and behavioural disorders';
    elif chp == 'G00–G99': return 'Diseases of the nervous system';
    elif chp == 'H00–H59': return 'Diseases of the eye and adnexa';
    elif chp == 'H60–H95': return 'Diseases of the ear and mastoid process';
    elif chp == 'I00–I99': return 'Diseases of the circulatory system';
    elif chp == 'J00–J99': return 'Diseases of the respiratory system';
    elif chp == 'K00–K93': return 'Diseases of the digestive system';
    elif chp == 'L00–L99': return 'Diseases of the skin and subcutaneous tissue';
    elif chp == 'M00–M99': return 'Diseases of the musculoskeletal system and connective tissue';
    elif chp == 'N00–N99': return 'Diseases of the genitourinary system';
    elif chp == 'O00–O99': return 'Pregnancy, childbirth and the puerperium';
    elif chp == 'P00–P96': return 'Certain conditions originating in the perinatal period';
    elif chp == 'Q00–Q99': return 'Congenital malformations, deformations and chromosomal abnormalities';
    elif chp == 'R00–R99': return 'Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified';
    elif chp == 'S00–T98': return 'Injury, poisoning and certain other consequences of external causes';
    elif chp == 'V01–Y98': return 'External causes of morbidity and mortality';
    elif chp == 'Z00–Z99': return 'Factors influencing health status and contact with health services';
    elif chp == 'U00–U99': return 'Codes for special purposes';
    else: return ' ';