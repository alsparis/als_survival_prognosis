# -*- coding: utf-8 -*-
# 20/12/2019
# Reboot to select most elems

import pandas as pd
import os
pd.set_option('display.max_rows', 20)
import numpy as np

for file_i in ['output','output/reboot']:
    try:
        os.mkdir(file_i)
    except:
        pass

#############################################################################
### AAA MISSS
##############################################################################


df = pd.read_csv('input/orig/patient_db_xsmall.csv',sep=';')

df.loc[df.source=='trophos','subject_id'] = df.loc[df.source=='trophos','subject_id'].apply(lambda x: 'trophos_'+x.split('_')[2])

df.set_index('subject_id',inplace=True)

df = df.loc[df.diag == 'ALS',:]
df.drop(['diag'],axis=1,inplace=True)

input_feats = ['sex','onset_spinal','age','time_disease','weight_baseline','ALSFRS_Total_baseline']
outsurv_feats = ['survived','ALSFRS_Total_final','death_unit']
outfunc_feats = ['ALSFRS_Total_final','Kings_score_final','MiToS_score_final','FT9_score_final','computeZones_hand_final','computeZones_leg_final',
                 'computeZones_trunk_final','computeZones_respiratory_final',
                 'computeZones_mouth_final']
'''['ALSFRS_Total_final','computeZones_hand_final','computeZones_leg_final',
                 'computeZones_trunk_final','computeZones_respiratory_final',
                 'computeZones_mouth_final']
'''
outfunc2 = ['Q1_Speech_final', 'Q2_Salivation_final',
                 'Q3_Swallowing_final', 'Q4_Handwriting_final', 'Q5_Cutting_final',
                  'Q5_Indic_final', 'Q5a_Cutting_without_Gastrostomy_final',
                  'Q5b_Cutting_with_Gastrostomy_final', 
                  'Q6_Dressing_and_Hygiene_final', 'Q7_Turning_in_Bed_final', 
                  'Q8_Walking_final', 'Q9_Climbing_Stairs_final',
                 'Q10_Respiratory_final','computeZones_hand_baseline','computeZones_leg_baseline',
                    'computeZones_mouth_baseline','computeZones_respiratory_baseline',
                    'computeZones_trunk_baseline']

input_miss_feats = ['computeZones_hand_baseline','computeZones_leg_baseline',
                    'computeZones_mouth_baseline','computeZones_respiratory_baseline',
                    'computeZones_trunk_baseline','pulse_baseline','height_baseline',
                    'bp_diastolic_baseline','bp_systolic_baseline',
                    'vital_capacity_percent_baseline','vital_capacity_baseline']

func_input =  ['Q10_Respiratory_baseline','Q1_Speech_baseline',
               'Q2_Salivation_baseline','Q3_Swallowing_baseline',
               'Q4_Handwriting_baseline','Q5_Cutting_baseline',
               'Q5_Indic_baseline','Q5a_Cutting_without_Gastrostomy_baseline',
               'Q5b_Cutting_with_Gastrostomy_baseline','Q6_Dressing_and_Hygiene_baseline',
               'Q7_Turning_in_Bed_baseline','Q8_Walking_baseline','Q9_Climbing_Stairs_baseline',
               'FT9_score_baseline','Kings_score_baseline','MiToS_score_baseline']

def checkSurvival(x):
    
    if pd.isnull(x['death_unit']) and pd.isnull(x['ALSFRS_Total_final']):
        y = np.nan
    elif pd.isnull(x['survived']) and ~pd.isnull(x['death_unit']):
        y = 0
    elif pd.isnull(x['survived']) and pd.isnull(x['death_unit']) and ~pd.isnull(x['ALSFRS_Total_final']):
        y = 1
    elif x['death_unit']==0:
        y = np.nan
    elif x['death_unit']<=12.1:
        y = 0
    elif x['death_unit']>12.1 or ~pd.isnull(x['ALSFRS_Total_final']):
        y = 1
    else:
        y = np.nan
    
    return y

#df.loc[:,'survived']
df.loc[:,'survived'] = df.apply(checkSurvival,axis=1)
#df.loc[:,['survived','ALSFRS_Total_final','death_unit']].to_csv('output/reboot/survival_check.csv',sep=';')

# WEIGHT CORRECTION NOT WORKING....
weight_correction = ['proact_9641','proact_31375','proact_38943','proact_103844',
                     'proact_110295','proact_115339','proact_252105','proact_269236',
                     'proact_297730','proact_415036','proact_428134','proact_428697',
                     'proact_442938','proact_448367','proact_460569','proact_501903',
                     'proact_534978','proact_535253','proact_544580','proact_550185',
                     'proact_555305','proact_558710','proact_576604','proact_626271',
                     'proact_630774','proact_635902','proact_669350','proact_739556',
                     'proact_750059','proact_754271','proact_760076','proact_765708',
                     'proact_775977','proact_782768','proact_793356','proact_802266',
                     'proact_820475','proact_891160','proact_947716','proact_968720',
                     'proact_973533','proact_983807','proact_990402','proact_997090']

df.reset_index(inplace=True)
df.loc[df.subject_id.isin(weight_correction),'weight_baseline'] = df.loc[df.subject_id.isin(weight_correction),'weight_baseline']*0.45
df.set_index('subject_id',inplace=True)

def cleanStage(x):
    if pd.isna(x['FT9_score_final']):
        return np.nan,np.nan
    else:
        return x['MiToS_score_final'],x['Kings_score_final']


df_input = df.loc[:,input_feats]
df_outs = df.loc[:,outsurv_feats]
df_outf = df.loc[:,outfunc_feats]

df['tmp'] = df.apply(lambda x: cleanStage(x),axis=1)
df['MiToS_score_final'] = df.loc[:,'tmp'].apply(lambda x: x[0] )
df['Kings_score_final'] = df.loc[:,'tmp'].apply(lambda x: x[1] )

df_misss = df.loc[:,input_feats+input_miss_feats+['source']+outsurv_feats]
df_missf = df.loc[:,input_feats+input_miss_feats+['source']+outfunc_feats]



mask_pre2008s = (((df_misss.source=='pre2008') & (~df_misss.ALSFRS_Total_baseline.isnull() & (~df_misss.age.isnull())) & (~df_misss.time_disease.isnull())) | (df_misss.source!='pre2008'))
mask_pre2008f = (((df_missf.source=='pre2008') & (~df_missf.ALSFRS_Total_baseline.isnull() & (~df_missf.age.isnull())) & (~df_missf.time_disease.isnull())) | (df_missf.source!='pre2008'))

df_misss = df_misss.loc[mask_pre2008s,:]
df_missf = df_missf.loc[mask_pre2008f,:]

def missAnalysis(x):
    
    def missCompute(df):
        source_list =  ['proact','exonhit','pre2008','trophos']
        df_columns = list(df.columns)
        list_miss = []
        for col_i in df_columns:
            miss_ratio = df.loc[:,col_i].isna().sum()/df.shape[0]
            list_tmp = [col_i,miss_ratio]
            for source_i in source_list:
                miss_ratio = df.loc[df.source==source_i,col_i].isna().sum()/df.loc[df.source==source_i,:].shape[0]
                list_tmp.append(miss_ratio)
            list_miss.append(list_tmp)
        df_miss = pd.DataFrame(list_miss,columns=['feature','miss_ratio']+['miss_ratio_'+x for x in source_list])
        return df_miss
    
    df_feature_level = missCompute(x)
    df_record_level = pd.DataFrame(x.isna().sum(axis=1).value_counts()).reset_index().rename({'index':'missing_per_records',0:'records_concerned'},axis='columns')
    mean_i = x.isna().sum(axis=1).mean()
    std_i = x.isna().sum(axis=1).std()
    median_i = x.isna().sum(axis=1).median()
    df_global_record = pd.DataFrame([mean_i,std_i,median_i],index=['mean','std','median'])  
    return df_record_level,df_feature_level,df_global_record

col_dict = {'source':-1,
 'sex':0,
 'onset_spinal':1,
 'age':2,
 'time_disease':3,
 'weight_baseline':4,
 'pulse_baseline':6,
 'height_baseline':5,
 'bp_diastolic_baseline':7,
 'bp_systolic_baseline':8,
 'vital_capacity_baseline':9,
 'vital_capacity_percent_baseline':10,
 'ALSFRS_Total_baseline':11,
 'computeZones_hand_baseline':12,
 'computeZones_leg_baseline':13,
 'computeZones_mouth_baseline':14,
 'computeZones_respiratory_baseline':15,
 'computeZones_trunk_baseline':16,
 'survived':17,
 'death_unit':18,
 'ALSFRS_Total_final':19,
 'computeZones_hand_final':20,
 'computeZones_leg_final':21,
 'computeZones_trunk_final':22,
 'computeZones_respiratory_final':23,
 'computeZones_mouth_final':24,
 'FT9_score_final':25,
 'Kings_score_final':26,
 'MiToS_score_final':27}


df_record,df_feature,df_overall = missAnalysis(df_misss)
df_feature['col_idx'] = df_feature.feature.apply(lambda x: col_dict[x])
df_feature.to_csv('output/reboot/article/survival/df_feature_survival.csv',sep=';')
df_record,df_feature,df_overall = missAnalysis(df_missf)
df_feature['col_idx'] = df_feature.feature.apply(lambda x: col_dict[x])
df_feature.to_csv('output/reboot/article/functional/df_feature_functional.csv',sep=';')

#print(df_misss.loc[:,['sex','source']].groupby('source').count())

def estimateSlope(x):
    
    if x['time_disease'] == 0:
        y = 0
    elif x['time_disease'] < 1:
        time_disease=1
        y = (x['ALSFRS_Total_baseline']-40)/time_disease
    else:
        time_disease=x['time_disease']
        y = (x['ALSFRS_Total_baseline']-40)/time_disease
        
    return y

df['ALSFRS_Total_estim_slope'] = df.loc[:,['ALSFRS_Total_baseline','time_disease']].apply(estimateSlope,axis=1)

# remove these two as the two had time_disease < 1 which led to a strong decline
df = df.loc[~df.index.isin(['pre2008_1923','pre2008_5408']),:]

df_survived = df.loc[:,input_feats+outsurv_feats+['source','ALSFRS_Total_estim_slope']].dropna(subset=input_feats+['survived'])
df_functional = df.loc[:,input_feats+outfunc_feats+['source','ALSFRS_Total_estim_slope']+func_input+outfunc2].dropna(subset=input_feats+['ALSFRS_Total_final'])
#df_functional.dropna(subset=['computeZones_'+x+'_final' for x in ['hand','leg','trunk','respiratory','mouth']],inplace=True)

#############################################################################
### BBB STATES
##############################################################################

df_survived.to_csv('output/reboot/input_for_process/df_survived.csv',sep=';')
df_functional.to_csv('output/reboot/input_for_process/df_functional.csv',sep=';')

exonhit_list = pd.read_excel('input/patient_exonhit_filtered.xlsx')
exonhit_list = list(exonhit_list.loc[:,'subject_id'])

# remove exonhit patients from output analysis if non placebo
# keep them both groups for initial projection
df_survived.reset_index(inplace=True)
patients_kept = (list(df_survived.loc[((df_survived.source=='exonhit')&(df_survived.subject_id.isin(exonhit_list))),'subject_id'])+ 
                list(df_survived.loc[df_survived.source=='proact','subject_id'])+
                list(df_survived.loc[df_survived.source=='pre2008','subject_id'])+
                list(df_survived.loc[df_survived.source=='trophos','subject_id']))
df_survived.set_index('subject_id',inplace=True)

patient_mask = (df_survived.index.isin(patients_kept))
df_survived = df_survived.loc[patient_mask,:]

dead_ALSFRS_mask = ((~df_survived.death_unit.isnull())&(df_survived.death_unit<=12.1))
df_survived.loc[dead_ALSFRS_mask,'ALSFRS_Total_final'] = 0

feats_functional_only = ['Kings_score_final','MiToS_score_final', 'FT9_score_final', 
                         'computeZones_hand_final','computeZones_leg_final',
                         'computeZones_trunk_final','computeZones_respiratory_final', 
                         'computeZones_mouth_final', 
                         'FT9_score_baseline', 'Kings_score_baseline', 'MiToS_score_baseline',
                         'computeZones_hand_baseline', 'computeZones_leg_baseline',
                         'computeZones_mouth_baseline', 'computeZones_respiratory_baseline',
                         'computeZones_trunk_baseline']

df_functional = pd.merge(df_survived,df_functional.loc[:,feats_functional_only],how='left',left_index=True,right_index=True)

mask_patients_stage = ((~df_functional.death_unit.isnull())&(df_functional.death_unit<=12.1)&(df_functional.source!='trophos'))

for stage_type_i in ['Kings','MiToS','FT9']:
    df_functional.loc[mask_patients_stage,stage_type_i+'_score_final'] = 5
for x in ['trunk','respiratory','mouth','leg','hand']:
    df_functional.loc[mask_patients_stage,'computeZones_'+x+'_final'] = 0

df_functional.dropna(subset=['ALSFRS_Total_final'],inplace=True)    

# COUNT FOR FUNCTIONAL LOSS
sur_func = df_survived.dropna(subset=['ALSFRS_Total_final']).reset_index().loc[:,['subject_id','source']].groupby('source').count()
sur_func.columns=['count_ALSFRS']
# COUNT FOR SURVIVAL (MONTHS)
sur_death = df_survived.dropna(subset=['death_unit']).reset_index().loc[:,['subject_id','source']].groupby('source').count()
sur_death.columns = ['count_death']
# COUNT FOR STAGING
func_stages = df_functional.dropna(subset=['Kings_score_final']).reset_index().loc[:,['subject_id','source']].groupby('source').count()
func_stages.columns = ['count_stages']

# SURVIVAL PER SOURCE
osur = df_survived.reset_index().groupby('source').agg(({'onset_spinal':['sum'],
                               'sex':['sum'],
                               'survived':['sum'],
                               'age':['mean','std','min','max'],
                               'time_disease':['mean','std','min','max'],
                               'weight_baseline':['mean','std','min','max'],
                               'ALSFRS_Total_baseline':['mean','std','min','max'],
                               'ALSFRS_Total_estim_slope':['mean','std','min','max'],
                               'subject_id':['count'],
                               'death_unit':['mean','std','min','max'],
                               'ALSFRS_Total_final':['mean','std','min','max'],}))

# ADD COUNT FOR DEATH (MONTHS) + FUNCTIONAL LOSS
osur = pd.merge(osur,sur_func,left_index=True,right_index=True) 
osur = pd.merge(osur,sur_death,left_index=True,right_index=True) 

# ADD OVERALL AGGREGATED VALUES
osur_all = [df_survived.onset_spinal.sum(),
                df_survived.sex.sum(),
                df_survived.survived.sum(),
                df_survived.age.mean(),
                df_survived.age.std(),
                df_survived.age.min(),
                df_survived.age.max(),
                df_survived.time_disease.mean(),
                df_survived.time_disease.std(),
                df_survived.time_disease.min(),
                df_survived.time_disease.max(),
                df_survived.weight_baseline.mean(),
                df_survived.weight_baseline.std(),
                df_survived.weight_baseline.min(),
                df_survived.weight_baseline.max(),
                df_survived.ALSFRS_Total_baseline.mean(),
                df_survived.ALSFRS_Total_baseline.std(),
                df_survived.ALSFRS_Total_baseline.min(),
                df_survived.ALSFRS_Total_baseline.max(),
                df_survived.ALSFRS_Total_estim_slope.mean(),
                df_survived.ALSFRS_Total_estim_slope.std(),
                df_survived.ALSFRS_Total_estim_slope.min(),
                df_survived.ALSFRS_Total_estim_slope.max(),
                df_survived.reset_index().subject_id.count(),
                df_survived.death_unit.mean(),
                df_survived.death_unit.std(),
                df_survived.death_unit.min(),
                df_survived.death_unit.max(),
                df_survived.ALSFRS_Total_final.mean(),
                df_survived.ALSFRS_Total_final.std(),
                df_survived.ALSFRS_Total_final.min(),
                df_survived.ALSFRS_Total_final.max(),
                df_survived.dropna(subset=['ALSFRS_Total_final']).shape[0],
                df_survived.dropna(subset=['death_unit']).shape[0]                
]


osur_all = pd.DataFrame(osur_all,columns=['all'],index=osur.columns).transpose()
osur.reset_index(inplace=True)
osur_all.reset_index(inplace=True)
osur = osur.append(osur_all,ignore_index=True,sort=False)

#osur.to_csv('output/reboot/article/survival/stats_article_sur.csv',sep=";")

pd.set_option('display.max_rows',40)
pd.set_option('display.max_columns',40)

col_outputs = ['source','n','Gender (male/female)','Onset (spinal/bulbar)',
               'age (year)','time since onset (month)','baseline weight (kg)',
               'baseline ALSFRS','baseline ALSFRS decline rate']

output = []
for idx,line_i in osur.iterrows():
    if line_i['source'] == 'pre2008':
        tmp = ['real-world']
    elif pd.isnull(line_i['source']):
        tmp = ['overall']
    else:
        tmp = [line_i['source']]
    tmp.append(int(line_i[('subject_id','count')]))
    for feat_i in ['sex','onset_spinal','age','time_disease','weight_baseline','ALSFRS_Total_baseline','ALSFRS_Total_estim_slope']:
        if feat_i not in ['sex','onset_spinal']:
            if feat_i == 'ALSFRS_Total_estim_slope':    
                tmp.append('{:.2f} +/- {:.2f} ({:.2f}-{:.2f}) '.format(line_i[(feat_i,'mean')],line_i[(feat_i,'std')],line_i[(feat_i,'min')],line_i[(feat_i,'max')]))
            else:
                tmp.append('{:.1f} +/- {:.1f} ({:.1f}-{:.1f}) '.format(line_i[(feat_i,'mean')],line_i[(feat_i,'std')],line_i[(feat_i,'min')],line_i[(feat_i,'max')]))
        else:
            tmp.append('{:.0f}/{:.0f}'.format(line_i[(feat_i,'sum')],line_i[('subject_id','count')]-line_i[(feat_i,'sum')]))
    
    output.append(tmp)

output = pd.DataFrame(output,columns=col_outputs)
output.set_index('source',inplace=True)
output = output.reindex(index=['proact','trophos','exonhit','real-world','overall'])
output.to_csv('output/reboot/article/survival/stats_article_sur_clean_input.csv',sep=';')

col_outputs2 = ['source','n 1-year survival','survival rate','n survival',
               'survival (month)','n functional loss','functional loss',
               ]

output2 = []
for idx,line_i in osur.iterrows():
    if line_i['source'] == 'pre2008':
        tmp = ['real-world']
    elif pd.isnull(line_i['source']):
        tmp = ['overall']
    else:
        tmp = [line_i['source']]
    tmp.append(int(line_i[('subject_id','count')]))
    for feat_i in ['survived','count_death','death_unit','count_ALSFRS','ALSFRS_Total_final']:
        if feat_i not in ['survived','count_death','count_ALSFRS']:
            tmp.append('{:.0f} +/- {:.0f} ({:.0f}:{:.0f}) '.format(line_i[(feat_i,'mean')],line_i[(feat_i,'std')],line_i[(feat_i,'min')],line_i[(feat_i,'max')]))
        elif feat_i == 'survived':
            tmp.append('{:.0f}%'.format(line_i[(feat_i,'sum')]/line_i[('subject_id','count')]*100))
        else:
            tmp.append('{:.0f}'.format(line_i[feat_i]))
    
    output2.append(tmp)

output2 = pd.DataFrame(output2,columns=col_outputs2)
output2.set_index('source',inplace=True)
output2 = output2.reindex(index=['proact','trophos','exonhit','real-world','overall'])
output2.to_csv('output/reboot/article/survival/stats_article_sur_clean_output.csv',sep=';')


# Additional checking for df_functional clinical staging

ofunc = df_functional.reset_index().groupby('source').agg(({'onset_spinal':['sum'],
                               'sex':['sum'],
                               'age':['mean','std','min','max'],
                               'time_disease':['mean','std','min','max'],
                               'weight_baseline':['mean','std','min','max'],
                               'ALSFRS_Total_baseline':['mean','std','min','max'],
                               'ALSFRS_Total_estim_slope':['mean','std','min','max'],
                               'subject_id':['count'],
                               'ALSFRS_Total_final':['mean','std','min','max'],
                               'computeZones_hand_baseline':['mean','std','min','max'],
                               'computeZones_leg_baseline':['mean','std','min','max'],
                               'computeZones_trunk_baseline':['mean','std','min','max'],
                               'computeZones_mouth_baseline':['mean','std','min','max'],
                               'computeZones_respiratory_baseline':['mean','std','min','max'],
                               'computeZones_hand_final':['mean','std','min','max'],
                               'computeZones_leg_final':['mean','std','min','max'],
                               'computeZones_trunk_final':['mean','std','min','max'],
                               'computeZones_mouth_final':['mean','std','min','max'],
                               'computeZones_respiratory_final':['mean','std','min','max'],
                               }))
           
ofunc = pd.merge(ofunc,func_stages,left_index=True,right_index=True,how='left') 


# ADD OVERALL AGGREGATED VALUES
ofunc_all = [df_functional.onset_spinal.sum(),
                df_functional.sex.sum(),
                df_functional.age.mean(),
                df_functional.age.std(),
                df_functional.age.min(),
                df_functional.age.max(),
                df_functional.time_disease.mean(),
                df_functional.time_disease.std(),
                df_functional.time_disease.min(),
                df_functional.time_disease.max(),
                df_functional.weight_baseline.mean(),
                df_functional.weight_baseline.std(),
                df_functional.weight_baseline.min(),
                df_functional.weight_baseline.max(),
                df_functional.ALSFRS_Total_baseline.mean(),
                df_functional.ALSFRS_Total_baseline.std(),
                df_functional.ALSFRS_Total_baseline.min(),
                df_functional.ALSFRS_Total_baseline.max(),
                df_functional.ALSFRS_Total_estim_slope.mean(),
                df_functional.ALSFRS_Total_estim_slope.std(),
                df_functional.ALSFRS_Total_estim_slope.min(),
                df_functional.ALSFRS_Total_estim_slope.max(),
                df_functional.dropna(subset=['ALSFRS_Total_final']).shape[0],
                df_functional.ALSFRS_Total_final.mean(),
                df_functional.ALSFRS_Total_final.std(),
                df_functional.ALSFRS_Total_final.min(),
                df_functional.ALSFRS_Total_final.max(),
                df_functional.computeZones_hand_baseline.mean(), 
                df_functional.computeZones_hand_baseline.std(), 
                df_functional.computeZones_hand_baseline.min(), 
                df_functional.computeZones_hand_baseline.max(),                 
                df_functional.computeZones_leg_baseline.mean(),
                df_functional.computeZones_leg_baseline.std(),
                df_functional.computeZones_leg_baseline.min(),
                df_functional.computeZones_leg_baseline.max(),                
                df_functional.computeZones_trunk_baseline.mean(),
                df_functional.computeZones_trunk_baseline.std(),
                df_functional.computeZones_trunk_baseline.min(),
                df_functional.computeZones_trunk_baseline.max(),
                df_functional.computeZones_mouth_baseline.mean(),
                df_functional.computeZones_mouth_baseline.std(),
                df_functional.computeZones_mouth_baseline.min(),
                df_functional.computeZones_mouth_baseline.max(),
                df_functional.computeZones_respiratory_baseline.mean(),
                df_functional.computeZones_respiratory_baseline.std(),
                df_functional.computeZones_respiratory_baseline.min(),
                df_functional.computeZones_respiratory_baseline.max(),
                df_functional.computeZones_hand_final.mean(), 
                df_functional.computeZones_hand_final.std(), 
                df_functional.computeZones_hand_final.min(), 
                df_functional.computeZones_hand_final.max(),                 
                df_functional.computeZones_leg_final.mean(),
                df_functional.computeZones_leg_final.std(),
                df_functional.computeZones_leg_final.min(),
                df_functional.computeZones_leg_final.max(),                
                df_functional.computeZones_trunk_final.mean(),
                df_functional.computeZones_trunk_final.std(),
                df_functional.computeZones_trunk_final.min(),
                df_functional.computeZones_trunk_final.max(),
                df_functional.computeZones_mouth_final.mean(),
                df_functional.computeZones_mouth_final.std(),
                df_functional.computeZones_mouth_final.min(),
                df_functional.computeZones_mouth_final.max(),
                df_functional.computeZones_respiratory_final.mean(),
                df_functional.computeZones_respiratory_final.std(),
                df_functional.computeZones_respiratory_final.min(),
                df_functional.computeZones_respiratory_final.max(),
                df_functional.dropna(subset=['MiToS_score_final']).shape[0],]

df_functional_stage = df_functional.loc[~df_functional.FT9_score_final.isnull(),:]

### STAGE AGGREGATION PER SOURCE 
columns = []
stages_results = pd.DataFrame([],index=['exonhit','pre2008','proact','trophos'])
for name_i in ['Kings','MiToS','FT9']:
    for time_i in ['baseline','final']:
        stages = list(df_functional_stage.loc[:,name_i+'_score_'+time_i].unique())
        stages.sort()
        for stage_i in stages:
           columns.append(name_i+'_'+time_i+'_'+str(stage_i))
           fc_tmp = df_functional_stage.loc[df_functional_stage[name_i+'_score_'+time_i]==stage_i,:].reset_index()
           print('{}_{} stage: {}: count = {}'.format(name_i,time_i,stage_i,fc_tmp.shape[0]))
           fc_tmp2 = fc_tmp.loc[:,['subject_id','source']].groupby('source').count()
           fc_tmp2 = fc_tmp2.rename(columns={'subject_id':name_i+'_'+time_i+'_'+str(stage_i)})
           print(fc_tmp2)
           if fc_tmp2.shape[0] != 0:
               stages_results[name_i+'_'+time_i+'_'+str(stage_i)] = fc_tmp2[name_i+'_'+time_i+'_'+str(stage_i)]

stages_results.columns = [
        'Kings_baseline_1.0', 
        'Kings_baseline_2.0',
        'Kings_baseline_3.0',
        'Kings_baseline_4.0',
         'Kings_final_1.0',
         'Kings_final_2.0',
         'Kings_final_3.0',
         'Kings_final_4.0',
         'Kings_final_4.5',
         'Kings_final_5.0',
         'MiToS_baseline_0.0',
         'MiToS_baseline_1.0',
         'MiToS_baseline_2.0',
         'MiToS_baseline_3.0',
         'MiToS_final_0.0',
         'MiToS_final_1.0',
         'MiToS_final_2.0',
         'MiToS_final_3.0',
         'MiToS_final_4.0',
         'MiToS_final_5.0',
         'FT9_baseline_0.0',
         'FT9_baseline_1.0',
         'FT9_baseline_2.0',
         'FT9_baseline_3.0',
         'FT9_baseline_4.0',
         'FT9_final_0.0',
        'FT9_final_1.0',
         'FT9_final_2.0',
         'FT9_final_3.0',
         'FT9_final_4.0',
         'FT9_final_5.0']

ofunc_all = pd.DataFrame(ofunc_all,columns=['all'],index=ofunc.columns).transpose()
ofunc.reset_index(inplace=True)
ofunc_all.reset_index(inplace=True)
ofunc = ofunc.append(ofunc_all,ignore_index=True,sort=False)

#ofunc.to_csv('output/reboot/article/functional/stats_article_func.csv',sep=";")

stages_results_out = stages_results.drop([x for x in list(stages_results.columns) if x.split('_')[1]=='baseline'],axis=1)
stages_results_in = stages_results.drop([x for x in list(stages_results.columns) if x.split('_')[1]=='final'],axis=1)
#stages_results.to_csv('output/reboot/article/functional/stats_article_func_stages.csv',sep=';')

col_outputs = (['source','n','Gender (male/female)','Onset (spinal/bulbar)',
               'age (year)','time since onset (month)','baseline weight (kg)',
               'baseline ALSFRS','baseline ALSFRS decline rate']+['baseline ALSFRS '+y for y in ['hand','leg','trunk','mouth','respiratory']]+
               ['baseline Kings - stage '+str(y) for y in [1,2,3,4,4.5]]+
                    ['baseline MiToS - stage '+str(y) for y in [0,1,2,3,4]]+
                    ['baseline FT9 '+str(y) for y in [0,1,2,3,4]])


output = []
for idx,line_i in ofunc.iterrows():
    if line_i['source'] == 'pre2008':
    #if line_i['source'].values[0] == 'pre2008':
        tmp = ['real-world']
    #elif pd.isnull(line_i['source'].values[0]):
    elif pd.isnull(line_i['source']):
        tmp = ['overall']
    else:
        tmp = [line_i['source']]#[line_i['source'].values[0]]
    tmp.append(int(line_i[('subject_id','count')]))
    for feat_i in ['sex','onset_spinal','age','time_disease','weight_baseline','ALSFRS_Total_baseline','ALSFRS_Total_estim_slope','computeZones_hand_baseline','computeZones_leg_baseline','computeZones_trunk_baseline','computeZones_mouth_baseline','computeZones_respiratory_baseline']:
        if feat_i not in ['sex','onset_spinal']:
            if feat_i == 'ALSFRS_Total_estim_slope':    
                tmp.append('{:.2f} +/- {:.2f} ({:.2f}-{:.2f}) '.format(line_i[(feat_i,'mean')],line_i[(feat_i,'std')],line_i[(feat_i,'min')],line_i[(feat_i,'max')]))
            else:
                tmp.append('{:.1f} +/- {:.1f} ({:.1f}-{:.1f}) '.format(line_i[(feat_i,'mean')],line_i[(feat_i,'std')],line_i[(feat_i,'min')],line_i[(feat_i,'max')]))
        else:
            tmp.append('{:.0f}/{:.0f}'.format(line_i[(feat_i,'sum')],line_i[('subject_id','count')]-line_i[(feat_i,'sum')]))
    #if  line_i['source'].values[0] in ['pre2008','exonhit','proact','trophos']:
    if  line_i['source'] in ['pre2008','exonhit','proact','trophos']:    
        #print(line_i['source'].values[0])
        for feat_i in (['Kings_'+x+'_'+str(y) for x in ['baseline'] for y in [1.0,2.0,3.0,4.0,4.5]]+
                    ['MiToS_'+x+'_'+str(y) for x in ['baseline'] for y in [0.0,1.0,2.0,3.0,4.0]]+
                    ['FT9_'+x+'_'+str(y) for x in ['baseline'] for y in [0.0,1.0,2.0,3.0,4.0]]):
            try:
                tmp.append(int(stages_results_in.loc[line_i['source'],feat_i]))
            except:
                print('feat_i: {}, line_i[source]: {}'.format(feat_i,line_i['source']))
                tmp.append(0)
                #print('no value for {}'.format(feat_i))
    else:
        #print('else: {}'.format(line_i['source'].values[0]))
        for feat_i in (['Kings_'+x+'_'+str(y) for x in ['baseline'] for y in [1.0,2.0,3.0,4.0,4.5]]+
                    ['MiToS_'+x+'_'+str(y) for x in ['baseline'] for y in [0.0,1.0,2.0,3.0,4.0]]+
                    ['FT9_'+x+'_'+str(y) for x in ['baseline'] for y in [0.0,1.0,2.0,3.0,4.0]]):
            try:
                tmp.append(stages_results_in.loc[:,feat_i].sum())
            except:
                tmp.append(0)
    output.append(tmp)

output = pd.DataFrame(output,columns=col_outputs)
output.set_index('source',inplace=True)
output = output.reindex(index=['proact','trophos','exonhit','real-world','overall'])
output.to_csv('output/reboot/article/functional/stats_article_func_clean_input.csv',sep=';')



col_outputs2 = (['source','n',' 1-year functional loss']+['final ALSFRS '+y for y in ['hand','leg','trunk','mouth','respiratory']]+['n (stage)']+
               ['1-year Kings - stage '+str(y) for y in [1,2,3,4,4.5,5]]+
                    ['1-year MiToS - stage '+str(y) for y in [0,1,2,3,4,5]]+
                    ['1-year FT9 '+str(y) for y in [0,1,2,3,4,5]])

output2 = []
for idx,line_i in ofunc.iterrows():
    #if line_i['source'].values[0] == 'pre2008':
    if line_i['source'] == 'pre2008':
        tmp = ['real-world']
    #elif pd.isnull(line_i['source'].values[0]):
    elif pd.isnull(line_i['source']):
        tmp = ['overall']
    else:
        #tmp = [line_i['source'].values[0]]
        tmp = [line_i['source']]
    tmp.append(int(line_i[('subject_id','count')]))
    for feat_i in ['ALSFRS_Total_final','computeZones_hand_final','computeZones_leg_final','computeZones_trunk_final','computeZones_mouth_final','computeZones_respiratory_final']:
        if feat_i not in ['count_clin']:
            tmp.append('{:.0f} +/- {:.0f} ({:.0f}-{:.0f}) '.format(line_i[(feat_i,'mean')],line_i[(feat_i,'std')],line_i[(feat_i,'min')],line_i[(feat_i,'max')]))
        else:
            tmp.append('{:.0f}'.format(line_i[feat_i]))
    #if line_i['source'].values[0] in ['pre2008','proact','exonhit','trophos']:
    try:
        tmp.append(int(line_i['count_stages']))
    except:
        tmp.append(0)
        print('skipped count_stages for {}'.format(line_i['source']))
    if line_i['source'] in ['pre2008','proact','exonhit','trophos']:
        for feat_i in (['Kings_'+x+'_'+str(y) for x in ['final'] for y in [1.0,2.0,3.0,4.0,4.5,5.0]]+
                    ['MiToS_'+x+'_'+str(y) for x in ['final'] for y in [0.0,1.0,2.0,3.0,4.0,5.0]]+
                    ['FT9_'+x+'_'+str(y) for x in ['final'] for y in [0.0,1.0,2.0,3.0,4.0,5.0]]):
            try:
                tmp.append(int(stages_results_out.loc[line_i['source'],feat_i]))
            except:
                tmp.append(0)
                #print('no value for {}'.format(feat_i))
    else:
        for feat_i in (['Kings_'+x+'_'+str(y) for x in ['final'] for y in [1.0,2.0,3.0,4.0,4.5,5.0]]+
                    ['MiToS_'+x+'_'+str(y) for x in ['final'] for y in [0.0,1.0,2.0,3.0,4.0,5.0]]+
                    ['FT9_'+x+'_'+str(y) for x in ['final'] for y in [0.0,1.0,2.0,3.0,4.0,5.0]]):
            try:
                tmp.append(stages_results_out.loc[:,feat_i].sum())
            except:
                tmp.append(0)
    output2.append(tmp)

output2 = pd.DataFrame(output2,columns=col_outputs2)
output2.set_index('source',inplace=True)
output2 = output2.reindex(index=['proact','trophos','exonhit','real-world','overall'])


output2.to_csv('output/reboot/article/functional/stats_article_func_clean_output.csv',sep=';')


def def3Zone(x):
    diff_up = x['y_axis']-(x['x_axis']*0.2+0.4)
    diff_down = x['y_axis']-(x['x_axis']*0.5-0.1)# 6/17+73/340
    if diff_up > 0:
        return 'top'
    elif diff_down < 0:
        return 'bottom'
    else:
        return 'mid'

### Pour recuperer les axis x et y

# WILL NEED TO CHANGE THE FUNCTIONAL_3ZONE_TESTING

print(aaaa)

df_axis = pd.read_csv('output/reboot/input_for_process/functional_3zone_testing.csv',sep=';')
df_axis = df_axis.loc[:,['x_axis','y_axis','subject_id']]
df_axis.set_index('subject_id',inplace=True)
df_functional = pd.merge(df_functional,df_axis,how='right',left_index=True,right_index=True)
df_functional['zone'] = df_functional.loc[:,['x_axis','y_axis']].apply(lambda x: def3Zone(x),axis=1)
df_functional.drop(['x_axis','y_axis','source'],axis=1,inplace=True)

ofunc_3z = df_functional.reset_index().groupby('zone').agg(({'onset_spinal':['sum'],
                               'sex':['sum'],
                               'age':['mean','std','min','max'],
                               'time_disease':['mean','std','min','max'],
                               'weight_baseline':['mean','std','min','max'],
                               'ALSFRS_Total_baseline':['mean','std','min','max'],
                               'ALSFRS_Total_estim_slope':['mean','std','min','max'],
                               'subject_id':['count'],
                               'computeZones_hand_baseline':['mean','std','min','max'],
                               'computeZones_leg_baseline':['mean','std','min','max'],
                               'computeZones_trunk_baseline':['mean','std','min','max'],
                               'computeZones_mouth_baseline':['mean','std','min','max'],
                               'computeZones_respiratory_baseline':['mean','std','min','max'], 
                               'ALSFRS_Total_final':['mean','std','min','max'],
                               }))
           
global_3z = [df_functional.onset_spinal.sum(),
             df_functional.sex.sum(),
             df_functional.age.mean(),
             df_functional.age.std(),
             df_functional.age.min(),
             df_functional.age.max(),
             df_functional.time_disease.mean(),
             df_functional.time_disease.std(),
             df_functional.time_disease.min(),
             df_functional.time_disease.max(),
             df_functional.weight_baseline.mean(),
             df_functional.weight_baseline.std(),
             df_functional.weight_baseline.min(),
             df_functional.weight_baseline.max(),
             df_functional.ALSFRS_Total_baseline.mean(),
             df_functional.ALSFRS_Total_baseline.std(),
             df_functional.ALSFRS_Total_baseline.min(),
             df_functional.ALSFRS_Total_baseline.max(),
             df_functional.ALSFRS_Total_estim_slope.mean(),
             df_functional.ALSFRS_Total_estim_slope.std(),
             df_functional.ALSFRS_Total_estim_slope.min(),
             df_functional.ALSFRS_Total_estim_slope.max(),
             df_functional.dropna(subset=['ALSFRS_Total_final']).shape[0],
             df_functional.computeZones_hand_baseline.mean(), 
             df_functional.computeZones_hand_baseline.std(), 
             df_functional.computeZones_hand_baseline.min(), 
             df_functional.computeZones_hand_baseline.max(),                 
             df_functional.computeZones_leg_baseline.mean(),
             df_functional.computeZones_leg_baseline.std(),
             df_functional.computeZones_leg_baseline.min(),
             df_functional.computeZones_leg_baseline.max(),                
             df_functional.computeZones_trunk_baseline.mean(),
             df_functional.computeZones_trunk_baseline.std(),
             df_functional.computeZones_trunk_baseline.min(),
             df_functional.computeZones_trunk_baseline.max(),
             df_functional.computeZones_mouth_baseline.mean(),
             df_functional.computeZones_mouth_baseline.std(),
             df_functional.computeZones_mouth_baseline.min(),
             df_functional.computeZones_mouth_baseline.max(),
             df_functional.computeZones_respiratory_baseline.mean(),
             df_functional.computeZones_respiratory_baseline.std(),
             df_functional.computeZones_respiratory_baseline.min(),
             df_functional.computeZones_respiratory_baseline.max(),
             df_functional.ALSFRS_Total_final.mean(),
             df_functional.ALSFRS_Total_final.std(),
             df_functional.ALSFRS_Total_final.min(),
             df_functional.ALSFRS_Total_final.max()]

global_3z = pd.DataFrame(global_3z,index= [x for x in ofunc_3z.columns if x != ('zone', '')])

ofunc_3z.to_csv('output/reboot/stats_article_func_3z_stat.csv',sep=';')

col_outputs_3z = (['zone','n','Gender (male/female)','Onset (spinal/bulbar)',
               'age (year)','symptom duration (month)','baseline weight (kg)',
               'baseline ALSFRS','baseline ALSFRS decline rate']+
    ['baseline ALSFRS '+y for y in ['hand','leg','trunk','mouth','respiratory']]+['ALSFRS_Total_final'])

ofunc_3z.reset_index(inplace=True)

output_3z = []
for idx,line_i in ofunc_3z.iterrows():
    
        
    tmp = [line_i['zone'].values[0]]
    tmp.append(int(line_i[('subject_id','count')]))
    for feat_i in ['sex','onset_spinal','age','time_disease','weight_baseline','ALSFRS_Total_baseline','ALSFRS_Total_estim_slope','computeZones_hand_baseline','computeZones_leg_baseline','computeZones_trunk_baseline','computeZones_mouth_baseline','computeZones_respiratory_baseline','ALSFRS_Total_final']:
        if feat_i not in ['sex','onset_spinal']:
            if feat_i == 'ALSFRS_Total_estim_slope':    
                tmp.append('{:.2f} +/- {:.2f} ({:.2f}-{:.2f}) '.format(line_i[(feat_i,'mean')],line_i[(feat_i,'std')],line_i[(feat_i,'min')],line_i[(feat_i,'max')]))
            else:
                tmp.append('{:.1f} +/- {:.1f} ({:.1f}-{:.1f}) '.format(line_i[(feat_i,'mean')],line_i[(feat_i,'std')],line_i[(feat_i,'min')],line_i[(feat_i,'max')]))
        else:
            tmp.append('{:.0f}/{:.0f}'.format(line_i[(feat_i,'sum')],line_i[('subject_id','count')]-line_i[(feat_i,'sum')]))
    output_3z.append(tmp)

tmp = ['overall']
for feat_i in ['sex','onset_spinal','age','time_disease','weight_baseline','ALSFRS_Total_baseline','ALSFRS_Total_estim_slope','computeZones_hand_baseline','computeZones_leg_baseline','computeZones_trunk_baseline','computeZones_mouth_baseline','computeZones_respiratory_baseline','ALSFRS_Total_final']:
    if feat_i not in ['sex','onset_spinal']:
        if feat_i == 'ALSFRS_Total_estim_slope':    
            tmp.append('{:.2f} +/- {:.2f} ({:.2f}-{:.2f}) '.format(global_3z[global_3z.index==(feat_i, 'mean')].values[0][0],
                                                                   global_3z[global_3z.index==(feat_i, 'std')].values[0][0],
                                                                   global_3z[global_3z.index==(feat_i, 'min')].values[0][0],
                                                                   global_3z[global_3z.index==(feat_i, 'max')].values[0][0],
                                                                   ))
        else:
            tmp.append('{:.1f} +/- {:.1f} ({:.1f}-{:.1f}) '.format(global_3z[global_3z.index==(feat_i, 'mean')].values[0][0],
                                                                   global_3z[global_3z.index==(feat_i, 'std')].values[0][0],
                                                                   global_3z[global_3z.index==(feat_i, 'min')].values[0][0],
                                                                   global_3z[global_3z.index==(feat_i, 'max')].values[0][0],
                                                                   ))
    else:
        tmp.append('{:.0f}/{:.0f}'.format(global_3z[global_3z.index==(feat_i, 'sum')].values[0][0],
                                          global_3z[global_3z.index==('subject_id', 'count')].values[0][0]-global_3z[global_3z.index==(feat_i, 'sum')].values[0][0]))
    output_3z.append(tmp)
    

output_3z = pd.DataFrame(output_3z,columns=col_outputs_3z)
output_3z.set_index('zone',inplace=True)
output_3z = output_3z.reindex(index=['top','mid','bottom','overall'])
output_3z.to_csv('output/reboot/article/functional/stats_article_func_clean_3z.csv',sep=';')



