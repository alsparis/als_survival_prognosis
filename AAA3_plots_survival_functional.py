# -*- coding: utf-8 -*-
# 02/01/2020

# Plot for survival and functional - second take
# Input features for plot 
# Output features for plot
# Grid analysis 
# Train and test analysis
# Advanced grid analysis

import pandas as pd
import math
import sklearn.preprocessing as sk_p
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.random import rand
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

umaps = {}
dfs = {}
umaps['dev_survival'] = pd.read_csv('output/reboot/input_for_process/Y_plot.csv',sep=';')
#umaps['dev_functional'] = pd.read_csv('output/reboot/Y_plot_func.csv',sep=';')
umaps['val_survival'] = pd.read_csv('output/reboot/input_for_process/Y_pre2008.csv',sep=';')
#umaps['val_functional'] = pd.read_csv('output/reboot/Y_pre2008_func.csv',sep=';')

dfs['dev_survival'] = pd.read_csv('output/reboot/input_for_process/df_survived.csv',sep=';')
dfs['val_survival'] = dfs['dev_survival'].loc[dfs['dev_survival'].source.isin(['pre2008']),:]
dfs['dev_survival'] = dfs['dev_survival'].loc[~dfs['dev_survival'].source.isin(['pre2008']),:]

dfs['val_functional'] = dfs['val_survival']
dfs['dev_functional'] = dfs['dev_survival']

functional_list = pd.read_csv('output/reboot/input_for_process/df_functional.csv',sep=';')

exonhit_list = pd.read_excel('input/patient_exonhit_filtered.xlsx')
exonhit_list = list(exonhit_list.loc[:,'subject_id'])

# remove exonhit patients from output analysis if non placebo
# keep them both groups for initial projection
patients_kept = (list(dfs['dev_survival'].loc[((dfs['dev_survival'].source=='exonhit')&(dfs['dev_survival'].subject_id.isin(exonhit_list))),'subject_id'])+ 
                list(dfs['dev_survival'].loc[dfs['dev_survival'].source=='proact','subject_id'])+
                list(dfs['dev_survival'].loc[dfs['dev_survival'].source=='trophos','subject_id']))
# Loss of 173


    
#dfs['dev_survival'].to_csv('test_ALSFRS_Total_final_dead.csv',sep=';')

# add patients which died within the year for the functional analysis (won't change the stage analysis)
#functional_dev = list(functional_list.loc[~functional_list.source.isin(['pre2008']),'subject_id'])+patients_to_add['dev']
#functional_val = list(functional_list.loc[functional_list.source.isin(['pre2008']),'subject_id'])+patients_to_add['val']
functional_dev = list(functional_list.loc[~functional_list.source.isin(['pre2008']),'subject_id'])
functional_val = list(functional_list.loc[functional_list.source.isin(['pre2008']),'subject_id'])
'''
print('list dev_func: {}={}+{} val_func:{}={}+{}'.format(len(functional_dev),
                                                        len(list(functional_list.loc[~functional_list.source.isin(['pre2008']),'subject_id'])),
                                                        len(patients_to_add['dev']),
                                                        len(functional_val),
                                                        len(list(functional_list.loc[functional_list.source.isin(['pre2008']),'subject_id'])),
                                                        len(patients_to_add['val'])))
'''
functional_list.set_index('subject_id',inplace=True)


feats_tested = {}

# Added sex and onset to minimal scope

feat_ALSFRS = {'hand':['Q4_Handwriting_baseline','Q5_Cutting_baseline',],
                       'trunk':['Q6_Dressing_and_Hygiene_baseline','Q7_Turning_in_Bed_baseline',],
                       'respiratory':['Q10_Respiratory_baseline',],
                       'leg':['Q8_Walking_baseline','Q9_Climbing_Stairs_baseline',],
                       'mouth':['Q1_Speech_baseline','Q2_Salivation_baseline','Q3_Swallowing_baseline',]}
        
def sumFeats(x,feats):
    y = np.nan
    idx = 0
    for feat_i in feats:
        if not np.isnan(x[feat_i]):
            if idx == 0:
                y = x[feat_i]
                idx = 1
            else:
                y += x[feat_i]
    return y
        
for elem_i in ['hand','trunk','respiratory','leg','mouth']:
    functional_list['computeZones_'+elem_i+'_baseline'] = functional_list.loc[:,feat_ALSFRS[elem_i]].apply(lambda x: sumFeats(x,feat_ALSFRS[elem_i]),axis=1)



    
for elem_i in [x+'_survival' for x in ['dev','val']]:
    umaps[elem_i].set_index('subject_id',inplace=True)

umaps['dev_functional'] = umaps['dev_survival'].loc[umaps['dev_survival'].index.isin(functional_dev),:]
umaps['val_functional'] = umaps['val_survival'].loc[umaps['val_survival'].index.isin(functional_val),:]
print('dev_func:{} and val_func: {}'.format(umaps['dev_functional'].shape,umaps['val_functional'].shape))

# Moved below so that functional analysis does not take in such information
# for ALSFRS analysis => add patients which died within the year 

patients_to_add = {}

for scope_i in ['dev','val']:
    dfs[scope_i+'_survival'].set_index('subject_id',inplace=True)
    # Tolerance added (12.1) as survived feature approximated for 12.1 month (hence )
    dead_ALSFRS_mask = ((~dfs[scope_i+'_survival'].death_unit.isnull())&(dfs[scope_i+'_survival'].death_unit<=12.1))
    #print('{} {}'.format(scope_i,dead_ALSFRS_mask))
    patients_to_add[scope_i] = list(dfs[scope_i+'_survival'].loc[dead_ALSFRS_mask,:].index)
    dfs[scope_i+'_survival'].loc[dead_ALSFRS_mask,'ALSFRS_Total_final'] = 0


for df_i in [x+'_functional' for x in ['val','dev']]:
    umaps[df_i].drop(['survived'],axis=1,inplace=True)
#    umaps[df_i].dropna(subset=['computeZones_'+x+'_final' for x in ['mouth','leg','hand','trunk','respiratory']],inplace=True)
    

#dfs['dev_functional'] = pd.read_csv('output/reboot/df_functional.csv',sep=';')
#dfs['val_functional'] = dfs['dev_functional'].loc[dfs['dev_functional'].source.isin(['pre2008']),:]
#dfs['dev_functional'] = dfs['functional'].loc[~dfs['dev_functional'].source.isin(['pre2008']),:]

params = {'rotate':{'survival':-120,'functional':-120},
          'input_title':{'survival':{'onset_spinal':'b. Onset \n ',
                               'sex':'c. Sex \n ',
                               'age':'d. Age \n (year)',
                               'time_disease':'e.Symptom duration \n (month)',
                               'weight_baseline':'f. Baseline \n weight \n (kg)',
                               'ALSFRS_Total_baseline':'g. Baseline \n ALSFRS \n(score)',
                               'ALSFRS_Total_estim_slope':'h. Est. baseline \nALSFRS slope \n'+ r'(score/month$^{-1})$'},
                    'functional':{'onset_spinal':'a. Onset \n ',
                                  'sex':'b. Sex \n ',
                                  'age':'c. Age \n (year)',
                                  'time_disease':'d. Symptom \n duration \n (month)',
                                  'weight_baseline':'e. Baseline \n weight \n (kg)',
                                  'ALSFRS_Total_baseline':'f. Baseline \n ALSFRS \n(score)',
                                  'ALSFRS_Total_estim_slope':'g. Est. baseline \nALSFRS slope \n'+ r'(score/month$^{-1})$',
                                  'computeZones_hand_baseline':'h. Baseline ALSFRS \n upper limb \n zone (score)',
                                  'computeZones_leg_baseline':'i. Baseline ALSFRS \n lower limb \n zone (score)',
                                  'computeZones_trunk_baseline':'j. Baseline ALSFRS \n trunk \n zone (score)',
                                  'computeZones_mouth_baseline':'k. Baseline ALSFRS \n bulbar \n zone (score)',
                                  'computeZones_respiratory_baseline':'l. Baseline ALSFRS \n respiratory \n zone (score)',
                                  }
                    },
        'input_feat':{'survival':['onset_spinal', 'sex', 'age','time_disease','weight_baseline', 'ALSFRS_Total_baseline','ALSFRS_Total_estim_slope'],
                           'functional':['onset_spinal', 'sex', 'age','time_disease','weight_baseline', 'ALSFRS_Total_baseline','ALSFRS_Total_estim_slope',
                                          'computeZones_hand_baseline','computeZones_leg_baseline','computeZones_trunk_baseline','computeZones_mouth_baseline',
                                          'computeZones_respiratory_baseline']},
        'input_subplot':{'survival':(2,4),'functional':(3,4)},
        'input_subplot_size':{'survival':(10.66,8),'functional':(8,12)},
        'ext_title':{'survival':{'death_unit':'a. overall survival \n (month)',
             'ALSFRS_Total_final':'c. 1-year ALSFRS \n (score)',
             'survived':'b. 1-year survival \n '},
                    'functional':{'ALSFRS_Total_final':'a. 1-year ALSFRS \n (score)',
                                  'Kings_score_final':'b. 1-year Kings \n (stage)',
                                  'MiToS_score_final':'c. 1-year MiToS \n (stage)',
                                  'FT9_score_final':'d. 1-year FT9 \n (stage)',},
                    },
        'ext_subplot':{'survival':(3,2),'functional':(2,4)},
        'ext_subplot_size':{'survival':(4.44,12),'functional':(8.88,8)},
        'output_subplot':{'survival':(1,3),'functional':(1,4)},
        'output_subplot_size':{'survival':(8,4),'functional':(8,4)},
        'output_feats':{'survival':['death_unit','survived','ALSFRS_Total_final'],
                        'functional':['ALSFRS_Total_final','Kings_score_final','MiToS_score_final','FT9_score_final']},
        'output_title':{'survival':{'death_unit':'a. Overall survival \n (month)',
             'ALSFRS_Total_final':'c. 1-year ALSFRS \n (score)',
             'Kings_score_final':'d. 1-year Kings \n (stage)',
             'MiToS_score_final':'e. 1-year MiToS \n (stage)',
             'FT9_score_final':'f. 1-year FT9 \n (stage)',
             'survived':'b. 1-year survival \n '},
                        'functional':{'ALSFRS_Total_final':'a. 1-year ALSFRS \n (score)',
                                      'Kings_score_final':'b. 1-year Kings \n (stage)',
                                      'MiToS_score_final':'c. 1-year MiToS \n (stage)',
                                      'FT9_score_final':'d. 1-year FT9 \n (stage)',
                                      'computeZones_hand_final':'e. 1-year ALSFRS upper limb \n subscore',
                                      'computeZones_leg_final':'c. 1-year ALSFRS lower limb \n subscore',
                                      'computeZones_trunk_final':'b. 1-year ALSFRS trunk \n subscore',
                                      'computeZones_respiratory_final':'d. 1-year ALSFRS respiratory \n subscore',
                                      'computeZones_mouth_final':'a. 1-year ALSFRS bulbar \n subscore',}
                        },
        'output_title2':{'functional':{'ALSFRS_Total_final':'a. 1-year ALSFRS \n (score)',
                                      'Kings_score_final':'b. 1-year Kings \n (stage)',
                                      'MiToS_score_final':'c. 1-year MiToS \n (stage)',
                                      'FT9_score_final':'d. 1-year FT9 \n (stage)',
                                      'computeZones_hand_final':'e. 1-year ALSFRS \n upper limb subscore',
                                      'computeZones_leg_final':'f. 1-year ALSFRS \n lower limb subscore',
                                      'computeZones_trunk_final':'g. 1-year ALSFRS \n trunk subscore',
                                      'computeZones_respiratory_final':'i. 1-year ALSFRS \n respiratory subscore',
                                      'computeZones_mouth_final':'h. 1-year ALSFRS \n bulbar subscore',}}
            }
def rotateData(x,angle):
    
    x_mod = x['x_axis']*math.cos(angle*math.pi/180)+x['y_axis']*math.sin(angle*math.pi/180)
    y_mod = -x['x_axis']*math.sin(angle*math.pi/180)+x['y_axis']*math.cos(angle*math.pi/180)
    
    return x_mod,y_mod

dict_input_spines = {'survival':{0:(0,1),
                                 1:(0,0),
                                 2:(0,0),
                                 3:(0,0),
                                 4:(1,1),
                                 5:(1,0),
                                 6:(1,0),
                                 7:(1,0)},
                     'functional':{0:(0,1),
                                  1:(0,0),
                                  2:(0,0),
                                  3:(0,0),
                                  4:(0,1),
                                  5:(0,0),
                                  6:(0,0),
                                  7:(0,0),
                                  8:(1,1),
                                  9:(1,0),
                                  10:(1,0),
                                  11:(1,0)}}

dict_output_spines = {'survival':{0:(1,1),
                                 1:(1,0),
                                 2:(1,0)},
                     'functional':{0:(1,1),
                                  1:(1,0),
                                  2:(1,0),
                                  3:(1,0),
                                  }}

output_dict_9ALSFRS = {0:(0,1),
                        1:(0,0),
                        2:(0,0),
                        3:(0,1),
                        4:(0,0),
                        5:(0,0),
                        6:(1,1),
                        7:(1,0),
                        8:(1,0)}
                       
for df_i in ['functional']:#['survival',['functional']: # 
    
    
    print('###################   {}   ##################'.format(df_i))
    
    if  df_i == 'survival':
        drop_feat = 'survived'
     
    for scope_i in ['dev','val']:
        try:
            umaps[scope_i+'_'+df_i].drop([drop_feat],axis=1,inplace=True)
        except:
            pass
        umaps[scope_i+'_'+df_i].rename(columns={'0':'x_axis','1':'y_axis'},inplace=True)
        angle_i = params['rotate'][df_i]
        umaps[scope_i+'_'+df_i]['tmp'] = np.nan
        umaps[scope_i+'_'+df_i].loc[:,'tmp'] = umaps[scope_i+'_'+df_i].apply(lambda x: rotateData(x,angle_i),axis=1)
        umaps[scope_i+'_'+df_i].loc[:,'x_axis'] = umaps[scope_i+'_'+df_i].loc[:,'tmp'].apply(lambda x: x[0])
        umaps[scope_i+'_'+df_i].loc[:,'y_axis'] = umaps[scope_i+'_'+df_i].loc[:,'tmp'].apply(lambda x: x[1])
        umaps[scope_i+'_'+df_i].drop(['tmp'],axis=1,inplace=True)
        #umaps[scope_i+'_'+df_i].to_csv('output/reboot/'+df_i+'_rotate_'+str(angle_i)+'.csv',sep=';')
        df_idx = umaps[scope_i+'_'+df_i].index
        if scope_i == 'dev':
            scaler = sk_p.MinMaxScaler()
            out = scaler.fit_transform(umaps[scope_i+'_'+df_i])
        else:
            out = scaler.transform(umaps[scope_i+'_'+df_i])
        umaps[scope_i+'_'+df_i] = pd.DataFrame(out,columns=['x_axis','y_axis'],index=df_idx)
        umaps[scope_i+'_'+df_i].loc[:,'x_axis'] = 1- umaps[scope_i+'_'+df_i].loc[:,'x_axis']
        #umaps[scope_i+'_'+df_i].to_csv('output/reboot/'+scope_i+'_'+df_i+'_rotate_'+str(angle_i)+'norm.csv',sep=';')
        #pd.merge1
        df = dfs[scope_i+'_'+df_i]
        try:
            df.set_index('subject_id',inplace=True)
        except:
            #print('index already set')
            pass
        
        #print('Validation df + umap merger')
        #print('{} pre merger with df: {}'.format(scope_i+'_'+df_i,umaps[scope_i+'_'+df_i].shape))
        umaps[scope_i+'_'+df_i] = pd.merge(umaps[scope_i+'_'+df_i],df,how='inner',left_index=True,right_index=True)
        #print('{} post merger with df: {}'.format(scope_i+'_'+df_i,umaps[scope_i+'_'+df_i].shape))
    
        if df_i == 'functional':

            #print('{} pre 2nd merger with df: {}'.format(scope_i+'_'+df_i,umaps[scope_i+'_'+df_i].shape))
            umaps[scope_i+'_'+df_i] = pd.merge(umaps[scope_i+'_'+df_i],
                                     functional_list.loc[:,[x+'_score_final' for x in ['Kings','MiToS','FT9']]+['computeZones_'+x+'_'+y for x in ['mouth','leg','trunk','respiratory','hand'] for y in ['baseline','final']]],
                                     how='left',
                                     left_index=True,
                                     right_index=True)
            
            # Skipped for functional analysis
            '''
            mask_patients_stage = ((~umaps[scope_i+'_'+df_i].death_unit.isnull())&(umaps[scope_i+'_'+df_i].death_unit<=12.1)&(umaps[scope_i+'_'+df_i].source!='trophos'))
            for stage_i in ['Kings','MiToS','FT9']:
                 umaps[scope_i+'_'+df_i].loc[mask_patients_stage,stage_i+'_score_final'] = 5
            for x in ['trunk','mouth','leg','hand','respiratory']:
                umaps[scope_i+'_'+df_i].loc[mask_patients_stage,'computeZones_'+x+'_final']=0
            '''
        
        
            #print('{} post 2nd merger with df: {}'.format(scope_i+'_'+df_i,umaps[scope_i+'_'+df_i].shape))
            
            '''
            feat_ALSFRS = {'hand':['Q4_Handwriting_baseline','Q5_Cutting_baseline',],
                       'trunk':['Q6_Dressing_and_Hygiene_baseline','Q7_Turning_in_Bed_baseline',],
                       'respiratory':['Q10_Respiratory_baseline',],
                       'leg':['Q8_Walking_baseline','Q9_Climbing_Stairs_baseline',],
                       'mouth':['Q1_Speech_baseline','Q2_Salivation_baseline','Q3_Swallowing_baseline',]}
        
            def sumFeats(x,feats):
                y = np.nan
                idx = 0
                for feat_i in feats:
                    if not np.isnan(x[feat_i]):
                        if idx == 0:
                            y = x[feat_i]
                            idx = 1
                    else:
                        y += x[feat_i]
                return y
        
            for elem_i in ['hand','trunk','respiratory','leg','mouth']:
                umaps[scope_i+'_'+df_i]['computeZones_'+elem_i+'_baseline'] = umaps[scope_i+'_'+df_i].loc[:,feat_ALSFRS[elem_i]].apply(lambda x: sumFeats(x,feat_ALSFRS[elem_i]),axis=1)
            '''
    
    #print('224: {}'.format('MiToS_score_baseline' in list(umaps['dev_'+df_i].columns)))
        
    ##########################################################################
    #                                 INPUT
    ##########################################################################

    print('Start input plot')
    
    if df_i == 'functional':
        '''
        fig,ax = plt.subplots(1,1,figsize=(2,4))
        ax.scatter(umaps['dev_'+df_i]['x_axis'],
                           umaps['dev_'+df_i]['y_axis'],
                           c=['xkcd:gray']*umaps['dev_'+df_i]['x_axis'].shape[0],
                           alpha=1,
                           s=1,
                           label='Patient projection')
        lgnd = ax.legend(loc='lower center',
                                 bbox_to_anchor=(0.5,-0.12),
                                 ncol=1,
                                 borderaxespad=0,
                                     frameon=False,
                                     fontsize='small',
                                     )
        for lgd_idx in range(len(lgnd.legendHandles)):
            lgnd.legendHandles[lgd_idx]._sizes = [20]
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.set_title('a. UMAP projection \n ')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # REMOVE AS INCLUDED IN INPUT
        #plt.savefig('output/reboot/func_plain_patient_proj.pdf')
        #plt.close()
        '''
    fig, axes = plt.subplots(params['input_subplot'][df_i][0],
                             params['input_subplot'][df_i][1],
                             #sharex=True,
                             #sharey=True,
                             figsize=(params['input_subplot_size'][df_i][0],
                                      params['input_subplot_size'][df_i][1]))
    
    combinations = {0:(0,0),1:(0,1),2:(0,2),3:(0,3),
                    4:(1,0),5:(1,1),6:(1,2),7:(1,3),
                    8:(2,0),9:(2,1),10:(2,2),11:(2,3)}
    colors = [(0/255,114/255,178/255),(230/255,159/255,0/255),'y','m','g','k','c']
    feat_labels = {'onset_spinal':['Bulbar onset','Spinal onset'],
                   'sex':['Female patient','Male patient']} 
    
    cdict_green2red = {'red':   ((0.0, 0/255,0/255 ),
                   (0.5,240/255,240/255),
                   (1.0,204/255,204/255 )),
    
         'blue':  ((0.0, 178/255, 178/255),
                   (0.5,66/255,66/255),
                   (1.0,167/255 ,167/255 )),
                   
         'green': ((0.0, 114/255,114/255 ),
                   (0.5,228/255,228/255),
                   (1.0,121/255 ,121/255 ))}

    cdict_red2green = {'red':   ((0.0, 204/255,204/255 ),
                   (0.5,240/255,240/255),
                   (1.0,0/255,0/255 )),
    
         'blue':  ((0.0, 167/255, 167/255),
                   (0.5,66/255,66/255),
                   (1.0,178/255 ,178/255 )),
                   
         'green': ((0.0, 121/255,121/255 ),
                   (0.5,228/255,228/255),
                   (1.0,114/255 ,114/255 ))}

    cmap_red2green = mcolors.LinearSegmentedColormap(
        'my_colormap', cdict_red2green, 100)

    cmap_green2red = mcolors.LinearSegmentedColormap(
        'my_colormap', cdict_green2red, 100)
       
    idx = 0
    if df_i == 'survival':
    
        ax = axes[combinations[idx][0]][combinations[idx][1]]
        cmap = plt.cm.rainbow
        
    
        ax.scatter(umaps['dev_'+df_i]['x_axis'],
                           umaps['dev_'+df_i]['y_axis'],
                           c=['xkcd:gray']*umaps['dev_'+df_i]['y_axis'].shape[0],
                           alpha=1,
                           s=1,
                           label='Patient projection')
        lgnd = ax.legend(loc='lower center',
                                 bbox_to_anchor=(0.5,-0.16),
                                 ncol=1,
                                 borderaxespad=0,
                                     frameon=False,
                                     fontsize='small',
                                     )
        for lgd_idx in range(len(lgnd.legendHandles)):
            lgnd.legendHandles[lgd_idx]._sizes = [20]
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.set_title('a. UMAP projection \n ')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if dict_input_spines[df_i][idx][0] == 0:
            ax.spines['bottom'].set_visible(False)
            ax.set_xticks([])
        if dict_input_spines[df_i][idx][1] == 0:
            ax.spines['left'].set_visible(False)  
            ax.set_yticks([])

        idx += 1
        
    nbs = {'weight_baseline':3,'time_disease':3,'ALSFRS_Total_estim_slope':3}
    
    for feat_i in params['input_feat'][df_i]: 
        ax = axes[combinations[idx][0]][combinations[idx][1]]#fig.add_subplot(idx)
        ax.set_title(params['input_title'][df_i][feat_i])
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        
        ss = [1,0.7,1,0.7]
        ss_idx =  0
        if feat_i in ['onset_spinal','sex']:
            color_idx = 0
            
            
            for elem_i in sorted(list(umaps['dev_'+df_i].loc[:,feat_i].unique())):
                ax.scatter(umaps['dev_'+df_i].loc[umaps['dev_'+df_i][feat_i]==elem_i,'x_axis'],
                            umaps['dev_'+df_i].loc[umaps['dev_'+df_i][feat_i]==elem_i,'y_axis'],
                            c=[colors[color_idx]]*umaps['dev_'+df_i].loc[umaps['dev_'+df_i][feat_i]==elem_i,'x_axis'].shape[0],
                            s=ss[ss_idx],label=feat_labels[feat_i][color_idx])
                color_idx += 1
                ss_idx += 1
            lgnd = ax.legend(loc='lower center',
                             bbox_to_anchor=(0.5,-0.17),
                             ncol=1,
                             borderaxespad=0,
                                 frameon=False,
                                 fontsize='small',
                                 )
                
            for lgd_idx in range(len(lgnd.legendHandles)):
                lgnd.legendHandles[lgd_idx]._sizes = [20]
                                        
        else:
            if feat_i == 'time_disease':
                df = umaps['dev_'+df_i].loc[umaps['dev_'+df_i][feat_i]<100,:]
            elif feat_i == 'weight_baseline':
                df = umaps['dev_'+df_i].loc[umaps['dev_'+df_i][feat_i]<130,:]
            elif feat_i == 'ALSFRS_Total_estim_slope':
                df = umaps['dev_'+df_i].loc[(umaps['dev_'+df_i][feat_i]>-1.5),:]
            else:
                df = umaps['dev_'+df_i]
            tmp = df[feat_i].apply(lambda x: x/abs(df[feat_i]).max())
            slope_i = abs(df[feat_i]).max()
            if feat_i in ['ALSFRS_Total_estim_slope','ALSFRS_Total_baseline','computeZones_hand_baseline',
                          'computeZones_respiratory_baseline','computeZones_mouth_baseline',
                          'computeZones_leg_baseline','computeZones_trunk_baseline',]:  
                points = ax.scatter(df['x_axis'],df['y_axis'],c=tmp,cmap = cmap_green2red,s=1)
            else:
                points = ax.scatter(df['x_axis'],df['y_axis'],c=tmp,cmap = cmap_red2green,s=1,vmin=0,vmax=1)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.1)
            
            lin_size = 5
            
            if feat_i != 'ALSFRS_Total_estim_slope':
                v1 = np.linspace(0,1,lin_size, endpoint=True)
            else:
                v1 = np.linspace(-1,0,lin_size, endpoint=True)
            cb = fig.colorbar(points,cax=cax,ticks=v1,orientation='vertical')
            if feat_i != 'ALSFRS_Total_estim_slope':
                cb.ax.set_yticklabels(['{:3.0f}'.format(i*slope_i) for i in v1])
            else:
                #print([i*slope_i for i in v1])
                a = ['{:3.2f}'.format(i*slope_i) for i in v1]
                #a.reverse()
                cb.ax.set_yticklabels(a)
                #cax.yaxis.set_ticks_position('left')
                
        if dict_input_spines[df_i][idx][0] == 0:
            ax.spines['bottom'].set_visible(False)
            ax.set_xticks([])
            ax.tick_params(bottom='off')
        else:
            ax.set_xticks([0,0.5,1])
        if dict_input_spines[df_i][idx][1] == 0:
            ax.spines['left'].set_visible(False)  
            ax.set_yticks([])
            ax.tick_params(left='off')
        else:
            ax.set_yticks([0,0.2,0.4,0.6,0.8,1])

        idx += 1
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False) 
    
    print('End plot input')
        
    fig.text(0.5, 0.07, 'Dim n1 (UMAP)', ha='center')
    fig.text(0.07, 0.5, 'Dim n2 (UMAP)', va='center', rotation='vertical')
    
    
    
    
    if df_i == 'functional':
        plt.subplots_adjust(wspace=0.47,hspace=0.45)
    else:
        plt.subplots_adjust(wspace=0.37,hspace=0.42)
    fig.savefig('output/reboot/article/'+df_i+'/pdf/'+df_i+'_input.pdf')
    fig.savefig('output/reboot/article/'+df_i+'/png/'+df_i+'_input.png',dpi=1500)
    #plt.close()        

    ##########################################################################
    #                                 Output
    ##########################################################################

    print('Start output plot')
    
    colors = [(230/255,159/255,0/255),(0/255,114/255,178/255),'y','m','g','k','c']

    feat_labels = {'survived':['Patient deceased','Patient survived'],
                   'source':['proact','exonhit','trophos','pre2008']}  
    
    slope = {}
    intercept = {}
    df
    slope['survived'] = 1
    intercept['survived'] = 0
    for zone_i in ['hand','leg','trunk']:
        slope['computeZones_'+zone_i+'_final'] = 8
        intercept['computeZones_'+zone_i+'_final'] = 0
    for score_i in ['MiToS','FT9']:
        slope[score_i+'_score_last'] = 4
        intercept[score_i+'_score_last'] = 0
    slope['computeZones_mouth_final'] = 12
    intercept['computeZones_mouth_final'] = 0
    slope['computeZones_respiratory_final'] = 4
    intercept['computeZones_respiratory_final'] = 0
    
    
    slope['Kings_score_last'] = 3.5
    slope['ALSFRS_Total_final'] = 40
    slope['death_unit'] = 13 #66.79 # based on df_with_last.csv (patient_db)
    slope['source'] = 1
    intercept['Kings_score_last'] = 1 
    intercept['ALSFRS_Total_final'] = 0
    intercept['death_unit'] = 0
    intercept['source'] = 0
    
    title = {}
    
    # Orange [230/255,159/255,0/255]
    # Blue [0/255,114/255,178/255]
    # Yellow [240/255,228/255,66/255]
    # Vermillion [213/255,94/255,0/255]
    # Bluish green [0/255,158/255,115/255]
    # Blue to Yellow to Vermillion 
    

    
    #print('461: {}'.format('MiToS_score_baseline' in list(umaps['dev_'+df_i].columns)))

        
    fig, axes = plt.subplots(params['output_subplot'][df_i][0],
                             params['output_subplot'][df_i][1],
                             #sharex=True,
                             #sharey=True,
                             figsize=(params['output_subplot_size'][df_i][0],
                                      params['output_subplot_size'][df_i][1]))
        #fig = plt.figure(figsize=(6,6))
        #plt.title('{}'.format(feat_i))
    
        #idx = 231
    idx = 0
        #plt.axis('off')
    combinations = {0:(0,0),1:(0,1),2:(0,2),3:(1,0),4:(1,1),5:(1,2),}
        
    for feat_i in params['output_feats'][df_i]:
    #'Kings_score_last','MiToS_score_last','FT9_score_last']: 
        print('feat_i:{}'.format(feat_i))
        # Categorical
        if len(umaps['dev_'+df_i].loc[:,feat_i].unique())<=4:
            color_idx = 0
            # remove exonhit patients 
            patient_mask = (umaps['dev_'+df_i].index.isin(patients_kept))
            #print('output: {} - pre selection patient ={}'.format(feat_i,umaps['dev_'+df_i].shape))

            df_dev = umaps['dev_'+df_i].loc[patient_mask,['x_axis','y_axis']+[feat_i]]
            #print('output: {} - pre drop={}'.format(feat_i,df_dev.shape))
            df_dev.dropna(inplace=True)
            print('output: {} - post drop={}'.format(feat_i,df_dev.shape))
            color_idx = 0
            # Overall case
            
        
            ss = {1:0.2,0:1}    
            ax = axes[idx]#fig.add_subplot(idx)
            ax.set_title(params['output_title'][df_i][feat_i])
            ax.set_xlim([0,1])
            ax.set_ylim([0,1])
            #print('f:{}, {},{}'.format(feat_i,idx,combinations[idx]))
            for elem_i in sorted(list(df_dev.loc[:,feat_i].unique())):
                ax.scatter(df_dev.loc[df_dev[feat_i]==elem_i,'x_axis'],
                            df_dev.loc[df_dev[feat_i]==elem_i,'y_axis'],
                            c=[colors[color_idx]]*df_dev.loc[df_dev[feat_i]==elem_i,'y_axis'].shape[0],
                            alpha=ss[elem_i],
                            s=1,label=feat_labels[feat_i][color_idx])
                color_idx += 1
                #plt.axis('off')
            lgnd = ax.legend(loc='lower center',
                             bbox_to_anchor=(0.5,-0.19),
                             ncol=1,
                             borderaxespad=0,
                                 frameon=False,
                                 fontsize='small',
                                 )
                
            for lgd_idx in range(len(lgnd.legendHandles)):
                lgnd.legendHandles[lgd_idx]._sizes = [20]
                            
            
            #plt.savefig('output/plot_color/all_'+feat_i+'.pdf')
            # Element based case
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False) 
                
        else:
            
            patient_mask = (umaps['dev_'+df_i].index.isin(patients_kept))
            #print('output: {} - pre selection patient ={}'.format(feat_i,umaps['dev_'+df_i].shape))
            
            #if feat_i == 'ALSFRS_Total_final':
                #print(ddd+5)
                
                
            max_feat = umaps['dev_'+df_i].loc[patient_mask,feat_i].max() 
            Y_tmp_norm = umaps['dev_'+df_i].loc[patient_mask,feat_i].apply(lambda x: x/max_feat)
            df_dev = pd.merge(umaps['dev_'+df_i].loc[patient_mask,['x_axis','y_axis']],Y_tmp_norm,how='left',left_index=True,right_index=True)
            #print('output: {} - pre drop={}'.format(feat_i,df_dev.shape))
            df_dev_gb_bef = df_dev.copy(deep=True)
            df_dev.dropna(inplace=True)
            df_dev_gb = df_dev.copy(deep=True)
            
            df_dev_gb_bef.reset_index(inplace=True)
            df_dev_gb.reset_index(inplace=True)
            
            df_dev_gb_bef['source'] = df_dev_gb_bef.loc[:,'subject_id'].apply(lambda x: x.split('_')[0])
            if feat_i == 'ALSFRS_Total_final':
                #df_dev_gb_bef.to_csv('output/reboot/'+df_i+'_check_discrepancies.csv',index=False,sep=';')
                pass
            #df_dev_gb_bef.loc[df_dev_gb_bef.source=='exonhit',:].to_csv('output/reboot/'+df_i+'_exonhit_bef.csv',index=False,sep=';')
            
            df_dev_gb['source'] = df_dev_gb.loc[:,'subject_id'].apply(lambda x: x.split('_')[0])
            #df_dev_gb.loc[df_dev_gb.source=='exonhit',:].to_csv('output/reboot/'+df_i+'_exonhit.csv',index=False,sep=';')

            if feat_i=='Kings_score_final':
                #df_dev_gb.to_csv('output/reboot/'+df_i+'_check_discrepancies_Kings.csv',sep=';',index=False)
                pass
            
            #print('BEF')
            #print(df_dev_gb_bef.loc[:,['subject_id','source']].groupby('source').count())
            #print('AFTER')
            print(df_dev_gb.loc[:,['subject_id','source']].groupby('source').count())

            print('output: {} - post drop={}'.format(feat_i,df_dev.shape))    
            #ax = fig.add_subplot(idx)
            ax = axes[idx]#axes[combinations[idx][0],combinations[idx][1]]
            ax.set_title(params['output_title'][df_i][feat_i])
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)    
            if feat_i in ['death_unit','ALSFRS_Total_final']:
                cmap_tmp = cmap_red2green
            else:
                cmap_tmp = cmap_green2red
            
            points = ax.scatter(df_dev['x_axis'],df_dev['y_axis'],c=df_dev[feat_i],cmap = cmap_tmp,s=1,vmin=0,vmax=1)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.1)
            
            #plt.axis('off')
            
            if feat_i == 'death_unit':
                lin_size = 6
            elif feat_i == 'survived':
                lin_size = 2
            elif feat_i == 'source':
                lin_size = 4
            else:
                lin_size = 6
                
            v1 = np.linspace(0,1,lin_size, endpoint=True)
            cb = fig.colorbar(points,cax=cax,ticks=v1,orientation='vertical')
            if feat_i == 'survived':
                cb.ax.set_yticklabels(['survived','dead'])
            elif feat_i in ['ALSFRS_Total_final','MiToS_score_last','FT9_score_last']:
                cb.ax.set_yticklabels(['{:3.0f}'.format(i*slope[feat_i]+intercept[feat_i]) for i in v1])
            elif feat_i in ['death_unit']:
                list_cb = ['{:3.0f}'.format(i*slope[feat_i]+intercept[feat_i]) for i in v1]
                list_cb.pop(-1)
                list_cb.append('13+')
                cb.ax.set_yticklabels(list_cb)
            else:
                cb.ax.set_yticklabels(['1','2','3','4','4.5','5'])
            ax.set_xlim([0,1])
            ax.set_ylim([0,1])
            
        if dict_output_spines[df_i][idx][0] == 0:
            ax.spines['bottom'].set_visible(False)
            ax.set_xticks([])
        if dict_output_spines[df_i][idx][1] == 0:
            ax.spines['left'].set_visible(False)
            ax.set_yticks([])
        idx += 1
    #axes[1].set_xlabel('Dim n1 (UMAP)',ha='center')
    fig.text(0.04, 0.5, 'Dim n2 (UMAP)', va='center', rotation='vertical')
    if df_i == 'survival':
        fig.text(0.5,-0.05, 'Dim n1 (UMAP)', ha='center')
    else:
        fig.text(0.5,-0.02, 'Dim n1 (UMAP)', ha='center')
    plt.subplots_adjust(wspace=0.37,hspace=0.42)
    fig.savefig('output/reboot/article/'+df_i+'/pdf/'+df_i+'_output.pdf', bbox_inches='tight')
    fig.savefig('output/reboot/article/'+df_i+'/png/'+df_i+'_output.png', dpi=1500)

    #print('580: {}'.format('MiToS_score_baseline' in list(umaps['dev_'+df_i].columns)))

    
    #plt.close()
    
    if df_i == 'functional':
    
        
        
        ####################################################################
        # REAL OUTPUT PLOT FUNCTIONAL - FUNCTIONAL OUTPUT PLOT
        ##################################################################
        # 9 output plot 
        fig, axes = plt.subplots(3,
                                 3,
                                 sharex=True,
                                 sharey=True,
                                 figsize=(8,
                                          12))
        idx = 0
        #plt.axis('off')
        combinations = {0:(0,0),
                        1:(0,1),
                        2:(0,2),
                        3:(1,0),
                        4:(1,1),
                        5:(1,2),
                        6:(2,0),
                        7:(2,1),
                        8:(2,2)}
        
        for feat_i in ['ALSFRS_Total_final','Kings_score_final','MiToS_score_final','FT9_score_final']+ ['computeZones_'+x+'_final' for x in ['hand','leg','trunk','mouth','respiratory',]]:
            
            # Categorical
            if len(umaps['dev_'+df_i].loc[:,feat_i].unique())<=4:
                color_idx = 0
                df_dev = umaps['dev_'+df_i].loc[patient_mask,['x_axis','y_axis']+[feat_i]]
                df_dev.dropna(inplace=True)
                
                color_idx = 0
                # Overall case
                
            
                ss = {1:0.2,0:1}    
                ax = axes[combinations[idx]]#fig.add_subplot(idx)
                ax.set_title(params['output_title2'][df_i][feat_i])
                ax.set_xlim([0,1])
                ax.set_ylim([0,1])
                #print('f:{}, {},{}'.format(feat_i,idx,combinations[idx]))
                for elem_i in sorted(list(df_dev.loc[:,feat_i].unique())):
                    ax.scatter(df_dev.loc[df_dev[feat_i]==elem_i,'x_axis'],
                                df_dev.loc[df_dev[feat_i]==elem_i,'y_axis'],
                                c=[colors[color_idx]]*df_dev.loc[df_dev[feat_i]==elem_i,'y_axis'].shape[0],
                                alpha=ss[elem_i],
                                s=1,label=feat_labels[feat_i][color_idx])
                    color_idx += 1
                    #plt.axis('off')
                lgnd = ax.legend(loc='lower center',
                                 bbox_to_anchor=(0.5,-0.19),
                                 ncol=1,
                                 borderaxespad=0,
                                     frameon=False,
                                     fontsize='small',
                                     )
                    
                for lgd_idx in range(len(lgnd.legendHandles)):
                    lgnd.legendHandles[lgd_idx]._sizes = [20]
                                
                
                #plt.savefig('output/plot_color/all_'+feat_i+'.pdf')
                # Element based case
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False) 
                    
            else:
                        
                umaps['dev_'+df_i].loc[:,['x_axis','y_axis']+[feat_i]]
                
                max_feat = umaps['dev_'+df_i][feat_i].max() 
                Y_tmp_norm = umaps['dev_'+df_i][feat_i].apply(lambda x: x/max_feat)
                df_dev = pd.merge(umaps['dev_'+df_i].loc[patient_mask,['x_axis','y_axis']],Y_tmp_norm,how='left',left_index=True,right_index=True)
                df_dev.dropna(inplace=True)            
        
                #ax = fig.add_subplot(idx)
                ax = axes[combinations[idx]]#axes[combinations[idx][0],combinations[idx][1]]
                ax.set_title(params['output_title2'][df_i][feat_i])
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)    
                if feat_i in ['death_unit','ALSFRS_Total_final','MiToS_score_final','FT9_score_final','Kings_score_final']:
                    cmap_tmp = cmap_green2red #cmap_red2green
                else:
                    cmap_tmp = cmap_green2red
                
                points = ax.scatter(df_dev['x_axis'],df_dev['y_axis'],c=df_dev[feat_i],cmap = cmap_tmp,s=1,vmin=0,vmax=1)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.1)
                
                #plt.axis('off')
                
                if feat_i == 'death_unit':
                    lin_size = 6
                elif feat_i == 'survived':
                    lin_size = 2
                elif feat_i == 'source':
                    lin_size = 4
                elif feat_i == 'computeZones_mouth_final':
                    lin_size = 13
                elif feat_i == 'computeZones_trunk_final':
                    lin_size = 9
                elif feat_i == 'computeZones_respiratory_final':
                    lin_size = 5
                elif feat_i == 'computeZones_hand_final':
                    lin_size= 9
                elif feat_i == 'computeZones_leg_final':
                    lin_size = 9
                elif feat_i in ['Kings_score_final','FT9_score_final','MiToS_score_final']:
                    lin_size = 6
                else:
                    # ALSFRS
                    lin_size = 5
                    
                v1 = np.linspace(0,1,lin_size, endpoint=True)
                cb = fig.colorbar(points,cax=cax,ticks=v1,orientation='vertical')
                if feat_i == 'survived':
                    cb.ax.set_yticklabels(['survived','dead'])
                elif feat_i in ['ALSFRS_Total_final','MiToS_score_last','FT9_score_last']:
                    cb.ax.set_yticklabels(['{:3.0f}'.format(i*slope[feat_i]+intercept[feat_i]) for i in v1])
                elif feat_i in ['death_unit']:
                    list_cb = ['{:3.0f}'.format(i*slope[feat_i]+intercept[feat_i]) for i in v1]
                    list_cb.pop(-1)
                    list_cb.append('13+')
                    cb.ax.set_yticklabels(list_cb)
                elif feat_i in ['computeZones_'+x+'_final' for x in ['mouth','trunk','leg','respiratory','hand']]:
                    cb.ax.set_yticklabels(['{:3.0f}'.format(i*slope[feat_i]+intercept[feat_i]) for i in v1])
                else:
                    cb.ax.set_yticklabels(['1','2','3','4','4.5','5'])
                ax.set_xlim([0,1])
                ax.set_ylim([0,1])
                if output_dict_9ALSFRS[idx][0] == 0:
                    ax.spines['bottom'].set_visible(False) 
                    #ax.set_xticks([])
                if output_dict_9ALSFRS[idx][1] == 0:
                    ax.spines['left'].set_visible(False) 
                    #ax.set_yticks([])
                
            idx += 1
        #axes[1].set_xlabel('Dim n1 (UMAP)',ha='center')
        fig.text(0.5,0.08, 'Dim n1 (UMAP)', ha='center')
        fig.text(0.04, 0.5, 'Dim n2 (UMAP)', va='center', rotation='vertical')
        plt.subplots_adjust(wspace=0.42,hspace=0.32)
        fig.savefig('output/reboot/article/'+df_i+'/pdf/'+df_i+'_output_9panels.pdf', bbox_inches='tight')
        fig.savefig('output/reboot/article/'+df_i+'/png/'+df_i+'_output_9panels.png', bpi=1500,bbox_inches='tight')

    print('End output plot')
    
    ##########################################################################
    #                                 Grid
    ##########################################################################

    print('Start grid plot')
    
    min_x = umaps['dev_'+df_i]['x_axis'].min()
    min_y = umaps['dev_'+df_i]['y_axis'].min()
    max_x = umaps['dev_'+df_i]['x_axis'].max()
    max_y = umaps['dev_'+df_i]['y_axis'].max()
    delta_x = max_x-min_x
    delta_y = max_y-min_y
    
    
    value_cells = 20
    sizeRec = 1/value_cells        
    frontiers_x = [round(min_x+x*delta_x/value_cells,2) for x in range(value_cells)]
    frontiers_y = [round(min_y+x*delta_y/value_cells,2) for x in range(value_cells)]
    frontiers = (frontiers_x,frontiers_y)
    
    
    
    if df_i == 'survival':
        
        df_dev = umaps['dev_'+df_i].loc[patient_mask,:]
        
        umaps['dev_'+df_i+'_dead'] = df_dev.loc[df_dev.survived==0,:]
        umaps['dev_'+df_i+'_alive'] = df_dev.loc[df_dev.survived==1,:]
        #print('{}: all: {} alive:{} dead:{}'.format(df_i,df_dev.shape,umaps['dev_'+df_i+'_alive'].shape,umaps['dev_'+df_i+'_dead'].shape))
        #umaps['dev_'+df_i].drop(['survived'],axis=1,inplace=True)
        
        fig, axes = plt.subplots(1,4, figsize=(8/3*4,4)) # sharex=True,sharey=True,
        
        
        
        ax = axes[0]
        
        ax.scatter(umaps['dev_survival_alive']['x_axis'],
                   umaps['dev_survival_alive']['y_axis'],
                   marker='.',
                   c=[(0/255,114/255,178/255)]*umaps['dev_survival_alive']['y_axis'].shape[0],
                   alpha=0.9,
                   s=1,
                   label='Patient survived')
        ax.scatter(umaps['dev_survival_dead']['x_axis'],
                   umaps['dev_survival_dead']['y_axis'],
                   marker='D',
                   c=[(230/255,159/255,0/255)]*umaps['dev_survival_dead']['y_axis'].shape[0],
                   alpha=1,
                   s=1,
                   label='Patient deceased')
        
        
        lgnd = ax.legend(loc='lower center',
                  bbox_to_anchor=(0.5,-0.20),
                  ncol=1,
                  borderaxespad=0,
                  frameon=False,
                  fontsize='small',
                  )
        
        #for lgd_idx in range(len(lgnd.legendHandles)):
        lgnd.legendHandles[0]._sizes = [80]
        lgnd.legendHandles[1]._sizes = [25]
        
        
        ax.set_ylabel('Dim n2 (UMAP)')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)  
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.set_title('a. 1-year survival \n ')  
        
        lowers = [0.3862]#[x/10000 for x in range(3860,3870)]
        highers = [0.6722]#[x/10000 for x in range(6720,6730)]
        
        '''
        patient_mask = (umaps['dev_'+df_i].index.isin(patients_kept))
        print('output: {} - pre selection patient ={}'.format(feat_i,umaps['dev_'+df_i].shape))

        df_dev = umaps['dev_'+df_i].loc[patient_mask,['x_axis','y_axis']+[feat_i]]
        print('output: {} - pre drop={}'.format(feat_i,df_dev.shape))
        df_dev.dropna(inplace=True)
        print('output: {} - post drop={}'.format(feat_i,df_dev.shape))
        '''
        patient_mask = (umaps['dev_'+df_i].index.isin(patients_kept))
        df_dev = umaps['dev_'+df_i].loc[patient_mask,:]
        #print('df_dev (bef finding split): {}'.format(df_dev.shape))
        for lower_i in lowers:
            for higher_i in highers:
                
                sub = df_dev.loc[df_dev.y_axis<lower_i,:]
                high = df_dev.loc[df_dev.y_axis>higher_i,:]
                mid =  df_dev.loc[((df_dev.y_axis>lower_i) & (df_dev.y_axis<higher_i)),:]
        
                #print('{}-{}:sub shape: {}, mid shape: {},high shape:{}'.format(lower_i,higher_i,sub.shape,mid.shape,high.shape))

        z_alpha = 1.96
        P = umaps['dev_'+df_i+'_dead'].shape[0]/df_dev.shape[0]
        
        df_dev['group'] =  np.nan
        df_dev.loc[df_dev.y_axis<=lowers[0],'group'] = 'low'
        df_dev.loc[((df_dev.y_axis<=highers[0])&(df_dev.y_axis>lowers[0])),'group']='mid'
        df_dev.loc[df_dev.y_axis>highers[0],'group'] = 'high'
        
        #df_dev.to_csv('output/reboot/umaps_dev.csv',sep=';')

        
        grid_groupby = df_dev.reset_index().groupby('group').agg(({'onset_spinal':['sum'],
                               'sex':['sum'],
                               'survived':['sum'],
                               'age':['mean','std','min','max'],
                               'time_disease':['mean','std','min','max'],
                               'weight_baseline':['mean','std','min','max'],
                               'ALSFRS_Total_baseline':['mean','std','min','max'],
                               'ALSFRS_Total_estim_slope':['mean','std','min','max'],
                               'subject_id':['count'],
                               'ALSFRS_Total_final':['mean','std','min','max'],}))
        #print(grid_groupby)
        #grid_groupby.to_csv('output/reboot/grid_groupby.csv',sep=';')
        grid_all = [df_dev.onset_spinal.sum(),
                df_dev.sex.sum(),
                df_dev.survived.sum(),
                df_dev.age.mean(),
                df_dev.age.std(),
                df_dev.age.min(),
                df_dev.age.max(),
                df_dev.time_disease.mean(),
                df_dev.time_disease.std(),
                df_dev.time_disease.min(),
                df_dev.time_disease.max(),
                df_dev.weight_baseline.mean(),
                df_dev.weight_baseline.std(),
                df_dev.weight_baseline.min(),
                df_dev.weight_baseline.max(),
                df_dev.ALSFRS_Total_baseline.mean(),
                df_dev.ALSFRS_Total_baseline.std(),
                df_dev.ALSFRS_Total_baseline.min(),
                df_dev.ALSFRS_Total_baseline.max(),
                df_dev.ALSFRS_Total_estim_slope.mean(),
                df_dev.ALSFRS_Total_estim_slope.std(),
                df_dev.ALSFRS_Total_estim_slope.min(),
                df_dev.ALSFRS_Total_estim_slope.max(),
                df_dev.reset_index().subject_id.count(),
                df_dev.ALSFRS_Total_final.mean(),
                df_dev.ALSFRS_Total_final.std(),
                df_dev.ALSFRS_Total_final.min(),
                df_dev.ALSFRS_Total_final.max(),
        ]

        grid_all = pd.DataFrame(grid_all,columns=['all'],index=grid_groupby.columns).transpose()
        grid_all.reset_index(inplace=True)
        grid_groupby.reset_index(inplace=True)
        grid_groupby = grid_groupby.append(grid_all,ignore_index=True,sort=False)
        grid_groupby.set_index('group',inplace=True)

        grid_output_col = ['group','n','Survived (yes/no)','Gender (male/female)','Onset (spinal/bulbar)',
               'age (year)','time since onset (month)','baseline weight (kg)',
               'baseline ALSFRS','baseline ALSFRS decline rate','1-year ALSFRS']

        group_name = {'low':'low survival rate zone','mid':'intermediate survival rate zone','high':'high survival rate zone','all':'overall'}

        grid_output = []
        for idx,line_i in grid_groupby.iterrows():
            #print(idx)
            if idx in ['mid','low','high']:
                tmp = [group_name[idx]]
            else:
                tmp = ['overall']
            tmp.append(int(line_i['subject_id']['count']))
            for feat_i in ['survived','sex','onset_spinal','age','time_disease','weight_baseline','ALSFRS_Total_baseline','ALSFRS_Total_estim_slope','ALSFRS_Total_final']:
                if feat_i not in ['survived','sex','onset_spinal']:
                    if feat_i == 'ALSFRS_Total_estim_slope':    
                        tmp.append('{:.2f} +/- {:.2f} ({:.2f}-{:.2f}) '.format(line_i[(feat_i,'mean')],line_i[(feat_i,'std')],line_i[(feat_i,'min')],line_i[(feat_i,'max')]))
                    else:
                        tmp.append('{:.1f} +/- {:.1f} ({:.1f}-{:.1f}) '.format(line_i[(feat_i,'mean')],line_i[(feat_i,'std')],line_i[(feat_i,'min')],line_i[(feat_i,'max')]))
                else:
                    tmp.append('{:.0f}/{:.0f}'.format(line_i[(feat_i,'sum')],line_i[('subject_id','count')]-line_i[(feat_i,'sum')]))
            
            grid_output.append(tmp)

        grid_output = pd.DataFrame(grid_output,columns=grid_output_col)
        grid_output.set_index('group',inplace=True)
        grid_output = grid_output.reindex(index=['high survival rate zone','intermediate survival rate zone','low survival rate zone','overall'])
        grid_output.to_csv('output/reboot/article/survival/grid_stats_survival.csv',sep=';')

        percent= {}
        width={}
        all_count = {}
        d_count= {}
        a_count = {}
        all_count['sub'] = df_dev.loc[df_dev.y_axis<lower_i,:].shape[0]
        d_count['sub'] = umaps['dev_'+df_i+'_dead'].loc[umaps['dev_'+df_i+'_dead'].y_axis<lower_i,:].shape[0]
        a_count['sub']= umaps['dev_'+df_i+'_alive'].loc[umaps['dev_'+df_i+'_alive'].y_axis<lower_i,:].shape[0]
        percent['sub'] = d_count['sub']/all_count['sub']
        width['sub'] = 2*z_alpha*math.sqrt(P*(1-P)/all_count['sub'])*100
        
        all_count['mid'] = df_dev.loc[((df_dev.y_axis>lower_i) & (df_dev.y_axis<higher_i)),:].shape[0]
        d_count['mid'] = umaps['dev_'+df_i+'_dead'].loc[((umaps['dev_'+df_i+'_dead'].y_axis>lower_i) & (umaps['dev_'+df_i+'_dead'].y_axis<higher_i)),:].shape[0]
        a_count['mid'] = umaps['dev_'+df_i+'_alive'].loc[((umaps['dev_'+df_i+'_alive'].y_axis>lower_i) & (umaps['dev_'+df_i+'_alive'].y_axis<higher_i)),:].shape[0]
        percent['mid'] = d_count['mid']/all_count['mid']
        width['mid'] = 2*z_alpha*math.sqrt(P*(1-P)/all_count['mid'])*100
        
        all_count['high'] = df_dev.loc[df_dev.y_axis>higher_i,:].shape[0]
        d_count['high'] = umaps['dev_'+df_i+'_dead'].loc[umaps['dev_'+df_i+'_dead'].y_axis>higher_i,:].shape[0]
        a_count['high'] = umaps['dev_'+df_i+'_alive'].loc[umaps['dev_'+df_i+'_alive'].y_axis>higher_i,:].shape[0]
        percent['high'] = d_count['high']/all_count['high']
        width['high'] = 2*z_alpha*math.sqrt(P*(1-P)/all_count['high'])*100
        
        #print('survival high: {:.0f}% +/-{:.0f}%, mid: {:.0f}% +/-{:.0f}%, low: {:.0f}% +/- {:.0f}%'.format((1-percent['high'])*100,width['high'],(1-percent['mid'])*100,width['mid'],(1-percent['sub'])*100,width['sub']))
        ax = axes[2]
        ax.set_yticks([])

        markers= {'dead':'D','alive':'.'}
        
        ax.scatter(df_dev['x_axis'],
                   df_dev['y_axis'],
                   marker=markers['alive'],
                   c=['xkcd:gray']*df_dev['y_axis'].shape[0],
                   alpha=0.8,
                   s=4)
        
        ax.plot([0,1],lowers*2,c='k',linewidth=1,
                    alpha=0.8,
                    linestyle=':')
        ax.plot([0,1],highers*2,c='k',linewidth=1,
                    alpha=0.8,
                    linestyle=':')
        
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.set_title('c. 1-year survival \n basic division')
    
        t1 = ax.text(0.30,
             0.85,
             'survival rate: \n 90% (+/- 4%)',
             c='k',
             weight='bold',
             multialignment = 'center',
             fontsize='x-small',)
 
        t2 = ax.text(0.30,
                     0.55,
                     'survival rate: \n 80% (+/- 4%)',
                     c='k',
                     weight='bold',
                     multialignment = 'center',
                     fontsize='x-small',)
        
        t4 = ax.text(0.30,
                     0.25,
                     'survival rate: \n 58% (+/- 4%)',
                     c='k',
                     weight='bold',
                     multialignment = 'center',
                     fontsize='x-small',) 
    
   
        x_high = [0,0,1,1]
        y_high = [highers[0],1,1,highers[0]]
        x_mid = [0,0,1,1]
        y_mid = [lowers[0],highers[0],highers[0],lowers[0]]
        x_low = [0,0,1,1]
        y_low = [0,lowers[0],lowers[0],0]
        
        ax.fill(x_high,y_high,facecolor=(0,114/255,178/255),alpha=0.3)
        ax.fill(x_mid,y_mid,facecolor=(240/255,228/255,66/255),alpha=0.3)
        ax.fill(x_low,y_low,facecolor=(204/255,121/255,167/255),alpha=0.3)
        
        ### Additional plot with external data example
        #####################################################################

        # Due to scaling differences (patient scope different might explain scaling differences)
        # Patients used for poster cannot be used here so information changed
        # former patients used were 320, 1832 and 508
        # new patients used: 
        # pre2008_2816 for low category
        # Woman with spinal onset, aged 78 years old, 11 month since symptom onset, initial weight 64kg, initial ALSFRS 19.5 initial slope -1.79
        # pre2008_429 for med cat
        # Man age 57 in since 13 month baseline weight 71, initial ALSFRS 33 estima slope -0.5
        # pre2008_4922 for high
        # Woman 41 year olds spinal onset, 84 kg, 6.5 month since symptom onset, 36 initial ALSFRS, -0.6 ALSFRS point loss 

        
        ax = axes[3]
        ax.set_yticks([])

        markers= {'dead':'D','alive':'.'}
        
        ax.scatter(df_dev['x_axis'],
                   df_dev['y_axis'],
                   marker=markers['alive'],
                   c=['xkcd:gray']*df_dev['y_axis'].shape[0],
                   alpha=0.8,
                   s=4)
        
        ax.plot([0,1],lowers*2,c='k',linewidth=1,
                    alpha=0.8,
                    linestyle=':')
        ax.plot([0,1],highers*2,c='k',linewidth=1,
                    alpha=0.8,
                    linestyle=':')
        
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.set_title('d. prognosis estimation \n for new data')
        
        '''
        t1 = ax.text(0.25,
             0.75,
             'high survival \n rate group',
             c='k',
             weight='bold',
             multialignment = 'center',
             fontsize='x-small',)
 
        t2 = ax.text(0.25,
                     0.45,
                     'intermediate \n survival rate group',
                     c='k',
                     weight='bold',
                     multialignment = 'center',
                     fontsize='x-small',)
        
        t4 = ax.text(0.25,
                     0.15,
                     'low survival \n rate group',
                     c='k',
                     weight='bold',
                     multialignment = 'center',
                     fontsize='x-small',) 
        '''
        '''
        print('location low: {}, med: {} and high: {}'.format(umaps['val_survival'].loc[umaps['val_survival'].index.isin(['pre2008_2816']),['x_axis','y_axis']].values,
                                                              umaps['val_survival'].loc[umaps['val_survival'].index.isin(['pre2008_429']),['x_axis','y_axis']].values,
                                                              umaps['val_survival'].loc[umaps['val_survival'].index.isin(['pre2008_4922']),['x_axis','y_axis']].values))
        '''
        #ax.grid(b=True)
        # Low
        ax.scatter(umaps['val_survival'].loc[umaps['val_survival'].index.isin(['pre2008_2816']),'x_axis'],
           umaps['val_survival'].loc[umaps['val_survival'].index.isin(['pre2008_2816']),'y_axis'],
               s=5,
           c= 'xkcd:black',
          label='New patient projected')
        # Med
        ax.scatter(umaps['val_survival'].loc[umaps['val_survival'].index.isin(['pre2008_429']),'x_axis'],
           umaps['val_survival'].loc[umaps['val_survival'].index.isin(['pre2008_429']),'y_axis'],
           s=5,
           #label='New patient assigned to the intermediate survival rate zone' ,
           c='xkcd:black')   
        # High
        ax.scatter(umaps['val_survival'].loc[umaps['val_survival'].index.isin(['pre2008_4922']),'x_axis'],
           umaps['val_survival'].loc[umaps['val_survival'].index.isin(['pre2008_4922']),'y_axis'],
           s=5,
           #label='New patient assigned to the high survival rate zone' ,
           c='xkcd:black')
        
        # pre2008_4922 for high
        # Woman 41 year olds spinal onset, 84 kg, 6.5 month since symptom onset, 36 initial ALSFRS, -0.6 ALSFRS point loss 
        thigh = ax.text(0.18,0.80,
         ('Woman aged 41, 84 kg,\n 6.5 months in, spinal onset,\n'+
          ' 36 baseline ALSFRS score,\n -0.6 est. decline rate'),
         c='k',
          weight='bold',
         fontsize='xx-small',
         )
         #backgroundcolor='green')
         #thigh.set_bbox(dict(alpha=0.3,facecolor='green',edgecolor='green'))
         # pre2008_429 for med cat
         # Man age 57 in since 13 month baseline weight 71, initial ALSFRS 33 estima slope -0.5
 
        tmed = ax.text(0.20,0.50,
         ('Man aged 57, 71 kg,\n 13 months in, spinal onset,\n'+
          ' 33 baseline ALSFRS score,\n -0.5 est. decline rate'),
         c='k',
          weight='bold',
         fontsize='xx-small',
         )
         #backgroundcolor='orange')
         #tmed.set_bbox(dict(alpha=0.3,facecolor='orange',edgecolor='orange'))
         # pre2008_2816 for low category
         # Woman with spinal onset, aged 78 years old, 11 month since symptom onset, initial weight 64kg, initial ALSFRS 19.5 initial slope -1.79
        tlow = ax.text(0.20,0.05,
         ('Woman aged 78, 64 kg,\n 11 months in, spinal onset,\n'+
          ' 19 baseline ALSFRS score,\n -1.8 est. decline rate'),
         c='k',
          weight='bold',
         fontsize='xx-small',
         )
        
        #backgroundcolor='red')
        #tlow.set_bbox(dict(alpha=0.3,facecolor='red',edgecolor='red'))
        #plt.grid(linestyle='dotted')
        lgnd = ax.legend(loc='lower center',
          bbox_to_anchor=(0.5,-0.15),
          ncol=1,
          borderaxespad=0,
          frameon=False,
          fontsize='small',
          )
        for lgd_idx in range(len(lgnd.legendHandles)):
             lgnd.legendHandles[lgd_idx]._sizes = [20]
        #ax.grid(linestyle='dotted')
        
    
        x_high = [0,0,1,1]
        y_high = [highers[0],1,1,highers[0]]
        x_mid = [0,0,1,1]
        y_mid = [lowers[0],highers[0],highers[0],lowers[0]]
        x_low = [0,0,1,1]
        y_low = [0,lowers[0],lowers[0],0]
        
        ax.fill(x_high,y_high,facecolor=(0,114/255,178/255),alpha=0.3)
        ax.fill(x_mid,y_mid,facecolor=(240/255,228/255,66/255),alpha=0.3)
        ax.fill(x_low,y_low,facecolor=(204/255,121/255,167/255),alpha=0.3)
        
        #ax.set_xlabel('Dim n1 (UMAP)')
        #ax.set_ylabel('Dim n2 (UMAP)')
        #plt.title('UMAP projection split into {}'.format(split_i))
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
       
        
        ####################### GRID ########################################
        
        ax = axes[1]
        ax.set_yticks([])

        for elem_x in range(1,20):
            ax.plot([elem_x/20]*100,
                          np.linspace(-0.07,1.07,100),
                          c='k',
                          linestyle=':',
                          linewidth=0.8)
        for elem_y in range(1,20):
            ax.plot(np.linspace(-0.07,1.07,100),
                         [elem_y/20]*100,
                         c='k',
                         linestyle=':',
                         linewidth=0.8)
        
        
        
        patches = []
        color_done = []
        labels_done = []
        handles_done = []
        
        stats = {}
        stats['dev_'+df_i+'_alive'] = {}
        stats['dev_'+df_i+'_dead'] = {}
        stats['dev_'+df_i+'_2'] = {}
        enum = 0
        
        umaps['dev_'+df_i+'_2'] = df_dev.copy(deep=True)
        
        #print('mask start')
        for df_j in ['dev_'+df_i+'_2','dev_'+df_i+'_dead','dev_'+df_i+'_alive']:
            stats[df_i] = {}
            for elem_x in range(len(frontiers[0])):
                for elem_y in range(len(frontiers[1])):                
                    mask_x = (umaps[df_j].x_axis>=frontiers[0][elem_x])
                    mask_y = (umaps[df_j].y_axis>=frontiers[1][elem_y])
                    if elem_x != len(frontiers[0])-1:
                        mask_x = mask_x & (umaps[df_j].x_axis<frontiers[0][elem_x+1])
                    if elem_y != len(frontiers[1])-1:
                        mask_y = mask_y & (umaps[df_j].y_axis<frontiers[1][elem_y+1])   
                
                    stats[df_j][str(frontiers[0][elem_x])+'-'+str(frontiers[1][elem_y])] = umaps[df_j].loc[mask_x & mask_y,'x_axis'].count()
                
        #print('mask end')
        stats_color = {}

        for elem_i in stats['dev_'+df_i+'_2'].keys():
            if stats['dev_'+df_i+'_2'][elem_i] != 0:
                pb = stats['dev_'+df_i+'_dead'][elem_i]/stats['dev_'+df_i+'_2'][elem_i]*100
            else:
                pb = 0
                
            if pb>=60:
                # Vermillion
                colRec = (213,94,0)
                labelRec = 'Survival probability lower than 40%'
            elif pb<60 and pb>=40:
                # Reddish purple
                colRec = (204,121,167)
                labelRec = 'Survival probability between 40% and 60%'
            elif pb>=20 and pb<40:
                # Orange
                colRec = (230,159,0)
                labelRec = 'Survival probability between 60% and 80%'
            elif pb>=10 and pb<20:
                # Yellow
                colRec = (240,228,66)
                labelRec = 'Survival probability between 80% and 90%'
            elif pb==0 and stats['dev_'+df_i+'_2'][elem_i]==0:
                colRec = 'xkcd:pale'
            else:
                # Bluish green
                colRec= (0,158,115)
                labelRec = 'Survival probability above 90%'
            stats_color[elem_i] = colRec
        
        '''
        arrays = {}
        indexes = {}
        
        for status_i in ['dead','alive']:
            for cat_i in ['low','med','high']:
                arrays['dev_'+df_i+'_'+status_i+'_'+cat_i] = []
                indexes['dev_'+df_i+'_'+status_i+'_'+cat_i] = []
        
        for status_i in ['dead','alive']:
            key = 'dev_'+df_i+'_'+status_i+'_'
            for i,row_i in umaps['dev_'+df_i+'_'+status_i].iterrows(): 
                print(i)
                if row_i.y_axis >= 0.80:
                    arrays[key+'high'].append(row_i)
                    indexes[key+'high'].append(i)
                if row_i.y_axis <0.80 and row_i.y_axis >= 0.75:
                    if row_i.x_axis >=0.3 and row_i.x_axis <0.5:
                        arrays[key+'med'].append(row_i)
                        indexes[key+'med'].append(i)
                    else:
                         arrays[key+'high'].append(row_i)
                         indexes[key+'high'].append(i)
                if row_i.y_axis <0.75 and row_i.y_axis >= 0.7:
                    if row_i.x_axis >=0.15 and row_i.x_axis <0.7:
                        arrays[key+'med'].append(row_i)
                        indexes[key+'med'].append(i)
                    else:
                         arrays[key+'high'].append(row_i)
                         indexes[key+'high'].append(i)
                if row_i.y_axis <0.70 and row_i.y_axis >= 0.6:
                    if row_i.x_axis >=0.15 and row_i.x_axis <0.8:
                        arrays[key+'med'].append(row_i)
                        indexes[key+'med'].append(i)
                    else:
                         arrays[key+'high'].append(row_i)
                         indexes[key+'high'].append(i)
                if row_i.y_axis <0.60 and row_i.y_axis >= 0.5:
                    if row_i.x_axis <0.9 and row_i.x_axis>=0.15:
                        arrays[key+'med'].append(row_i)
                        indexes[key+'med'].append(i)
                    else:
                         arrays[key+'high'].append(row_i)
                         indexes[key+'high'].append(i)
                
                if row_i.y_axis <0.50 and row_i.y_axis >= 0.45:
                    if row_i.x_axis >= 0.15: #row_i.x_axis >=0.2 and row_i.x_axis <0.45:
                        arrays[key+'med'].append(row_i)
                        indexes[key+'med'].append(i)
                    else:
                         arrays[key+'high'].append(row_i)
                         indexes[key+'high'].append(i)
                if row_i.y_axis <0.45 and row_i.y_axis >= 0.35:
                    if row_i.x_axis <0.9:
                        arrays[key+'med'].append(row_i)
                        indexes[key+'med'].append(i)
                    else:
                         arrays[key+'med'].append(row_i)
                         indexes[key+'med'].append(i)
                if row_i.y_axis <0.35 and row_i.y_axis >= 0.3:
                    if row_i.x_axis >=0.9 :
                        arrays[key+'med'].append(row_i)
                        indexes[key+'med'].append(i)
                    else:
                        if row_i.x_axis >=0.35 and row_i.x_axis <0.45:
                             arrays[key+'low'].append(row_i)
                             indexes[key+'low'].append(i)
                        else:
                            arrays[key+'med'].append(row_i)
                            indexes[key+'med'].append(i)
                if row_i.y_axis <0.30 and row_i.y_axis >= 0.25:
                    if row_i.x_axis >=0.15 and row_i.x_axis <0.55:
                        arrays[key+'low'].append(row_i)
                        indexes[key+'low'].append(i)
                    else:
                         arrays[key+'med'].append(row_i)
                         indexes[key+'med'].append(i)
                if row_i.y_axis <0.25 and row_i.y_axis >= 0.2:
                    if row_i.x_axis<0.55: 
                        arrays[key+'low'].append(row_i)
                        indexes[key+'low'].append(i)
                    else:
                         arrays[key+'med'].append(row_i)
                         indexes[key+'med'].append(i)
                if row_i.y_axis <0.20:
                    arrays[key+'low'].append(row_i)
                    indexes[key+'low'].append(i)
            
                umaps['dev_'+df_i+'_'+status_i+'_low'] = pd.DataFrame(arrays[key+'low'],index=indexes[key+'low'],columns=umaps['dev_survival'].columns)
                umaps['dev_'+df_i+'_'+status_i+'_med'] = pd.DataFrame(arrays[key+'med'],index=indexes[key+'med'],columns=umaps['dev_survival'].columns)
                umaps['dev_'+df_i+'_'+status_i+'_high'] = pd.DataFrame(arrays[key+'high'],index=indexes[key+'high'],columns=umaps['dev_survival'].columns)
        
            
        for status_i in ['dead','alive']:
            for range_i in ['low','med','high']:
                umaps['dev_'+df_i+'_'+status_i+'_'+range_i].to_csv('output/reboot/input_for_process/dev_survival_'+status_i+'_'+range_i+'.csv',sep=';')
        '''
        
        
        for status_i in ['dead','alive']:
            for range_i in ['low','med','high']:
                try:
                    umaps['dev_'+df_i+'_'+status_i+'_'+range_i] = pd.read_csv('output/reboot/input_for_process/dev_survival_'+status_i+'_'+range_i+'.csv',sep=';')     
                except:
                    #print('dev_'+df_i+'_'+status_i+'_'+range_i+' not found')
                    pass
        
        pb_high =umaps['dev_'+df_i+'_alive_high'].shape[0]/(umaps['dev_'+df_i+'_dead_high'].shape[0]+umaps['dev_'+df_i+'_alive_high'].shape[0])*100
        pb_mid = umaps['dev_'+df_i+'_alive_med'].shape[0]/(umaps['dev_'+df_i+'_dead_med'].shape[0]+umaps['dev_'+df_i+'_alive_med'].shape[0])*100
        pb_low =umaps['dev_'+df_i+'_alive_low'].shape[0]/(umaps['dev_'+df_i+'_dead_low'].shape[0]+umaps['dev_'+df_i+'_alive_low'].shape[0])*100
        width_high = 2*z_alpha*math.sqrt(P*(1-P)/(umaps['dev_'+df_i+'_dead_high'].shape[0]+umaps['dev_'+df_i+'_alive_high'].shape[0]))*100
        width_mid = 2*z_alpha*math.sqrt(P*(1-P)/(umaps['dev_'+df_i+'_dead_med'].shape[0]+umaps['dev_'+df_i+'_alive_med'].shape[0]))*100
        width_low = 2*z_alpha*math.sqrt(P*(1-P)/(umaps['dev_'+df_i+'_dead_low'].shape[0]+umaps['dev_'+df_i+'_alive_low'].shape[0]))*100
        '''
        print('survival enhanced: high: {:.0f}%+/-{:.0f}%, mid: {:.0f}%+/-{:.0f}%, low: {:.0f}%+/-{:.0f}%'.format(pb_high,
              width_high,pb_mid,width_mid,pb_low,width_low))
         '''
        patches = []
        color_done = []
        labels_done = []
        handles_done = []
        
        #elem_i ='0.35-0.45'
        
        for elem_i in stats['dev_survival_2'].keys():        
            stat_all = stats['dev_survival_2'][elem_i]
            stat_alive = stats['dev_survival_alive'][elem_i]
            stat_dead = stats['dev_survival_dead'][elem_i]
            if stat_all != 0:
                pb = round(stat_dead/stat_all*100,1)
            else:
                pb = 0.0
            
            if pb>=60:
                colRec = (204/255,121/255,167/255)
                labelRec = r'            survival pb $\leq$ 40%'
            elif pb<60 and pb>=40:
                colRec = (213/255,94/255,0)
                labelRec = r'40% < survival pb $\leq$ 60%'
            elif pb>=20 and pb<40:
                colRec = (240/255,228/255,66/255)
                labelRec = r'60% < survival pb $\leq$ 80%'
            elif pb>=10 and pb<20:
                colRec = (0/255,158/255,115/255)
                labelRec = r'80% < survival pb $\leq$ 90%'
            elif pb==0 and stats['dev_survival_2'][elem_i]==0:
                colRec = 'xkcd:white'
            else:
                colRec=(0,114/255,178/255)
                labelRec = '90% < survival pb'
                
            first_time = {(204/255,121/255,167/255):0,
                          (213/255,94/255,0):0,
                          (240/255,228/255,66/255):0,
                          (0/255,158/255,115/255):0,
                          (0,114/255,178/255):0}
                
            rect = mpatches.Rectangle((float(elem_i.split('-')[0]),
                                   float(elem_i.split('-')[1])),
                                   sizeRec,
                                   sizeRec,
                                   facecolor=colRec,
                                   alpha=0.5)
            ax.add_artist(rect)
                        
            if colRec not in color_done and colRec != 'xkcd:white':
                color_done.append(colRec)
                handles_done.append(rect)
                labels_done.append(labelRec)
                
            # reorder for proper survival evolution
        #print(labels_done)
        labels_done2 =  [labels_done[0]]+[labels_done[4]]+[labels_done[2]]+[labels_done[3]]+[labels_done[1]]
        handles_done2 = [handles_done[0]]+[handles_done[4]]+[handles_done[2]]+[handles_done[3]]+[handles_done[1]]
                    
        ax.legend(handles_done2,
                      labels_done2,
                      loc='lower center',
                      labelspacing = 0.1,
                  bbox_to_anchor=(0.5,-0.40),
                  ncol=1,
                  borderaxespad=0,
                  frameon=False,
                  fontsize='small',
                  )
            
        

        
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.set_title('b. 1-year survival \n grid analysis')    

        fig.text(0.5, -0.23, 'Dim n1 (UMAP)', ha='center')
        plt.subplots_adjust(wspace=0.37)
        
        plt.savefig('output/reboot/article/survival/pdf/survival_grid_simple.pdf', bbox_inches='tight')
        plt.savefig('output/reboot/article/survival/png/survival_grid_simple.png', bpi=1500, bbox_inches='tight')

        #plt.close()

        def findGroup(x,lowers,highers):
            
            if x < lowers[0]:
                return 'low'
            elif x>= lowers[0] and x < highers[0]:
                return 'mid'
            else:
                return 'high'
        
        

        umaps['val_survival']['group'] = umaps['val_survival'].loc[:,'y_axis'].apply(lambda x: findGroup(x,lowers,highers))
        umaps['val_survival'].reset_index(inplace=True)
        val_gb = umaps['val_survival'].loc[:,['group','survived','subject_id']].groupby('group').agg({'survived':'sum','subject_id':'count'})
        val_gb['pb'] = val_gb['survived']/val_gb['subject_id']
        val_gb.to_csv('output/reboot/article/survival/validation_data_distribution.csv',sep=';')

        print('End grid plot')
    
    ##########################################################################
    #                                 Advanced grid
    ##########################################################################

    print('Start advanced grid plot')
    
    if df_i == 'survival':
        
        fig, axes = plt.subplots(1,3, sharex=True,sharey=True,figsize=(8,4))

        ax = axes[0]
        
        ax.scatter(umaps['dev_survival_alive']['x_axis'],
                   umaps['dev_survival_alive']['y_axis'],
                   marker='.',
                   c=[(0/255,114/255,178/255)]*umaps['dev_survival_alive']['y_axis'].shape[0],
                   alpha=0.9,
                   s=1,
                   label='Patient survived')
        ax.scatter(umaps['dev_survival_dead']['x_axis'],
                   umaps['dev_survival_dead']['y_axis'],
                   marker='D',
                   c=[(230/255,159/255,0/255)]*umaps['dev_survival_dead']['x_axis'].shape[0],
                   alpha=1,
                   s=1,
                   label='Patient deceased')
        
        
        lgnd = ax.legend(loc='lower center',
                  bbox_to_anchor=(0.5,-0.20),
                  ncol=1,
                  borderaxespad=0,
                  frameon=False,
                  fontsize='small',
                  )
        
        #for lgd_idx in range(len(lgnd.legendHandles)):
        lgnd.legendHandles[0]._sizes = [80]
        lgnd.legendHandles[1]._sizes = [25]
        
        
        ax.set_ylabel('Dim n2 (UMAP)')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)  
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.set_title('a. 1-year survival \n ') 
        
        
        ax = axes[2]
        ax.set_yticks([])

        markers= {'dead':'D','alive':'.'}
        
        for status_i in ['dead','alive']:
            for cat_i in ['low','med','high']:
                key = 'dev_survival_'+status_i+'_'+cat_i
                legend_string = status_i+' patients'
                ax.scatter(umaps[key]['x_axis'],
                           umaps[key]['y_axis'],
                           marker=markers[status_i],
                           c=['xkcd:gray']*umaps[key]['x_axis'].shape[0],
                           alpha=0.8,
                           s=4)
        
        
        list_lines = [[(0,0.25),(0.15,0.25)],
                        [(0.15,0.25),(0.15,0.3)],
                        [(0.15,0.3),(0.35,0.3)],
                        [(0.35,0.3),(0.35,0.35)],
                        [(0.35,0.35),(0.45,0.35)],
                        [(0.45,0.35),(0.45,0.30)],
                        [(0.45,0.30),(0.55,0.30)],
                        [(0.55,0.30),(0.55,0.20)],
                        [(0.55,0.2),(1,0.20)],
                        [(0,0.45),(0.15,0.45)],
                        [(0.15,0.45),(0.15,0.75)],
                        [(0.15,0.75),(0.3,0.75)],
                        [(0.3,0.75),(0.3,0.8)],
                        [(0.3,0.8),(0.5,0.8)],
                        [(0.5,0.8),(0.5,0.75)],
                        [(0.5,0.75),(0.7,0.75)],
                        [(0.7,0.75),(0.7,0.7)],
                        [(0.7,0.7),(0.8,0.7)],
                        [(0.8,0.7),(0.8,0.6)],      
                        [(0.8,0.6),(0.9,0.6)],
                        [(0.9,0.6),(0.9,0.5)],
                        [(0.9,0.5),(1,0.5)]
                        ]
        
        lines_zones = {}
        
        lines_zones['low'] =  [  [(0,0.25),(0.15,0.25)],
                        [(0.15,0.25),(0.15,0.3)],
                        [(0.15,0.3),(0.35,0.3)],
                        [(0.35,0.3),(0.35,0.35)],
                        [(0.35,0.35),(0.45,0.35)],
                        [(0.45,0.35),(0.45,0.30)],
                        [(0.45,0.30),(0.55,0.30)],
                        [(0.55,0.30),(0.55,0.20)],
                        [(0.55,0.2),(1,0.20)],
                        [(1,0.2),(1,0)],
                        [(1,0),(0,0)],
                        [(0,0),(0,0.25)]
                        ]
        lines_zones['med'] =  [[(0,0.25),(0.15,0.25)],
                        [(0.15,0.25),(0.15,0.3)],
                        [(0.15,0.3),(0.35,0.3)],
                        [(0.35,0.3),(0.35,0.35)],
                        [(0.35,0.35),(0.45,0.35)],
                        [(0.45,0.35),(0.45,0.30)],
                        [(0.45,0.30),(0.55,0.30)],
                        [(0.55,0.30),(0.55,0.20)],
                        [(0.55,0.2),(1,0.20)],
                        [(1,0.2),(1,0.5)],
                        [(1,0.5),(0.9,0.5)],
                        [(0.9,0.5),(0.9,0.6)],
                        [(0.9,0.6),(0.8,0.6)],
                        [(0.8,0.6),(0.8,0.7)],      
                        [(0.8,0.7),(0.7,0.7)],
                        [(0.7,0.7),(0.7,0.75)],
                        [(0.7,0.75),(0.5,0.75)],
                        [(0.5,0.75),(0.5,0.8)],
                        [(0.5,0.8),(0.3,0.8)],
                        [(0.3,0.8),(0.3,0.75)],
                        [(0.3,0.75),(0.15,0.75)],
                        [(0.15,0.75),(0.15,0.45)],
                        [(0.15,0.45),(0,0.45)],
                        [(0,0.45),(0,0.25)],
                        ]
        lines_zones['high'] = [  [(0,0.45),(0.15,0.45)],
                        [(0.15,0.45),(0.15,0.75)],
                        [(0.15,0.75),(0.3,0.75)],
                        [(0.3,0.75),(0.3,0.8)],
                        [(0.3,0.8),(0.5,0.8)],
                        [(0.5,0.8),(0.5,0.75)],
                        [(0.5,0.75),(0.7,0.75)],
                        [(0.7,0.75),(0.7,0.7)],
                        [(0.7,0.7),(0.8,0.7)],
                        [(0.8,0.7),(0.8,0.6)],      
                        [(0.8,0.6),(0.9,0.6)],
                        [(0.9,0.6),(0.9,0.5)],
                        [(0.9,0.5),(1,0.5)],
                        [(1,0.5),(1,1)],
                        [(1,1),(0,1)],
                        [(0,1),(0,0.46)],
                        ]
        
        x_lines = {}
        y_lines = {}
        
        for line_cat in lines_zones.keys():
            x_lines[line_cat] = []
            y_lines[line_cat] = []
            lines_tmp = lines_zones[line_cat]
            for elem_i in lines_tmp:
                x_lines[line_cat].append(elem_i[0][0])
                x_lines[line_cat].append(elem_i[1][0])
                y_lines[line_cat].append(elem_i[0][1])
                y_lines[line_cat].append(elem_i[1][1])
        
        ax.fill(x_lines['high'],y_lines['high'],facecolor=(0,114/255,178/255),alpha=0.3)
        ax.fill(x_lines['med'],y_lines['med'],facecolor=(240/255,228/255,66/255),alpha=0.3)
        ax.fill(x_lines['low'],y_lines['low'],facecolor=(204/255,121/255,167/255),alpha=0.3)
        
        for line_i in list_lines:
            ax.plot(np.linspace(line_i[0][0],line_i[1][0],100),
                    np.linspace(line_i[0][1],line_i[1][1],100),
                    c='k',
                    linewidth=1,
                    alpha=0.8,
                    linestyle=':')
        
        t1 = ax.text(0.25,
                     0.87,
                     'survival rate: \n 90% (+/- 4%)',
                     c='k',
                     weight='bold',
                     multialignment = 'center',
                     fontsize='x-small',)
         
        t2 = ax.text(0.25,
                     0.4,
                     'survival rate: \n 77% (+/- 4%)',
                     c='k',
                     weight='bold',
                     multialignment = 'center',
                     fontsize='x-small',)
        
        t4 = ax.text(0.25,
                     0.12,
                     'survival rate: \n 50% (+/- 5%)',
                     c='k',
                     weight='bold',
                     multialignment = 'center',
                     fontsize='x-small',) 
        
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.set_title('c. 1-year survival \n enhanced division')
        
        ax = axes[1]
        ax.set_yticks([])

        for elem_x in range(1,20):
            ax.plot([elem_x/20]*100,
                          np.linspace(-0.07,1.07,100),
                          c='k',
                          linestyle=':',
                          linewidth=0.8)
        for elem_y in range(1,20):
            ax.plot(np.linspace(-0.07,1.07,100),
                         [elem_y/20]*100,
                         c='k',
                         linestyle=':',
                         linewidth=0.8)
        
        
        patches = []
        color_done = []
        labels_done = []
        handles_done = []
        
        for elem_i in stats['dev_survival_2'].keys():        
            stat_all = stats['dev_survival_2'][elem_i]
            stat_alive = stats['dev_survival_alive'][elem_i]
            stat_dead = stats['dev_survival_dead'][elem_i]
            if stat_all != 0:
                pb = round(stat_dead/stat_all*100,1)
            else:
                pb = 0.0
            if pb>=60:
                colRec = (204/255,121/255,167/255)
                labelRec = r'            survival pb $\leq$ 40%'
            elif pb<60 and pb>=40:
                colRec = (213/255,94/255,0)
                labelRec = r'40% < survival pb $\leq$ 60%'
            elif pb>=20 and pb<40:
                colRec = (240/255,228/255,66/255)
                labelRec = r'60% < survival pb $\leq$ 80%'
            elif pb>=10 and pb<20:
                colRec = (0/255,158/255,115/255)
                labelRec = r'80% < survival pb $\leq$ 90%'
            elif pb==0 and stats['dev_survival_2'][elem_i]==0:
                colRec = 'xkcd:white'
            else:
                colRec=(0,114/255,178/255)
                labelRec = '90% < survival pb'
                
            first_time = {(204/255,121/255,167/255):0,
                          (213/255,94/255,0):0,
                          (240/255,228/255,66/255):0,
                          (0/255,158/255,115/255):0,
                          (0,114/255,178/255):0}
                
            rect = mpatches.Rectangle((float(elem_i.split('-')[0]),
                                   float(elem_i.split('-')[1])),
                                   sizeRec,
                                   sizeRec,
                                   facecolor=colRec,
                                   alpha=0.5)
            ax.add_artist(rect)
            
            '''
            color_done = [(204/255,121/255,167/255),
                          (213/255,94/255,0),
                          (240/255,228/255,66/255),
                          (0/255,158/255,115/255),
                          (0,114/255,178/255)]
            
            labels_done = [r'           survival pb $\leq$ 40%',
                           r'40% < survival pb $\leq$ 60%',
                           r'60% < survival pb $\leq$ 80%',
                           r'80% < survival pb $\leq$ 90%',
                           '90% < survival pb']
            '''
            
            
            if colRec not in color_done and colRec != 'xkcd:white':
                color_done.append(colRec)
                handles_done.append(rect)
                labels_done.append(labelRec)
                
        # reorder for proper survival evolution
        #print(labels_done)
        labels_done2 =  [labels_done[0]]+[labels_done[4]]+[labels_done[2]]+[labels_done[3]]+[labels_done[1]]
        handles_done2 = [handles_done[0]]+[handles_done[4]]+[handles_done[2]]+[handles_done[3]]+[handles_done[1]]
                
        ax.legend(handles_done2,
                  labels_done2,
                  loc='lower center',
                  labelspacing = 0.1,
              bbox_to_anchor=(0.5,-0.40),
              ncol=1,
              borderaxespad=0,
              frameon=False,
              fontsize='small',
              )
        
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.set_title('b. 1-year survival \n grid analysis')    
        
        fig.text(0.5, -0.23, 'Dim n1 (UMAP)', ha='center')
        plt.subplots_adjust(wspace=0.37)

        fig.savefig('output/reboot/article/survival/pdf/survival_enh_grid.pdf', bbox_inches='tight')
        fig.savefig('output/reboot/article/survival/png/survival_enh_grid.png', bpi=1500, bbox_inches='tight')

        #plt.close()
    
    print('End advanced grid plot')
    
    ##########################################################################
    #                                 Functinal plot split (FT9/ALSFRS)
    ##########################################################################
    
    
    if df_i == 'functional':
        
        
       #print('1684: {}'.format('MiToS_score_baseline' in list(umaps['dev_'+df_i].columns)))
       #print('1684x: {}'.format('MiToS_score_baseline_x' in list(umaps['dev_'+df_i].columns)))

        
       #  Analysis 3 split 
       #[x/10000 for x in range(7090,7100)]
       # Based on survival values
       lowers = [0.3862]#[x/10000 for x in range(3860,3870)]
       highers = [0.6722]
       
       for lower_i in lowers:
           for higher_i in highers:
        
               sub = umaps['dev_'+df_i].loc[umaps['dev_'+df_i].y_axis<lower_i,:]
               high = umaps['dev_'+df_i].loc[umaps['dev_'+df_i].y_axis>higher_i,:]
               mid =  umaps['dev_'+df_i].loc[((umaps['dev_'+df_i].y_axis>lower_i) & (umaps['dev_'+df_i].y_axis<higher_i)),:]
        
               #print('{}-{}:sub shape: {}, mid shape: {},high shape:{}'.format(lower_i,higher_i,sub.shape,mid.shape,high.shape))
               
       
            
    
       score_value = {'MiToS':[0,1,2,3,4],
                      'Kings':[1,2,3,4,4.5],
                      'FT9':[0,1,2,3,4]}
       spa_mask = {}
       spa_mask['low'] = (umaps['dev_'+df_i].y_axis<lower_i)
       spa_mask['mid'] = ((umaps['dev_'+df_i].y_axis>lower_i) & (umaps['dev_'+df_i].y_axis<higher_i))
       spa_mask['high']= (umaps['dev_'+df_i].y_axis>higher_i)
      
       als_mask = {}
       als_mask[10] = (umaps['dev_'+df_i].ALSFRS_Total_final<=10)
       als_mask[20] = ((umaps['dev_'+df_i].ALSFRS_Total_final<=20) & (umaps['dev_'+df_i].ALSFRS_Total_final>10))
       als_mask[30] = ((umaps['dev_'+df_i].ALSFRS_Total_final<=30) & (umaps['dev_'+df_i].ALSFRS_Total_final>20))
       als_mask[40] = (umaps['dev_'+df_i].ALSFRS_Total_final>30)
       
       masks_analysis = {}
       for score_i in ['FT9','MiToS','Kings','ALSFRS_Total_final']:
           masks_analysis[score_i] = {}

           if score_i in ['ALSFRS_Total_final']:
              for elem_i in [10,20,30,40]:
                  masks_analysis[score_i][elem_i] = {}
                  for cat_i in ['low','mid','high']:
                          masks_analysis[score_i][elem_i][cat_i] = (spa_mask[cat_i] & (als_mask[elem_i]))
           else:
               for elem_i in score_value[score_i]:
                       masks_analysis[score_i][elem_i] = {}
                       for cat_i in ['low','mid','high']:
                           masks_analysis[score_i][elem_i][cat_i] = (spa_mask[cat_i] & (umaps['dev_'+df_i][score_i+'_score_final']==elem_i))
       
       count_functional = []  
       sum_overall = umaps['dev_'+df_i].shape[0]
       for score_i in ['FT9','MiToS','Kings','ALSFRS_Total_final']:
           for elem_i in masks_analysis[score_i].keys():
               for cat_i in ['low','mid','high']:
                   
                   tmp= [score_i,elem_i,cat_i]
                   sum_zone = umaps['dev_'+df_i].loc[spa_mask[cat_i],:].shape[0]
                   if score_i == 'ALSFRS_Total_final':
                       sum_stage = umaps['dev_'+df_i].loc[als_mask[elem_i],:].shape[0] 
                   else:
                       sum_stage = umaps['dev_'+df_i].loc[umaps['dev_'+df_i][score_i+'_score_final']==elem_i,:].shape[0]
                   val_count = umaps['dev_'+df_i].loc[masks_analysis[score_i][elem_i][cat_i],:].shape[0]
                   tmp.append(val_count)
                   tmp.append(val_count/sum_zone)
                   tmp.append(val_count/sum_overall)
                   tmp.append(val_count/sum_stage)
                   tmp.append(sum_zone)
                   tmp.append(sum_stage)
                   count_functional.append(tmp)
                   
       count_functional = pd.DataFrame(count_functional,columns=['score','stage','cat','count','freq_zone','freq_overall','freq_stage','count_zone','count_stage'])
       #count_functional.to_csv('output/reboot/functional_stage_overview_3fold.csv',sep=';')
       
       #  Analysis 2 split 
       mid = [0.58138]#[x/100000 for x in range(58130,58140)]
       
       for mid_i in mid:
        
           sub = umaps['dev_'+df_i].loc[umaps['dev_'+df_i].y_axis<mid_i,:]
           high =  umaps['dev_'+df_i].loc[umaps['dev_'+df_i].y_axis>=mid_i,:]
           #print('{}:sub shape: {},high shape:{}'.format(mid_i,sub.shape,high.shape))
       
       score_value = {'MiToS':[0,1,2,3,4],
                      'Kings':[1,2,3,4,4.5],
                      'FT9':[0,1,2,3,4]}
       spa_mask = {}
       spa_mask['low'] = (umaps['dev_'+df_i].y_axis<mid_i)
       spa_mask['high']= (umaps['dev_'+df_i].y_axis>=mid_i)
      
       als_mask = {}
       als_mask[10] = (umaps['dev_'+df_i].ALSFRS_Total_final<=10)
       als_mask[20] = ((umaps['dev_'+df_i].ALSFRS_Total_final<=20) & (umaps['dev_'+df_i].ALSFRS_Total_final>10))
       als_mask[30] = ((umaps['dev_'+df_i].ALSFRS_Total_final<=30) & (umaps['dev_'+df_i].ALSFRS_Total_final>20))
       als_mask[40] = (umaps['dev_'+df_i].ALSFRS_Total_final>30)
       
       masks_analysis = {}
       for score_i in ['FT9','MiToS','Kings','ALSFRS_Total_final']:
           masks_analysis[score_i] = {}

           if score_i in ['ALSFRS_Total_final']:
              for elem_i in [10,20,30,40]:
                  masks_analysis[score_i][elem_i] = {}
                  for cat_i in ['low','high']:
                          masks_analysis[score_i][elem_i][cat_i] = (spa_mask[cat_i] & (als_mask[elem_i]))
           else:
               for elem_i in score_value[score_i]:
                       masks_analysis[score_i][elem_i] = {}
                       for cat_i in ['low','high']:
                           masks_analysis[score_i][elem_i][cat_i] = (spa_mask[cat_i] & (umaps['dev_'+df_i][score_i+'_score_final']==elem_i))
       
       count_functional = []  
       sum_overall = umaps['dev_'+df_i].shape[0]
       for score_i in ['FT9','MiToS','Kings','ALSFRS_Total_final']:
           for elem_i in masks_analysis[score_i].keys():
               for cat_i in ['low','high']:
                   
                   tmp= [score_i,elem_i,cat_i]
                   sum_zone = umaps['dev_'+df_i].loc[spa_mask[cat_i],:].shape[0]
                   if score_i == 'ALSFRS_Total_final':
                       sum_stage = umaps['dev_'+df_i].loc[als_mask[elem_i],:].shape[0] 
                   else:
                       sum_stage = umaps['dev_'+df_i].loc[umaps['dev_'+df_i][score_i+'_score_final']==elem_i,:].shape[0]
                   val_count = umaps['dev_'+df_i].loc[masks_analysis[score_i][elem_i][cat_i],:].shape[0]
                   tmp.append(val_count)
                   tmp.append(val_count/sum_zone)
                   tmp.append(val_count/sum_overall)
                   tmp.append(val_count/sum_stage)
                   tmp.append(sum_zone)
                   tmp.append(sum_stage)
                   
                   count_functional.append(tmp)
                   
       count_functional = pd.DataFrame(count_functional,columns=['score','stage','cat','count','freq_zone','freq_overall','freq_stage','count_zone','count_stage'])
       #count_functional.to_csv('output/reboot/functional_stage_overview_2fold.csv',sep=';')
       
       
       ##################### MEGA Grid (combination of 2 previous)

       masks = {}
       masks[0] = (umaps['dev_functional']['ALSFRS_Total_final']<=10)
       masks[1] = ((umaps['dev_functional']['ALSFRS_Total_final']<=20)&(umaps['dev_functional']['ALSFRS_Total_final']>10))
       masks[2] = ((umaps['dev_functional']['ALSFRS_Total_final']<=30)&(umaps['dev_functional']['ALSFRS_Total_final']>20))
       masks[3] = (umaps['dev_functional']['ALSFRS_Total_final']>30)
       

       titles = {0:[r'a. 1-year ALSFRS$\leq$10','('+str(umaps['dev_functional'].loc[masks[0],:].shape[0])+')'],
                 1:[r'b. 10$<$1-year ALSFRS$\leq$20','('+str(umaps['dev_functional'].loc[masks[1],:].shape[0])+')'],
                 2:[r'c. 20$<$1-year ALSFRS$\leq$30','('+str(umaps['dev_functional'].loc[masks[2],:].shape[0])+')'],
                 3:[r'd. 30 < 1-year ALSFRS','('+str(umaps['dev_functional'].loc[masks[3],:].shape[0])+')'],
                 5:[r'f. 1-year ALSFRS$\leq$10','('+str(umaps['dev_functional'].loc[masks[0],:].shape[0])+')'],
                 6:[r'g. 10$<$1-year ALSFRS$\leq$20','('+str(umaps['dev_functional'].loc[masks[0],:].shape[0])+')'],
                 7:[r'h. 20$<$1-year ALSFRS$\leq$30','('+str(umaps['dev_functional'].loc[masks[0],:].shape[0])+')'],
                 8:[r'i. 30 < 1-year ALSFRS','('+str(umaps['dev_functional'].loc[masks[0],:].shape[0])+')'],}


       fig = plt.figure(figsize=(25.6,12.8))
       #fig = plt.figure(figsize=(51.2,25.6))

       axes = {}
       
       axes[0] = plt.subplot2grid((4,12), (0, 0))
       axes[1] = plt.subplot2grid((4,12), (0, 1))
       axes[2] = plt.subplot2grid((4,12), (1, 0))
       axes[3] = plt.subplot2grid((4,12), (1, 1))
       axes[4] = plt.subplot2grid((4,12), (0, 2), colspan=2, rowspan=2)
       axes[5] = plt.subplot2grid((4,12), (0, 4))
       axes[6] = plt.subplot2grid((4,12), (0, 5))
       axes[7] = plt.subplot2grid((4,12), (1, 4))
       axes[8] = plt.subplot2grid((4,12), (1, 5))
              
       axes[0].spines['bottom'].set_visible(False)
       axes[1].spines['bottom'].set_visible(False)
       axes[1].spines['left'].set_visible(False)
       axes[3].spines['left'].set_visible(False)
       
       axes[5].spines['bottom'].set_visible(False)
       axes[6].spines['bottom'].set_visible(False)
       axes[5].spines['left'].set_visible(False)
       axes[6].spines['left'].set_visible(False)
       axes[7].spines['left'].set_visible(False)
       axes[8].spines['left'].set_visible(False)

       x = np.linspace(-0.05,1.05,200)
       y1 = 0.4+0.2*x
       y2 = -0.1 + 0.5*x
       
       for i in range(4):
           ax = axes[i]
           ax.set_title('{} \n '.format(titles[i][0]),
                        fontsize='small')
           ax.scatter(umaps['dev_functional'].loc[masks[i],'x_axis'],
                      umaps['dev_functional'].loc[masks[i],'y_axis'],
                      marker='.',
                      c=['grey']*umaps['dev_functional'].loc[masks[i],'y_axis'].shape[0],
                      alpha=0.6,
                      s=1,
                      )
           ax.spines['right'].set_visible(False)
           ax.spines['top'].set_visible(False)  
           ax.set_xlim([0,1])
           ax.set_ylim([0,1])
           #ax.grid(alpha=0.5)
       
       

       x = np.linspace(-0.05,1.05,200)
       y1 = 0.4+0.2*x
       y2 = -0.1 + 0.5*x
       
       count_4z = {0:{'top':'XX%',
                      'mid':'XX%',
                      'bottom':'XX%',},
                   1:{'top':'XX%',
                      'mid':'XX%',
                      'bottom':'XX%',},
                   2:{'top':'XX%',
                      'mid':'XX%',
                      'bottom':'XX%',},
                   3:{'top':' XX%',
                      'mid':'XX%',
                      'bottom':'XX%',}}
       
       axes[0].set_xticks([])
       axes[1].set_xticks([])
       axes[1].set_yticks([])
       axes[3].set_yticks([])
       
       axes[4].scatter(umaps['dev_functional'].loc[:,'x_axis'],
                      umaps['dev_functional'].loc[:,'y_axis'],
                      marker='.',
                      c=['grey']*umaps['dev_functional'].loc[:,'y_axis'].shape[0],
                      alpha=0.7,
                      s=1,
                      )
       axes[4].spines['right'].set_visible(False)
       axes[4].spines['top'].set_visible(False)  
       axes[4].set_xlim([0,1])
       axes[4].set_ylim([0,1])
       axes[4].plot(x,y1,c='k',alpha=0.7)
       axes[4].plot(x,y2,c='k',alpha=0.7)
       axes[4].fill([0,1,1,0],[0.4,0.6,1,1],facecolor=(0,114/255,178/255),alpha=0.3)  
       axes[4].fill([0,0.2,1,1,0],[0,0,0.4,0.6,0.4],facecolor=(240/255,228/255,66/255),alpha=0.3)  
       axes[4].set_title('e. 1-year functional loss \n division')
       axes[4].fill([0.2,1,1,0.2],[0,0.4,0,0],facecolor=(204/255,121/255,167/255),alpha=0.3)  
       axes[4].text(0.3,
                 0.7,
                 'marginal loss: \n {} \n XX% (+/- XX%)'.format(r'P(20$\leq$1-year ALSFRS)'),
                 c='k',
                 weight='bold',
                 multialignment = 'center',
                 fontsize='small',) 
       axes[4].text(0.25,
                 0.3,
                 'intermediate loss: \n {} \n XX% (+/- XX%)'.format(r'P(10$\leq$1-year ALSFRS<30)'),
                 c='k',
                 weight='bold',
                 multialignment = 'center',
                 fontsize='small',) 
       axes[4].text(0.45,
                 0.05,
                 'significant loss: \n {} \n XX% (+/- XX%)'.format(r'P(1-year ALSFRS<20)'),
                 c='k',
                 weight='bold',
                 multialignment = 'center',
                 fontsize='small',) 
       
       plt.subplots_adjust(wspace=0.38,hspace=0.30)
       
       for i in range(4):
           ax = axes[i+5]
           ax.set_title('{} \n division'.format(titles[i+5][0]),
                        fontsize='small')
           ax.scatter(umaps['dev_functional'].loc[masks[i],'x_axis'],
                      umaps['dev_functional'].loc[masks[i],'y_axis'],
                      marker='.',
                      c=['grey']*umaps['dev_functional'].loc[masks[i],'y_axis'].shape[0],
                      alpha=0.6,
                      s=1,
                      )
           ax.spines['right'].set_visible(False)
           ax.spines['top'].set_visible(False)  
           ax.set_xlim([0,1])
           ax.set_ylim([0,1])
           #ax.grid(alpha=0.5)      
       
       for i in range(4):
           ax = axes[i+5]
           ax.plot(x,y1,c='k',alpha=0.7)
           ax.plot(x,y2,c='k',alpha=0.7)
           ax.fill([0,1,1,0],[0.4,0.6,1,1],facecolor=(0,114/255,178/255),alpha=0.3)  
           ax.fill([0,0.2,1,1,0],[0,0,0.4,0.6,0.4],facecolor=(240/255,228/255,66/255),alpha=0.3)  
           ax.fill([0.2,1,1,0.2],[0,0.4,0,0],facecolor=(204/255,121/255,167/255),alpha=0.3)  
           ax.text(0.3,
                     0.7,
                     'marginal \n loss: \n {}'.format(count_4z[i]['top']),
                     c='k',
                     weight='bold',
                     multialignment = 'center',
                     fontsize='x-small') 
           ax.text(0.20,
                     0.25,
                     'intermediate \n loss: \n {}'.format(count_4z[i]['mid']),
                     c='k',
                     weight='bold',
                     multialignment = 'center',
                     fontsize='x-small',) 
           ax.text(0.52,
                     0.02,
                     'significant \n loss: \n {}'.format(count_4z[i]['bottom']),
                     c='k',
                     weight='bold',
                     multialignment = 'center',
                     fontsize='x-small',)
       
       axes[5].set_xticks([])
       axes[6].set_xticks([])
       axes[6].set_yticks([])
       axes[8].set_yticks([])
       
       axes[5].set_yticks([])
       axes[7].set_yticks([])

       
       fig.text(0.32,0.49, 'Dim n1 (UMAP)', ha='center')
       fig.text(0.1, 0.7, 'Dim n2 (UMAP)', va='center', rotation='vertical')
       #### SELECTED
       fig.savefig('output/reboot/article/functional/pdf/functional_grid_mega.pdf', bbox_inches='tight')
       fig.savefig('output/reboot/article/functional/png/functional_grid_mega.png', dpi=1500, bbox_inches='tight')

       
       # FT9 2 missing as incomplete (but still usable for others -> add 2 in biggest cat)
       
       
       score_value = {'MiToS':[0,1,2,3,4,5],
                      'Kings':[1,2,3,4,4.5,5],
                      'FT9':[0,1,2,3,4,5]}
       
       
       val_pscore = {'Kings':{1:370,
                              2:779,
                              3:1288,
                              4:327,
                              4.5:17,
                              5:1014},
                     'MiToS':{0:1178,
                              1:1054,
                              2:376,
                              3:121,
                              4:52,
                              5:1014},
                     'FT9':{0:73,
                            1:450,
                            2:830,
                            3:803,
                            4:625,
                            5:1014}}
       
       
       #print('2466: {}'.format('MiToS_score_baseline' in list(umaps['dev_'+df_i].columns)))
       #print('2466x: {}'.format('MiToS_score_baseline_x' in list(umaps['dev_'+df_i].columns)))

       #print('2471: {}'.format(df_i))
       iternum = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v']
                   
       #print('2526: {}'.format(df_i))
       ##### FUNCTIONAL APPENDIX
       #### SELECTED
       
       
       
       fig, axes = plt.subplots(3,6, sharex=True,sharey=True,figsize=(10,12))
       axes_idx = [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),
                   (1,0),(1,1),(1,2),(1,3),(1,4),(1,5),
                   (2,0),(2,1),(2,2),(2,3),(2,4),(2,5)
                   ]
       masks = {}
       titles = {}
          
       axes[0][0].spines['bottom'].set_visible(False)
       axes[0][1].spines['bottom'].set_visible(False)
       axes[0][2].spines['bottom'].set_visible(False)
       axes[0][3].spines['bottom'].set_visible(False)
       axes[0][4].spines['bottom'].set_visible(False)
       axes[0][5].spines['bottom'].set_visible(False)
       axes[1][0].spines['bottom'].set_visible(False)
       axes[1][1].spines['bottom'].set_visible(False)
       axes[1][2].spines['bottom'].set_visible(False)
       axes[1][3].spines['bottom'].set_visible(False)
       axes[1][4].spines['bottom'].set_visible(False)
       axes[1][5].spines['bottom'].set_visible(False)
       
       axes[0][1].spines['left'].set_visible(False)
       axes[0][2].spines['left'].set_visible(False)
       axes[0][3].spines['left'].set_visible(False)
       axes[0][4].spines['left'].set_visible(False)
       axes[0][5].spines['left'].set_visible(False)
       axes[1][1].spines['left'].set_visible(False)
       axes[1][2].spines['left'].set_visible(False)
       axes[1][3].spines['left'].set_visible(False)
       axes[1][4].spines['left'].set_visible(False)
       axes[1][5].spines['left'].set_visible(False)
       axes[2][1].spines['left'].set_visible(False)
       axes[2][2].spines['left'].set_visible(False)
       axes[2][3].spines['left'].set_visible(False)
       axes[2][4].spines['left'].set_visible(False)
       axes[2][5].spines['left'].set_visible(False)
       
       for i in range(0,15):
           if i<5:
               score_i = 'Kings'
           elif i>= 5 and i<10:
               score_i = 'MiToS'
           else:
               score_i = 'FT9'
           masks[i] = (umaps['dev_functional'][score_i+'_score_final']==score_value[score_i][i%5])
           titles[i] = '{}. 1-year {} stage {}'.format(iternum[i],score_i,
                 str(score_value[score_i][i%5]),
                 val_pscore[score_i][score_value[score_i][i%5]])

       
       for i in range(18):
           if i<6:
               score_i = 'Kings'
           elif i>= 6 and i<12:
               score_i = 'MiToS'
           else:
               score_i = 'FT9'
           ax = axes[axes_idx[i][0]][axes_idx[i][1]]
           ax.set_title(titles[i],fontsize='small')
           ax.scatter(umaps['dev_functional'].loc[masks[i],'x_axis'],
                      umaps['dev_functional'].loc[masks[i],'y_axis'],
                      marker='.',
                      c=['k']*umaps['dev_functional'].loc[masks[i],'y_axis'].shape[0],#(0/255,114/255,178/255),
                      alpha=1,
                      s=1,
                      )
           ax.spines['right'].set_visible(False)
           ax.spines['top'].set_visible(False)  
           ax.set_xlim([0,1])
           ax.set_ylim([0,1])
           
       fig.text(0.5, 0.08, 'Dim n1 (UMAP)', ha='center')
       fig.text(0.07, 0.5, 'Dim n2 (UMAP)', va='center', rotation='vertical')   
       plt.subplots_adjust(wspace=0.35,hspace=0.25)
       fig.savefig('output/reboot/article/functional/pdf/functional_scores_sep_split.pdf', bbox_inches='tight')
       fig.savefig('output/reboot/article/functional/png/functional_scores_sep_split.png', bpi=1500, bbox_inches='tight')

       #print('2572: {}'.format(df_i))

       #plt.close()
       
       count_4z_stages = {'MiToS':{0:{'top':'83% ',
                                      'mid':'16% ',
                                      'bottom':'1% ',},
                                   1:{'top':'47% ',
                                      'mid':'41% ',
                                      'bottom':'12% ',},
                                   2:{'top':'39% ',
                                      'mid':'37% ',
                                      'bottom':'24% ',},
                                   3:{'top':'24.5% ',
                                      'mid':'34.5% ',
                                      'bottom':'31% ',},
                                   4:{'top':' 21% ',
                                      'mid':'26% ',
                                      'bottom':'53% ',}},
                      'FT9':{0:{'top':'100% ',
                              'mid':'0% ',
                              'bottom':'0% ',},
                           1:{'top':'94% ',
                              'mid':'5.8% ',
                              'bottom':'0.2% ',},
                           2:{'top':'67% ',
                              'mid':'29% ',
                              'bottom':'4% ',},
                           3:{'top':' 47% ',
                              'mid':'38.5% ',
                              'bottom':'14.5% ',},
                           4:{'top':' 37.5% ',
                              'mid': '40% ',
                              'bottom':'22.5% ',}},
                      'Kings':{0:{'top':'96.6% ',
                                  'mid':'3.2% ',
                                  'bottom':'0.2% ',},
                               1:{'top':'73% ',
                                  'mid':'24% ',
                                  'bottom':'3% ',},
                               2:{'top':'46% ',
                                  'mid':'39% ',
                                  'bottom':'15% ',},
                               3:{'top':' 46% ',
                                  'mid':'34% ',
                                  'bottom':'20% ',},
                               4:{'top':'41% ',
                                  'mid':'47% ',
                                  'bottom':'12% ',}}}
    
        #### DELTA ANALYSIS    
       #print('2668: {}'.format(df_i))
       df_info_scores = pd.read_csv('input/proc/df_functional.csv',sep=';')
       df_info_scores.set_index('subject_id',inplace=True)
       df_info_scores = df_info_scores.loc[:,[x+'_score_baseline' for x in ['MiToS','FT9','Kings']]]
       umaps['dev_'+df_i] = pd.merge(umaps['dev_'+df_i],
                                     df_info_scores,
                                     how='left',
                                     left_index=True,
                                     right_index=True)
       # One figure
       fig, axes = plt.subplots(3,5,figsize=(10,10)) # sharex=True,sharey=True,
       axes_idx = [(0,0),(0,1),(0,2),(0,3),(0,4),
                       (1,0),(1,1),(1,2),(1,3),(1,4),
                       (2,0),(2,1),(2,2),(2,3),(2,4),
                       ]
           
       '''fsh = {0:(0,1),1:(0,0),2:(0,0),3:(0,0),4:(0,0),5:(0,0),
              6:(0,1),7:(0,0),8:(0,0),9:(0,0),10:(0,0),11:(0,0),
              12:(1,1),13:(1,0),14:(1,0),15:(1,0),16:(1,0),17:(1,0)}
       '''
       fsh = {0:(0,1),1:(0,0),2:(0,0),3:(0,0),4:(0,0),
              5:(0,1),6:(0,0),7:(0,0),8:(0,0),9:(0,0),
              10:(1,1),11:(1,0),12:(1,0),13:(1,0),14:(1,0)}
       
       masks = {}
       titles = {}
       for i in range(0,15):
           if i<5:
               score_i = 'Kings'
           elif i>= 5 and i<10:
               score_i = 'MiToS'
           else:
               score_i = 'FT9'
           masks[i] = (umaps['dev_functional'][score_i+'_score_final']==score_value[score_i][i%5])
           titles[i] = '{}. 1-year {} stage {}'.format(iternum[i],score_i,
                 str(score_value[score_i][i%5]),
                 val_pscore[score_i][score_value[score_i][i%5]])
        
       for i in range(15):
           if i<5:
               score_i = 'Kings'
           elif i>= 5 and i<10:
               score_i = 'MiToS'
           else:
               score_i = 'FT9'
           ax = axes[axes_idx[i][0]][axes_idx[i][1]]
           ax.set_title(titles[i],fontsize='small')
           ax.scatter(umaps['dev_functional'].loc[masks[i],'x_axis'],
                      umaps['dev_functional'].loc[masks[i],'y_axis'],
                      marker='.',
                      c=['grey']*umaps['dev_functional'].loc[masks[i],'y_axis'].shape[0],#(0/255,114/255,178/255),
                      alpha=0.8,
                      s=1,
                      )
           ax.spines['right'].set_visible(False)
           ax.spines['top'].set_visible(False)  
           ax.set_xlim([0,1])
           ax.set_ylim([0,1])
           x = np.linspace(-0.05,1.05,200)
           y1 = 0.4+0.2*x
           y2 = -0.1 + 0.5*x

           ax.plot(x,y1,c='k',alpha=0.7)
           ax.plot(x,y2,c='k',alpha=0.7)
           ax.fill([0,1,1,0],[0.4,0.6,1,1],facecolor=(0,114/255,178/255),alpha=0.3)  
           ax.fill([0,0.2,1,1,0],[0,0,0.4,0.6,0.4],facecolor=(240/255,228/255,66/255),alpha=0.3)  
           ax.fill([0.2,1,1,0.2],[0,0.4,0,0],facecolor=(204/255,121/255,167/255),alpha=0.3)  
           ax.text(0.3,
                 0.7,
                 'marginal \n loss: \n {}'.format(count_4z_stages[score_i][i%5]['top']),
                 c='k',
                 weight='bold',
                 multialignment = 'center',
                 fontsize='x-small') 
           ax.text(0.2,
                 0.3,
                 'intermediate \n loss: \n {}'.format(count_4z_stages[score_i][i%5]['mid']),
                 c='k',
                 weight='bold',
                 multialignment = 'center',
                 fontsize='x-small',) 
           ax.text(0.52,
                 0.05,
                 'significant \n loss: \n {}'.format(count_4z_stages[score_i][i%5]['bottom']),
                 c='k',
                 weight='bold',
                 multialignment = 'center',
                 fontsize='x-small',)
           
           '''
           if fsh[idx][0] == 0:
               ax.spines['bottom'].set_visible(False)  
               ax.set_xticks([])
           if fsh[idx][1] == 1:
               ax.spines['left'].set_visible(False)  
               ax.set_yticks([])
          '''
    
       fig.text(0.5, 0.08, 'Dim n1 (UMAP)', ha='center')
       fig.text(0.07, 0.5, 'Dim n2 (UMAP)', va='center', rotation='vertical')   
       plt.subplots_adjust(wspace=0.35,hspace=0.13)
       axes[0][0].spines['bottom'].set_visible(False)
       axes[0][1].spines['bottom'].set_visible(False)
       axes[0][2].spines['bottom'].set_visible(False)
       axes[0][3].spines['bottom'].set_visible(False)
       axes[0][4].spines['bottom'].set_visible(False)
       #axes[0][5].spines['bottom'].set_visible(False)
       axes[1][0].spines['bottom'].set_visible(False)
       axes[1][1].spines['bottom'].set_visible(False)
       axes[1][2].spines['bottom'].set_visible(False)
       axes[1][3].spines['bottom'].set_visible(False)
       axes[1][4].spines['bottom'].set_visible(False)
       #axes[1][5].spines['bottom'].set_visible(False)
       
       axes[0][1].spines['left'].set_visible(False)
       axes[0][2].spines['left'].set_visible(False)
       axes[0][3].spines['left'].set_visible(False)
       axes[0][4].spines['left'].set_visible(False)
       #axes[0][5].spines['left'].set_visible(False)
       axes[1][1].spines['left'].set_visible(False)
       axes[1][2].spines['left'].set_visible(False)
       axes[1][3].spines['left'].set_visible(False)
       axes[1][4].spines['left'].set_visible(False)
       #axes[1][5].spines['left'].set_visible(False)
       axes[2][1].spines['left'].set_visible(False)
       axes[2][2].spines['left'].set_visible(False)
       axes[2][3].spines['left'].set_visible(False)
       axes[2][4].spines['left'].set_visible(False)
       #axes[2][5].spines['left'].set_visible(False)  
       axes[0][0].set_xticks([])
       axes[0][1].set_xticks([])
       axes[0][2].set_xticks([])
       axes[0][3].set_xticks([])
       axes[0][4].set_xticks([])
       #axes[0][5].set_xticks([])
       axes[1][0].set_xticks([])
       axes[1][1].set_xticks([])
       axes[1][2].set_xticks([])
       axes[1][3].set_xticks([])
       axes[1][4].set_xticks([])
       #axes[1][5].set_xticks([])
       
       axes[0][1].set_yticks([])
       axes[0][2].set_yticks([])
       axes[0][3].set_yticks([])
       axes[0][4].set_yticks([])
       #axes[0][5].set_yticks([])
       axes[1][1].set_yticks([])
       axes[1][2].set_yticks([])
       axes[1][3].set_yticks([])
       axes[1][4].set_yticks([])
       #axes[1][5].set_yticks([])
       axes[2][1].set_yticks([])
       axes[2][2].set_yticks([])
       axes[2][3].set_yticks([])
       axes[2][4].set_yticks([])
       #axes[2][5].set_yticks([])
        ##### FUNCTIONAL APPENDIX
        #### SELECTED
        
       fig.savefig('output/reboot/article/functional/pdf/functional_scores_horizontal.pdf', bbox_inches='tight')
       fig.savefig('output/reboot/article/functional/png/functional_scores_horizontal.png', bbox_inches='tight',dpi=1500)    
        
       def computeScoreDelta(x,score_name):
           return x[score_name+'_score_final']-x[score_name+'_score_baseline']
       
       def computeScoreChange(x,score_name):
           delta = x[score_name+'_score_final']-x[score_name+'_score_baseline']
           if delta != 0:
               return 1
           else:
               return 0
       def computeScoreChange2(x,score_name):
           delta = x[score_name+'_score_final']-x[score_name+'_score_baseline']
           if delta == 0:
               return 0
           elif delta==1:
               return 1
           else:
               return 2
    
       def computeScoreChange3(x,score_name):
           delta = x[score_name+'_score_final']-x[score_name+'_score_baseline']
           if delta > 1:
               return 1
           else:
               return 0
    
       def computeScoreChangeALS(x):
           delta = abs((x['ALSFRS_Total_final']-x['ALSFRS_Total_baseline']))/max(x['ALSFRS_Total_final'],x['ALSFRS_Total_baseline'])*100
           #print(delta)
           if delta > 30:
               return 1
           else:
               return 0
       def computeScoreChangeALS3(x):
           delta = abs((x['ALSFRS_Total_final']-x['ALSFRS_Total_baseline']))
           #print(delta)
           if delta > 12:
               return 1
           else:
               return 0
           
       for score_i in ['MiToS','Kings','FT9']:
           umaps['dev_'+df_i][score_i+'_delta'] = umaps['dev_'+df_i].apply(lambda x: computeScoreDelta(x,score_i),axis=1)
           umaps['dev_'+df_i][score_i+'_change'] = umaps['dev_'+df_i].apply(lambda x: computeScoreChange(x,score_i),axis=1)
           umaps['dev_'+df_i][score_i+'_change2'] = umaps['dev_'+df_i].apply(lambda x: computeScoreChange2(x,score_i),axis=1)
           umaps['dev_'+df_i][score_i+'_change3'] = umaps['dev_'+df_i].apply(lambda x: computeScoreChange3(x,score_i),axis=1)
       umaps['dev_'+df_i]['ALSFRS_Total_change'] = umaps['dev_'+df_i].apply(lambda x: computeScoreChangeALS(x),axis=1) 
       umaps['dev_'+df_i]['ALSFRS_Total_change3'] = umaps['dev_'+df_i].apply(lambda x: computeScoreChangeALS3(x),axis=1) 

       #umaps['dev_'+df_i].loc[:,[x+'_'+y for x in ['MiToS','FT9','Kings'] for y in ['delta','score_final','score_baseline','change','change2']]].to_csv('output/reboot/check_score_dif.csv',sep=';')
       
        #print('runtime warning 2949 with operation from computeALSFRSsubChangeVar')
       def computeALSFRSsubChangeVar(x,zone):
           return abs(x['computeZones_'+zone+'_final']-x['computeZones_'+zone+'_baseline'])/max(x['computeZones_'+zone+'_final'],x['computeZones_'+zone+'_baseline'])
           
       
       def computeALSFRSsubChange(x,zone):
           delta = abs(x['computeZones_'+zone+'_final']-x['computeZones_'+zone+'_baseline'])/max(x['computeZones_'+zone+'_final'],x['computeZones_'+zone+'_baseline'])
           
           if zone in ['hand','trunk','leg']:
               if delta >= 0.25: #2
                   return 1
               else:
                   return 0
           if zone in ['mouth']:
               if delta >= 0.25: #3
                   return 1
               else:
                   return 0
           if zone in ['respiratory']:
               if delta >= 0.25: #1
                   return 1
               else:
                   return 0
               
         
       for zone_i in ['trunk','mouth','respiratory','leg','hand']:
           umaps['dev_'+df_i]['computeZones_'+zone_i+'_change'] = umaps['dev_'+df_i].loc[:,['computeZones_'+zone_i+'_baseline','computeZones_'+zone_i+'_final']].apply(lambda x: computeALSFRSsubChange(x,zone_i),axis=1)
           umaps['dev_'+df_i]['computeZones_'+zone_i+'_changeVar'] = umaps['dev_'+df_i].loc[:,['computeZones_'+zone_i+'_baseline','computeZones_'+zone_i+'_final']].apply(lambda x: computeALSFRSsubChangeVar(x,zone_i),axis=1)

       masks = {}
       masks2 = {}
       titles = {}
       for score_i in ['FT9','MiToS','Kings','ALSFRS_Total']:
           masks[score_i] = {}
           masks2[score_i] = {}
           if score_i != 'ALSFRS_Total':
               titles[score_i] = '1-year '+score_i+' stage evolution'
           else: 
               titles[score_i] = '1-year '+score_i+' evolution'
           for change_i in range(2):
               masks[score_i][change_i] = (umaps['dev_functional'][score_i+'_change']==change_i)
               masks2[score_i][change_i] = (umaps['dev_functional'][score_i+'_change3']==change_i)

               
       colors = {0:(0/255,114/255,178/255), #Blue
                 1:(240/255,228/255,66/255),#Yellow
                 2:(213/255,94/255,0/255)}        
       
       
       
      

       
       
       
       ### HORIZONTAL SPLIT
       mid = [0.58138] #[x/100000 for x in range(58130,58140)]
       mid_i = 0.58138
       
       spa_mask = {}
       spa_mask['low'] = (umaps['dev_'+df_i].y_axis<mid_i)
       spa_mask['high']= (umaps['dev_'+df_i].y_axis>=mid_i)
       
       masks_analysis = {}
       masks_analysis2 = {}
       for score_i in ['FT9','MiToS','Kings','ALSFRS_Total']:
           masks_analysis[score_i] = {}
           masks_analysis2[score_i] = {}
           for cat_i in ['low','high']:
               masks_analysis[score_i][cat_i] = {}
               masks_analysis2[score_i][cat_i] = {}
               for elem_i in [0,1]:
                   masks_analysis[score_i][cat_i][elem_i] = (spa_mask[cat_i] & (umaps['dev_'+df_i][score_i+'_change']==elem_i))
                   masks_analysis2[score_i][cat_i][elem_i] = (spa_mask[cat_i] & (umaps['dev_'+df_i][score_i+'_change3']==elem_i))

       count_functional = []  
       sum_overall = umaps['dev_'+df_i].shape[0]
       for score_i in ['FT9','MiToS','Kings','ALSFRS_Total']:
           for cat_i in ['low','high']:
               for elem_i in [0,1]:
                   tmp= [score_i,cat_i,elem_i]
                   sum_zone = umaps['dev_'+df_i].loc[spa_mask[cat_i],:].shape[0]
                   sum_change = umaps['dev_'+df_i].loc[umaps['dev_'+df_i][score_i+'_change']==elem_i,:].shape[0]
                   val_count = umaps['dev_'+df_i].loc[masks_analysis[score_i][cat_i][elem_i],:].shape[0]
                   tmp.append(val_count)
                   tmp.append(val_count/sum_zone)
                   tmp.append(val_count/sum_overall)
                   tmp.append(val_count/sum_change)
                   tmp.append(sum_zone)
                   tmp.append(sum_change)
                   
                   count_functional.append(tmp)
                   
       count_functional = pd.DataFrame(count_functional,columns=['score','cat','stage','count','freq_zone','freq_overall','freq_change','count_zone','count_change'])
       #count_functional.to_csv('output/reboot/functional_binary_change.csv',sep=';')
       
       # CHANGE 3
       count_functional = []  
       sum_overall = umaps['dev_'+df_i].shape[0]
       for score_i in ['FT9','MiToS','Kings','ALSFRS_Total']:
           for cat_i in ['low','high']:
               for elem_i in [0,1]:
                   tmp= [score_i,cat_i,elem_i]
                   sum_zone = umaps['dev_'+df_i].loc[spa_mask[cat_i],:].shape[0]
                   sum_change = umaps['dev_'+df_i].loc[umaps['dev_'+df_i][score_i+'_change3']==elem_i,:].shape[0]
                   val_count = umaps['dev_'+df_i].loc[masks_analysis2[score_i][cat_i][elem_i],:].shape[0]
                   tmp.append(val_count)
                   tmp.append(val_count/sum_zone)
                   tmp.append(val_count/sum_overall)
                   tmp.append(val_count/sum_change)
                   tmp.append(sum_zone)
                   tmp.append(sum_change)
                   
                   count_functional.append(tmp)
                   
       count_functional = pd.DataFrame(count_functional,columns=['score','cat','stage','count','freq_zone','freq_overall','freq_change','count_zone','count_change'])
       #count_functional.to_csv('output/reboot/functional_binary_change3.csv',sep=';')
     
       ### DIAGONAL SPLIT
       def calcDiag(x):
           
           diff = x['y_axis']-(x['x_axis']*6/17+73/340) # 6/17+73/340
           if diff < 0:
               return 'low'
           else:
               return 'high'
       
       umaps['dev_'+df_i]['diag'] = umaps['dev_'+df_i].loc[:,['x_axis','y_axis']].apply(lambda x: calcDiag(x),axis=1)
       
       spa_mask2 = {}
       spa_mask2['low'] = (umaps['dev_'+df_i].diag=='low')
       spa_mask2['high']= (umaps['dev_'+df_i].diag=='high')

       masks_analysis2 = {}
       for score_i in ['FT9','MiToS','Kings','ALSFRS_Total']:
           masks_analysis2[score_i] = {}
           for cat_i in ['low','high']:
               masks_analysis2[score_i][cat_i] = {}
               for elem_i in [0,1]:
                   masks_analysis2[score_i][cat_i][elem_i] = (spa_mask2[cat_i] & (umaps['dev_'+df_i][score_i+'_change']==elem_i))
       
       # Changement en diagonal 
        
       count_functional2 = []  
       sum_overall = umaps['dev_'+df_i].shape[0]
       for score_i in ['FT9','MiToS','Kings','ALSFRS_Total']:
           for cat_i in ['low','high']:
               for elem_i in [0,1]:
                   tmp= [score_i,cat_i,elem_i]
                   sum_zone = umaps['dev_'+df_i].loc[spa_mask2[cat_i],:].shape[0]
                   sum_change = umaps['dev_'+df_i].loc[umaps['dev_'+df_i][score_i+'_change']==elem_i,:].shape[0]
                   val_count = umaps['dev_'+df_i].loc[masks_analysis2[score_i][cat_i][elem_i],:].shape[0]
                   tmp.append(val_count)
                   tmp.append(val_count/sum_zone)
                   tmp.append(val_count/sum_overall)
                   tmp.append(val_count/sum_change)
                   tmp.append(sum_zone)
                   tmp.append(sum_change)
                   
                   count_functional2.append(tmp)
                   
       count_functional2 = pd.DataFrame(count_functional2,columns=['score','cat','stage','count','freq_zone','freq_overall','freq_change','count_zone','count_change'])
       #count_functional2.to_csv('output/reboot/functional_binary_change_diag.csv',sep=';') 
        
       
        
       
       
       ##### LOW ZONE (MiToS 0/1 - FT9 0/1 - King 1/2 ALSFRS >= 30)
       # Voir le combiné des 3 et 4 (ALSFRS exc puis inc)
       ####" CHANGED TO MiTOS 0 only pour avoir bon resultats
       
       comb_lz = {0:(0,0),1:(0,1),2:(1,0),
                  3:(1,1)}
       
       
       # Count for low zone analysis
      
       
       def calcDiag_low(x):
           
           diff = x['y_axis']-(x['x_axis']*0.2+0.4) # 6/17+73/340
           if diff < 0:
               return 'bottom'
           else:
               return 'top'
       
       umaps['dev_'+df_i]['diag_low'] = umaps['dev_'+df_i].loc[:,['x_axis','y_axis']].apply(lambda x: calcDiag_low(x),axis=1)
       
       umaps['val_'+df_i]['diag_low'] = umaps['val_'+df_i].loc[:,['x_axis','y_axis']].apply(lambda x: calcDiag_low(x),axis=1)

       
       spa_mask_low = {}
       spa_mask_low['bottom'] = (umaps['dev_'+df_i].diag_low=='bottom')
       spa_mask_low['top']= (umaps['dev_'+df_i].diag_low=='top')
       
       spa_mask_low_val = {}
       spa_mask_low_val['bottom'] = (umaps['val_'+df_i].diag_low=='bottom')
       spa_mask_low_val['top']= (umaps['val_'+df_i].diag_low=='top')
       
       mask_lz_val = {'FT9':{0:umaps['val_functional'].FT9_score_final<=1,
                         1:umaps['val_functional'].FT9_score_final>1},
                  'MiToS':{0:umaps['val_functional'].MiToS_score_final<1,
                           1:umaps['val_functional'].MiToS_score_final>=1},
                  'Kings':{0:umaps['val_functional'].Kings_score_final<=2,
                           1:umaps['val_functional'].Kings_score_final>2},
                  'ALSFRS_Total':{0:umaps['val_functional'].ALSFRS_Total_final>=30,
                                  1:umaps['val_functional'].ALSFRS_Total_final<30}
                  }


       masks_analysis_low = {}
       masks_analysis_low_val = {}

       mask_lz = {'FT9':{0:umaps['dev_functional'].FT9_score_final<=1,
                         1:umaps['dev_functional'].FT9_score_final>1},
                  'MiToS':{0:umaps['dev_functional'].MiToS_score_final<1,
                           1:umaps['dev_functional'].MiToS_score_final>=1},
                  'Kings':{0:umaps['dev_functional'].Kings_score_final<=2,
                           1:umaps['dev_functional'].Kings_score_final>2},
                  'ALSFRS_Total':{0:umaps['dev_functional'].ALSFRS_Total_final>=30,
                                  1:umaps['dev_functional'].ALSFRS_Total_final<30}
                  }

       


       for score_i in ['FT9','MiToS','Kings','ALSFRS_Total']:
           masks_analysis_low[score_i] = {}
           masks_analysis_low_val[score_i] = {}

           for cat_i in ['bottom','top']:
               masks_analysis_low[score_i][cat_i] = {}
               masks_analysis_low_val[score_i][cat_i] = {}

               for elem_i in [0,1]:
                   masks_analysis_low[score_i][cat_i][elem_i] = (spa_mask_low[cat_i] & (mask_lz[score_i][elem_i]))
                   masks_analysis_low_val[score_i][cat_i][elem_i] = (spa_mask_low_val[cat_i] & (mask_lz_val[score_i][elem_i]))

        # Changement en diagonal 
        
       count_functional_low = []  
       sum_overall = umaps['dev_'+df_i].shape[0]
       for score_i in ['FT9','MiToS','Kings','ALSFRS_Total']:
           for cat_i in ['top','bottom']:
               for elem_i in [0,1]:
                   tmp= [score_i,cat_i,elem_i]
                   sum_zone = umaps['dev_'+df_i].loc[spa_mask_low[cat_i],:].shape[0]
                   sum_change = umaps['dev_'+df_i].loc[mask_lz[score_i][elem_i],:].shape[0]
                   val_count = umaps['dev_'+df_i].loc[masks_analysis_low[score_i][cat_i][elem_i],:].shape[0]
                   tmp.append(val_count)
                   tmp.append(val_count/sum_zone)
                   tmp.append(val_count/sum_overall)
                   tmp.append(val_count/sum_change)
                   tmp.append(sum_zone)
                   tmp.append(sum_change)
                   
                   count_functional_low.append(tmp)
                   
       count_functional_low = pd.DataFrame(count_functional_low,columns=['score','cat','stage','count','freq_zone','freq_overall','freq_change','count_zone','count_change'])
       #count_functional_low.to_csv('output/reboot/functional_low_zone_count.csv',sep=';') 
       
       count_functional_low_val = []  
       sum_overall = umaps['val_'+df_i].shape[0]
       for score_i in ['FT9','MiToS','Kings','ALSFRS_Total']:
           for cat_i in ['top','bottom']:
               for elem_i in [0,1]:
                   tmp= [score_i,cat_i,elem_i]
                   sum_zone = umaps['val_'+df_i].loc[spa_mask_low_val[cat_i],:].shape[0]
                   sum_change = umaps['val_'+df_i].loc[mask_lz_val[score_i][elem_i],:].shape[0]
                   val_count = umaps['val_'+df_i].loc[masks_analysis_low_val[score_i][cat_i][elem_i],:].shape[0]
                   tmp.append(val_count)
                   tmp.append(val_count/sum_zone)
                   tmp.append(val_count/sum_overall)
                   tmp.append(val_count/sum_change)
                   tmp.append(sum_zone)
                   tmp.append(sum_change)
                   count_functional_low_val.append(tmp)
                   
       count_functional_low_val = pd.DataFrame(count_functional_low_val,columns=['score','cat','stage','count','freq_zone','freq_overall','freq_change','count_zone','count_change'])
       #count_functional_low_val.to_csv('output/reboot/functional_low_zone_count_val.csv',sep=';') 
       
       ############# MANUAL DIVISION FOR FUNCTIONAL (3 zones) - 
       
       def def3Zone(x):
           
           diff_up = x['y_axis']-(x['x_axis']*0.2+0.4)
           diff_down = x['y_axis']-(x['x_axis']*0.5-0.1)# 6/17+73/340
           if diff_up > 0:
               return 'top'
           elif diff_down < 0:
               return 'bottom'
           else:
               return 'mid'
       
       umaps['dev_'+df_i]['diag_3z'] = umaps['dev_'+df_i].loc[:,['x_axis','y_axis']].apply(lambda x: def3Zone(x),axis=1)
       #umaps['dev_'+df_i].to_csv('output/reboot/functional_3zone_testing.csv',sep=';') 
       
       umaps['val_'+df_i]['diag_3z'] = umaps['val_'+df_i].loc[:,['x_axis','y_axis']].apply(lambda x: def3Zone(x),axis=1)
       
       # Test normality of ALSFRS_Total_final to see if standard error means anything

       
       spa_mask_3z = {}
       for cat_i in ['dev','val']:
           spa_mask_3z[cat_i] = {}
           spa_mask_3z[cat_i]['bottom'] = (umaps[cat_i+'_'+df_i].diag_3z=='bottom')
           spa_mask_3z[cat_i]['top']= (umaps[cat_i+'_'+df_i].diag_3z=='top')
           spa_mask_3z[cat_i]['mid']= (umaps[cat_i+'_'+df_i].diag_3z=='mid')

       
       mask_func = {}
       for cat_i in ['dev','val']:
           mask_func[cat_i] = {'ALSFRS4_10':(umaps[cat_i+'_'+df_i].ALSFRS_Total_final<10),
                    'ALSFRS4_20':((umaps[cat_i+'_'+df_i].ALSFRS_Total_final>=10)&(umaps[cat_i+'_'+df_i].ALSFRS_Total_final<20)),
                    'ALSFRS4_30':((umaps[cat_i+'_'+df_i].ALSFRS_Total_final>=20)&(umaps[cat_i+'_'+df_i].ALSFRS_Total_final<30)),
                    'ALSFRS4_40':(umaps[cat_i+'_'+df_i].ALSFRS_Total_final>=30),
                    'ALSFRS8_5':(umaps[cat_i+'_'+df_i].ALSFRS_Total_final<5),
                    'ALSFRS8_10':((umaps[cat_i+'_'+df_i].ALSFRS_Total_final>=5)&(umaps[cat_i+'_'+df_i].ALSFRS_Total_final<10)),
                    'ALSFRS8_15':((umaps[cat_i+'_'+df_i].ALSFRS_Total_final>=10)&(umaps[cat_i+'_'+df_i].ALSFRS_Total_final<15)),
                    'ALSFRS8_20':((umaps[cat_i+'_'+df_i].ALSFRS_Total_final>=15)&(umaps[cat_i+'_'+df_i].ALSFRS_Total_final<20)),
                    'ALSFRS8_25':((umaps[cat_i+'_'+df_i].ALSFRS_Total_final>=20)&(umaps[cat_i+'_'+df_i].ALSFRS_Total_final<25)),
                    'ALSFRS8_30':((umaps[cat_i+'_'+df_i].ALSFRS_Total_final>=25)&(umaps[cat_i+'_'+df_i].ALSFRS_Total_final<30)),
                    'ALSFRS8_35':((umaps[cat_i+'_'+df_i].ALSFRS_Total_final>=30)&(umaps[cat_i+'_'+df_i].ALSFRS_Total_final<35)),
                    'ALSFRS8_40':(umaps[cat_i+'_'+df_i].ALSFRS_Total_final>=35),
                    }         
           for stage_i in ['Kings','FT9','MiToS']:
               if stage_i == 'Kings':
                   value_stages = [1,2,3,4,4.5]
               else:  
                   value_stages = [0,1,2,3,4]
               for value_i in value_stages:
                   mask_func[cat_i][stage_i+'_'+str(value_i)] = (umaps[cat_i+'_'+df_i][stage_i+'_score_final']==value_i)
       
       masks_3z = {}
       for cat_i in ['dev','val']:
           masks_3z[cat_i] = {}
           for zone_i in ['bottom','top','mid']:
               masks_3z[cat_i][zone_i] = {}
               for elem_i in mask_func['dev'].keys():
                   masks_3z[cat_i][zone_i][elem_i] = (spa_mask_3z[cat_i][zone_i] & (mask_func[cat_i][elem_i]))

       count_functional_3z = []     

       for cat_i in ['dev','val']: 
           
           for zone_i in ['top','bottom','mid']:
               for elem_i in mask_func['dev'].keys():
                   tmp= [cat_i,zone_i,elem_i]
                   if elem_i.split('_')[0] in ['FT9','MiToS','Kings']:
                       sum_overall = umaps[cat_i+'_'+df_i].loc[:,elem_i.split('_')[0]+'_score_final'].dropna().shape[0]
                       sum_zone = umaps[cat_i+'_'+df_i].loc[spa_mask_3z[cat_i][zone_i],elem_i.split('_')[0]+'_score_final'].dropna().shape[0]
                       sum_change = umaps[cat_i+'_'+df_i].loc[mask_func[cat_i][elem_i],elem_i.split('_')[0]+'_score_final'].dropna().shape[0]
                       val_count = umaps[cat_i+'_'+df_i].loc[masks_3z[cat_i][zone_i][elem_i],:].shape[0]
                   else:
                       sum_overall = umaps[cat_i+'_'+df_i].shape[0]
                       sum_zone = umaps[cat_i+'_'+df_i].loc[spa_mask_3z[cat_i][zone_i],:].shape[0]
                       sum_change = umaps[cat_i+'_'+df_i].loc[mask_func[cat_i][elem_i],:].shape[0]
                       val_count = umaps[cat_i+'_'+df_i].loc[masks_3z[cat_i][zone_i][elem_i],:].shape[0]
                   tmp.append(val_count)
                   tmp.append(val_count/sum_zone)
                   tmp.append(val_count/sum_overall)
                   tmp.append(val_count/sum_change)
                   tmp.append(sum_zone)
                   tmp.append(sum_change)
                   tmp.append(sum_overall)
                   tmp.append(umaps[cat_i+'_'+df_i].loc[spa_mask_3z[cat_i][zone_i],'ALSFRS_Total_final'].mean())
                   tmp.append(umaps[cat_i+'_'+df_i].loc[spa_mask_3z[cat_i][zone_i],'ALSFRS_Total_final'].std())
                   tmp.append(umaps[cat_i+'_'+df_i].loc[spa_mask_3z[cat_i][zone_i],'ALSFRS_Total_final'].quantile(q=0.1))
                   tmp.append(umaps[cat_i+'_'+df_i].loc[spa_mask_3z[cat_i][zone_i],'ALSFRS_Total_final'].quantile(q=0.2))
                   tmp.append(umaps[cat_i+'_'+df_i].loc[spa_mask_3z[cat_i][zone_i],'ALSFRS_Total_final'].quantile(q=0.25))
                   tmp.append(umaps[cat_i+'_'+df_i].loc[spa_mask_3z[cat_i][zone_i],'ALSFRS_Total_final'].quantile(q=0.30))
                   tmp.append(umaps[cat_i+'_'+df_i].loc[spa_mask_3z[cat_i][zone_i],'ALSFRS_Total_final'].quantile(q=0.33))
                   tmp.append(umaps[cat_i+'_'+df_i].loc[spa_mask_3z[cat_i][zone_i],'ALSFRS_Total_final'].quantile(q=0.40))
                   tmp.append(umaps[cat_i+'_'+df_i].loc[spa_mask_3z[cat_i][zone_i],'ALSFRS_Total_final'].quantile(q=0.50))
                   tmp.append(umaps[cat_i+'_'+df_i].loc[spa_mask_3z[cat_i][zone_i],'ALSFRS_Total_final'].quantile(q=0.60))
                   tmp.append(umaps[cat_i+'_'+df_i].loc[spa_mask_3z[cat_i][zone_i],'ALSFRS_Total_final'].quantile(q=0.66))
                   tmp.append(umaps[cat_i+'_'+df_i].loc[spa_mask_3z[cat_i][zone_i],'ALSFRS_Total_final'].quantile(q=0.70))
                   tmp.append(umaps[cat_i+'_'+df_i].loc[spa_mask_3z[cat_i][zone_i],'ALSFRS_Total_final'].quantile(q=0.75))
                   tmp.append(umaps[cat_i+'_'+df_i].loc[spa_mask_3z[cat_i][zone_i],'ALSFRS_Total_final'].quantile(q=0.80))
                   tmp.append(umaps[cat_i+'_'+df_i].loc[spa_mask_3z[cat_i][zone_i],'ALSFRS_Total_final'].quantile(q=0.90))
                   count_functional_3z.append(tmp)
                   
       count_functional_3z = pd.DataFrame(count_functional_3z,columns=['range',
                                                                         'zone',
                                                                         'scope',
                                                                         'count',
                                                                         'freq_zone',
                                                                         'freq_overall',
                                                                         'freq_change',
                                                                         'count_zone',
                                                                         'count_change',
                                                                         'count_overall',
                                                                         'mean',
                                                                         'std',
                                                                         'q10',
                                                                         'q20',
                                                                         'q25',
                                                                         'q30',
                                                                         'q33',
                                                                         'q40',
                                                                         'q50',
                                                                         'q60',
                                                                         'q66',
                                                                         'q70',
                                                                         'q75',
                                                                         'q80',
                                                                         'q90'])
       #count_functional_3z.to_csv('output/reboot/functional_3z_count.csv',sep=';') 
      
##### STATS FIRST TABLE

feats_kept = ['x_axis', 'y_axis', 'sex', 'onset_spinal', 'age', 'time_disease',
              'weight_baseline', 'ALSFRS_Total_baseline','ALSFRS_Total_final',
              'source','ALSFRS_Total_estim_slope']

out_feat = {'functional':'ALSFRS_Total_final',
            'survival':'survived'}

feats_func = {'functional':['computeZones_{}_{}'.format(x,y) for y in ['baseline','final'] for x in ['trunk','mouth','respiratory','leg','hand']]+[x+'_score_final' for x in ['Kings','MiToS','FT9']],
              'survival':['death_unit']}

dict_feats = {'survival':{'onset_spinal':['sum'],
                                                                   'sex':['sum'],                                                           
                                                                   'age':['mean','std','min','max'],
                                                                   'time_disease':['mean','std','min','max'],
                                                                   'weight_baseline':['mean','std','min','max'],
                                                                   'ALSFRS_Total_baseline':['mean','std','min','max'],
                                                                   'ALSFRS_Total_estim_slope':['mean','std','min','max'],
                                                                   'subject_id':['count'],
                                                                   'ALSFRS_Total_final':['mean','std','min','max'],
                                                                   'survived':['sum'],
                                                                   'death_unit':['mean','std','min','max']},
              'functional':{'onset_spinal':['sum'],
                                                                   'sex':['sum'],
                                                                   'age':['mean','std','min','max'],
                                                                   'time_disease':['mean','std','min','max'],
                                                                   'weight_baseline':['mean','std','min','max'],
                                                                   'ALSFRS_Total_baseline':['mean','std','min','max'],
                                                                   'ALSFRS_Total_estim_slope':['mean','std','min','max'],
                                                                   'subject_id':['count'],
                                                                   'ALSFRS_Total_final':['mean','std','min','max'],
                                                                   'computeZones_trunk_baseline':['mean','std','min','max'],
                                                                   'computeZones_mouth_baseline':['mean','std','min','max'],
                                                                   'computeZones_respiratory_baseline':['mean','std','min','max'],
                                                                   'computeZones_leg_baseline':['mean','std','min','max'],
                                                                   'computeZones_hand_baseline':['mean','std','min','max'],
                                                                   'computeZones_trunk_final':['mean','std','min','max'],
                                                                   'computeZones_mouth_final':['mean','std','min','max'],
                                                                   'computeZones_respiratory_final':['mean','std','min','max'],
                                                                   'computeZones_leg_final':['mean','std','min','max'],
                                                                   'computeZones_hand_final':['mean','std','min','max'],
                                                                   }}

'''

for df_i in ['functional','survival']:
    
    
    
    print('df_i: {}'.format(df_i))
    if df_i == 'functional':
        feats_tmp = feats_kept+[out_feat['functional']]+feats_func['functional']
        concat_df = pd.concat([umaps['dev_'+df_i].loc[patient_mask,:],umaps['val_'+df_i]])

    else:
        feats_tmp = feats_kept + [out_feat['survival']]+feats_func['survival']
        tmp = pd.merge(umaps['dev_'+df_i],pd.DataFrame([],index=patients_kept,columns=['x']),left_index=True,right_index=True,how='right')
        tmp.drop(['x'],axis=1,inplace=True)
        concat_df = pd.concat([tmp,umaps['val_'+df_i]])

    try:
        umaps['val_'+df_i].set_index('subject_id',inplace=True)
    except:
        pass
    concat_df = concat_df.loc[:,feats_tmp]
    concat_df.dropna(subset=[out_feat[df_i]],inplace=True)
    print('df_i: {} - concat_df:{}'.format(df_i,concat_df.shape[0]))
    concat_df.reset_index(inplace=True)
    grid_groupby = concat_df.groupby('source').agg((dict_feats[df_i]))
    print(grid_groupby)
    #grid_groupby.to_csv('output/reboot/grid_groupby.csv',sep=';')
    grid_all = [concat_df.onset_spinal.sum(),
                concat_df.sex.sum(),
                concat_df.age.mean(),
                concat_df.age.std(),
                concat_df.age.min(),
                concat_df.age.max(),
                concat_df.time_disease.mean(),
                concat_df.time_disease.std(),
                concat_df.time_disease.min(),
                concat_df.time_disease.max(),
                concat_df.weight_baseline.mean(),
                concat_df.weight_baseline.std(),
                concat_df.weight_baseline.min(),
                concat_df.weight_baseline.max(),
                concat_df.ALSFRS_Total_baseline.mean(),
                concat_df.ALSFRS_Total_baseline.std(),
                concat_df.ALSFRS_Total_baseline.min(),
                concat_df.ALSFRS_Total_baseline.max(),
                concat_df.ALSFRS_Total_estim_slope.mean(),
                concat_df.ALSFRS_Total_estim_slope.std(),
                concat_df.ALSFRS_Total_estim_slope.min(),
                concat_df.ALSFRS_Total_estim_slope.max(),
                concat_df.reset_index().subject_id.count(),
                concat_df.ALSFRS_Total_final.mean(),
                concat_df.ALSFRS_Total_final.std(),
                concat_df.ALSFRS_Total_final.min(),
                concat_df.ALSFRS_Total_final.max(),
        ]

    grid_output_col = ['group','n','Gender (male/female)','Onset (spinal/bulbar)',
           'age (year)','time since onset (month)','baseline weight (kg)',
           'baseline ALSFRS','baseline ALSFRS decline rate','1-year ALSFRS']

    if df_i == 'survival':
        grid_all = grid_all+[
                    concat_df.survived.sum(),
                    concat_df.death_unit.mean(),
                    concat_df.death_unit.std(),
                    concat_df.death_unit.min(),
                    concat_df.death_unit.max()]
        grid_output_col += ['survival','survival (months)']
    else:
        grid_all = grid_all+[concat_df.computeZones_trunk_baseline.mean(),
                            concat_df.computeZones_trunk_baseline.std(),
                            concat_df.computeZones_trunk_baseline.min(),
                            concat_df.computeZones_trunk_baseline.max(),
                            concat_df.computeZones_mouth_baseline.mean(),
                            concat_df.computeZones_mouth_baseline.std(),
                            concat_df.computeZones_mouth_baseline.min(),
                            concat_df.computeZones_mouth_baseline.max(),
                            concat_df.computeZones_respiratory_baseline.mean(),
                            concat_df.computeZones_respiratory_baseline.std(),
                            concat_df.computeZones_respiratory_baseline.min(),
                            concat_df.computeZones_respiratory_baseline.max(),
                            concat_df.computeZones_leg_baseline.mean(),
                            concat_df.computeZones_leg_baseline.std(),
                            concat_df.computeZones_leg_baseline.min(),
                            concat_df.computeZones_leg_baseline.max(),
                            concat_df.computeZones_hand_baseline.mean(),
                            concat_df.computeZones_hand_baseline.std(),
                            concat_df.computeZones_hand_baseline.min(),
                            concat_df.computeZones_hand_baseline.max(),
                            concat_df.computeZones_trunk_final.mean(),
                            concat_df.computeZones_trunk_final.std(),
                            concat_df.computeZones_trunk_final.min(),
                            concat_df.computeZones_trunk_final.max(),
                            concat_df.computeZones_mouth_final.mean(),
                            concat_df.computeZones_mouth_final.std(),
                            concat_df.computeZones_mouth_final.min(),
                            concat_df.computeZones_mouth_final.max(),
                            concat_df.computeZones_respiratory_final.mean(),
                            concat_df.computeZones_respiratory_final.std(),
                            concat_df.computeZones_respiratory_final.min(),
                            concat_df.computeZones_respiratory_final.max(),
                            concat_df.computeZones_leg_final.mean(),
                            concat_df.computeZones_leg_final.std(),
                            concat_df.computeZones_leg_final.min(),
                            concat_df.computeZones_leg_final.max(),
                            concat_df.computeZones_hand_final.mean(),
                            concat_df.computeZones_hand_final.std(),
                            concat_df.computeZones_hand_final.min(),
                            concat_df.computeZones_hand_final.max()]
        grid_output_col += ['baseline trunk sub score (score)',
                            'baseline bulbar sub score (score)',
                            'baseline respiratory sub score (score)',
                            'baseline upper limb sub score (score)',
                            'baseline lower limb sub score (score)',
                            '1-year trunk sub score (score)',
                            '1-year bulbar sub score (score)',
                            '1-year respiratory sub score (score)',
                            '1-year upper limb sub score (score)',
                            '1-year lower limb sub score (score)',]

    grid_all = pd.DataFrame(grid_all,columns=['all'],index=grid_groupby.columns).transpose()
    grid_all.reset_index(inplace=True)
    grid_groupby.reset_index(inplace=True)
    grid_groupby = grid_groupby.append(grid_all,ignore_index=True,sort=False)
    grid_groupby.set_index('group',inplace=True)

    group_name = {'low':'low survival rate zone','mid':'intermediate survival rate zone','high':'high survival rate zone','all':'overall'}
    
    
    
    ### en commentaire avant
    grid_output = []
    for idx,line_i in grid_groupby.iterrows():
        #print(idx)
        if idx in ['mid','low','high']:
            tmp = [group_name[idx]]
        else:
            tmp = ['overall']
        tmp.append(int(line_i['subject_id']['count']))
        for feat_i in ['survived','sex','onset_spinal','age','time_disease','weight_baseline','ALSFRS_Total_baseline','ALSFRS_Total_estim_slope','ALSFRS_Total_final']:
            if feat_i not in ['survived','sex','onset_spinal']:
                if feat_i == 'ALSFRS_Total_estim_slope':    
                    tmp.append('{:.2f} +/- {:.2f} ({:.2f}-{:.2f}) '.format(line_i[(feat_i,'mean')],line_i[(feat_i,'std')],line_i[(feat_i,'min')],line_i[(feat_i,'max')]))
                else:
                    tmp.append('{:.1f} +/- {:.1f} ({:.1f}-{:.1f}) '.format(line_i[(feat_i,'mean')],line_i[(feat_i,'std')],line_i[(feat_i,'min')],line_i[(feat_i,'max')]))
            else:
                tmp.append('{:.0f}/{:.0f}'.format(line_i[(feat_i,'sum')],line_i[('subject_id','count')]-line_i[(feat_i,'sum')]))
        
        grid_output.append(tmp)

    grid_output = pd.DataFrame(grid_output,columns=grid_output_col)
    grid_output.set_index('group',inplace=True)
    grid_output = grid_output.reindex(index=['high survival rate zone','intermediate survival rate zone','low survival rate zone','overall'])
    grid_output.to_csv('output/reboot/article/survival/grid_stats_survival.csv',sep=';')
    # en commentaire avant (fin)

'''

'''
plots_n = 9
fig, axes = plt.subplots(1,plots_n,figsize=(8*plots_n,8))

masks = {}
masks[0] = (umaps['dev_functional']['ALSFRS_Total_final']==0)
masks[1] = ((umaps['dev_functional']['ALSFRS_Total_final']<=5)&(umaps['dev_functional']['ALSFRS_Total_final']>0))
masks[2] = ((umaps['dev_functional']['ALSFRS_Total_final']<=10)&(umaps['dev_functional']['ALSFRS_Total_final']>5))
masks[3] = ((umaps['dev_functional']['ALSFRS_Total_final']<=15)&(umaps['dev_functional']['ALSFRS_Total_final']>10))
masks[4] = ((umaps['dev_functional']['ALSFRS_Total_final']<=20)&(umaps['dev_functional']['ALSFRS_Total_final']>15))
masks[5] = ((umaps['dev_functional']['ALSFRS_Total_final']<=25)&(umaps['dev_functional']['ALSFRS_Total_final']>20))
masks[6] = ((umaps['dev_functional']['ALSFRS_Total_final']<=30)&(umaps['dev_functional']['ALSFRS_Total_final']>25))
masks[7] = ((umaps['dev_functional']['ALSFRS_Total_final']<=35)&(umaps['dev_functional']['ALSFRS_Total_final']>30))
masks[8] = (umaps['dev_functional']['ALSFRS_Total_final']>35)

for i in range(plots_n):
    ax = axes[i]
    ax.grid(b=True)
    ax.set_title('{} \n '.format(i),
                 fontsize='small')
    ax.scatter(umaps['dev_functional'].loc[masks[i],'x_axis'],
               umaps['dev_functional'].loc[masks[i],'y_axis'],
               marker='.',
               c=['black']*umaps['dev_functional'].loc[masks[i],'y_axis'].shape[0],
               alpha=1,
               s=10,
               )
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_xticks([x/10 for x in range(0,11)])
    ax.set_yticks([x/10 for x in range(0,11)])
         
plt.savefig('output/reboot/article/functional/test_grid_functional8.pdf')

fig, axes = plt.subplots(1,2,figsize=(8*2,8))

masks = {}
masks[0] = (umaps['dev_functional']['ALSFRS_Total_final']<=20)
masks[1] = (umaps['dev_functional']['ALSFRS_Total_final']>20)

for i in range(2):
    ax = axes[i]
    ax.grid(b=True)
    ax.set_title('{} \n '.format(i),
                 fontsize='small')
    ax.scatter(umaps['dev_functional'].loc[masks[i],'x_axis'],
               umaps['dev_functional'].loc[masks[i],'y_axis'],
               marker='.',
               c=['black']*umaps['dev_functional'].loc[masks[i],'y_axis'].shape[0],
               alpha=1,
               s=10,
               )
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_xticks([x/10 for x in range(0,11)])
    ax.set_yticks([x/10 for x in range(0,11)])
         
plt.savefig('output/reboot/article/functional/test_grid_functional2.pdf')

'''

def zoneDefinition_postRebuttal3z(x):
           
    diff_up = x['y_axis']-0.5#(x['x_axis']*0.1+0.4)
    diff_down = x['y_axis']-(-x['x_axis']+0.5)# 6/17+73/340
    if diff_up > 0:
        return 'top'
    elif diff_down < 0:
        return 'bottom'
    else:
        return 'mid'
    


# WARNING DROP EXONHIT PATIENTS THAT FOLLOWED TREATMENT (END AS INPUT PLOT DONE)
#patient_mask = (umaps['dev_functional'].index.isin(patients_kept))
umaps['dev_functional'] = umaps['dev_functional'].loc[patient_mask,:]

umaps['dev_functional']['zone_rebuttal'] = umaps['dev_functional'].loc[:,['x_axis','y_axis']].apply(lambda x: zoneDefinition_postRebuttal3z(x),axis=1)
#umaps['dev_'+df_i].to_csv('output/reboot/functional_3zone_testing.csv',sep=';') 

umaps['val_functional']['zone_rebuttal'] = umaps['val_functional'].loc[:,['x_axis','y_axis']].apply(lambda x: zoneDefinition_postRebuttal3z(x),axis=1)

spa_mask_zoneReb = {}
for cat_i in ['dev','val']:
    spa_mask_zoneReb[cat_i] = {}
    spa_mask_zoneReb[cat_i]['bottom'] = (umaps[cat_i+'_functional'].zone_rebuttal=='bottom')
    spa_mask_zoneReb[cat_i]['top']= (umaps[cat_i+'_functional'].zone_rebuttal=='top')
    spa_mask_zoneReb[cat_i]['mid']= (umaps[cat_i+'_functional'].zone_rebuttal=='mid')


mask_func_zoneReb = {}
for cat_i in ['dev','val']:
    mask_func_zoneReb[cat_i] = {
             'ALSFRS4_0':(umaps[cat_i+'_functional'].ALSFRS_Total_final==0),
             'ALSFRS4_10':((umaps[cat_i+'_functional'].ALSFRS_Total_final<=10)&(umaps[cat_i+'_functional'].ALSFRS_Total_final>0)),
             'ALSFRS4_20':((umaps[cat_i+'_functional'].ALSFRS_Total_final>10)&(umaps[cat_i+'_functional'].ALSFRS_Total_final<=20)),
             'ALSFRS4_30':((umaps[cat_i+'_functional'].ALSFRS_Total_final>20)&(umaps[cat_i+'_functional'].ALSFRS_Total_final<=30)),
             'ALSFRS4_40':(umaps[cat_i+'_functional'].ALSFRS_Total_final>30),
             'ALSFRS8_0':(umaps[cat_i+'_functional'].ALSFRS_Total_final==0),
             'ALSFRS8_5':(umaps[cat_i+'_functional'].ALSFRS_Total_final<=5)&(umaps[cat_i+'_functional'].ALSFRS_Total_final>0),
            'ALSFRS8_10':((umaps[cat_i+'_functional'].ALSFRS_Total_final>5)&(umaps[cat_i+'_functional'].ALSFRS_Total_final<=10)),
            'ALSFRS8_15':((umaps[cat_i+'_functional'].ALSFRS_Total_final>10)&(umaps[cat_i+'_functional'].ALSFRS_Total_final<=15)),
            'ALSFRS8_20':((umaps[cat_i+'_functional'].ALSFRS_Total_final>15)&(umaps[cat_i+'_functional'].ALSFRS_Total_final<=20)),
            'ALSFRS8_25':((umaps[cat_i+'_functional'].ALSFRS_Total_final>20)&(umaps[cat_i+'_functional'].ALSFRS_Total_final<=25)),
            'ALSFRS8_30':((umaps[cat_i+'_functional'].ALSFRS_Total_final>25)&(umaps[cat_i+'_functional'].ALSFRS_Total_final<=30)),
            'ALSFRS8_35':((umaps[cat_i+'_functional'].ALSFRS_Total_final>30)&(umaps[cat_i+'_functional'].ALSFRS_Total_final<=35)),
            'ALSFRS8_40':(umaps[cat_i+'_functional'].ALSFRS_Total_final>35),}
    for stage_i in ['Kings','FT9','MiToS']:
        if stage_i == 'Kings':
            value_stages = [1,2,3,4,4.5]
        else:  
            value_stages = [0,1,2,3,4]
        for value_i in value_stages:
            mask_func_zoneReb[cat_i][stage_i+'_'+str(value_i)] = (umaps[cat_i+'_functional'][stage_i+'_score_final']==value_i)

masks_3z_zoneReb = {}
for cat_i in ['dev','val']:
    masks_3z_zoneReb[cat_i] = {}
    for zone_i in ['bottom','top','mid']:
        masks_3z_zoneReb[cat_i][zone_i] = {}
        for elem_i in mask_func_zoneReb['dev'].keys():
            masks_3z_zoneReb[cat_i][zone_i][elem_i] = (spa_mask_zoneReb[cat_i][zone_i] & (mask_func_zoneReb[cat_i][elem_i]))

count_functional_zoneReb = []     

for cat_i in ['dev','val']: 
    
    for zone_i in ['top','bottom','mid']:
        for elem_i in mask_func_zoneReb['dev'].keys():
            tmp= [cat_i,zone_i,elem_i]
            if elem_i.split('_')[0] in ['FT9','MiToS','Kings']:
                tmp.append('score')
                sum_overall = umaps[cat_i+'_functional'].loc[:,elem_i.split('_')[0]+'_score_final'].dropna().shape[0]
                sum_zone = umaps[cat_i+'_functional'].loc[spa_mask_zoneReb[cat_i][zone_i],elem_i.split('_')[0]+'_score_final'].dropna().shape[0]
                sum_change = umaps[cat_i+'_functional'].loc[mask_func_zoneReb[cat_i][elem_i],elem_i.split('_')[0]+'_score_final'].dropna().shape[0]
                val_count = umaps[cat_i+'_functional'].loc[masks_3z_zoneReb[cat_i][zone_i][elem_i],:].shape[0]
            else:
                tmp.append('func')
                sum_overall = umaps[cat_i+'_functional'].shape[0]
                sum_zone = umaps[cat_i+'_functional'].loc[spa_mask_zoneReb[cat_i][zone_i],:].shape[0]
                sum_change = umaps[cat_i+'_functional'].loc[mask_func_zoneReb[cat_i][elem_i],:].shape[0]
                val_count = umaps[cat_i+'_functional'].loc[masks_3z_zoneReb[cat_i][zone_i][elem_i],:].shape[0]
            tmp.append(val_count)
            tmp.append(val_count/sum_zone)
            tmp.append(val_count/sum_overall)
            tmp.append(val_count/sum_change)
            tmp.append(sum_zone)
            tmp.append(sum_change)
            tmp.append(sum_overall)
            tmp.append(umaps[cat_i+'_functional'].loc[spa_mask_zoneReb[cat_i][zone_i],'ALSFRS_Total_final'].mean())
            tmp.append(umaps[cat_i+'_functional'].loc[spa_mask_zoneReb[cat_i][zone_i],'ALSFRS_Total_final'].std())
            tmp.append(umaps[cat_i+'_functional'].loc[spa_mask_zoneReb[cat_i][zone_i],'ALSFRS_Total_final'].quantile(q=0.1))
            tmp.append(umaps[cat_i+'_functional'].loc[spa_mask_zoneReb[cat_i][zone_i],'ALSFRS_Total_final'].quantile(q=0.2))
            tmp.append(umaps[cat_i+'_functional'].loc[spa_mask_zoneReb[cat_i][zone_i],'ALSFRS_Total_final'].quantile(q=0.25))
            tmp.append(umaps[cat_i+'_functional'].loc[spa_mask_zoneReb[cat_i][zone_i],'ALSFRS_Total_final'].quantile(q=0.30))
            tmp.append(umaps[cat_i+'_functional'].loc[spa_mask_zoneReb[cat_i][zone_i],'ALSFRS_Total_final'].quantile(q=0.33))
            tmp.append(umaps[cat_i+'_functional'].loc[spa_mask_zoneReb[cat_i][zone_i],'ALSFRS_Total_final'].quantile(q=0.40))
            tmp.append(umaps[cat_i+'_functional'].loc[spa_mask_zoneReb[cat_i][zone_i],'ALSFRS_Total_final'].quantile(q=0.50))
            tmp.append(umaps[cat_i+'_functional'].loc[spa_mask_zoneReb[cat_i][zone_i],'ALSFRS_Total_final'].quantile(q=0.60))
            tmp.append(umaps[cat_i+'_functional'].loc[spa_mask_zoneReb[cat_i][zone_i],'ALSFRS_Total_final'].quantile(q=0.66))
            tmp.append(umaps[cat_i+'_functional'].loc[spa_mask_zoneReb[cat_i][zone_i],'ALSFRS_Total_final'].quantile(q=0.70))
            tmp.append(umaps[cat_i+'_functional'].loc[spa_mask_zoneReb[cat_i][zone_i],'ALSFRS_Total_final'].quantile(q=0.75))
            tmp.append(umaps[cat_i+'_functional'].loc[spa_mask_zoneReb[cat_i][zone_i],'ALSFRS_Total_final'].quantile(q=0.80))
            tmp.append(umaps[cat_i+'_functional'].loc[spa_mask_zoneReb[cat_i][zone_i],'ALSFRS_Total_final'].quantile(q=0.90))
            count_functional_zoneReb.append(tmp)
            
count_functional_zoneReb = pd.DataFrame(count_functional_zoneReb,columns=['range',
                                                                  'zone',
                                                                  'scope',
                                                                  'cat',
                                                                  'count',
                                                                  'freq_zone',
                                                                  'freq_overall',
                                                                  'freq_change',
                                                                  'count_zone',
                                                                  'count_change',
                                                                  'count_overall',
                                                                  'mean',
                                                                  'std',
                                                                  'q10',
                                                                  'q20',
                                                                  'q25',
                                                                  'q30',
                                                                  'q33',
                                                                  'q40',
                                                                  'q50',
                                                                  'q60',
                                                                  'q66',
                                                                  'q70',
                                                                  'q75',
                                                                  'q80',
                                                                  'q90'])

count_functional_zoneReb.to_csv('output/reboot/article/functional/zoneRebuttal_analysis.csv',sep=';',index=False)

def zoneDefinition_postRebuttal2z(x):
           
    diff_up = x['y_axis']-(x['x_axis']*0.1+0.4)
    if diff_up > 0:
        return 'top'
    else:
        return 'bottom'
    

umaps['dev_functional']['zone_rebuttal2z'] = umaps['dev_functional'].loc[:,['x_axis','y_axis']].apply(lambda x: zoneDefinition_postRebuttal2z(x),axis=1)

#umaps['dev_'+df_i].to_csv('output/reboot/functional_3zone_testing.csv',sep=';') 

umaps['val_functional']['zone_rebuttal2z'] = umaps['val_functional'].loc[:,['x_axis','y_axis']].apply(lambda x: zoneDefinition_postRebuttal2z(x),axis=1)

spa_mask_zoneReb2z = {}
for cat_i in ['dev','val']:
    spa_mask_zoneReb2z[cat_i] = {}
    spa_mask_zoneReb2z[cat_i]['bottom'] = (umaps[cat_i+'_functional'].zone_rebuttal2z=='bottom')
    spa_mask_zoneReb2z[cat_i]['top']= (umaps[cat_i+'_functional'].zone_rebuttal2z=='top')

masks_2z_zoneReb = {}
for cat_i in ['dev','val']:
    masks_2z_zoneReb[cat_i] = {}
    for zone_i in ['bottom','top']:
        masks_2z_zoneReb[cat_i][zone_i] = {}
        for elem_i in mask_func_zoneReb['dev'].keys():
            masks_2z_zoneReb[cat_i][zone_i][elem_i] = (spa_mask_zoneReb2z[cat_i][zone_i] & (mask_func_zoneReb[cat_i][elem_i]))

count_functional_zoneReb2z = []     

for cat_i in ['dev','val']: 
    
    for zone_i in ['top','bottom']:
        for elem_i in mask_func_zoneReb['dev'].keys():
            tmp= [cat_i,zone_i,elem_i]
            if elem_i.split('_')[0] in ['FT9','MiToS','Kings']:
                tmp.append('score')
                sum_overall = umaps[cat_i+'_functional'].loc[:,elem_i.split('_')[0]+'_score_final'].dropna().shape[0]
                sum_zone = umaps[cat_i+'_functional'].loc[spa_mask_zoneReb2z[cat_i][zone_i],elem_i.split('_')[0]+'_score_final'].dropna().shape[0]
                sum_change = umaps[cat_i+'_functional'].loc[mask_func_zoneReb[cat_i][elem_i],elem_i.split('_')[0]+'_score_final'].dropna().shape[0]
                val_count = umaps[cat_i+'_functional'].loc[masks_2z_zoneReb[cat_i][zone_i][elem_i],:].shape[0]
            else:
                tmp.append('func')
                sum_overall = umaps[cat_i+'_functional'].shape[0]
                
                sum_zone = umaps[cat_i+'_functional'].loc[spa_mask_zoneReb2z[cat_i][zone_i],:].shape[0]
                sum_change = umaps[cat_i+'_functional'].loc[mask_func_zoneReb[cat_i][elem_i],:].shape[0]
                val_count = umaps[cat_i+'_functional'].loc[masks_2z_zoneReb[cat_i][zone_i][elem_i],:].shape[0]
            tmp.append(val_count)
            tmp.append(val_count/sum_zone)
            tmp.append(val_count/sum_overall)
            tmp.append(val_count/sum_change)
            tmp.append(sum_zone)
            tmp.append(sum_change)
            tmp.append(sum_overall)
            tmp.append(umaps[cat_i+'_functional'].loc[spa_mask_zoneReb2z[cat_i][zone_i],'ALSFRS_Total_final'].mean())
            tmp.append(umaps[cat_i+'_functional'].loc[spa_mask_zoneReb2z[cat_i][zone_i],'ALSFRS_Total_final'].std())
            tmp.append(umaps[cat_i+'_functional'].loc[spa_mask_zoneReb2z[cat_i][zone_i],'ALSFRS_Total_final'].quantile(q=0.1))
            tmp.append(umaps[cat_i+'_functional'].loc[spa_mask_zoneReb2z[cat_i][zone_i],'ALSFRS_Total_final'].quantile(q=0.2))
            tmp.append(umaps[cat_i+'_functional'].loc[spa_mask_zoneReb2z[cat_i][zone_i],'ALSFRS_Total_final'].quantile(q=0.25))
            tmp.append(umaps[cat_i+'_functional'].loc[spa_mask_zoneReb2z[cat_i][zone_i],'ALSFRS_Total_final'].quantile(q=0.30))
            tmp.append(umaps[cat_i+'_functional'].loc[spa_mask_zoneReb2z[cat_i][zone_i],'ALSFRS_Total_final'].quantile(q=0.33))
            tmp.append(umaps[cat_i+'_functional'].loc[spa_mask_zoneReb2z[cat_i][zone_i],'ALSFRS_Total_final'].quantile(q=0.40))
            tmp.append(umaps[cat_i+'_functional'].loc[spa_mask_zoneReb2z[cat_i][zone_i],'ALSFRS_Total_final'].quantile(q=0.50))
            tmp.append(umaps[cat_i+'_functional'].loc[spa_mask_zoneReb2z[cat_i][zone_i],'ALSFRS_Total_final'].quantile(q=0.60))
            tmp.append(umaps[cat_i+'_functional'].loc[spa_mask_zoneReb2z[cat_i][zone_i],'ALSFRS_Total_final'].quantile(q=0.66))
            tmp.append(umaps[cat_i+'_functional'].loc[spa_mask_zoneReb2z[cat_i][zone_i],'ALSFRS_Total_final'].quantile(q=0.70))
            tmp.append(umaps[cat_i+'_functional'].loc[spa_mask_zoneReb2z[cat_i][zone_i],'ALSFRS_Total_final'].quantile(q=0.75))
            tmp.append(umaps[cat_i+'_functional'].loc[spa_mask_zoneReb2z[cat_i][zone_i],'ALSFRS_Total_final'].quantile(q=0.80))
            tmp.append(umaps[cat_i+'_functional'].loc[spa_mask_zoneReb2z[cat_i][zone_i],'ALSFRS_Total_final'].quantile(q=0.90))
            count_functional_zoneReb2z.append(tmp)
            
count_functional_zoneReb2z = pd.DataFrame(count_functional_zoneReb2z,columns=['range',
                                                                  'zone',
                                                                  'scope',
                                                                  'cat',
                                                                  'count',
                                                                  'freq_zone',
                                                                  'freq_overall',
                                                                  'freq_change',
                                                                  'count_zone',
                                                                  'count_change',
                                                                  'count_overall',
                                                                  'mean',
                                                                  'std',
                                                                  'q10',
                                                                  'q20',
                                                                  'q25',
                                                                  'q30',
                                                                  'q33',
                                                                  'q40',
                                                                  'q50',
                                                                  'q60',
                                                                  'q66',
                                                                  'q70',
                                                                  'q75',
                                                                  'q80',
                                                                  'q90'])

count_functional_zoneReb2z.to_csv('output/reboot/article/functional/zoneRebuttal_analysis2z.csv',sep=';',index=False)
