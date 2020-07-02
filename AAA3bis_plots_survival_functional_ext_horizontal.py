# -*- coding: utf-8 -*-
# 02/01/2020

# Plot for survival and functional - second take
# Input features for plot 
# Output features for plot
# Grid analysis 
# Train and test analysis
# Advanced grid analysis

#%matplotlib inline

import pandas as pd
import math
import sklearn.preprocessing as sk_p
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.random import rand
import matplotlib.patches as mpatches
import ndtest
import scipy.stats as sc_s
import seaborn as sns
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn
seaborn.set_style(style='white')
#seaborn.set()
import warnings
warnings.filterwarnings("ignore")
from pandas import ExcelWriter
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns; sns.set_style(style='white')
#import SeabornFig2Grid as sfg
from matplotlib.ticker import NullFormatter
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



exonhit_list = pd.read_excel('input/patient_exonhit_filtered.xlsx')
exonhit_list = list(exonhit_list.loc[:,'subject_id'])

# remove exonhit patients from output analysis if non placebo
# keep them both groups for initial projection
patients_kept = (list(dfs['dev_survival'].loc[((dfs['dev_survival'].source=='exonhit')&(dfs['dev_survival'].subject_id.isin(exonhit_list))),'subject_id'])+ 
                list(dfs['dev_survival'].loc[dfs['dev_survival'].source=='proact','subject_id'])+
                list(dfs['dev_survival'].loc[dfs['dev_survival'].source=='trophos','subject_id']))

# Loss of 173

# for ALSFRS analysis => add patients which died within the year 

patients_to_add = {}

    
    


functional_list = pd.read_csv('output/reboot/input_for_process/df_functional.csv',sep=';')

#### CHANGE TO REMOVE DEAD
#functional_dev = list(functional_list.loc[functional_list.subject_id.isin(patients_kept),'subject_id'])+patients_to_add['dev']
#functional_val = list(functional_list.loc[functional_list.source.isin(['pre2008']),'subject_id'])+patients_to_add['val']
functional_dev = list(functional_list.loc[~functional_list.subject_id.isin(['pre2008']),'subject_id'])
functional_val = list(functional_list.loc[functional_list.source.isin(['pre2008']),'subject_id'])


func_csv = functional_dev + functional_val
func_csv = pd.DataFrame(func_csv)
#func_csv.to_csv('output/reboot/functional_ext_list_subject.csv',sep=';')

'''
print('list dev_func: {}={}+{} val_func:{}={}+{}'.format(len(functional_dev),
                                                        len(list(functional_list.loc[functional_list.subject_id.isin(patients_kept),'subject_id'])),
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

for scope_i in ['dev','val']:
    dfs[scope_i+'_survival'].set_index('subject_id',inplace=True)
    # Tolerance added (12.1) as survived feature approximated for 12.1 month (hence )
    dead_ALSFRS_mask = ((~dfs[scope_i+'_survival'].death_unit.isnull())&(dfs[scope_i+'_survival'].death_unit<=12.1))
    patients_to_add[scope_i] = list(dfs[scope_i+'_survival'].loc[dead_ALSFRS_mask,:].index)
    dfs[scope_i+'_survival'].loc[dead_ALSFRS_mask,'ALSFRS_Total_final'] = 0

for df_i in [x+'_functional' for x in ['val','dev']]:
    umaps[df_i].drop(['survived'],axis=1,inplace=True)
#    umaps[df_i].dropna(subset=['computeZones_'+x+'_final' for x in ['mouth','leg','hand','trunk','respiratory']],inplace=True)
    
    

#dfs['dev_functional'] = pd.read_csv('output/reboot/df_functional.csv',sep=';')
#dfs['val_functional'] = dfs['dev_functional'].loc[dfs['dev_functional'].source.isin(['pre2008']),:]
#dfs['dev_functional'] = dfs['functional'].loc[~dfs['dev_functional'].source.isin(['pre2008']),:]

params = {'rotate':{'survival':-120,'functional':-120},
        'ext_title':{'survival':{'death_unit':'a. Survie \n (mois)',
             'ALSFRS_Total_final':'c. ALSFRS à 1 an\n (score)',
             'survived':'b. Survie à 1 an\n '},
                    'functional':{'ALSFRS_Total_final':'a. ALSFRS à 1 an\n (score)',
                                  'Kings_score_final':'b. Kings à 1 an\n (étape)',
                                  'MiToS_score_final':'c. MiToS à 1 an\n (étape)',
                                  'FT9_score_final':'d. FT9 à 1 an \n (étape)',},
                    },
        'output_feats':{'survival':['death_unit','survived','ALSFRS_Total_final'],
                        'functional':['ALSFRS_Total_final','Kings_score_final','MiToS_score_final','FT9_score_final']},
        'ext_subplot':{'survival':(2,3),'functional':(2,4)},
        'ext_subplot_size':{'survival':(8,8),'functional':(10.66,8)},
       
            }
def rotateData(x,angle):
    
    x_mod = x['x_axis']*math.cos(angle*math.pi/180)+x['y_axis']*math.sin(angle*math.pi/180)
    y_mod = -x['x_axis']*math.sin(angle*math.pi/180)+x['y_axis']*math.cos(angle*math.pi/180)
    
    return x_mod,y_mod


scopes_df = ['survival','functional'] #['functional'] #  #['functional'] #
writer = ExcelWriter('output/reboot/patient_scope_dev.xlsx')


for df_i in scopes_df:##
    
    
    print('####################### {} #######################'.format(df_i))
    
    if  df_i == 'survival':
        drop_feat = 'survived'
     
    for scope_i in ['dev','val']:
        try:
            umaps[scope_i+'_'+df_i].drop([drop_feat],axis=1,inplace=True)
        except:
            pass
        umaps[scope_i+'_'+df_i].rename(columns={'0':'x_axis','1':'y_axis'},inplace=True)
        angle_i = params['rotate'][df_i]
        umaps[scope_i+'_'+df_i]['tmp'] = umaps[scope_i+'_'+df_i].apply(lambda x: rotateData(x,angle_i),axis=1)
        umaps[scope_i+'_'+df_i].loc[:,'x_axis'] = umaps[scope_i+'_'+df_i].loc[:,'tmp'].apply(lambda x: x[0])
        umaps[scope_i+'_'+df_i].loc[:,'y_axis'] = umaps[scope_i+'_'+df_i].loc[:,'tmp'].apply(lambda x: x[1])
        umaps[scope_i+'_'+df_i].drop(['tmp'],axis=1,inplace=True)
        #umaps[scope_i+'_'+df_i].to_csv('output/reboot_FR/'+df_i+'_rotate_'+str(angle_i)+'.csv',sep=';')
        df_idx = umaps[scope_i+'_'+df_i].index
        if scope_i == 'dev':
            scaler = sk_p.MinMaxScaler()
            out = scaler.fit_transform(umaps[scope_i+'_'+df_i])
            #print('AAA3bis_min: {} - AAA3bis_scale:{}, {} len: {}'.format(scaler.min_,scaler.scale_,scope_i+'_'+df_i,umaps[scope_i+'_'+df_i].shape))
        else:
            out = scaler.transform(umaps[scope_i+'_'+df_i])
            

        umaps[scope_i+'_'+df_i] = pd.DataFrame(out,columns=['x_axis','y_axis'],index=df_idx)
        umaps[scope_i+'_'+df_i].loc[:,'x_axis'] = 1- umaps[scope_i+'_'+df_i].loc[:,'x_axis']
        #umaps[scope_i+'_'+df_i].to_csv('output/reboot_FR/'+scope_i+'_'+df_i+'_rotate_'+str(angle_i)+'norm.csv',sep=';')
        df = dfs[scope_i+'_'+df_i]
        try:
            df.set_index('subject_id',inplace=True)
        except:
            #print('df already indexed at subject_id?')
            pass
        umaps[scope_i+'_'+df_i] = pd.merge(umaps[scope_i+'_'+df_i],df,how='left',left_index=True,right_index=True)
    
        
    
        if df_i == 'functional':
            
            umaps[scope_i+'_'+df_i] = pd.merge(umaps[scope_i+'_'+df_i],
                                     functional_list.loc[:,[x+'_score_final' for x in ['Kings','MiToS','FT9']]+['computeZones_'+x+'_'+y for x in ['mouth','leg','trunk','respiratory','hand'] for y in ['baseline','final']]],
                                     how='left',
                                     left_index=True,
                                     right_index=True)
            
            #for stage_i in ['Kings','MiToS','FT9']:
            #     mask_patients_stage = ((~umaps[scope_i+'_'+df_i].death_unit.isnull())&(umaps[scope_i+'_'+df_i].death_unit<=12.1))
            #     umaps[scope_i+'_'+df_i].loc[mask_patients_stage,stage_i+'_score_final'] = 5

            
    ##########################################################################
    #                                 External
    ##########################################################################
    
    print('Start external plot')
    
    # No change in blue so 0 to 1, 0 in y0 to y1
    # light orange 
    # #fdaa48 rgb(253,170,72)
    
    ###" EXTERNAL STAT
                    
        
    cdict = {'red':   ((0.0, 140/255, 140/255),
                       (0.5,(140+4)/(2*255),(140+4)/(2*255)),
                       (1.0, 4/255, 4/255)),
             'blue':  ((0.0, 5/255, 5/255),
                       (0.5,(5+15)/(2*255),(5+15)/(2*255)),
                       (1.0, 15/255, 15/255)),
             'green': ((0.0, 0.0, 0.0),
                       (0.5,(74)/(2*255),(74)/(2*255)),
                       (1.0, 74/255, 74/255))}
    
    cdict = {'red':   ((0.0, 0/255,0/255 ),
                       (0.5,240/255,240/255),
                       (1.0,204/255,204/255 )),
        
             'blue':  ((0.0, 178/255, 178/255),
                       (0.5,66/255,66/255),
                       (1.0,167/255 ,167/255 )),
                       
             'green': ((0.0, 114/255,114/255 ),
                       (0.5,228/255,228/255),
                       (1.0,121/255 ,121/255 ))}
    
    cmap_red2green = mcolors.LinearSegmentedColormap(
            'my_colormap', cdict, 100)
    
    cmap_green2red = mcolors.LinearSegmentedColormap(
            'my_colormap2', cdict, 100)
    
    
    colors = [(230/255,159/255,0/255),(0/255,114/255,178/255),'y','m','g','k','c']
    
    feat_labels = {'survived':['Patient deceased','Patient survived'],
                   'source':['proact','exonhit','trophos','pre2008']}  
    slope = {}
    intercept = {}
    
    slope['survived'] = 1
    intercept['survived'] = 0
    for zone_i in ['hand','leg','trunk']:
        slope['computeZones_'+zone_i+'_last'] = 8
        intercept['computeZones_'+zone_i+'_last'] = 0
    for score_i in ['MiToS','FT9']:
        slope[score_i+'_score_last'] = 4
        intercept[score_i+'_score_last'] = 0
    slope['computeZones_mouth_last'] = 12
    intercept['computeZones_mouth_last'] = 0
    slope['computeZones_respiratory_last'] = 4
    intercept['computeZones_respiratory_last'] = 0
    
    
    slope['Kings_score_last'] = 3.5
    slope['ALSFRS_Total_final'] = 40
    slope['death_unit'] = 13 #66.79 # based on df_with_last.csv (patient_db)
    slope['source'] = 1
    intercept['Kings_score_last'] = 1 
    intercept['ALSFRS_Total_final'] = 0
    intercept['death_unit'] = 0
    intercept['source'] = 0
    
    fig, axes = plt.subplots(params['ext_subplot'][df_i][0],
                             params['ext_subplot'][df_i][1],
                             sharex=True,
                             sharey=True,
                             figsize=(params['ext_subplot_size'][df_i][0],
                                      params['ext_subplot_size'][df_i][1]))
    #fig = plt.figure(figsize=(6,8))
    #plt.axis('off')
    
    idx = 0
    '''
    idx_plots = {0:[(0,0),(1,0)],
                 1:[(0,1),(1,1)],
                 2:[(0,2),(1,2)],
                 3:[(0,3),(1,3)]
                 }
    '''
    idx_plots = {0:(0,0),
                 1:(1,0),
                 2:(0,1),
                 3:(1,1),
                 4:(0,2),
                 5:(1,2),
                 6:(0,3),
                 7:(1,3),
                 }
    
    name_plots = {0:'a',
                  1:'a',
                  2:'b',
                  3:'b',
                  4:'c',
                  5:'c',
                  6:'d',
                  7:'d'}
    
    
    pos_plots = {0:0.4,
                 1:0.4,
                 2:0.4,
                 3:0.4,
                 }
    
      
    params['ext_title'] = {'survival': {'death_unit': 'overall survival (month)',
                          'ALSFRS_Total_final': '1-year ALSFRS (score)',
                          'survived': '1-year survival '},
                         'functional': {'ALSFRS_Total_final': '1-year ALSFRS (score)',
                          'Kings_score_final': '1-year Kings (stage)',
                          'MiToS_score_final': '1-year MiToS (stage)',
                          'FT9_score_final': '1-year FT9 (stage)'}}
    max_val_dict = {}
    #axarr[0,0].xaxis.set_visible(False) 
    
    dict_axis = {0:(0,1),
                 1:(1,1),
                 2:(0,0),
                 3:(1,0),
                 4:(0,0),
                 5:(1,0),
                 6:(0,0),
                 7:(1,0)
                 }
    
    for feat_i in params['output_feats'][df_i]: 
        
        for scope_i in ['dev','val']:
            # Categorical
            
            
            if len(umaps[scope_i+'_'+df_i].loc[:,feat_i].unique())<=4:
                color_idx = 0
                if scope_i == 'dev' and df_i =='survival':
                    df_dev = umaps[scope_i+'_'+df_i].loc[patients_kept,['x_axis','y_axis']+[feat_i]]
                else:
                    df_dev = umaps[scope_i+'_'+df_i].loc[:,['x_axis','y_axis']+[feat_i]]
                df_dev.dropna(inplace=True)
                
                print('df: {} - scope: {} - feat: {} => patients: {}'.format(df_i,scope_i,feat_i,df_dev.shape[0]))
                
                #print('patient kept feat_i: {} : {}'.format(feat_i,df_dev.shape[0]))
                ax = axes[idx_plots[idx][0],idx_plots[idx][1]]#fig.add_subplot(idx)
                #ax.set_title(params['ext_title'][df_i][feat_i],position=(1.3,1.05))
                if idx_plots[idx][0] == 0:
                    sub_text = '{}1. {} \n development data '.format(name_plots[idx],params['ext_title'][df_i][feat_i])
                else:
                    sub_text = '{}2. {} \n validation data'.format(name_plots[idx],params['ext_title'][df_i][feat_i])
                ax.set_title(sub_text,fontsize='small')
                ax.set_xlim([0,1])
                ax.set_ylim([0,1])
                ss = {1:0.2,0:1}    
        
                
                for elem_i in sorted(list(df_dev.loc[:,feat_i].unique())):
                    ax.scatter(df_dev.loc[df_dev[feat_i]==elem_i,'x_axis'],
                                df_dev.loc[df_dev[feat_i]==elem_i,'y_axis'],
                                c=[colors[color_idx]],
                                alpha=ss[elem_i],
                                s=1,label=feat_labels[feat_i][color_idx])
                    color_idx += 1
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False) 
                #ax.spines['left'].set_visible(False)
                #ax.spines['bottom'].set_visible(False)
                #ax.set_xticks([])
                #ax.set_yticks([])
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='10%', pad=0.15)
                cax.set_xticks([])
                cax.set_yticks([])
                cax.spines['right'].set_visible(False)
                cax.spines['top'].set_visible(False) 
                cax.spines['left'].set_visible(False)
                cax.spines['bottom'].set_visible(False)
                            
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='10%', pad=0.15)
                cax.set_xticks([])
                cax.set_yticks([])
                cax.spines['right'].set_visible(False)
                cax.spines['top'].set_visible(False) 
                cax.spines['left'].set_visible(False)
                cax.spines['bottom'].set_visible(False)
                    #plt.axis('off')
                lgnd = ax.legend(loc='lower left',
                                 bbox_to_anchor=(-0.0,-0.25),
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
                if dict_axis[idx][0] == 0:
                    ax.spines['bottom'].set_visible(False)
                    #ax.set_xticks([])
                if dict_axis[idx][1] == 0:
                    ax.spines['left'].set_visible(False)
                    #ax.set_yticks([])
                
            else:
                
                umaps[scope_i+'_'+df_i].loc[:,['x_axis','y_axis']+[feat_i]]
                
                if scope_i == 'dev' and df_i =='survival':
                
                    max_feat = umaps[scope_i+'_'+df_i][feat_i].max() 
                    Y_tmp_norm = umaps[scope_i+'_'+df_i][feat_i].apply(lambda x: x/max_feat)
                    
                    df_patients_kept = pd.DataFrame([],index=patients_kept,columns=['x'])
                    df_dev = pd.merge(umaps[scope_i+'_'+df_i].loc[:,['x_axis','y_axis']],df_patients_kept,how='right',left_index=True,right_index=True)
                    df_dev = pd.merge(df_dev,Y_tmp_norm,how='left',left_index=True,right_index=True)
                    df_dev.drop(['x'],axis=1,inplace=True)
                    df_dev.dropna(inplace=True)
                    max_val_dict[feat_i] = max_feat
                    df_dev.to_excel(writer,feat_i)
                    if feat_i in ['{}_score_final'.format(x) for x in ['Kings','MiToS','FT9']]:
                        
                        #df_tmp = df_dev.copy(deep=True)
                        df_dev.reset_index(inplace=True)
                        df_dev['source'] = df_dev.loc[:,'index'].apply(lambda x: x.split('_')[0])
                        df_dev = df_dev.loc[~df_dev.source.isin(['trophos']),:]
                        df_dev.drop(['source'],axis=1,inplace=True)
                        df_dev.set_index('index',inplace=True)
                        #print(df_tmp.loc[:,['source','index']].groupby('source').count())
                    
                    print('df: {} - scope: {} - feat: {} => patients: {}'.format(df_i,scope_i,feat_i,df_dev.shape[0]))

                else:
                    if scope_i =='dev':
                        max_val_dict[feat_i] = umaps[scope_i+'_'+df_i][feat_i].max() 
                    Y_tmp_val_norm = umaps[scope_i+'_'+df_i][feat_i].apply(lambda x: x/max_val_dict[feat_i])
                    df_dev = pd.merge(umaps[scope_i+'_'+df_i].loc[:,['x_axis','y_axis']],Y_tmp_val_norm,how='left',left_index=True,right_index=True)
                    df_dev.dropna(inplace=True)
                    print('df: {} - scope: {} - feat: {} => patients: {}'.format(df_i,scope_i,feat_i,df_dev.shape[0]))
                #print('patient kept feat_i: {} : {}'.format(feat_i,df_dev.shape[0]))
                if feat_i =='ALSFRS_Total_final':
                    #df_dev.to_csv('output/reboot/survival_check_patients_kept_ext_functional.csv',sep=';')
                    df_tmp = df_dev.copy(deep=True)
                    df_tmp.reset_index(inplace=True)
                    #df_tmp['source'] = df_tmp.loc[:,'subject_id'].apply(lambda x: x.split('_')[0])
                    #print(df_tmp.loc[:,['source','subject_id']].groupby('source').count())
                if feat_i =='Kings_score_final':
                    #df_dev.to_csv('output/reboot/functional_check_patients_kept_ext_functional_kings.csv',sep=';')
                    df_tmp = df_dev.copy(deep=True)
                    df_tmp.reset_index(inplace=True)
                    #df_tmp['source'] = df_tmp.loc[:,'subject_id'].apply(lambda x: x.split('_')[0])
                    #print(df_tmp.loc[:,['source','subject_id']].groupby('source').count())
                
                #ax = fig.add_subplot(idx)
                ax_tmp1 = axes[idx_plots[idx][0],idx_plots[idx][1]]#fig.add_subplot(idx)
                #ax.set_title(params['ext_title'][df_i][feat_i],position=(1.3,1.05))
                if idx_plots[idx][0] == 0:
                    sub_text = '{}1. {} \n development data'.format(name_plots[idx],params['ext_title'][df_i][feat_i])
                else:
                    sub_text = '{}2. {} \n validation data'.format(name_plots[idx],params['ext_title'][df_i][feat_i])
                ax_tmp1.set_title(sub_text,fontsize='small')
                
                
                ax_tmp1.spines['right'].set_visible(False)
                ax_tmp1.spines['top'].set_visible(False) 
                if dict_axis[idx][0] == 0:
                    ax_tmp1.spines['bottom'].set_visible(False)
                    #ax_tmp1.set_xticks([])
                if dict_axis[idx][1] == 0:
                    ax_tmp1.spines['left'].set_visible(False)
                    #ax_tmp1.set_yticks([])
                #ax_tmp1.spines['left'].set_visible(False)
                #ax_tmp1.spines['bottom'].set_visible(False)
                #ax_tmp1.set_xticks([])
                #ax_tmp1.set_yticks([])
                divider = make_axes_locatable(ax_tmp1)
                cax = divider.append_axes('right', size='10%', pad=0.15)
                cax.set_xticks([])
                cax.set_yticks([])
                cax.spines['right'].set_visible(False)
                cax.spines['top'].set_visible(False) 
                cax.spines['left'].set_visible(False)
                cax.spines['bottom'].set_visible(False)
                
                if feat_i in ['death_unit','ALSFRS_Total_last']:
                        cmap_tmp = cmap_red2green
                else:
                        cmap_tmp = cmap_green2red
                    
                
                
                if feat_i == 'ALSFRS_Total_final':
                    vmin_i=0
                elif feat_i == 'death_unit':
                    vmin_i=0
                elif feat_i == 'MiToS_score_final' or feat_i == 'FT9_score_final':
                    vmin_i = 0
                else:
                    vmin_i = 0
                #print('scatter')
                points = ax_tmp1.scatter(df_dev['x_axis'],df_dev['y_axis'],c=df_dev[feat_i],cmap = cmap_tmp,s=1,vmin=vmin_i,vmax=1)
                
                #print('end scatter')
                
                
                
                divider = make_axes_locatable(ax_tmp1)
                cax = divider.append_axes('right', size='10%', pad=0.15)
                
                #plt.axis('off')
                
                if feat_i == 'death_unit':
                    lin_size = 6
                elif feat_i == 'survived':
                    lin_size = 2
                elif feat_i == 'source':
                    lin_size = 4
                elif feat_i == 'ALSFRS_Total_final':
                    lin_size = 5
                else:
                    lin_size = 6
                    
                # ax=axes[idx_plots[idx][0][0]:idx_plots[idx][0][0]+1,idx_plots[idx][0][1]:idx_plots[idx][0][1]+1]
                v1 = np.linspace(0,1,lin_size, endpoint=True)
                
                cb = fig.colorbar(points,cax=cax,ticks=v1,orientation='vertical') #
                '''
                if feat_i == 'survived':
                    cb.ax.set_yticklabels(['survived','dead'])
                elif feat_i in ['ALSFRS_Total_final','MiToS_score_final','FT9_score_final']:
                    cb.ax.set_yticklabels(['{:3.0f}'.format(i*slope[feat_i]+intercept[feat_i]) for i in v1])
                elif feat_i in ['death_unit']:
                    list_cb = ['{:3.0f}'.format(i*slope[feat_i]+intercept[feat_i]) for i in v1]
                    list_cb.pop(-1)
                    list_cb.append('13+')
                    cb.ax.set_yticklabels(list_cb)
                else:
                    cb.ax.set_yticklabels(['1','2','3','4','4.5'])
                '''
                #ax.set_xlim([0,1])
                #ax.set_ylim([0,1])
                if feat_i == 'survived':
                    cb.ax.set_yticklabels(['survived','dead'])
                elif feat_i == 'ALSFRS_Total_final':
                    cb.ax.set_yticklabels(['0','10','20','30','40'])
                elif feat_i == 'death_unit':
                    cb.ax.set_yticklabels(['0','3','5','8','10','13+'])
                elif feat_i == 'MiToS_score_final' or feat_i == 'FT9_score_final':
                    print('idx: {} - feat_i: {}'.format(idx,feat_i))
                    cb.ax.set_yticklabels(['0','1','2','3','4','5'])
                else:
                    cb.ax.set_yticklabels(['1','2','3','4','4.5','5'])
            
            
                
            idx += 1
        
    #axes[idx_plots[2][1][0]][idx_plots[2][1][1]].set_xlabel('Dim n1 (UMAP)',ha='center')
    #axes[idx_plots[2][1][0]][idx_plots[2][1][1]].xaxis.set_label_coords(-0.23, -0.10)
    #fig.text(0.5, 0.02, 'Dim n1 (UMAP)', ha='center')
    #fig.text(0.04, 0.5, 'Dim n2 (UMAP)', va='center', rotation='vertical')
    if df_i == 'survival':
        hspace_i = 0.40
    else:
        hspace_i = 0.3
    plt.subplots_adjust(wspace=0.25,hspace=hspace_i)
    if df_i == 'survival':
        fig.text(0.5, 0.03, 'Dim n1 (UMAP)', ha='center')
        fig.text(.06, 0.5, 'Dim n2 (UMAP)', va='center', rotation='vertical')
    else:
        fig.text(0.5, 0.05, 'Dim n1 (UMAP)', ha='center')
        fig.text(.06, 0.5, 'Dim n2 (UMAP)', va='center', rotation='vertical')
    
    fig.savefig('output/reboot/article/'+df_i+'/pdf/'+df_i+'_ext.pdf', bbox_inches='tight')
    fig.savefig('output/reboot/article/'+df_i+'/png/'+df_i+'_ext.png', bbox_inches='tight',bpi=1500)    
    
    print('End external plot')    
    
writer.save()

##############################################################################
#                               ML MODEL COMPARISON
##############################################################################
    
import sklearn.ensemble as sk_e
import sklearn.linear_model as sk_lm
import sklearn.metrics as sk_m
import sklearn.preprocessing as sk_p

results_sur = []
results_func = []

for df_i in scopes_df:
    
    models = {}
    models['lr_2'] = sk_lm.LogisticRegression(max_iter=10000)
    models['lr'] = sk_lm.LogisticRegression(max_iter=10000)
    models['rf_2'] = sk_e.RandomForestClassifier(n_estimators=10)
    models['rf'] = sk_e.RandomForestClassifier(n_estimators=10)
    
    feats_trunc = ['age','ALSFRS_Total_baseline']
    feats_remain = ['sex','onset_spinal','time_disease',
                    'ALSFRS_Total_estim_slope','weight_baseline']
    
    scaler = sk_p.MinMaxScaler()
    norm_umaps = pd.DataFrame(scaler.fit_transform(umaps['dev_'+df_i].loc[:,feats_trunc+feats_remain]),columns=[feats_trunc+feats_remain])
    norm_umaps_val = pd.DataFrame(scaler.transform(umaps['val_'+df_i].loc[:,feats_trunc+feats_remain]),columns=[feats_trunc+feats_remain])
    print('norm_umaps.shape: {}'.format(norm_umaps.shape))
    # Normalization in this fashion
    # Creates potential values higher than 1 
    # Not that significant on model as close to one though             
                  
    X_train_trunc = norm_umaps.loc[:,feats_trunc]
    X_train = norm_umaps.loc[:,feats_trunc+feats_remain]
    
    X_test_trunc = norm_umaps_val.loc[:,feats_trunc]
    X_test = norm_umaps_val.loc[:,feats_trunc+feats_remain]
    
    if df_i == 'survival':         
        Y_train = umaps['dev_'+df_i].loc[:,'survived']
        Y_test = umaps['val_'+df_i].loc[:,'survived']
    else:
        
        
        def catFinalALSFRS(x):
            if x <= 10:
                return '10less'
            elif x>10 and x<=20:
                return '10to20'
            elif x>20 and x<=30:
                return '20to30'
            else:
                return '30more'
        
        umaps['dev_'+df_i]['ALSFRS_Total_final_cat'] = umaps['dev_'+df_i].loc[:,'ALSFRS_Total_final'].apply(lambda x: catFinalALSFRS(x))
        umaps['val_'+df_i]['ALSFRS_Total_final_cat'] = umaps['val_'+df_i].loc[:,'ALSFRS_Total_final'].apply(lambda x: catFinalALSFRS(x))
        
        scaler2 = sk_p.MinMaxScaler()
        
        #Y_train_reg = pd.DataFrame(scaler2.fit_transform(umaps['dev_'+df_i].loc[:,'ALSFRS_Total_final'].reshape(-1,1)),columns=['ALSFRS_Total_final'])
        #Y_test_reg = pd.DataFrame(scaler2.transform(umaps['val_'+df_i].loc[:,'ALSFRS_Total_final'].reshape(-1,1)),columns=['ALSFRS_Total_final'])
        
        Y_train = umaps['dev_'+df_i].loc[:,'ALSFRS_Total_final_cat']
        Y_test = umaps['val_'+df_i].loc[:,'ALSFRS_Total_final_cat']
        
        #models['reg_lr_2'] = sk_lm.Ridge()
        #models['reg_lr'] = sk_lm.Ridge()
        #models['reg_rf_2'] = sk_e.RandomForestRegressor(n_estimators=10)
        #models['reg_rf'] = sk_e.RandomForestRegressor(n_estimators=10)
        
    for model_i in models.keys():
        if model_i.find('2') == -1:
            X_train_tmp = X_train
            X_test_tmp = X_test
            tag_trunc = 'all'
        else:
            X_train_tmp = X_train_trunc
            X_test_tmp = X_test_trunc
            tag_trunc = 'trunc'
        '''
        if model_i.split('_')[0] == 'reg':
            models[model_i].fit(X_tmp,Y_train_reg)
            Y_pred = models[model_i].predict(X_tmp_test)
        '''
        
        print('df_i: {} model:{} => train: {} - test: {}'.format(df_i,model_i,X_train_tmp.shape[0],X_test_tmp.shape[0]))
        
        models[model_i].fit(X_train_tmp,Y_train)
        Y_pred = models[model_i].predict(X_test_tmp)
        
        Y_pred = pd.DataFrame(Y_pred,columns=['predicted'],index=Y_test.index)
        
        Y_comb = pd.merge(Y_test,Y_pred,left_index=True,right_index=True,how='left')
        Y_rename = {}
        for col_i in Y_comb.columns:
            Y_rename[col_i] = col_i+'_'+model_i
        Y_comb.rename(columns=Y_rename,inplace=True)
        
        if df_i == 'survival':
            confuse_matrix = sk_m.confusion_matrix(Y_test, Y_pred,labels=[0,1])
            results_sur.append(['survival',model_i,tag_trunc,
                                confuse_matrix[0][0],
                                confuse_matrix[0][1],
                                confuse_matrix[1][0],
                                confuse_matrix[1][1]])
        else:
            confuse_matrix = sk_m.confusion_matrix(Y_test, Y_pred,labels=['10less','10to20','20to30','30more'])
            results_func.append(['functional',model_i,tag_trunc,
                                confuse_matrix[0][0],
                                confuse_matrix[0][1],
                                confuse_matrix[0][2],
                                confuse_matrix[0][3],
                                confuse_matrix[1][0],
                                confuse_matrix[1][1],
                                confuse_matrix[1][2],
                                confuse_matrix[1][3],
                                confuse_matrix[2][0],
                                confuse_matrix[2][1],
                                confuse_matrix[2][2],
                                confuse_matrix[2][3],
                                confuse_matrix[3][0],
                                confuse_matrix[3][1],
                                confuse_matrix[3][2],
                                confuse_matrix[3][3]])
            
        print(Y_test)
            
        try:
            tmp = pd.merge(tmp,Y_comb,left_index=True,right_index=True,how='left')
        except:
            tmp = Y_comb
            
        def findZone(x):
            #y1 = 0.4+0.2*x
            #y2 = -0.1 + 0.5*x
            if (x['y_axis']-(x['x_axis']*0.2+0.4))>0:
                return 'high'
            elif (x['y_axis']-(x['x_axis']*0.2+0.4))<0 and (x['y_axis']-(x['x_axis']*0.5-0.1))>0:
                return 'mid'
            else:
                return 'low'
            
        out = pd.merge(Y_test,Y_pred,how='left',left_index=True,right_index=True)
        out = pd.merge(out,umaps['val_functional'].loc[:,['x_axis','y_axis']],how='left',left_index=True,right_index=True)
        out.rename(columns={'ALSFRS_Total_final_cat':'label_true','predicted':'label_predicted'},inplace=True)
        out['zone'] = out.apply(lambda x: findZone(x),axis=1)
        out.to_csv('output/reboot/article/functional/ML_outputs.csv',sep=';')
        
        #print(confuse_matrix)
        if df_i == 'functional':
            tmp = pd.merge(tmp,umaps['val_functional'].loc[:,['x_axis','y_axis']],how='left',left_index=True,right_index=True)
            tmp.rename(columns={'ALSFRS_Total_final_cat_lr_2':'ALSFRS_Total_final_cat'},inplace=True)
            tmp.drop(['ALSFRS_Total_final_{}'.format(x) for x in ['lr','rf','rf_2']],axis=1,inplace=True)
        tmp.to_csv('output/reboot/article/model_output_{}.csv'.format(df_i),sep=';')
    
    
confusion_survival = pd.DataFrame(results_sur,columns=['df','model','trunc','tp','fp','fn','tn'])

def compAcc(x):
    return (x['tp']+x['tn'])/(x['tp']+x['fp']+x['fn']+x['tn'])
def compPre(x):
    return x['tp']/(x['tp']+x['fp'])
def compSpe(x):
    return x['tn']/(x['tn']+x['fp'])
def compRec(x):
    return x['tp']/(x['tp']+x['fn'])
def compBalAcc(x):
    return (x['rec']+x['spe'])/2
def compF1(x):
    return 2*(x['rec']*x['pre'])/(x['rec']+x['pre'])

confusion_survival['acc'] = confusion_survival.apply(lambda x: compAcc(x),axis=1)
confusion_survival['pre'] = confusion_survival.apply(lambda x: compPre(x),axis=1)
confusion_survival['spe'] = confusion_survival.apply(lambda x: compSpe(x),axis=1)
confusion_survival['rec'] = confusion_survival.apply(lambda x: compRec(x),axis=1)
confusion_survival['bal_acc'] = confusion_survival.apply(lambda x: compBalAcc(x),axis=1)
confusion_survival['F1'] = confusion_survival.apply(lambda x: compF1(x),axis=1)


confusion_functional = pd.DataFrame(results_func,columns=['df','model','trunc','00','01',
                                                         '02','03','10','11','12',
                                                         '13','20','21','22','23',
                                                         '30','31','32','33'])

def McompAcc(x):
    return (x['00']+x['11']+x['22']+x['33'])/(x['00']+x['01']+x['02']+x['03']+x['10']+x['11']+x['12']+x['13']+x['20']+x['21']+x['22']+x['23']+x['30']+x['31']+x['32']+x['33'])

######### FINISH IMPLEMENTATION OF MULTICLASS PRE, REC, BAL ACC AND SPE (F1)
# https://towardsdatascience.com/multi-class-metrics-made-simple-part-i-precision-and-recall-9250280bddc2

def McompPre(x,i):
    return x[str(i)+str(i)]/(x[str(i)+'0']+x[str(i)+'1']+x[str(i)+'2']+x[str(i)+'3'])
def McompRec(x,i):
    try:
        return x[str(i)+str(i)]/(x['0'+str(i)]+x['1'+str(i)]+x['2'+str(i)]+x['3'+str(i)])
    except:
        return 'NA'
def McompSpe(x,i):
    list_elems= [0,1,2,3]
    list_elems.pop(i)
    num = 0
    # if for cat 0
    # loop to get tn == 11,12,13,21,22,23,31,32,33
    
    for elem_i in list_elems:
        for elem_j in list_elems:
            num += x[str(elem_i)+str(elem_j)]
    denom = 0
    for elem_i in list_elems:
        denom += x[str(elem_i)+'0']+x[str(elem_i)+'1']+x[str(elem_i)+'2']+x[str(elem_i)+'3']
    # predicted negative
    return num/denom

confusion_functional['acc'] = confusion_functional.apply(lambda x: McompAcc(x),axis=1)
for i in range(0,4):
    confusion_functional['pre_'+str(i)] = confusion_functional.apply(lambda x: McompPre(x,i),axis=1)
    confusion_functional['rec_'+str(i)] = confusion_functional.apply(lambda x: McompRec(x,i),axis=1)
    confusion_functional['spe_'+str(i)] = confusion_functional.apply(lambda x: McompSpe(x,i),axis=1)
    
writer = ExcelWriter('output/reboot/article/performance_external_both_articles.xlsx')
confusion_survival.to_excel(writer,'survival',index=False)
confusion_functional.to_excel(writer,'functional',index=False)

writer.save()
        
###################### 2D 2-sample Kolmogorov Smirnov test

p = ndtest.ks2d2s(umaps['dev_functional'].loc[:,'x_axis'].values,
                     umaps['dev_functional'].loc[:,'y_axis'].values,
                     umaps['val_functional'].loc[:,'x_axis'].values,
                     umaps['val_functional'].loc[:,'y_axis'].values)


p_y  = sc_s.ks_2samp(umaps['dev_functional'].loc[:,'y_axis'].values,
            umaps['val_functional'].loc[:,'y_axis'].values)

p_x = sc_s.ks_2samp(umaps['dev_functional'].loc[:,'x_axis'].values,
            umaps['val_functional'].loc[:,'x_axis'].values)
#print('2D: {}, x_axis: {}, y_axis: {}'.format(p,p_y.pvalue,p_x.pvalue))


###################################################################

# Plot distributions


class SeabornFig2Grid():

    def __init__(self, seaborngrid, fig,  subplot_spec,title,xlabel=False,ylabel=False,xoffset=False,yoffset=False,offset_x=0,offset_y=0):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        self.title=title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xoffset = xoffset
        self.yoffset = yoffset
        self.offset_x = offset_x
        self.offset_y  =offset_y
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        #self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1],xlabel=self.xlabel,
                                                               ylabel=self.ylabel,
                                                               xoffset=self.xoffset,
                                                               offset_x=self.offset_x,
                                                               yoffset=self.yoffset,
                                                               offset_y=self.offset_y)
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1],xlabel=False,
                                                               ylabel=False,
                                                               xoffset=False,
                                                               offset_x=0,
                                                               yoffset=False,
                                                               offset_y=0,
                                                               joint=True)
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1],xlabel=False,
                                                               ylabel=False,
                                                               xoffset=False,
                                                               offset_x=0,
                                                               yoffset=False,
                                                               offset_y=0)
        #self.sg.ax_joint.set_title(self.title)

    def _moveaxes(self, ax, gs,xlabel,ylabel,xoffset,offset_x,yoffset,offset_y,joint=False):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)
        if xlabel==True:
            ax.set_xlabel('Dim n1 (UMAP)')#,x=2,y=-0.025) Created weird space between plots
            if xoffset==True:
                ax.yaxis.set_label_coords(1+offset_x, -0.025)
                #print('xlabel:{}xoffset:{}'.format(xlabel,xoffset))
        if ylabel==True:
            ax.set_ylabel('Dim n2 (UMAP)')
            if xoffset==True:
                ax.xaxis.set_label_coords(1.5+offset_y, -0.025)
                
        if joint==True:
            ax.set_title(self.title)
        
    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())



col_blind = [(230/255,159/255,0/255),(0/255,114/255,178/255)]

def multivariateGrid(col_x,
                     col_y, 
                     col_k, 
                     df, 
                     bins_in,
                     k_is_color=False,
                     scatter_alpha=.5,
                     title='',
                     keep_xticks=True,
                     keep_yticks=True,
                     keep_x=False,
                     keep_y=False):
    def colored_scatter(x, y, c=None):
        def scatter(*args, **kwargs):
            args = (x, y)
            if c is not None:
                kwargs['c'] = c
            kwargs['alpha'] = scatter_alpha
            plt.scatter(*args, **kwargs)

        return scatter

    g = sns.JointGrid(
        x=col_x,
        y=col_y,
        data=df,
        xlim=(-0.1,1.1),
        ylim=(-0.1,1.1),
        ratio=3
    )
    
    g.fig.set_figwidth(2)
    g.fig.set_figheight(4)
    
    #g.fig.suptitle(title)
    color = None
    legends=[]
    for name, df_group in df.groupby(col_k):
        #print(name)
        legends.append(name)
        if k_is_color:
            color=name
        g.plot_joint(
            colored_scatter(df_group[col_x],df_group[col_y],color),
        )
        sns.distplot(
            df_group[col_x].values,
            ax=g.ax_marg_x,
            color=color,
            bins=bins_in[0]
        )
        sns.distplot(
            df_group[col_y].values,
            ax=g.ax_marg_y,
            color=color,            
            vertical=True,
            bins=bins_in[1]
        )
    '''
    # Do also global Hist:
    sns.distplot(
        df[col_x].values,
        ax=g.ax_marg_x,
        color='grey',
        bins=bins_in[0]
    )
    sns.distplot(
        df[col_y].values.ravel(),
        ax=g.ax_marg_y,
        color='grey',
        vertical=True,
        bins=bins_in[1]
    )
    '''
    #plt.legend(legends)
    g.ax_joint.set_xlabel('')
    g.ax_joint.set_ylabel('')
    if keep_xticks==True:
        g.ax_joint.set_xticks([0,0.5,1])
        
    else:
        #g.ax_joint.set_xticks([])
        g.ax_joint.set_xticks([0,0.5,1])
        g.ax_joint.set_xticklabels([])
        #g.ax_joint.xaxis.set_major_formatter(NullFormatter())
        
    if keep_yticks==True:
        g.ax_joint.set_yticks([0,0.5,1])
    else:
        #g.ax_joint.set_yticks([])
        g.ax_joint.set_yticks([0,0.5,1])
        g.ax_joint.set_yticklabels([])
        #g.ax_joint.yaxis.set_major_formatter(NullFormatter())
        
    if keep_x == 0:
        g.ax_joint.spines['bottom'].set_visible(False)
    if keep_y == 0:
        g.ax_joint.spines['left'].set_visible(False)

    g.ax_marg_y.spines['left'].set_visible(False)

    g.ax_marg_x.spines['bottom'].set_visible(False)

    return g 



# 2 distributions on same joint plot
# https://stackoverflow.com/questions/31539815/plotting-two-distributions-in-seaborn-jointplot
# https://stackoverflow.com/questions/35920885/how-to-overlay-a-seaborn-jointplot-with-a-marginal-distribution-histogram-fr
# Make jointplot plot rectangular
# https://stackoverflow.com/questions/29909515/how-to-plot-non-square-seaborn-jointplot-or-jointgrid
# Multiple plot of jointplots
# https://stackoverflow.com/questions/35042255/how-to-plot-multiple-seaborn-jointplot-in-subplot

'''
tmp_x = np.asarray(umaps['val_survival'].loc[:,'x_axis'])
tmp_y = np.asarray(umaps['val_survival'].loc[:,'y_axis'])

q1_x = sc_s.scoreatpercentile(tmp_x, 25)
q3_x = sc_s.scoreatpercentile(tmp_x, 75)
q1_y = sc_s.scoreatpercentile(tmp_y, 25)
q3_y = sc_s.scoreatpercentile(tmp_y, 75)

iqr_x = q3_x-q1_x
iqr_y = q3_y-q1_y

len_tmp_x = len(tmp_x)
len_tmp_y = len(tmp_y)

h_y = 2* iqr_y/(len_tmp_y**(1/3))
h_x = 2* iqr_x/(len_tmp_x**(1/3))

nb_x = int(np.ceil((tmp_x.max() - tmp_x.min()) / h_x))
nb_y = int(np.ceil((tmp_y.max() - tmp_y.min()) / h_y))

nb_bins = min(nb_x,nb_y)
'''

def defineBinValue(df):
    q1_x = sc_s.scoreatpercentile(df.loc[:,'x_axis'], 25)
    q3_x = sc_s.scoreatpercentile(df.loc[:,'x_axis'], 75)
    q1_y = sc_s.scoreatpercentile(df.loc[:,'y_axis'], 25)
    q3_y = sc_s.scoreatpercentile(df.loc[:,'y_axis'], 75)
    
    iqr_x = q3_x-q1_x
    iqr_y = q3_y-q1_y
    
    len_tmp_x = len(df.loc[:,'x_axis'].values)
    len_tmp_y = len(df.loc[:,'y_axis'].values)
    
    h_y = 2* iqr_y/(len_tmp_y**(1/3))
    h_x = 2* iqr_x/(len_tmp_x**(1/3))
    
    nb_x = int(np.ceil((df.loc[:,'x_axis'].max() - df.loc[:,'x_axis'].min()) / h_x))
    nb_y = int(np.ceil((df.loc[:,'y_axis'].max() - df.loc[:,'y_axis'].min()) / h_y))
    
    bin_x = [0+x*1/nb_x for x in range(0,nb_x+1)]
    bin_y = [0+x*1/nb_y for x in range(0,nb_y+1)]
    
    return bin_x,bin_y


### SEABORN CONFIGURATION FOR GREY BACKGROUND REMOVAL    
## REMOVE SPINES FOR CENTRAL GRIDS (also previous plots)
#https://seaborn.pydata.org/tutorial/aesthetics.html

umaps['dev_functional']['kind'] = 'development'
umaps['val_functional']['kind'] = 'validation'
umaps['dev_survival']['kind'] = 'development'
umaps['val_survival']['kind'] = 'validation'

df_patients_kept = pd.DataFrame([],index=patients_kept,columns=['x'])

umap_dev_func = umaps['dev_functional'].loc[:,['x_axis','y_axis','ALSFRS_Total_final','kind']]
umap_dev_func = pd.merge(umap_dev_func,df_patients_kept,how='right',left_index=True,right_index=True)
umap_dev_func.drop(['x'],axis=1,inplace=True)
umap_dev_func.dropna(inplace=True)

umap_dev_surv = umaps['dev_survival'].loc[:,['x_axis','y_axis','survived','kind']]
umap_dev_surv = pd.merge(umap_dev_surv,df_patients_kept,how='right',left_index=True,right_index=True)
umap_dev_surv.drop(['x'],axis=1,inplace=True)
umap_dev_surv.dropna(inplace=True)

umap_dev_stages = umaps['dev_functional'].loc[:,['x_axis','y_axis','Kings_score_final','FT9_score_final','MiToS_score_final','kind']]
umap_dev_stages = pd.merge(umap_dev_stages,df_patients_kept,how='right',left_index=True,right_index=True)
umap_dev_stages.drop(['x'],axis=1,inplace=True)
umap_dev_stages.reset_index(inplace=True)
umap_dev_stages['source'] = umap_dev_stages.loc[:,'index'].apply(lambda x: x.split('_')[0])  
umap_dev_stages = umap_dev_stages.loc[umap_dev_stages.source!='trophos',:]
umap_dev_stages.set_index('index',inplace=True)
umap_dev_stages.drop(['source'],axis=1,inplace=True)
umap_dev_stages.dropna(inplace=True)

umap_dev_death = umaps['dev_survival'].loc[:,['x_axis','y_axis','death_unit','kind']]
umap_dev_death = pd.merge(umap_dev_death,df_patients_kept,how='right',left_index=True,right_index=True)
umap_dev_death.drop(['x'],axis=1,inplace=True)
umap_dev_death.dropna(inplace=True)

test_joint_functional = pd.concat([umap_dev_func,umaps['val_functional']])
test_joint_survival = pd.concat([umap_dev_surv,umaps['val_survival']])
test_joint_stages = pd.concat([umap_dev_stages,umaps['val_functional']])
test_joint_death = pd.concat([umap_dev_death,umaps['val_survival'].dropna(subset=['death_unit'])])

bin_x_stand = [0+x*1/6 for x in range(0,6+1)]
bin_y_stand = [0+x*1/10 for x in range(0,10+1)]

graph_main = multivariateGrid('x_axis', 'y_axis', 'kind',df=test_joint_survival,
                 bins_in=(bin_x_stand,bin_y_stand),
                 keep_yticks=True,
                 keep_xticks=False,
                 keep_y = True)
#plt.savefig('output/reboot/sp_diff_input.pdf')
plt.close()
# 1-year survival
graph_survival = {}



val_a = test_joint_survival.loc[test_joint_survival.survived==0,:].shape[0]
val_b = test_joint_survival.loc[test_joint_survival.survived==1,:].shape[0]
print('deceased: {} - survived: {} => {}'.format(val_a,val_b,val_a+val_b))

graph_survival['dead'] = multivariateGrid('x_axis',
                                          'y_axis', 
                                          'kind', 
                                          df=test_joint_survival.loc[test_joint_survival.survived==0,:],
                                          bins_in=(bin_x_stand,bin_y_stand),#defineBinValue(umaps['val_functional'].loc[umaps['val_functional'].survived==0,:]),
                                          title='Deceased within 1 year',
                                          keep_yticks=False,
                                          keep_xticks=False,
                                          )
#plt.savefig('output/reboot/sp_diff_dead.pdf')
plt.close()
graph_survival['alive'] = multivariateGrid('x_axis', 'y_axis', 'kind', df=test_joint_survival.loc[test_joint_survival.survived==1,:],
                                           bins_in=(bin_x_stand,bin_y_stand),#defineBinValue(umaps['val_functional'].loc[umaps['val_functional'].survived==1,:]),
                                           title='Survived 1 year',
                                           keep_yticks=False,
                                           keep_xticks=False)
#plt.savefig('output/reboot/sp_diff_alive.pdf')
plt.close()

######
# Dead and Alive distribution based on zone

lowers = [0.3862]
highers = [0.6722]
graph_survival['dead_low'] = multivariateGrid('x_axis',
                                          'y_axis', 
                                          'kind', 
                                          df=test_joint_survival.loc[((test_joint_survival.survived==0)&(test_joint_survival.y_axis<=lowers[0])),:],
                                          bins_in=(bin_x_stand,bin_y_stand),#defineBinValue(umaps['val_functional'].loc[umaps['val_functional'].survived==0,:]),
                                          title='Deceased within 1 year',
                                          keep_yticks=True,
                                          keep_xticks=False,
                                          keep_y=True)
#plt.savefig('output/reboot/sp_diff_dead_low.pdf')
plt.close()

graph_survival['dead_mid'] = multivariateGrid('x_axis',
                                          'y_axis', 
                                          'kind', 
                                          df=test_joint_survival.loc[((test_joint_survival.survived==0)&(test_joint_survival.y_axis>lowers[0])&(test_joint_survival.y_axis<=highers[0])),:],
                                          bins_in=(bin_x_stand,bin_y_stand),#defineBinValue(umaps['val_functional'].loc[umaps['val_functional'].survived==0,:]),
                                          title='Deceased within 1 year',
                                          keep_yticks=False,
                                          keep_xticks=False)
#plt.savefig('output/reboot/sp_diff_dead_mid.pdf')
plt.close()

graph_survival['dead_high'] = multivariateGrid('x_axis',
                                          'y_axis', 
                                          'kind', 
                                          df=test_joint_survival.loc[((test_joint_survival.survived==0)&(test_joint_survival.y_axis>highers[0])),:],
                                          bins_in=(bin_x_stand,bin_y_stand),#defineBinValue(umaps['val_functional'].loc[umaps['val_functional'].survived==0,:]),
                                          title='Deceased within 1 year',
                                          keep_yticks=False,
                                          keep_xticks=False)
#plt.savefig('output/reboot/sp_diff_dead_high.pdf')
plt.close()

graph_survival['alive_low'] = multivariateGrid('x_axis',
                                          'y_axis', 
                                          'kind', 
                                          df=test_joint_survival.loc[((test_joint_survival.survived==1)&(test_joint_survival.y_axis<=lowers[0])),:],
                                          bins_in=(bin_x_stand,bin_y_stand),#defineBinValue(umaps['val_functional'].loc[umaps['val_functional'].survived==0,:]),
                                          title='Deceased within 1 year',
                                          keep_yticks=True,
                                          keep_xticks=True,
                                          keep_x = True,
                                          keep_y = True)
#plt.savefig('output/reboot/sp_diff_alive_low.pdf')
plt.close()

graph_survival['alive_mid'] = multivariateGrid('x_axis',
                                          'y_axis', 
                                          'kind', 
                                          df=test_joint_survival.loc[((test_joint_survival.survived==1)&(test_joint_survival.y_axis>lowers[0])&(test_joint_survival.y_axis<=highers[0])),:],
                                          bins_in=(bin_x_stand,bin_y_stand),#defineBinValue(umaps['val_functional'].loc[umaps['val_functional'].survived==0,:]),
                                          title='Deceased within 1 year',
                                          keep_yticks=False,
                                          keep_xticks=True,
                                          keep_x=True)
#plt.savefig('output/reboot/sp_diff_alive_mid.pdf')
plt.close()

graph_survival['alive_high'] = multivariateGrid('x_axis',
                                          'y_axis', 
                                          'kind', 
                                          df=test_joint_survival.loc[((test_joint_survival.survived==1)&(test_joint_survival.y_axis>highers[0])),:],
                                          bins_in=(bin_x_stand,bin_y_stand),#defineBinValue(umaps['val_functional'].loc[umaps['val_functional'].survived==0,:]),
                                          title='Deceased within 1 year',
                                          keep_yticks=False,
                                          keep_x = True,
                                          keep_xticks=True)
#plt.savefig('output/reboot/sp_diff_alive_high.pdf')
plt.close()



# Functional loss 4 group split

graph_ALSFRS = {}

graph_ALSFRS[10] = multivariateGrid('x_axis', 'y_axis', 'kind', df=test_joint_functional.loc[test_joint_functional.ALSFRS_Total_final<10,:],
                                    bins_in=(bin_x_stand,bin_y_stand),#defineBinValue(umaps['val_functional'].loc[umaps['val_functional'].ALSFRS_Total_final<10,:]),title='1-year ALSFRS < 10',
                                    keep_yticks=True,
                                    keep_xticks=True,
                                    keep_x = True,
                                    keep_y = True)
#plt.savefig('output/reboot/sp_diff_ALSFRSbelow10.pdf')
plt.close()
graph_ALSFRS[20] = multivariateGrid('x_axis', 'y_axis', 'kind', df=test_joint_functional.loc[((test_joint_functional.ALSFRS_Total_final>=10)&(test_joint_functional.ALSFRS_Total_final<20)),:],
                                    bins_in=(bin_x_stand,bin_y_stand),#defineBinValue(umaps['val_functional'].loc[((umaps['val_functional'].ALSFRS_Total_final>=10)&(umaps['val_functional'].ALSFRS_Total_final<20)),:]),
                                    title = r'20 > 1-year ALSFRS $\geq$ 10',
                                    keep_yticks=False,
                                    keep_xticks=True,
                                    keep_x = True)
#plt.savefig('output/reboot/sp_diff_ALSFRSbetween10and20.pdf')
plt.close()
graph_ALSFRS[30] = multivariateGrid('x_axis', 'y_axis', 'kind', df=test_joint_functional.loc[((test_joint_functional.ALSFRS_Total_final>=20)&(test_joint_functional.ALSFRS_Total_final<30)),:],
                                    bins_in=(bin_x_stand,bin_y_stand),#defineBinValue(umaps['val_functional'].loc[((umaps['val_functional'].ALSFRS_Total_final>=20)&(umaps['val_functional'].ALSFRS_Total_final<30)),:]),
                                    title = r'30 > 1-year ALSFRS $\geq$ 20',keep_yticks=False,
                                    keep_xticks=True,
                                    keep_x = True)
#plt.savefig('output/reboot/sp_diff_ALSFRSbetween20and30.pdf')
plt.close()
graph_ALSFRS[40] = multivariateGrid('x_axis', 'y_axis', 'kind', df=test_joint_functional.loc[test_joint_functional.ALSFRS_Total_final>=30,:],
                                    bins_in=(bin_x_stand,bin_y_stand),#defineBinValue(umaps['val_functional'].loc[umaps['val_functional'].ALSFRS_Total_final>=30,:]),
                                    title = r'1-year ALSFRS $\geq$ 30',keep_yticks=False,
                                    keep_xticks=True,
                                    keep_x = True)
#plt.savefig('output/reboot/sp_diff_ALSFRSabove30.pdf')
plt.close()

val_a = test_joint_functional.loc[test_joint_functional.ALSFRS_Total_final<10,:].shape[0]
val_b = test_joint_functional.loc[((test_joint_functional.ALSFRS_Total_final>=10)&(test_joint_functional.ALSFRS_Total_final<20)),:].shape[0]
val_c = test_joint_functional.loc[((test_joint_functional.ALSFRS_Total_final>=20)&(test_joint_functional.ALSFRS_Total_final<30)),:].shape[0]
val_d = test_joint_functional.loc[test_joint_functional.ALSFRS_Total_final>=30,:].shape[0]
print('ALSFRS below 10: {} - ALSFRS between 10/20: {}'.format(val_a,val_b))
print('ALSFRS between 20/30: {} - ALSFRS above 30: {}'.format(val_c,val_d))
print('=> {}'.format(val_a+val_b+val_c+val_d))

graph_death = {}

graph_death[3] = multivariateGrid('x_axis', 'y_axis', 'kind', df=test_joint_death.loc[test_joint_death.death_unit<=3,:],
                                  bins_in=(bin_x_stand,bin_y_stand),#defineBinValue(umaps['val_functional'].loc[umaps['val_functional'].death_unit<=3,:]),
                                  title= r'survival $\leq$ 3 months',keep_yticks=True,
                                    keep_xticks=False,
                                    keep_y= True)
#plt.savefig('output/reboot/sp_diff_deadUnitBelow3m.pdf')
plt.close()
graph_death[6] = multivariateGrid('x_axis', 'y_axis', 'kind', df=test_joint_death.loc[((test_joint_death.death_unit<=6)&(test_joint_death.death_unit>3)),:],
                                  bins_in=(bin_x_stand,bin_y_stand),#defineBinValue(umaps['val_functional'].loc[((umaps['val_functional'].death_unit>3)&(umaps['val_functional'].death_unit<=6)),:]),
                                  title= r'6 > survival $\geq$ 3 months',
                                  keep_yticks=False,
                                    keep_xticks=False)
#plt.savefig('output/reboot/sp_diff_deadUnitBelow6mAbove3m.pdf',)
plt.close()
graph_death[12] = multivariateGrid('x_axis', 'y_axis', 'kind', df=test_joint_death.loc[((test_joint_death.death_unit<=12)&(test_joint_death.death_unit>6)),:],
                                   bins_in=(bin_x_stand,bin_y_stand),#defineBinValue(umaps['val_functional'].loc[((umaps['val_functional'].death_unit>6)&(umaps['val_functional'].death_unit<=12)),:]),
                                   title= r'12 > survival $\geq$ 6 months',
                                   keep_yticks=False,
                                    keep_xticks=False)
#plt.savefig('output/reboot/sp_diff_deadUnitBelow12mAbove6m.pdf')
plt.close()
graph_death[13] = multivariateGrid('x_axis', 'y_axis', 'kind', df=test_joint_death.loc[test_joint_death.death_unit>12,:],
                                   bins_in=(bin_x_stand,bin_y_stand),#defineBinValue(umaps['val_functional'].loc[umaps['val_functional'].death_unit>12,:]),
                                   title= r'survival $\geq$ 12',keep_yticks=False,
                                    keep_xticks=False)
#plt.savefig('output/reboot/sp_diff_deadUnitAbove12m.pdf')
plt.close()

val_a = test_joint_death.loc[test_joint_death.death_unit<=3,:].shape[0]
val_b = test_joint_death.loc[((test_joint_death.death_unit<=6)&(test_joint_death.death_unit>3)),:].shape[0]
val_c = test_joint_death.loc[((test_joint_death.death_unit<=12)&(test_joint_death.death_unit>6)),:].shape[0]
val_d = test_joint_death.loc[test_joint_death.death_unit>12,:].shape[0]
print('death below 3: {} - death between 3/6: {}'.format(val_a,val_b))
print('death between 6/12: {} - death above 12: {}'.format(val_c,val_d))
print('=> {}'.format(val_a+val_b+val_c+val_d))
graphs_stages = {}

keep_ticks = {'Kings_1':(False,True),
              'Kings_2':(False,False),
              'Kings_3':(False,False),
              'Kings_4':(False,False),
              'Kings_4.5':(False,False),
              'Kings_5':(False,False),
              'MiToS_0':(False,True),
              'MiToS_1':(False,False),
              'MiToS_2':(False,False),
              'MiToS_3':(False,False),
              'MiToS_4':(False,False),
              'MiToS_5':(False,False),
              'FT9_0':(True,True),
              'FT9_1':(True,False),
              'FT9_2':(True,False),
              'FT9_3':(True,False),
              'FT9_4':(True,False),
              'FT9_5':(True,False),}

stage_dev_count = pd.DataFrame([],index=['Kings','FT9','MiToS'],columns=[0,1,2,3,4,4.5,5])

for stage_i in ['Kings','MiToS','FT9']:
    
       
    if stage_i == 'Kings':
        stage_values = [1,2,3,4,4.5] # [1,2,3,4,4.5,5]
        val_a = test_joint_stages.loc[test_joint_stages[stage_i+'_score_final']==1,:].shape[0]
        val_b = test_joint_stages.loc[test_joint_stages[stage_i+'_score_final']==2,:].shape[0]
        val_c = test_joint_stages.loc[test_joint_stages[stage_i+'_score_final']==3,:].shape[0]
        val_d = test_joint_stages.loc[test_joint_stages[stage_i+'_score_final']==4,:].shape[0]
        val_e = test_joint_stages.loc[test_joint_stages[stage_i+'_score_final']==4.5,:].shape[0]
        #val_f = test_joint_stages.loc[test_joint_stages[stage_i+'_score_final']==5,:].shape[0]
        print('{} stage 1: {} stage 2: {}, stage 3: {}, stage 4: {}, stage 4.5: {}'.format(stage_i,val_a,val_b,val_c,val_d,val_e))
        #print('{} stage 1: {} stage 2: {}, stage 3: {}, stage 4: {}, stage 4.5: {}, stage 5: {}'.format(stage_i,val_a,val_b,val_c,val_d,val_e,val_f))
        #print('=> {}'.format(val_a+val_b+val_c+val_d+val_e+val_f))
        print('=> {}'.format(val_a+val_b+val_c+val_d+val_e))
    else:
        stage_values = [0,1,2,3,4] # [0,1,2,3,4,5]
        val_a = test_joint_stages.loc[test_joint_stages[stage_i+'_score_final']==0,:].shape[0]
        val_b = test_joint_stages.loc[test_joint_stages[stage_i+'_score_final']==1,:].shape[0]
        val_c = test_joint_stages.loc[test_joint_stages[stage_i+'_score_final']==2,:].shape[0]
        val_d = test_joint_stages.loc[test_joint_stages[stage_i+'_score_final']==3,:].shape[0]
        val_e = test_joint_stages.loc[test_joint_stages[stage_i+'_score_final']==4,:].shape[0]
        #val_f = test_joint_stages.loc[test_joint_stages[stage_i+'_score_final']==5,:].shape[0]
        #print('{} stage 0: {}, stage 1: {} stage 2: {}, stage 3: {}, stage 4: {}, stage 5: {}'.format(stage_i,val_a,val_b,val_c,val_d,val_e,val_f))
        print('{} stage 0: {}, stage 1: {} stage 2: {}, stage 3: {}, stage 4: {}'.format(stage_i,val_a,val_b,val_c,val_d,val_e))
        print('=> {}'.format(val_a+val_b+val_c+val_d+val_e))
        #print('=> {}'.format(val_a+val_b+val_c+val_d+val_e+val_f))
    for stage_val_i in stage_values:
        
        stage_dev_count.loc[stage_i,stage_val_i] = test_joint_stages.loc[((test_joint_stages.kind=='development')&(test_joint_stages[stage_i+'_score_final']==stage_val_i)),:].shape[0]
        
        graphs_stages['{}_{}'.format(stage_i,stage_val_i)] = multivariateGrid('x_axis', 
                                                                             'y_axis', 
                                                                             'kind', 
                                                                             df=test_joint_stages.loc[test_joint_stages[stage_i+'_score_final']==stage_val_i,:],
                                                                             bins_in=(bin_x_stand,bin_y_stand),#defineBinValue(umaps['val_functional'].loc[umaps['val_functional'][stage_i+'_score_final']==stage_val_i,:]),
                                                                             title='1-year {} at {}'.format(stage_i,stage_val_i),
                                                                             keep_xticks=keep_ticks['{}_{}'.format(stage_i,stage_val_i)][0],
                                                                             keep_x = keep_ticks['{}_{}'.format(stage_i,stage_val_i)][0],
                                                                             keep_yticks= keep_ticks['{}_{}'.format(stage_i,stage_val_i)][1],
                                                                             keep_y = keep_ticks['{}_{}'.format(stage_i,stage_val_i)][1])
        #plt.savefig('output/reboot/sp_diff_stages_{}_{}.pdf'.format(stage_i,stage_val_i))
        plt.close()

# Check if division matches with one from AAA3_plot_survival_functional.py
x = np.linspace(-0.05,1.05,200)
a1=0.2
b1=0.4
a2=0.5
b2=-0.1
y1 = a1*x+b1
y2 = a2*x+b2

def findCat(row,a1,b1,a2,b2):
    
    if row['y_axis'] > row['x_axis']*a1+b1:
        return 'high'
    elif row['y_axis'] <= row['x_axis']*a1+b1 and row['y_axis']> row['x_axis']*a2+b2:
        return 'med'
    else:
        return 'low'
    

test_joint_functional['cat'] = test_joint_functional.loc[:,['x_axis','y_axis']].apply(lambda x: findCat(x,a1,b1,a2,b2), axis=1)

perim_ALSFRS = {10:(test_joint_functional.ALSFRS_Total_final<=10),
                20:((test_joint_functional.ALSFRS_Total_final>10)&(test_joint_functional.ALSFRS_Total_final<=20)),
                30:((test_joint_functional.ALSFRS_Total_final>20)&(test_joint_functional.ALSFRS_Total_final<=30)),
                40:((test_joint_functional.ALSFRS_Total_final>30))}

title_ALSFRS = {'10_low':r'a. ALSFRS $\leq$ 10'+'\n significant \n loss zone',
                '20_low':r'b. 10 < ALSFRS $\leq$ 20'+' \n significant \n loss zone',
                '30_low':r'c. 20 < ALSFRS $\leq$ 30'+' \n significant \n loss zone',
                '40_low':r'd. 30 < ALSFRS'+' \n significant \n loss zone',
                '10_med':r'd. ALSFRS $\leq$ 10'+' \n intermediate \n loss zone',
                '20_med':r'e. 10 < ALSFRS $\leq$ 20'+' \n intermediate \n loss zone',
                '30_med':r'f. 20 < ALSFRS $\leq$ 30'+' \n intermediate \n loss zone',
                '40_med':r'g. 30 < ALSFRS'+' \n intermediate \n loss zone',
                '10_high':r'h. ALSFRS $\leq$ 10'+' \n marginal \n loss zone',
                '20_high':r'i. 10 < ALSFRS $\leq$ 20'+' \n marginal \n loss zone',
                '30_high':r'j. 20 < ALSFRS $\leq$ 30'+' \n marginal \n loss zone',
                '40_high':r'k. 30 < ALSFRS'+' \n marginal \n loss zone'
              }

ticks_labels_ALSFRS = {'10_low':(0,1),
                '20_low':(0,0),
                '30_low':(0,0),
                '40_low':(0,0),
                '10_med':(0,1),
                '20_med':(0,0),
                '30_med':(0,0),
                '40_med':(0,0),
                '10_high':(1,1),
                '20_high':(1,0),
                '30_high':(1,0),
                '40_high':(1,0)
                    }



tot_val = 0
val_cat = pd.DataFrame([],index=perim_ALSFRS.keys(),columns=['low','med','high'])

for perim_i in perim_ALSFRS.keys():
    for cat_i in ['low','med','high']:
        val_tmp = test_joint_functional.loc[((test_joint_functional.cat==cat_i)&(perim_ALSFRS[perim_i])),:].shape[0]
        tot_val += val_tmp
        print('{}-{}: {}'.format(perim_i,cat_i,val_tmp))
        graph_ALSFRS['{}_{}'.format(perim_i,cat_i)] =  multivariateGrid('x_axis',
                                                                        'y_axis', 
                                                                        'kind', 
        df=test_joint_functional.loc[((test_joint_functional.cat==cat_i)&(perim_ALSFRS[perim_i])),:],
        bins_in=(bin_x_stand,bin_y_stand),#defineBinValue(umaps['val_functional'].loc[umaps['val_functional'].survived==0,:]),
        title=title_ALSFRS['{}_{}'.format(perim_i,cat_i)],
        keep_xticks=ticks_labels_ALSFRS['{}_{}'.format(perim_i,cat_i)][0],
        keep_x=ticks_labels_ALSFRS['{}_{}'.format(perim_i,cat_i)][0],
        keep_yticks=ticks_labels_ALSFRS['{}_{}'.format(perim_i,cat_i)][1],
        keep_y=ticks_labels_ALSFRS['{}_{}'.format(perim_i,cat_i)][1])
        #plt.savefig('output/reboot/sp_diff_ALSFRS_{}_{}.pdf'.format(perim_i,cat_i))
        plt.close()
        val_cat.loc[perim_i,cat_i] = test_joint_functional.loc[((test_joint_functional.kind=='validation')&(test_joint_functional.cat==cat_i)&(perim_ALSFRS[perim_i])),:].shape[0]
        
val_cat.to_csv('output/reboot/article/functional/functional_ALSFRS_zone_allocation.csv',sep=';')
        
print('=> {}'.format(tot_val))

#plt.close()


'''
f = plt.figure()
for J in [graph_survival['dead'],graph_survival['alive']]:
    for A in J.fig.axes:
        f._axstack.add(f._make_key(A),A)
        
f.axes[0].set_position([0.05, 0.05, 0.4,  0.4])
f.axes[1].set_position([0.05, 0.45, 0.4,  0.05])
f.axes[2].set_position([0.45, 0.05, 0.05, 0.4])
f.axes[3].set_position([0.55, 0.05, 0.4,  0.4])
f.axes[4].set_position([0.55, 0.45, 0.4,  0.05])
f.axes[5].set_position([0.95, 0.05, 0.05, 0.4])
'''

# (3,4) -> (12,12) -> 12*1.5 = 18
fig1 = plt.figure(figsize=(12,18))
gs_1 = gridspec.GridSpec(3, 4)

# Decaler le Dim n1 
# Add une legende
# Synchroniser les bins

# Do modification on GIMP...
# https://en.wikibooks.org/wiki/GIMP/Removal_of_Unwanted_Elements_in_the_Image

mg_1 = SeabornFig2Grid(graph_main, fig1, gs_1[0],'a. overall',ylabel=True)
mg0 = SeabornFig2Grid(graph_survival['dead'], fig1, gs_1[1],'b. deceased within 1 year')
mg1 = SeabornFig2Grid(graph_survival['alive'], fig1, gs_1[2],'c. survived 1 year')
mg2 = SeabornFig2Grid(graph_death[3], fig1, gs_1[4],r'd. survival $\leq$ 3 months',ylabel=True)#,ylabel=True)
mg3 = SeabornFig2Grid(graph_death[6], fig1, gs_1[5],r'e. 3 < survival $\geq$ 6 months')
mg4 = SeabornFig2Grid(graph_death[12], fig1, gs_1[6],r'f. 6 < survival $\leq$ 12 months')
mg5 = SeabornFig2Grid(graph_death[13], fig1, gs_1[7],r'g. survival > 12 month')
mg6 = SeabornFig2Grid(graph_ALSFRS[10], fig1, gs_1[8],'h. 1-year ALSFRS < 10',ylabel=True,xlabel=True)#,ylabel=False)
mg7 = SeabornFig2Grid(graph_ALSFRS[20], fig1, gs_1[9],r'i. 20 > 1-year ALSFRS $\geq$ 10',xlabel=True)
mg8 = SeabornFig2Grid(graph_ALSFRS[30], fig1, gs_1[10],r'j. 30 > 1-year ALSFRS $\geq$ 20',xlabel=True)
mg9 = SeabornFig2Grid(graph_ALSFRS[40], fig1, gs_1[11],r'k. 1-year ALSFRS $\geq$ 30',xlabel=True)
#gs_1.update(bottom=0.6)
#fig1.subplots_adjust(bottom=0.5)

gs_1.tight_layout(fig1)
#fig1.text(s='Dim n1 (UMAP)',x=0.45,y=0)

fig1.legend(['Development data','Validation data'],loc='lower center',bbox_to_anchor=(0.85,0.68),
            ncol=1,borderaxespad=0,frameon=False,fontsize='small',)
#
#fig1.axes[4*3].set_ylabel('Dim n2 (UMAP)')
#fig1.axes[10*3].set_xlabel('Dim n1 (UMAP)')

#gs_1.tight_layout(fig1)

plt.savefig('output/reboot/article/survival/pdf/distribution_outcomes_s.pdf',bbox_inches='tight')
plt.savefig('output/reboot/article/survival/png/distribution_outcomes_s.png',dpi=1500,bbox_inches='tight')

# (1,4) -> (12,4) -> 4*1.5 = 6
fig2 = plt.figure(figsize=(12,6))
gs_2 = gridspec.GridSpec(1, 4)

mg_a = SeabornFig2Grid(graph_ALSFRS[10], fig2, gs_2[0],title='a. 1-year ALSFRS < 10',ylabel=True,xlabel=True)
mg_b = SeabornFig2Grid(graph_ALSFRS[20], fig2, gs_2[1],title=r'b. 20 > 1-year ALSFRS $\geq$ 10',xlabel=False)
mg_c = SeabornFig2Grid(graph_ALSFRS[30], fig2, gs_2[2],title=r'c. 30 > 1-year ALSFRS $\geq$ 20',xlabel=False)
mg_d = SeabornFig2Grid(graph_ALSFRS[40], fig2, gs_2[3],title=r'd. 1-year ALSFRS $\geq$ 30',xlabel=True)

gs_2.tight_layout(fig2)

fig2.legend(['Development data','Validation data'],loc='lower center',bbox_to_anchor=(0.5,0.00),
            ncol=2,borderaxespad=0,frameon=False,fontsize='small',)
#fig2.text(s='Dim n1 (UMAP)',x=0.5,y=0)
#fig2.axes[6].set_xlabel('Dim n1 (UMAP)')



plt.savefig('output/reboot/article/functional/pdf/distributution_outcomes_f.pdf',bbox_inches='tight')
plt.savefig('output/reboot/article/functional/png/distributution_outcomes_f.png',dpi=1500)

#(3,6) -> (18,12) -> 12*1.5 = 18
fig3 = plt.figure(figsize=(18,18))
gs_3 = gridspec.GridSpec(3, 5)

mg10 = SeabornFig2Grid(graphs_stages['Kings_1'], fig3, gs_3[0],'a. Kings stage 1',ylabel=True)
mg11 = SeabornFig2Grid(graphs_stages['Kings_2'], fig3, gs_3[1],'b. Kings stage 2')
mg12 = SeabornFig2Grid(graphs_stages['Kings_3'], fig3, gs_3[2],'c. Kings stage 3')
mg13 = SeabornFig2Grid(graphs_stages['Kings_4'], fig3, gs_3[3],'d. Kings stage 4')
mg14 = SeabornFig2Grid(graphs_stages['Kings_4.5'], fig3, gs_3[4],'e. Kings stage 4.5')
#mg15 = SeabornFig2Grid(graphs_stages['Kings_5'], fig3, gs_3[5],'f. Kings stage 5')

mg16 = SeabornFig2Grid(graphs_stages['MiToS_0'], fig3, gs_3[5],'f. MiToS stage 0',ylabel=True)
mg17 = SeabornFig2Grid(graphs_stages['MiToS_1'], fig3, gs_3[6],'g. MiToS stage 1')
mg18 = SeabornFig2Grid(graphs_stages['MiToS_2'], fig3, gs_3[7],'h. MiToS stage 2')
mg19 = SeabornFig2Grid(graphs_stages['MiToS_3'], fig3, gs_3[8],'i. MiToS stage 3')
mg20 =SeabornFig2Grid(graphs_stages['MiToS_4'], fig3, gs_3[9],'j. MiToS stage 4')
#mg21 = SeabornFig2Grid(graphs_stages['MiToS_5'], fig3, gs_3[11],'l. MiToS stage 5')

mg12 = SeabornFig2Grid(graphs_stages['FT9_0'], fig3, gs_3[10],'k. FT9 stage 0',ylabel=True)
mg13 = SeabornFig2Grid(graphs_stages['FT9_1'], fig3, gs_3[11],'l. FT9 stage 1',xlabel=True)
mg14 = SeabornFig2Grid(graphs_stages['FT9_2'], fig3, gs_3[12],'m. FT9 stage 2')
mg15 = SeabornFig2Grid(graphs_stages['FT9_3'], fig3, gs_3[13],'n. FT9 stage 3')
mg16 = SeabornFig2Grid(graphs_stages['FT9_4'], fig3, gs_3[14],'o. FT9 stage 4',xlabel=True)
#mg17 = SeabornFig2Grid(graphs_stages['FT9_5'], fig3, gs_3[17],'r. FT9 stage 5')

#fig.text(0.5,-0.05, 'Dim n1 (UMAP)', ha='center')
#fig.text(0.04, 0.5, 'Dim n2 (UMAP)', va='center', rotation='vertical')
gs_3.tight_layout(fig3)

#fig3.text(s='Dim n1 (UMAP)',x=0.5,y=0)
fig3.legend(['Development data','Validation data'],loc='lower center',bbox_to_anchor=(0.5,0.005),
            ncol=2,borderaxespad=0,frameon=False,fontsize='small',)

#fig3.axes[13*3].set_xlabel('Dim n1 (UMAP)')
plt.savefig('output/reboot/article/functional/pdf/distribution_outcomes_f_stages.pdf',bbox_inches='tight')
plt.savefig('output/reboot/article/functional/png/distribution_outcomes_f_stages.png',dpi=1500,bbox_inches='tight')

#####################

#(2,3) -> (9,8) -> 8*1.5 = 12 (12,12)
fig4 = plt.figure(figsize=(9,12))
gs_4 = gridspec.GridSpec(2, 3)

mg_18 = SeabornFig2Grid(graph_survival['dead_low'], fig4, gs_4[0],'a. deceased within 1 year \n low survival \n rate zone',ylabel=True)
mg_19 = SeabornFig2Grid(graph_survival['dead_mid'], fig4, gs_4[1],'b. deceased within 1 year \n intermediate survival \n rate zone')
mg_20 = SeabornFig2Grid(graph_survival['dead_high'], fig4, gs_4[2],'c. deceased within 1 year \n high survival \n rate zone')
mg_21 = SeabornFig2Grid(graph_survival['alive_low'], fig4, gs_4[3],'d. survived 1 year \n low survival \n rate zone',xlabel=True,ylabel=True)
mg_22 = SeabornFig2Grid(graph_survival['alive_mid'], fig4, gs_4[4],'e. survived 1 year \n intermediate survival \n rate zone')
mg_23 = SeabornFig2Grid(graph_survival['alive_high'], fig4, gs_4[5],'f. survived 1 year \n high survival \n rate zone',xlabel=True)
#gs_1.update(bottom=0.6)
#fig1.subplots_adjust(bottom=0.5)

gs_4.tight_layout(fig4)
#fig1.text(s='Dim n1 (UMAP)',x=0.45,y=0)

fig4.legend(['Development data','Validation data'],loc='lower center',bbox_to_anchor=(0.5,0.01),
            ncol=2,borderaxespad=0,frameon=False,fontsize='small',)
#
#fig4.axes[3*3].set_ylabel('Dim n2 (UMAP)')
#fig4.axes[0].set_ylabel('Dim n2 (UMAP)')
#fig4.axes[4*3].set_xlabel('Dim n1 (UMAP)')
plt.savefig('output/reboot/article/survival/pdf/distribution_grid_s.pdf',bbox_inches='tight')
plt.savefig('output/reboot/article/survival/png/distribution_grid_s.png',dpi=1500,bbox_inches='tight')

#(3,4) -> (12,12) -> 12*1.5 = 18 (18,12)
fig5 = plt.figure(figsize=(12,18))
gs_5 = gridspec.GridSpec(3, 4)

mg_24 = SeabornFig2Grid(graph_ALSFRS['10_low'], fig5, gs_5[0],title_ALSFRS['10_low'],ylabel=True)
mg_25 = SeabornFig2Grid(graph_ALSFRS['20_low'], fig5, gs_5[1],title_ALSFRS['20_low'])
mg_26 = SeabornFig2Grid(graph_ALSFRS['30_low'], fig5, gs_5[2],title_ALSFRS['30_low'])
#mg_27 = SeabornFig2Grid(graph_ALSFRS['40_low'], fig5, gs_5[3],title_ALSFRS['40_low'])
mg_28 = SeabornFig2Grid(graph_ALSFRS['10_med'], fig5, gs_5[4],title_ALSFRS['10_med'],ylabel=True)
mg_29 = SeabornFig2Grid(graph_ALSFRS['20_med'], fig5, gs_5[5],title_ALSFRS['20_med'])
mg_30 = SeabornFig2Grid(graph_ALSFRS['30_med'], fig5, gs_5[6],title_ALSFRS['30_med'])
mg_31 = SeabornFig2Grid(graph_ALSFRS['40_med'], fig5, gs_5[7],title_ALSFRS['40_med'])
mg_32 = SeabornFig2Grid(graph_ALSFRS['10_high'], fig5, gs_5[8],title_ALSFRS['10_high'],xlabel=True,ylabel=True)
mg_33 = SeabornFig2Grid(graph_ALSFRS['20_high'], fig5, gs_5[9],title_ALSFRS['20_high'])
mg_34 = SeabornFig2Grid(graph_ALSFRS['30_high'], fig5, gs_5[10],title_ALSFRS['30_high'])
mg_35 = SeabornFig2Grid(graph_ALSFRS['40_high'], fig5, gs_5[11],title_ALSFRS['40_high'],xlabel=True)

#gs_1.update(bottom=0.6)
#fig1.subplots_adjust(bottom=0.5)

gs_5.tight_layout(fig5)
#fig1.text(s='Dim n1 (UMAP)',x=0.45,y=0)

fig5.legend(['Development data','Validation data'],loc='lower center',bbox_to_anchor=(0.5,0.01),
            ncol=2,borderaxespad=0,frameon=False,fontsize='small',)
#
#fig4.axes[3*3].set_ylabel('Dim n2 (UMAP)')
#fig4.axes[0].set_ylabel('Dim n2 (UMAP)')
#fig4.axes[4*3].set_xlabel('Dim n1 (UMAP)')
plt.savefig('output/reboot/article/functional/pdf/distribution_grid_f.pdf',bbox_inches='tight')
plt.savefig('output/reboot/article/functional/png/distribution_grid_f.png',dpi=1500,bbox_inches='tight')



'''
graph_d = sns.jointplot(umaps['dev_functional'].loc[umaps['dev_functional'].survived==0,'x_axis'],
                      umaps['dev_functional'].loc[umaps['dev_functional'].survived==0,'y_axis'],
                      color=col_blind[1],)
graph_d.x = umaps['val_functional'].loc[umaps['val_functional'].survived==0,'x_axis']
graph_d.y = umaps['val_functional'].loc[umaps['val_functional'].survived==0,'y_axis']
graph_d.plot_joint(plt.scatter,marker='x',c=col_blind[0],s=50)

graph_d.fig.set_figwidth(2)
graph_d.fig.set_figheight(4)

graph_a = sns.jointplot(umaps['dev_functional'].loc[umaps['dev_functional'].survived==1,'x_axis'],
                      umaps['dev_functional'].loc[umaps['dev_functional'].survived==1,'y_axis'],
                      color=col_blind[1],)
graph_a.x = umaps['val_functional'].loc[umaps['val_functional'].survived==1,'x_axis']
graph_a.y = umaps['val_functional'].loc[umaps['val_functional'].survived==1,'y_axis']
graph_a.plot_joint(plt.scatter,marker='x',c=col_blind[0],s=50)

graph_a.fig.set_figwidth(2)
graph_a.fig.set_figheight(4)


fig = plt.figure(figsize=(4,4)) #plt.subplots(2,1,sharex=True,sharey=True,figsize=(4,4))

for plot_i in [graph_d,graph_a]:
    for axis_i in plot_i.fig.axes:
        fig._axstack.add(fig._make_key(axis_i),axis_i)

# set_position(left,bottom,width,height)
# 3 for each - marginal x, marginal y then scatter 
fig.axes[0].set_position([0.05, 0.05, 0.4,  0.4])
fig.axes[1].set_position([0.05, 0.45, 0.4,  0.05])
fig.axes[2].set_position([0.45, 0.05, 0.05, 0.4])
#
fig.axes[3].set_position([0.55, 0.05, 0.4,  0.4])
fig.axes[4].set_position([0.55, 0.45, 0.4,  0.05])
fig.axes[5].set_position([0.95, 0.05, 0.05, 0.4])


fig.savefig('output/reboot/joint_distribution_comparison.pdf', bbox_inches='tight')
fig.savefig('output/reboot/joint_distribution_comparison.png', bbox_inches='tight')

'''

#################################################################
'''
fig, axes = plt.subplots(2,1,sharex=True,sharey=True,figsize=(4,4))

col_blind = [(230/255,159/255,0/255),(0/255,114/255,178/255)]

tmp = np.asarray(umaps['val_functional'].loc[:,'x_axis'])
q1 = sc_s.scoreatpercentile(tmp, 25)
q3 = sc_s.scoreatpercentile(tmp, 75)
iqr = q3-q1
len_tmp = len(tmp)
h = 2* iqr/(len_tmp**(1/3))
nb = int(np.ceil((tmp.max() - tmp.min()) / h))

sns.distplot(umaps['dev_functional'].loc[:,'x_axis'],
             color=col_blind[0],
             ax=axes[0],
             bins=nb)
sns.distplot(umaps['val_functional'].loc[:,'x_axis'],
             color=col_blind[1],
             ax=axes[0],
             bins=nb)

tmp = np.asarray(umaps['val_functional'].loc[:,'y_axis'])
q1 = sc_s.scoreatpercentile(tmp, 25)
q3 = sc_s.scoreatpercentile(tmp, 75)
iqr = q3-q1
len_tmp = len(tmp)
h = 2* iqr/(len_tmp**(1/3))
nb = int(np.ceil((tmp.max() - tmp.min()) / h))


sns.distplot(umaps['dev_functional'].loc[:,'y_axis'],
             color=col_blind[0],
             ax=axes[1],
             bins=nb)
sns.distplot(umaps['val_functional'].loc[:,'y_axis'],
             color=col_blind[1],
             ax=axes[1],
             bins=nb,
             )

fig.text(.06, 0.5, 'Probability density function', va='center', rotation='vertical')
fig.savefig('output/reboot/article/functional/pdf/distribution_comparison.pdf', bbox_inches='tight')
fig.savefig('output/reboot/article/functional/png/distribution_comparison.png', bbox_inches='tight',dpi=1500)
plt.close()
'''

lowers = [0.3862]
highers = [0.6722]

import itertools

masks = {}
masks_out = {}

for scope_i in ['dev','val']:
    masks[scope_i] = {'low':umaps[scope_i+'_survival'].y_axis<=lowers[0],
                      'mid':((umaps[scope_i+'_survival'].y_axis<=highers[0])&(umaps[scope_i+'_survival'].y_axis>lowers[0])),
                      'high':umaps[scope_i+'_survival'].y_axis>highers[0]}
    masks_out[scope_i] = {'dead':umaps[scope_i+'_survival'].survived==0,
                      'alive':umaps[scope_i+'_survival'].survived==1,
                      'func10':umaps[scope_i+'_survival'].ALSFRS_Total_final<=10,
                      'func20':((umaps[scope_i+'_survival'].ALSFRS_Total_final>10)&(umaps[scope_i+'_survival'].ALSFRS_Total_final<=20)),
                      'func30':((umaps[scope_i+'_survival'].ALSFRS_Total_final>20)&(umaps[scope_i+'_survival'].ALSFRS_Total_final<=30)),
                      'func40':umaps[scope_i+'_survival'].ALSFRS_Total_final>30,
                      'death3':umaps[scope_i+'_survival'].death_unit<=3,
                      'death6':((umaps[scope_i+'_survival'].death_unit>3)&(umaps[scope_i+'_survival'].death_unit<=6)),
                      'death12':((umaps[scope_i+'_survival'].death_unit>6)&(umaps[scope_i+'_survival'].death_unit<=12)),
                      'death13+':umaps[scope_i+'_survival'].death_unit>12}

overview_division = []
for scope_i in ['dev','val']:
    for space_i in ['low','mid','high']:
        for out_i in ['dead','alive']:
            overview_division.append([scope_i,
                                      space_i,
                                      out_i,
                                      umaps[scope_i+'_survival'].loc[((masks[scope_i][space_i]) & (masks_out[scope_i][out_i])),:].shape[0]
                                      ])
                  
overview_division = pd.DataFrame(overview_division,columns=['scope','zone','outcome','count'])
overview_division.to_csv('output/reboot/article/survival/zone_division.csv',sep=';',index=False)                    


testing_file = []
combinations = list(itertools.product(masks['dev'].keys(),masks_out['dev'].keys()))
cat_out = {'dead':'1y_survival',
           'alive':'1y_survival',
           'func10':'func',
           'func20':'func',
           'func30':'func',
           'func40':'func',
           'death3':'survival',
           'death6':'survival',
           'death12':'survival',
           'death13+':'survival',}

for elem_i in combinations:
    try:
        p, D = ndtest.ks2d2s(umaps['dev_survival'].loc[((masks['dev'][elem_i[0]])&(masks_out['dev'][elem_i[1]])),'x_axis'].values,
                         umaps['dev_survival'].loc[((masks['dev'][elem_i[0]])&(masks_out['dev'][elem_i[1]])),'y_axis'].values,
                         umaps['val_survival'].loc[((masks['val'][elem_i[0]])&(masks_out['val'][elem_i[1]])),'x_axis'].values,
                         umaps['val_survival'].loc[((masks['val'][elem_i[0]])&(masks_out['val'][elem_i[1]])),'y_axis'].values,
                         extra=True)
    except:
        p = 'NA'
        D = 'NA'
    try:
        
        D_y, p_y = sc_s.ks_2samp(umaps['dev_survival'].loc[((masks['dev'][elem_i[0]])&(masks_out['dev'][elem_i[1]])),'y_axis'].values,
                                 umaps['val_survival'].loc[((masks['val'][elem_i[0]])&(masks_out['val'][elem_i[1]])),'y_axis'].values)
    except:
        D_y = 'NA'
        p_y = 'NA'
    
    try:
        D_x, p_x = sc_s.ks_2samp(umaps['dev_survival'].loc[((masks['dev'][elem_i[0]])&(masks_out['dev'][elem_i[1]])),'x_axis'].values,
                                 umaps['val_survival'].loc[((masks['val'][elem_i[0]])&(masks_out['val'][elem_i[1]])),'x_axis'].values)
    except:
        D_x = 'NA'
        p_x = 'NA'
        

    testing_file.append([elem_i[0],
                         cat_out[elem_i[1]],
                         elem_i[1],
                         p,
                         D,
                         p_y,
                         D_y,
                         p_x,
                         D_x])



testing_file = pd.DataFrame(testing_file,columns=['scope','out','out_cat','p','KS_stat','p_x','KS_stat_x','p_y','KS_stat_y'])
testing_file.to_csv('output/reboot/KS_external.csv',sep=';',index=False)

# KL leibler divergence (bootstrap on p distribution)

testing_KL = []

n = 10

dev_survival = umaps['dev_survival'].loc[patients_kept,:]
q = dev_survival.loc[:,['x_axis','y_axis']]
q = q.loc[((q['y_axis']!=0)&(q['x_axis']!=0)),:]
p = umaps['val_survival'].loc[:,['x_axis','y_axis']].sample(q.shape[0],replace=True)
#p = p.loc[p['y_axis']!=0,:]
initial_entropy = sc_s.entropy(p.values,q.values)

testing_KL.append(['overall','','',q.shape[0],initial_entropy[0]/n,initial_entropy[1]/n])

# Outcome specific
for out_i in masks_out['dev'].keys():
    q = dev_survival.loc[((masks_out['dev'][out_i])),['x_axis','y_axis']]
    q = q.loc[((q['y_axis']!=0)&(q['x_axis']!=0)),:]
    p = umaps['val_survival'].loc[(masks_out['val'][out_i]),['x_axis','y_axis']].sample(q.shape[0],replace=True)

    entropy = sc_s.entropy(p.values,q.values)
    testing_KL.append([cat_out[out_i],out_i,'',q.shape[0],entropy[0]/n,entropy[1]/n])

# Outcome and zone specific
for elem_i in combinations:
    q = dev_survival.loc[((masks['dev'][elem_i[0]])&(masks_out['dev'][elem_i[1]])),['x_axis','y_axis']]
    q = q.loc[((q['y_axis']!=0)&(q['x_axis']!=0)),:]
    try:    
        p = umaps['val_survival'].loc[((masks['val'][elem_i[0]])&(masks_out['val'][elem_i[1]])),['x_axis','y_axis']].sample(q.shape[0],replace=True)
        entropy = sc_s.entropy(p.values,q.values)
        testing_KL.append([cat_out[elem_i[1]],elem_i[1],elem_i[0],p.shape[0],entropy[0]/n,entropy[1]/n])
    except:
        pass
        print('exception for p for elem {}'.format(elem_i))
        print('p.shape: {} q.shape: {}'.format(p.shape[0],q.shape[0]))
    
    
testing_KL = pd.DataFrame(testing_KL,columns=['outcome','outcome_val','zone','n','div_x','div_y'])

for i in range(1,n):

    q = dev_survival.loc[:,['x_axis','y_axis']]
    q = q.loc[((q['y_axis']!=0)&(q['x_axis']!=0)),:]
    p = umaps['val_survival'].loc[:,['x_axis','y_axis']].sample(q.shape[0],replace=True)
    #p = p.loc[p['y_axis']!=0,:]
    initial_entropy = sc_s.entropy(p.values,q.values)
    
    testing_KL.loc[testing_KL.outcome=='overall','div_x'] = testing_KL.loc[testing_KL.outcome=='overall','div_x'] + initial_entropy[0]/n
    testing_KL.loc[testing_KL.outcome=='overall','div_y'] = testing_KL.loc[testing_KL.outcome=='overall','div_y'] + initial_entropy[1]/n
    
    # Outcome specific
    for out_i in masks_out['dev'].keys():
        q = dev_survival.loc[(masks_out['dev'][out_i]),['x_axis','y_axis']]
        q = q.loc[((q['y_axis']!=0)&(q['x_axis']!=0)),:]
        p = umaps['val_survival'].loc[(masks_out['val'][out_i]),['x_axis','y_axis']].sample(q.shape[0],replace=True)
    
        entropy = sc_s.entropy(p.values,q.values)
        #testing_KL.append([cat_out[out_i],out_i,'',q.shape[0],entropy[0]/n,entropy[1]/n])
        mask_outcome = ((testing_KL.outcome_val==out_i)&(testing_KL.zone==''))
        testing_KL.loc[mask_outcome,'div_x'] = testing_KL.loc[mask_outcome,'div_x'] + entropy[0]/n
        testing_KL.loc[mask_outcome,'div_y'] = testing_KL.loc[mask_outcome,'div_y'] + entropy[1]/n

    # Outcome and zone specific
    for elem_i in combinations:
        q = dev_survival.loc[((masks['dev'][elem_i[0]])&(masks_out['dev'][elem_i[1]])),['x_axis','y_axis']]
        q = q.loc[((q['y_axis']!=0)&(q['x_axis']!=0)),:]
        
        mask_combinations = ((testing_KL.outcome_val==elem_i[0])&(testing_KL.zone==elem_i[1]))
        
        try:    
            p = umaps['val_survival'].loc[((masks['val'][elem_i[0]])&(masks_out['val'][elem_i[1]])),['x_axis','y_axis']].sample(q.shape[0],replace=True)
            entropy = sc_s.entropy(p.values,q.values)
            #testing_KL.append([cat_out[out_i],elem_i[1],elem_i[0],p.shape[0],entropy[0]/n,entropy[1]/n])
            testing_KL.loc[mask_combinations,'div_x'] = testing_KL.loc[mask_combinations,'div_x'] + entropy[0]/n
            testing_KL.loc[mask_combinations,'div_y'] = testing_KL.loc[mask_combinations,'div_y'] + entropy[1]/n
        except:
            pass
            print('exception for p for elem {}'.format(elem_i))
            print('p.shape: {} q.shape: {}'.format(p.shape[0],q.shape[0]))


testing_KL.to_csv('output/reboot/KL_results_bootstrap_{}.csv'.format(n),sep=';',index=False)

'''
before (truncated) - version with subsampling q distribution
testing_KL.append(['overall','','',umaps['val_survival'].shape[0],initial_entropy[0],initial_entropy[1]])

for out_i in masks_out['dev'].keys():
    
    p = umaps['val_survival'].loc[(masks_out['val'][out_i]),['x_axis','y_axis']]
    entropy = sc_s.entropy(p.values,umaps['dev_survival'].loc[(masks_out['dev'][out_i]),['x_axis','y_axis']].sample(p.shape[0]).values)
    testing_KL.append([cat_out[out_i],out_i,'',p.shape[0],entropy[0],entropy[1]])

for elem_i in combinations:
    
    p = umaps['val_survival'].loc[((masks['val'][elem_i[0]])&(masks_out['val'][elem_i[1]])),['x_axis','y_axis']]
    entropy = sc_s.entropy(p.values,umaps['dev_survival'].loc[((masks['dev'][elem_i[0]])&(masks_out['dev'][elem_i[1]])),['x_axis','y_axis']].sample(p.shape[0]).values)
    testing_KL.append([cat_out[out_i],elem_i[1],elem_i[0],p.shape[0],entropy[0],entropy[1]])
    
testing_KL = pd.DataFrame(testing_KL,columns=['outcome','outcome_val','zone','n','div_x','div_y'])
testing_KL.to_csv('output/reboot/KL_results.csv',sep=';',index=False)
'''

#print('2D: {}, x_axis: {}, y_axis: {}'.format(p,p_y.pvalue,p_x.pvalue))

# Compute bin size for validation
# Seaborn uses the freedman-diaconis rule for automatic bin number calculation
# bin width = 2* IQR(x)/cubic root of n 
# IQR interquartile range of data
# h = 2 * iqr(a) / (len(a) ** (1 / 3)) (q3-q1 for interpercentil range)
#nb = int(np.ceil((a.max() - a.min()) / h))



tmp_x = np.asarray(umaps['val_survival'].loc[:,'x_axis'])
tmp_y = np.asarray(umaps['val_survival'].loc[:,'y_axis'])

q1_x = sc_s.scoreatpercentile(tmp_x, 25)
q3_x = sc_s.scoreatpercentile(tmp_x, 75)
q1_y = sc_s.scoreatpercentile(tmp_y, 25)
q3_y = sc_s.scoreatpercentile(tmp_y, 75)

iqr_x = q3_x-q1_x
iqr_y = q3_y-q1_y

len_tmp_x = len(tmp_x)
len_tmp_y = len(tmp_y)

h_y = 2* iqr_y/(len_tmp_y**(1/3))
h_x = 2* iqr_x/(len_tmp_x**(1/3))

nb_x = int(np.ceil((tmp_x.max() - tmp_x.min()) / h_x))
nb_y = int(np.ceil((tmp_y.max() - tmp_y.min()) / h_y))

nb_bins = min(nb_x,nb_y)
# Compute histogram for val et dev

H_val, xedges_val, yedges_val = np.histogram2d(x=umaps['val_survival'].loc[:,'x_axis'],
                                               y=umaps['val_survival'].loc[:,'y_axis'],
                                               bins=5)#nb_bins)
total_val_cases = H_val.sum()


H_dev, xedges_dev, yedges_dev = np.histogram2d(x=umaps['dev_survival'].loc[:,'x_axis'],
                                               y=umaps['dev_survival'].loc[:,'y_axis'],
                                               bins=5)#nb_bins)
total_dev_cases = H_dev.sum()

H_dev_percent = H_dev*(1/total_dev_cases)   
H_val_percent = H_val*(1/total_val_cases)

H_diff = np.zeros(shape=H_dev_percent.shape)

for elem_i in range(H_dev_percent.shape[0]):
    for elem_j in range(H_dev_percent.shape[1]):
        H_diff[elem_i][elem_j] = abs(H_dev_percent[elem_i][elem_j]-H_val_percent[elem_i][elem_j])/max(H_dev_percent[elem_i][elem_j],H_val_percent[elem_i][elem_j])*100
        
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
        
'''
fig, ax = plt.subplots()
ax.matshow(H_diff, cmap=cmap_green2red)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.1)
lin_size = 5
v1 = np.linspace(0,1,lin_size, endpoint=True)
cb = fig.colorbar(points,cax=cax,ticks=v1,orientation='vertical')
cb.ax.set_yticklabels(['{:3.0f}%'.format(i*100) for i in v1])
ax.set_yticks([])
ax.set_xticks([])
ax.set_title('Relative gap between \n development and validation \n projection distribution')
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
cax2 = divider.append_axes('left', size='2%', pad=0.05)
cax3 = divider.append_axes('bottom', size='2%', pad=0.05)

cax2.spines['right'].set_visible(False)
#cax2.spines['left'].set_visible(False)
cax2.spines['top'].set_visible(False)
cax2.spines['bottom'].set_visible(False)
cax2.set_xticks([])
cax2.set_yticks([0,1])

cax3.spines['right'].set_visible(False)
cax3.spines['left'].set_visible(False)
cax3.spines['top'].set_visible(False)
#cax3.spines['bottom'].set_visible(False)
cax3.set_yticks([])
cax3.set_xticks([0,1])

cax3.set_xlabel('Dim 1 (UMAP)')
cax2.set_ylabel('Dim 2 (UMAP)')

#plt.savefig('output/reboot/ext_distrib_differences.pdf')
#plt.close()
################################

tmp_x = np.asarray(umaps['val_survival'].loc[masks_out['val']['dead'],'x_axis'])
tmp_y = np.asarray(umaps['val_survival'].loc[masks_out['val']['dead'],'y_axis'])

q1_x = sc_s.scoreatpercentile(tmp_x, 25)
q3_x = sc_s.scoreatpercentile(tmp_x, 75)
q1_y = sc_s.scoreatpercentile(tmp_y, 25)
q3_y = sc_s.scoreatpercentile(tmp_y, 75)

iqr_x = q3_x-q1_x
iqr_y = q3_y-q1_y

len_tmp_x = len(tmp_x)
len_tmp_y = len(tmp_y)

h_y = 2* iqr_y/(len_tmp_y**(1/3))
h_x = 2* iqr_x/(len_tmp_x**(1/3))

nb_x = int(np.ceil((tmp_x.max() - tmp_x.min()) / h_x))
nb_y = int(np.ceil((tmp_y.max() - tmp_y.min()) / h_y))

nb_bins = min(nb_x,nb_y)

H_val, xedges_val, yedges_val = np.histogram2d(x=umaps['val_survival'].loc[masks_out['val']['dead'],'x_axis'],
                                               y=umaps['val_survival'].loc[masks_out['val']['dead'],'y_axis'],
                                               bins=5)#nb_bins)
total_val_cases = H_val.sum()


H_dev, xedges_dev, yedges_dev = np.histogram2d(x=umaps['dev_survival'].loc[masks_out['dev']['dead'],'x_axis'],
                                               y=umaps['dev_survival'].loc[masks_out['dev']['dead'],'y_axis'],
                                               bins=5)#nb_bins)
total_dev_cases = H_dev.sum()

H_dev_percent = H_dev*(1/total_dev_cases)   
H_val_percent = H_val*(1/total_val_cases)

H_diff = np.zeros(shape=H_dev_percent.shape)

for elem_i in range(H_dev_percent.shape[0]):
    for elem_j in range(H_dev_percent.shape[1]):
        H_diff[elem_i][elem_j] = abs(H_dev_percent[elem_i][elem_j]-H_val_percent[elem_i][elem_j])/max(H_dev_percent[elem_i][elem_j],H_val_percent[elem_i][elem_j])*100

fig, ax = plt.subplots()
ax.matshow(H_diff, cmap=cmap_green2red)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.1)
lin_size = 5
v1 = np.linspace(0,1,lin_size, endpoint=True)
cb = fig.colorbar(points,cax=cax,ticks=v1,orientation='vertical')
cb.ax.set_yticklabels(['{:3.0f}%'.format(i*100) for i in v1])
ax.set_yticks([])
ax.set_xticks([])
ax.set_title('Relative gap between \n development and validation \n projection distribution')
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
cax2 = divider.append_axes('left', size='2%', pad=0.05)
cax3 = divider.append_axes('bottom', size='2%', pad=0.05)

cax2.spines['right'].set_visible(False)
#cax2.spines['left'].set_visible(False)
cax2.spines['top'].set_visible(False)
cax2.spines['bottom'].set_visible(False)
cax2.set_xticks([])
cax2.set_yticks([0,1])

cax3.spines['right'].set_visible(False)
cax3.spines['left'].set_visible(False)
cax3.spines['top'].set_visible(False)
#cax3.spines['bottom'].set_visible(False)
cax3.set_yticks([])
cax3.set_xticks([0,1])

cax3.set_xlabel('Dim 1 (UMAP)')
cax2.set_ylabel('Dim 2 (UMAP)')

#plt.savefig('output/reboot/ext_distrib_differences_dead.pdf')
#plt.close()
'''

'''
pear_x = sc_s.pearsonr(umaps['dev_survival'].loc[:,'x_axis'].values,
                       umaps['val_survival'].loc[:,'x_axis'].values)

pear_y = sc_s.pearsonr(umaps['dev_survival'].loc[:,'y_axis'].values,
                       umaps['val_survival'].loc[:,'y_axis'].values)

spear_x = sc_s.spearmanr(umaps['dev_survival'].loc[:,'x_axis'].values,
                         umaps['val_survival'].loc[:,'x_axis'].values)

spear_y = sc_s.spearmanr(umaps['dev_survival'].loc[:,'y_axis'].values,
                         umaps['val_survival'].loc[:,'y_axis'].values)

print('pear_x: {}, pear_y: {}, spear_x: {}, spear_y: {}'.format(pear_x,pear_y,spear_x,spear_y))
'''


# Pearson correlation for LINEAR relationships
# Spearman correlation for MONOTONIC relationships
# Pearson + accurate than Spearman in its ability to describe how related the 2 variables are
# Spearman will go -1/+1/0 
# Hoeffdings D correlation 
# non-linear correlation => association (non linear association)
# linear correlation => strength of linear association
# CANT WORK AS DIFFERENT LENGTH FOR DATASETS

# Seaborn uses the freedman-diaconis rule for automatic bin number calculation
# bin width = 2* IQR(x)/cubic root of n 
# IQR interquartile range of data
# h = 2 * iqr(a) / (len(a) ** (1 / 3)) (q3-q1 for interpercentil range)
#nb = int(np.ceil((a.max() - a.min()) / h))

#####################################################################





