# -*- coding: utf-8 -*-
# Python 3.6
# 27/09/2019
# UMAP on all sets on 3 scopes (2000,4000,shape[0]-1) for min_dist=0.01 and 0.05

# From input/proc to output/umap

# -*- coding: utf-8 -*-
# 30/09/2019
# E2_umap_full version to compute specifically min, with slope and with estimated slope
# Based on E3_umap_full_PF_scope but no split between categorical and numerical
# Python 3.6

import pandas as pd
import numpy as np
import os
import umap
from umap import umap_ as umap2

import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import sklearn.preprocessing as sk_pre
import warnings
warnings.filterwarnings("ignore")
import datetime
import joblib

local = input('Is local? (Y/N) \n')

if local == 'N':
    path = '/data/vgrollem/als/preproc/'
else:
    path = ''

for file_i in ['umap_v5']:
    try:
        os.mkdir(path+'output/'+file_i)
    except:
        pass


output_path = path+'output/umap_v5'

# WARNING CHECK FEAT_i and FEATS_i -> resolve in issues with test_sample
    

df = pd.read_csv(path+'input/proc/df_survived.csv',sep=';')

df.set_index('subject_id',inplace=True)

feats_tested = {}

# Added sex and onset to minimal scope


feats_tested['slope_tbef'] = ['age','time_disease','ALSFRS_Total_baseline',
                            'weight_baseline','sex','onset_spinal',
                            'ALSFRS_Total_estim_slope']
'''
feats_tested['slope_t3'] = ['age','time_disease','ALSFRS_Total_baseline',
                            'weight_baseline','sex','onset_spinal',
                            'ALSFRS_Total_start_slope']

feats_tested['min'] = ['age','time_disease','ALSFRS_Total_baseline',
                            'weight_baseline','ALSFRS_Total_estim_slope']
'''

#df.loc[:,feats_tested['slope_tbef_t3']+['sex','onset_spinal']].dropna().to_csv(path+'output/umap_min_v3/test_samples.csv',sep=';')

for feats_i in feats_tested.keys(): #[x for x in feats_tested.keys() if x != 'testing']:

    print(feats_i)
    
    
    df_tmp = df.loc[~df.source.isin(['pre2008']),feats_tested[feats_i]]
    #test_sample = df.loc[df.source.isin(['pre2008']),feats_tested[feats_i]].sample(int(round(df.loc[df.source.isin(['pre2008']),feats_tested[feats_i]].shape[0]/4,0)))
    #df_tmp = df.loc[~df.index.isin(list(test_sample.index)),feats_tested[feats_i]]
    df_col = df_tmp.columns
    df_idx = df_tmp.index
    scaler = sk_pre.MinMaxScaler()
    df_tmp = scaler.fit_transform(df_tmp)
    
    scaling_info = [list(scaler.min_),
                    list(scaler.scale_),
                    list(scaler.data_min_),
                    list(scaler.data_max_),
                    list(scaler.data_range_),
                    ]
    scaling_info = pd.DataFrame(scaling_info,
                                columns = df_col,
                                index=['min','scale','data_min',
                                         'data_max','data_range'])
    scaling_info.to_csv(output_path+'/norm_umap_info.csv',sep=';')
    
    df_tmp = pd.DataFrame(df_tmp,columns=df_col,index=df_idx)
    tmp_shape= df_tmp.shape
    df_tmp.dropna(inplace=True)
    #print('df {} , df_tmp: {}, test_sample: {}, df_tmp post drop: {}'.format(df.shape,
    #      tmp_shape,test_sample.shape,df_tmp.shape))
    
    X = df_tmp.drop(['onset_spinal','sex'],axis=1)
    x = X.values
    x = x.astype(np.float32, order='A')
    X_cat = df_tmp.loc[:,['onset_spinal','sex']]
    x_cat = X_cat.values
    x_cat = x_cat.astype(np.float32,order='A')
        
    #test_col = test_sample.columns
    #test_idx = test_sample.index
    #test_sample = scaler.transform(test_sample)
    #test_sample = pd.DataFrame(test_sample,columns=test_col,index=test_idx)
    #X_test = test_sample
    #x_test = X_test.values
    #x_test = x_test.astype(np.float32,order='A')
    
    
    #X_raw = df.loc[:,input_feat_raw']]
    #x_raw = X_raw.values
    #x_raw = x_raw.astype(np.float32, order='A')
    
    
    params = {'n_neighbors':15,'n_components':2,'metric':'euclidean',
                  'n_epochs':None,'learning_rate':1.0,'init':'spectral',
                  'min_dist':0.1,'spread':1.0,'set_op_mix_ratio':1.0,
                  'local_connectivity':1.0,'repulsion_strength':1,
                  'negative_sample_rate':5,'transform_queue_size':1.0,'a':None, 
                  'b':None,'random_state':None,'metric_kwds':None,
                  'angular_rp_forest':False,'target_n_neighbors':-1,
                  'target_metric':'categorical','target_metric_kwds':None,
                  'target_weight':0.5,'transform_seed':26,'verbose':True}
    
    output = {}
    output_raw = {}
    
    for component_i in [2]:#,3]: #[3]: #
        output[component_i] = {}
        output_raw[component_i] = {}
        for n_neighbors_i in [df_tmp.shape[0]-1]: #[4000]: #: #[1000]:##[4000]: ##[100]:#,:#[4000]:#[100,200,500,1000,2000,4000]: #[100]: 
            output[component_i][n_neighbors_i] = {}
            output_raw[component_i][n_neighbors_i] = {}
            for min_dist_i in [0.1]:#,0.05]:#,0.1,0.2,0.3]: #[0.01]: #
                
                print("###################################")
                print('Starting for c={},n={} and md={}'.format(component_i,n_neighbors_i,min_dist_i))

                component_j = np.int(component_i)
                n_neighbors_j = np.int32(n_neighbors_i)
                min_dist_j = np.float32(min_dist_i)
                
                params['n_components'] = component_j    
                params['n_neighbors'] = n_neighbors_j
                params['min_dist'] = min_dist_j
                
                reducer_num = umap.UMAP(**params)
                params['metric'] = 'dice'
                reducer_cat = umap.UMAP(**params)
                print('Start fit : {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                #embedding = reducer_num.fit_transform(x)
                
                fit1 = reducer_num.fit(x)
                print('Start fit categorical: {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                fit2 = reducer_cat.fit(x_cat)
                
                intersection = umap2.general_simplicial_set_intersection(fit1.graph_, fit2.graph_, weight=0.5)
                intersection = umap2.reset_local_connectivity(intersection)
                # initial_alpha became learning_rate
                # gamma became repulsion_strength
            
                print('Start for simplicial_set_embedding: {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            
                sse_params = {'data':fit1._raw_data,
                          'graph':intersection,
                          'n_components':np.int32(fit1.n_components),
                          'initial_alpha':np.float32(fit1.learning_rate),
                          'a':np.float32(fit1._a),
                          'b':np.float32(fit1._b),
                          'gamma':np.float32(fit1.repulsion_strength),
                          'negative_sample_rate':np.int32(fit1.negative_sample_rate),
                          'n_epochs':np.int32(200),
                          'init':'random',
                          'random_state':np.random,
                          'metric':fit1.metric,
                          'metric_kwds':fit1._metric_kwds,
                          'verbose':False}
            
                embedding = umap2.simplicial_set_embedding(**sse_params)
                
                output[component_i][n_neighbors_i][min_dist_i] = embedding
                output[component_i][n_neighbors_i][min_dist_i] = pd.DataFrame(output[component_i][n_neighbors_i][min_dist_i],index=X.index)
                output[component_i][n_neighbors_i][min_dist_i].to_csv(output_path+'/norm_'+feats_i+'_components_c'+str(component_i)+'_n'+str(n_neighbors_i)+'_d'+str(min_dist_i)+'.csv',sep=';')
                
                '''
                test_embedding = []
                for elem_i in range(x_test.shape[0]):
                print('Start transform for sample {}: {}'.format(elem_i,datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                    test_embedding.append([reducer_num.transform(x_test[elem_i].reshape(1, -1)).tolist()])
                print('End transform for sample {}: {}'.format(elem_i,datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                
                print('Start transform for test: {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                test_embedding = reducer_num.transform(x_test)
                print('End transform for test: {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                
                output[component_i][n_neighbors_i][min_dist_i] = embedding
                output[component_i][n_neighbors_i][min_dist_i] = pd.DataFrame(output[component_i][n_neighbors_i][min_dist_i],index=X.index)
                
                
                output_test = pd.DataFrame(test_embedding,index=X_test.index)
                output_test[1] = output_test.loc[:,0].apply(lambda x: x[0][1])
                output_test.loc[:,0] = output_test.loc[:,0].apply(lambda x: x[0][0])
                
                
                output_test = pd.DataFrame(test_embedding,index=X_test.index)
                output_test.to_csv(output_path+'/norm_test_'+feats_i+'_components_c'+str(component_i)+'_n'+str(n_neighbors_i)+'_d'+str(min_dist_i)+'.csv',sep=';')
                #output_raw[component_i][n_neighbors_i][min_dist_i] = reducer.fit_transform(x_raw)
                #output_raw[component_i][n_neighbors_i][min_dist_i] = pd.DataFrame(output_raw[component_i][n_neighbors_i][min_dist_i],index=X.index)
                #output_raw[component_i][n_neighbors_i][min_dist_i].to_csv(path+'output/umap/components_c'+str(component_i)+'_n'+str(n_neighbors_i)+'_d'+str(min_dist_i)+'_raw.csv',sep=';')
                '''
                joblib.dump(reducer_num,(output_path+'/umap_model_c'+
                                         str(component_i)+'_n'+str(n_neighbors_i)
                                         +'_m'+str(min_dist_i)+'.joblib'))
                
                if component_i == 2:
                    plt.figure()
                    plt.scatter(output[component_i][n_neighbors_i][min_dist_i][0],output[component_i][n_neighbors_i][min_dist_i][1],c='b',s=1)
                    #plt.scatter(output_test[0],output_test[1],c='r',s=1)
                    plt.savefig(output_path+'/norm_'+feats_i+'_plot_c'+str(component_i)+'_n'+str(n_neighbors_i)+'_d'+str(min_dist_i)+'.pdf')
                    plt.close()
                    #plt.figure()
                    #plt.scatter(output_raw[component_i][n_neighbors_i][min_dist_i][0],output_raw[component_i][n_neighbors_i][min_dist_i][1],c='b',s=1)
                    #plt.savefig(path+'output/umap/plot_c'+str(component_i)+'_n'+str(n_neighbors_i)+'_d'+str(min_dist_i)+'_raw.pdf')
    
                                