#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 12:22:11 2022

@author: XXXX
"""
from time import time
init_time = time()
import argparse # 1.1
from numba import njit, prange # 0.54.1
import numpy as np # 1.20.3
import os
import pandas as pd # 1.3.5
from scipy.linalg import cho_solve,cho_factor # 1.7.3
from scipy.optimize import minimize_scalar # 1.7.3
from scipy.special import gamma # 1.7.3
from scipy.spatial.distance import cdist
import sys

#%% 

parser = argparse.ArgumentParser()
parser.add_argument("-p","--positions",nargs='+', type=int, help="postions of aminoacids taken in consideration",
                    action="store",default=[0,1,2,3,4,5])
parser.add_argument("-v","--voc_aa",nargs='+', help="postions of aminoacids taken in consideration",
                    action="store",default=['C','N','O'])#C D W
parser.add_argument("-d","--data", help="data used",
                    action="store",default='linear+cyclic'),#linear, cyclic or linear_cyclic
parser.add_argument("-i","--indices",nargs='+', help="delimiter indeces used for training, validation and test. They can be floats (based on the number of mutations), integers (hard indeces) or None. If the validation minimum (indices[2]) is None then the validation set is of variable length (training [indices[0]:i≤indices[1]], validation [i:indices[3]])",
                    action="store",default=['0.','3.','3.','4.','4.','5.'])
parser.add_argument("-e","--encoding", help="encoding used, default ohe",
                    action="store",default='ohe')
parser.add_argument("--err_kind", help="error used, default MAE",
                    action="store",default='MAE')
parser.add_argument("-r","--randomisation", help="mutagenesis, randomised mutagenesis or random sampling of training set, default 'mutagenesis'",
                    action="store",default='mutagenesis')
parser.add_argument("--randomisation_valid_test", type=bool, help="mutagenesis or random sampling of validation and test sets, default False",
                    action="store",default=False)
parser.add_argument("--n_lc", type=int, help="number of training points between different mutants used for the learning curves (arg_train_max-arg_train_min)*n_interp, default=20",
                                        action="store",default=20)
parser.add_argument("--replicates", type=int, help="number of replicates, default 1",
                    action="store",default=1)
parser.add_argument("--cluster", type=int, help="Does it run on a slurm cluster?, default False",
                    action="store",default=0)
parser.add_argument("--time_limit", type=str, help="time limit, ignored if 'cluster'==False, use DD-HH:MM:SS format or none if infinite",
                    action="store",default=None)
args = parser.parse_args()

WT_seq = 'CCCCCC'

positions = np.sort(args.positions).tolist()
voc_aa = np.sort(args.voc_aa).tolist()
data = args.data
indices = args.indices
encoding = args.encoding
err_kind = args.err_kind
randomisation = args.randomisation
randomisation_valid_test = args.randomisation_valid_test
n_lc = args.n_lc
replicates = args.replicates
cluster = args.cluster
if cluster:
    time_limit = args.time_limit
else:
    time_limit = args.time_limit

temp=[]
for i,temp_idx in enumerate(indices):
    if '.' in temp_idx: temp.append(float(temp_idx))
    elif ((temp_idx=='None')|(temp_idx=='none'))&(i==2): temp.append(np.nan)
    else: temp_idx.append(int(temp_idx))
indices = np.array(temp); del temp

#%% functions

def export_name(positions,voc_aa,data,encoding,randomisation,randomisation_valid_test):
    temp = ''
    for i in positions: temp+=str(i)
    temp+='_'
    for i in voc_aa: temp+=str(i)
    temp+='_'
    temp+=data+'_'+encoding+'_'+randomisation+'_'+str(randomisation_valid_test)
    return temp

@njit( parallel=True )
def subtraction(x, a):
    for i in prange(x.shape[0]):
        for ii in prange(x.shape[1]):
            x[i,ii]=a-x[i,ii]
    return x

def calc_d(x,idx=None,idx_train=None):
    temp = x.sum(axis=1)
    if (temp-temp[0]>1e-6).any():
        raise ValueError('The norm of x is not constant along axis = 1, the euclidian distance needs to be calculated without matrix multiplication')
    if np.any(idx==None):
        temp=x.sum(axis=1)
        temp=temp.reshape(-1,1)
        if idx_train==None:
            temp1 = x.T
            out_temp = x@temp1
            out_temp = subtraction(out_temp, temp[0,0])
            return out_temp
        else:
            temp1 = x[idx_train[0]:idx_train[1]]
            temp1 = temp1.T
            out_temp = x@temp1
            out_temp = subtraction(out_temp, temp[0,0])
            return out_temp
    else:
        temp1=x[idx]
        temp=temp1.sum(axis=1)
        temp=temp.reshape(-1,1)
        if idx_train==None:
            temp2 = temp1.T
            out_temp = temp1@temp2
            out_temp = subtraction(out_temp, temp[0,0])
            return out_temp
        else:
            temp2 = idx[idx_train[0]:idx_train[1]]
            temp2 = x[temp2]
            temp2 = temp2.T
            out_temp = temp1@temp2
            out_temp = subtraction(out_temp, temp[0,0])
            return out_temp

def f_time_limit(t=''):
    try:
        temp = np.zeros(4,dtype=int).astype('<U2')
        if len(t.split('-')[0])>0: temp[0]=t.split('-')[0]
        if len(t.split('-')[1].split(':')[0])>0: temp[1]=t.split('-')[1].split(':')[0]
        if len(t.split('-')[1].split(':')[1])>0: temp[2]=t.split('-')[1].split(':')[1]
        if len(t.split('-')[1].split(':')[2])>0: temp[3]=t.split('-')[1].split(':')[2]
        return np.sum(temp.astype(int)*np.array([24*60*60,60*60,60,1]))
    except:
        return np.inf

def _real_n_mut_float(n,p,voc_size=20):
    oo=0
    for pp in np.arange(p,-1,-1):
        oo+=gamma(n+1)/(gamma(n-pp+1)*gamma(pp+1))*(voc_size-1)**pp
        pp-=1
    return oo
#%% load the data
if (sys.platform != 'darwin'):
    try:
        prefix = os.environ["TMPDIR"]+'/'
        df_labels = pd.read_csv(prefix+'out_seq_total.txt',sep=' ',index_col=0)
    except:
        prefix = os.environ["HOME"]+'/'
        df_labels = pd.read_csv(prefix+'out_seq_total.txt',sep=' ',index_col=0)
    df_energies = pd.read_csv(prefix+'output_energies_total.txt',sep=' ',header=None,index_col=0,names=['energy'])

else:
    prefix = ''
    df_labels = pd.read_csv(prefix+'out_seq_total.txt',sep=' ',index_col=0)
    df_energies = pd.read_csv(prefix+'output_energies_total.txt',sep=' ',header=None,index_col=0,names=['energy'])

    
# expand sequences
df_temp = df_labels.query('cyclic==0')['sequence'].str.split("", expand=True).iloc[:,1:-1]
temp = []
for i in range(df_temp.shape[1]): temp.append(f'aa_{i}')
df_temp.columns = temp
del temp
df_temp = pd.concat((df_temp,df_temp),axis=0)
df_temp.index=df_labels['sequence'].values

df_labels.set_index('sequence',inplace=True)
df_labels.index.name = None
#concatenate
df = pd.concat((df_labels,df_energies,df_temp),axis=1,join='inner')
del df_temp

df.index.name = 'sequence'
df.reset_index(inplace=True)

#check duplicates in sequences
if 1:
    temp = df.drop_duplicates('sequence').shape[0]
    if temp!=df.shape[0]:
        raise AttributeError('DataFrame contains duplicates in "sequence" column')
    del temp


#%% modify dataset
df_modified = df.copy()

for i,wt_aa in enumerate(list(WT_seq)):
    if i in positions:
        # df_modified = df_modified.query(f'aa_{i} in {list(voc_aa)}')
        df_modified = df_modified.query(f'aa_{i} in {list(voc_aa)} | aa_{i} == "{wt_aa}"')
    else:
        df_modified = df_modified.query(f'aa_{i} == "{wt_aa}"')

# np.random.seed(2)
# df_modified = df_modified.reset_index().sample(frac=1).sort_values('n_mutations')
#%% encoding (ohe)
if data == 'linear':
    df_modified = df_modified.query('cyclic==0')
elif data == 'cyclic':
    df_modified = df_modified.query('cyclic==1')
elif (data == 'linear_cyclic')|(data == 'linear+cyclic'):
    df_modified = df_modified.query('(cyclic==0)|(cyclic==1)')

m_temp = df_modified.loc[:,df_modified.columns.str.contains("aa_", case=False)].values
x_ohe = np.ones((m_temp.shape[0],m_temp.shape[1],len(np.unique(m_temp))))*np.nan
for i,aa in enumerate(np.unique(m_temp)):
    temp = np.zeros(len(np.unique(m_temp)))
    temp[i]=1
    x_ohe[m_temp==aa,...]=temp.copy()
del m_temp,temp

x_ohe_flatten = x_ohe.reshape((df_modified.shape[0],-1))
if (data == 'linear_cyclic')|(data == 'linear+cyclic'):
    x_ohe_flatten = np.concatenate((x_ohe_flatten,df_modified.cyclic.values.reshape(-1,1),np.abs(df_modified.cyclic.values.reshape(-1,1)-1)),axis=1)

if encoding=='ohe':
    a=x_ohe_flatten.copy()
else:
    from qml import Compound
    from qml.representations import *
if encoding=='coulomb_matrix':
    a=[]
    for i,s in enumerate(df_modified.sequence.values[:]):
        c = Compound('./dir_xyz/'+s+'.xyz')
        nuclear_charges=c.nuclear_charges
        coordinates=c.coordinates
        a.append(generate_coulomb_matrix(nuclear_charges, coordinates,size=20, sorting="row-norm"))
    a=np.array(a)
elif encoding=='eigenvalue_coulomb_matrix':
    a=[]
    for i,s in enumerate(df_modified.sequence.values[:]):
        c = Compound('./dir_xyz/'+s+'.xyz')
        nuclear_charges=c.nuclear_charges
        coordinates=c.coordinates
        a.append(generate_eigenvalue_coulomb_matrix(nuclear_charges, coordinates,size=20))
    a=np.array(a)
elif encoding=='bob_matrix':
    a=[]
    for i,s in enumerate(df_modified.sequence.values[:]):
        c = Compound('./dir_xyz/'+s+'.xyz')
        nuclear_charges=c.nuclear_charges
        coordinates=c.coordinates
        atomtypes=c.atomtypes
        a.append(generate_bob(nuclear_charges, coordinates,atomtypes,size=20))
    a=np.array(a)
elif encoding=='fchl':
    a=[]
    for i,s in enumerate(df_modified.sequence.values[:]):
        from qml.fchl import *
        c = Compound('./dir_xyz/'+s+'.xyz')
        nuclear_charges=c.nuclear_charges
        coordinates=c.coordinates
        a.append(generate_representation(coordinates, nuclear_charges,max_size=20,neighbors=20).flatten())
    a=np.array(a)

y = df_modified.energy.values
#%% sequence

if type(indices[0])==np.int64:
    arg_train_min = indices[0]
    arg_train_max = indices[1]
    arg_valid_min = indices[2]
    arg_valid_max = indices[3]
    arg_test_min = indices[4]
    arg_test_max = indices[5]#3125
elif type(indices[0])==np.float64:
    temp3=[]
    for i in positions:temp3.append(f'aa_{i}')
    temp=[]
    for i,p in enumerate(indices):
        if (i==2)&np.isnan(p): temp.append(np.nan)
        elif (i==0)&(p==0.): temp.append(0) 
        else:
            if df_modified.cyclic.unique().size==1:
                temp.append(np.round(_real_n_mut_float(len(positions),p,np.unique(df_modified.loc[:,temp3]).size)).astype(int))
            elif df_modified.cyclic.unique().size==2:
                temp.append(np.round(2*_real_n_mut_float(len(positions),p,np.unique(df_modified.loc[:,temp3]).size)).astype(int))
    arg_train_min, arg_train_max, arg_valid_min, arg_valid_max, arg_test_min, arg_test_max = temp
    del temp,temp3

#https://math.stackexchange.com/questions/1236465/euclidean-distance-and-dot-product/1236477
# d = x_ohe_flatten.sum(axis=1).reshape((-1,1))-x_ohe_flatten@x_ohe_flatten[arg_train_min:arg_train_max].T

# temp = x_ohe_flatten/np.linalg.norm(x_ohe_flatten,ord=2,axis=1).reshape((-1,1))
# d = 2*(1-temp@temp.T)

#%%
# check if 
filename=export_name(positions,voc_aa,data,encoding,randomisation,randomisation_valid_test)
print(filename)

temp=np.cumsum(np.unique(df_modified.n_mutations.values,return_counts=True)[1])
temp2 = []
temp3=[]
for i in positions:temp3.append(f'aa_{i}')
for p in np.arange(len(positions)+1).astype(float):
    if df_modified.cyclic.unique().size==1:
        temp2.append(_real_n_mut_float(len(positions),p,np.unique(df_modified.loc[:,temp3]).size))
    elif df_modified.cyclic.unique().size==2:
        temp2.append(2*_real_n_mut_float(len(positions),p,np.unique(df_modified.loc[:,temp3]).size))
if np.any(temp.astype(float)!=np.array(temp2)):
    raise AttributeError(f'n_mutation in df_modified is inconsistent with ns_train_norm calculation: \n{temp.astype(float)} \nvs \n{np.array(temp2)}')
del temp,temp2

if indices.dtype==np.float64:
    ns_train_norm=np.linspace(indices[0],indices[1],np.round((indices[1]-indices[0])*n_lc).astype(int)+1)
else:
    if df_modified.cyclic.unique().size==1:
        temp0 = minimize_scalar(lambda p: np.abs(_real_n_mut_float(len(positions),p,np.unique(df_modified.loc[:,temp3]).size)-indices[0]))['x']
        temp1 = minimize_scalar(lambda p: np.abs(_real_n_mut_float(len(positions),p,np.unique(df_modified.loc[:,temp3]).size)-indices[1]))['x']
    elif df_modified.cyclic.unique().size==2:
        temp0 = minimize_scalar(lambda p: np.abs(2*_real_n_mut_float(len(positions),p,np.unique(df_modified.loc[:,temp3]).size)-indices[0]))['x']
        temp1 = minimize_scalar(lambda p: np.abs(2*_real_n_mut_float(len(positions),p,np.unique(df_modified.loc[:,temp3]).size)-indices[1]))['x']
    ns_train_norm=np.linspace(temp0,temp1,np.round((temp1-temp0)*n_lc).astype(int)+1)
    del temp0,temp1
ns_train=[]
ns_train_float=[]
for i in ns_train_norm:
    if df_modified.cyclic.unique().size==1:
        ns_train_float.append(_real_n_mut_float(len(positions),i,np.unique(df_modified.loc[:,temp3]).size))
    elif df_modified.cyclic.unique().size==2:
        ns_train_float.append(2*_real_n_mut_float(len(positions),i,np.unique(df_modified.loc[:,temp3]).size))
    ns_train.append(np.round(ns_train_float[-1]).astype(int))
ns_train=np.array(ns_train)
ns_train_float=np.array(ns_train_float)                      
del temp3


df_old = df_modified.reset_index().loc[:,['sequence','n_mutations']].copy().sort_values('n_mutations')
# df_old['train'] = False; 
# df_old['valid'] = False;
# df_old['test'] = False;


num_time_limit = f_time_limit(time_limit)

res = np.ones((replicates,len(ns_train)))*np.nan
res_tot = np.ones((*res.shape,3))*np.nan
res_tot_mut = np.ones((*res.shape,3,len(WT_seq)+1))*np.nan
l_opt = res.copy()
ls=0.1*2**np.linspace(0,25,250)#0.1*2**np.linspace(0,30,300)
idx_seeds = np.ones((res.shape[0],x_ohe_flatten.shape[0]))*np.nan
alpha_opt = []
valid_errs = np.ones((*res.shape,len(ls)))*np.nan
test_errs = np.ones((*res.shape,len(ls)))*np.nan

break_loop=False
for i_seed in range(res.shape[0]):
    print(i_seed)
    np.random.seed(i_seed)
    
    if randomisation=='mutagenesis':
        if randomisation_valid_test:
            df_new = pd.concat((df_old.iloc[arg_train_min:arg_train_max,:].sample(frac=1).sort_values('n_mutations'),
                                df_old.iloc[arg_train_max:,:].sample(frac=1)),axis=0).copy()
        else:
            df_new = pd.concat((df_old.iloc[arg_train_min:arg_train_max,:].sample(frac=1).sort_values('n_mutations'),
                                df_old.iloc[arg_train_max:,:]),axis=0).copy()
    elif randomisation=='random':
        df_new = df_old.sample(frac=1).copy()
        
    elif (randomisation=='randomised_mutagenesis')|(randomisation=='randomized_mutagenesis'):
        if randomisation_valid_test:
            df_new = pd.concat((df_old.iloc[arg_train_min:arg_train_max,:].sample(frac=1),
                                df_old.iloc[arg_train_max:,:].sample(frac=1)),axis=0).copy()
        else:
            df_new = pd.concat((df_old.iloc[arg_train_min:arg_train_max,:].sample(frac=1),
                                df_old.iloc[arg_train_max:,:]),axis=0).copy()
    
    # df_new.train.iloc[arg_train_min:arg_train_max]=True
    # if np.isnan(arg_valid_min): df_new.valid.iloc[ns_train[1]:arg_valid_max]=True
    # else: df_new.valid.iloc[arg_valid_min:arg_valid_max]=True
    # df_new.test.iloc[arg_test_min:]=True
    
    idx_seed = df_new.index.values
    idx_seeds[i_seed,:]=idx_seed
    
    # if randomisation=='random':
    #     temp = x_ohe_flatten[idx_seed]
    #     d_new = temp.sum(axis=1).reshape((-1,1))-temp@temp[arg_train_min:arg_train_max].T
    #     del temp
    # else:
    #     d_new = d[idx_seed,:][:,idx_seed[arg_train_min:arg_train_max]]
    
    # d_new = calc_d(x_ohe_flatten,idx=idx_seed,idx_train=[arg_train_min,arg_train_max])
    d_new = cdist(a[idx_seed],a[idx_seed][arg_train_min:1+arg_train_max],'cityblock')

    y_new = y[idx_seed]
    
    temp_valid_errs=[]
    temp_alpha_opt = []
    for ii,(aa,bb) in enumerate(zip(ns_train_norm,ns_train)):
        alphas=[]
        err=np.ones(ls.shape)*np.nan
        err_test=err.copy()
        d_tr_tr = d_new[arg_train_min:bb,:][:,:bb]
        if np.isnan(arg_valid_min): d_va_tr = d_new[bb:arg_valid_max,:][:,:bb]
        else: d_va_tr = d_new[arg_valid_min:arg_valid_max,:][:,:bb]
        d_te_tr = d_new[arg_test_min:arg_test_max,:][:,:bb]
        y_tr = y_new[:bb]
        if np.isnan(arg_valid_min): y_va = y_new[bb:arg_valid_max]
        else: y_va = y_new[arg_valid_min:arg_valid_max]
        y_te = y_new[arg_test_min:arg_test_max]
        
        for iii,l in enumerate(ls):
                    K_tr = np.exp(-d_tr_tr/2/l)
                    K_va = np.exp(-d_va_tr/2/l)
                    c, low = cho_factor(K_tr+np.eye(K_tr.shape[0])*1e-8)
                    alphas.append(cho_solve((c, low), y_tr))
                    err[iii]=np.mean(np.abs(y_va-K_va@alphas[-1]))
                    err_test[iii]=np.mean(np.abs(y_te-np.exp(-d_te_tr/2/l)@alphas[-1]))
                    if num_time_limit<(time()-init_time):
                        break_loop=True
                        break
        l = ls[np.nanargmin(err)]
        alpha = alphas[np.nanargmin(err)]
        K_tot = np.exp(-d_new[:,:bb]/2/l)
        y_pred_tot = K_tot@alpha
        
        if np.isnan(arg_valid_min): arg_valid_min_temp = bb
        else: arg_valid_min_temp = arg_valid_min
        temp = df_new.n_mutations.values
        temp1 = np.delete(np.arange(d_new.shape[0]),np.concatenate((np.arange(arg_train_min,arg_train_max),
                                                                    np.arange(arg_valid_min_temp,arg_valid_max))))
        for iii in range(len(WT_seq)+1):
            df_temp = df_new.n_mutations
            res_tot_mut[i_seed,ii,0,iii] = np.mean(np.abs(y_pred_tot[arg_train_min:bb][temp[arg_train_min:bb]==iii]-y_new[arg_train_min:bb][temp[arg_train_min:bb]==iii]))
            res_tot_mut[i_seed,ii,1,iii] = np.mean(np.abs(y_pred_tot[arg_valid_min_temp:arg_valid_max][temp[arg_valid_min_temp:arg_valid_max]==iii]-y_new[arg_valid_min_temp:arg_valid_max][temp[arg_valid_min_temp:arg_valid_max]==iii]))
            res_tot_mut[i_seed,ii,2,iii] = np.mean(np.abs(y_pred_tot[temp1][temp[temp1]==iii]-y_new[temp1][temp[temp1]==iii]))        
        res_tot[i_seed,ii,0] = np.mean(np.abs(y_pred_tot[arg_train_min:bb]-y_new[arg_train_min:bb]))
        res_tot[i_seed,ii,1] = np.mean(np.abs(y_pred_tot[arg_valid_min_temp:arg_valid_max]-y_new[arg_valid_min_temp:arg_valid_max]))
        res_tot[i_seed,ii,2] = np.mean(np.abs(y_pred_tot[temp1]-y_new[temp1]))
        del temp,temp1

        
        res[i_seed,ii] = np.mean(np.abs(y_pred_tot[arg_test_min:arg_test_max]-y_new[arg_test_min:arg_test_max]))
        l_opt[i_seed,ii] = l
        temp_alpha_opt.append(alpha)
        valid_errs[i_seed,ii,:] = err
        test_errs[i_seed,ii,:] = err_test
        print(i_seed,aa,bb,res[i_seed,ii],num_time_limit,(time()-init_time))
        if break_loop:
            break
    alpha_opt.append(temp_alpha_opt)
    if break_loop:
        break
    
kwargs = {'initial_parameters':np.array([{'positions':positions,
                                          'voc_aa':voc_aa,
                                          'data':data,
                                          'encoding':encoding,
                                          'indices':indices,
                                          'err_kind':err_kind,
                                          'randomisation':randomisation,
                                          'randomisation_valid_test':randomisation_valid_test,
                                          'n_lc':n_lc,
                                          'replicates':replicates,
                                          'cluster':cluster,
                                          'time_limit':time_limit
                                          }],dtype=object),
    'ns_train':ns_train,
    'ns_train_float':ns_train_float,
    'ns_train_norm':ns_train_norm,         
    'res':res,
    'res_tot':res_tot,
    'res_tot_mut':res_tot_mut,
    'l_opt':l_opt,
    'ls':ls,
    'idx_seeds':idx_seeds,
    'alpha_opt':np.array(alpha_opt,dtype=object),
    'valid_errs':valid_errs,
    'test_errs':test_errs,
    'tr_va_te_limits':np.array([arg_train_min, arg_train_max, arg_valid_min, arg_valid_max, arg_test_min, arg_test_max],dtype=object),
    'info':{'initial_parameters':'Parameters used to initialize the script',
            'l_opt':'Optimal kernel length used during the test',
            'ls':'Kernel lengths used for grid search',
            'idx_seeds':'Indices used to reshuffle the data. If one want to rebild the initial order use np.argsort(idx_seeds)',
            'alpha_opt':'Optimal regression parameters used to calculate the test error. To sort the data use alpha_opt[i][ii][np.argsort(idx_seeds[ii,0:arg_train_max].astype(int)[:ns_train[i]].'
            }
    }
if (sys.platform != 'darwin'):
    if cluster:
        np.savez(os.environ["HOME"]+'/file_'+filename+'_'+os.environ.get("SLURM_JOB_ID"),**kwargs)
    else:
        np.savez(os.environ["HOME"]+'/file_'+filename,**kwargs)
        
