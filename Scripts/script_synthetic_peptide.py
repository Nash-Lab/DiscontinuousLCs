#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 12:22:11 2022

@author: XXXX
"""
#%% Import Libraries

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
import sys

#%% Inputs
 
parser = argparse.ArgumentParser()
parser.add_argument("-p","--positions",nargs='+', type=int, help="postions of amino acids taken in consideration",
                    action="store",default=[0,1,2,3,4,5,6])
parser.add_argument("-v","--voc_aa",nargs='+', help="amino acid vacabulary",
                    action="store",default=['A', 'F', 'G', 'R', 'S'])
parser.add_argument("-m","--model", help="model used",
                    action="store",default='unrelaxed')
parser.add_argument("-e","--encoding", help="encoding used, possible values: ohe, aaindex, zscore5_binned",
                    action="store",default='ohe')
parser.add_argument("-l","--l_distance", type=float, help="l used to calculate p_distance**l_distance",
                    action="store",default=1.)
parser.add_argument("-i","--indices",nargs='+', help="delimiter indeces used for training, validation and test. They can be floats (based on the number of mutations), integers (hard indeces) or None. If the validation minimum (indices[2]) is None then the validation set is of variable length (training [indices[0]:i≤indices[1]], validation [i:indices[3]])",
                    action="store",default=['0.','3.','3.','4.','4.','5.'])
parser.add_argument("--err_kind", help="error function used, default MAE",
                    action="store",default='MAE')
parser.add_argument("-r","--randomisation", help="mutagenesis, randomised mutagenesis or random sampling of training set, default 'mutagenesis'",
                    action="store",default='mutagenesis')
parser.add_argument("--randomisation_valid_test", type=bool, help="mutagenesis or random sampling of validation and test sets, default False",
                    action="store",default=False)
parser.add_argument("--n_lc", type=int, help="number of training points between different mutants used for the learning curves (arg_train_max-arg_train_min)*n_interp, default=20",
                                        action="store",default=20)
parser.add_argument("--replicates", type=int, help="number of replicates, default 1",
                    action="store",default=1)
parser.add_argument("--cluster", type=int, help="Does the script run on a slurm cluster?, default False",
                    action="store",default=0)
parser.add_argument("--time_limit", type=str, help="time limit, ignored if 'cluster'==False, use DD-HH:MM:SS format or none if infinite",
                    action="store",default=None)
args = parser.parse_args()

WT_seq = 'FFFSARG'

# convert inputs
positions = np.sort(args.positions).tolist()
voc_aa = np.sort(args.voc_aa).tolist()
model = args.model
encoding = args.encoding
l_distance = args.l_distance
indices = args.indices
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

#%% Functions

def export_name(positions,voc_aa,model,encoding,l_distance,randomisation,randomisation_valid_test):
    temp = ''
    for i in positions: temp+=str(i)
    temp+='_'
    for i in voc_aa: temp+=str(i)
    temp+='_'
    temp+=model+'_'+encoding+'_'+str(l_distance)+'_'+randomisation+'_'+str(randomisation_valid_test)
    return temp

def body1(x_ohe):
    np.random.seed(0)
    temp = np.random.rand(1,1,x_ohe.shape[-1])
    # np.array([[[0.5488135 , 0.71518937, 0.60276338, 0.54488318, 0.4236548 ,0.64589411, 0.43758721, 0.891773]]])
    return (temp*x_ohe).sum(axis=-1).sum(axis=-1)

def body2(x_ohe):
    np.random.seed(1)
    temp = np.random.rand(1,1,x_ohe.shape[-1])
    # np.array([[[4.17022005e-01, 7.20324493e-01, 1.14374817e-04, 3.02332573e-01,1.46755891e-01, 9.23385948e-02, 1.86260211e-01, 3.45560727e-01]]])
    y_temp = np.zeros(x_ohe.shape[0])
    x_temp = (temp*x_ohe).sum(axis=-1)
    for i in range(x_ohe.shape[1]-1):
        for ii in range(i+1,x_ohe.shape[1]):
            y_temp += x_temp[:,i]*x_temp[:,ii]/(ii-i)**1
    return y_temp

@njit( parallel=True )
def subtraction(x, a):
    for i in prange(x.shape[0]):
        for ii in prange(x.shape[1]):
            x[i,ii]=a-x[i,ii]
    return x

@njit( parallel=True )
def calc_d_dummy(out, temp, temp1,l):
    for i in prange(temp.shape[0]):
        for ii in prange(temp1.shape[0]):
            t=0
            for iii in prange(temp1.shape[1]):
                t+=abs(temp[i,iii]-temp1[ii,iii])**l
            out[i,ii]=t
    return out/2

try:
    #if qml library is installed
    from qml.distance import manhattan_distance,l2_distance,p_distance
    def calc_d(x,idx=None,idx_train=None,l=1.):
        if np.any(idx==None):
            temp = x
            if idx_train==None:
                temp1 = x
            else:
                temp1 = x[idx[idx_train[0]:idx_train[1]]]
        else:
            temp = x[idx]
            if idx_train==None:
                temp1 = x[idx]
            else:
                temp1 = x[idx[idx_train[0]:idx_train[1]]]
        if l == 1.:
            out_temp = manhattan_distance(temp,temp1)
        elif l == 2.:
            out_temp = l2_distance(temp,temp1)**2.
        else:
            out_temp = p_distance(temp,temp1,l)**l
        return out_temp
except:
    def calc_d(x,idx=None,idx_train=None,l=1):
        #https://math.stackexchange.com/questions/1236465/euclidean-distance-and-dot-product/1236477
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
            else:
                temp1 = x[idx_train[0]:idx_train[1]]
                temp1 = temp1.T
                out_temp = x@temp1
                out_temp = subtraction(out_temp, temp[0,0])
        else:
            temp1=x[idx]
            temp=temp1.sum(axis=1)
            temp=temp.reshape(-1,1)
            if idx_train==None:
                temp2 = temp1.T
                out_temp = temp1@temp2
                out_temp = subtraction(out_temp, temp[0,0])
            else:
                temp2 = idx[idx_train[0]:idx_train[1]]
                temp2 = x[temp2]
                temp2 = temp2.T
                out_temp = temp1@temp2
                out_temp = subtraction(out_temp, temp[0,0])
        return out_temp**l

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

# combinatorial scale
def _real_n_mut_float(n,p,voc_size=20):
    oo=0
    for pp in np.arange(p,-1,-1):
        oo+=gamma(n+1)/(gamma(n-pp+1)*gamma(pp+1))*(voc_size-1)**pp
        pp-=1
    return oo

#%% Load the data
if (sys.platform != 'darwin'):
    try:
        prefix = os.environ["TMPDIR"]+'/'
        df_labels = pd.read_csv(prefix+'out_seq_total.txt',sep=' ',index_col=0)
    except:
        prefix = os.environ["HOME"]+'/'
        df_labels = pd.read_csv(prefix+'out_seq_total.txt',sep=' ',index_col=0)
else:
    prefix = ''
    df_labels = pd.read_csv(prefix+'out_seq_total.txt',sep=' ',index_col=0)

if model == 'unrelaxed':
    df_energies = pd.read_csv(prefix+'output_energies_total.txt',sep=' ',header=None,index_col=0,names=['energy'])
elif model == 'relaxed':
    df_energies = pd.read_csv(prefix+'output_energies_total_repaired.txt',sep=' ',header=None,index_col=0,names=['energy'])
else:
    df_energies = pd.read_csv(prefix+'output_energies_total.txt',sep=' ',header=None,index_col=0,names=['energy'])
    
# expand sequences
df_temp = df_labels['sequence'].str.split("", expand=True).iloc[:,1:-1]
temp = []
for i in range(df_temp.shape[1]): temp.append(f'aa_{i}')
df_temp.columns = temp
del temp

#concatenate
df = pd.concat((df_labels,df_energies,df_temp),axis=1,join='inner')
del df_temp

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
        df_modified = df_modified.query(f'aa_{i} in {list(voc_aa)} | aa_{i} == "{wt_aa}"')
    else:
        df_modified = df_modified.query(f'aa_{i} == "{wt_aa}"')

#%% encoding (ohe)
m_temp = df_modified.loc[:,df_modified.columns.str.contains("aa_", case=False)].values
    
if encoding=='ohe':
    x = np.ones((m_temp.shape[0],m_temp.shape[1],len(np.unique(m_temp))))*np.nan
    d_encoder={}
    for i,aa in enumerate(np.unique(m_temp)):
        temp = np.zeros(len(np.unique(m_temp)))
        temp[i]=1
        d_encoder[aa]=temp
    del temp
elif encoding=='aaindex':
    d_encoder = {'A': np.array([0.47643547, 0.57367069, 1.        , 0.47854889, 0.62914672,   0.58986869, 0.20987087, 0.35168915, 0.62439324, 0.24154633,   0.51092499, 0.56800074, 0.51980193, 0.62638274, 0.98076892,   0.85285402, 0.64470677, 0.18126958]),
                 'R': np.array([0.26079703, 0.        , 0.43150555, 0.45250432, 0.        ,   0.01207181, 0.2368323 , 0.03622721, 0.14136082, 0.        ,   0.86326866, 0.30842974, 0.19965029, 0.46380858, 0.64697034,   0.41913677, 0.41822572, 0.41515698]),
                 'N': np.array([0.08015426, 0.4209555 , 0.33482139, 0.27167376, 0.35428641,   0.65775339, 0.60119517, 0.7529993 , 0.31686854, 0.72029562,   0.6051326 , 0.3129469 , 0.0062735 , 0.10393986, 1.        ,   0.48439746, 0.32070657, 0.23082088]),
                 'D': np.array([0.        , 0.34373397, 0.46598648, 0.38079544, 0.76143104,   0.78839123, 1.        , 0.30371391, 0.15583811, 0.30311533,   0.54485642, 0.35730357, 0.12258042, 1.        , 0.18494585,   0.47129562, 0.69543897, 0.40311301]),
                 'C': np.array([0.680962  , 0.64150144, 0.2381341 , 0.        , 1.        ,   0.        , 0.32475838, 0.07004567, 0.2179411 , 0.67793885,   0.63469091, 0.49444237, 0.4770351 , 0.41865618, 0.40057099,   0.48680365, 0.42378126, 0.35125734]),
                 'Q': np.array([0.26867635, 0.17939673, 0.52588002, 0.42967362, 0.53124866,   0.33889091, 0.36699167, 0.42366292, 0.36623207, 0.42039494,   0.        , 0.21235681, 0.54242481, 0.        , 0.40684311,   0.30187497, 1.        , 0.46646852]),
                 'E': np.array([0.1613067 , 0.11000224, 0.82921962, 0.52247445, 0.88601526,   0.67776504, 0.73699392, 0.18020248, 0.10428898, 0.32135234,   0.44242947, 0.61648808, 0.64397859, 0.07584217, 0.53068948,   0.35109319, 0.        , 0.48597018]),
                 'G': np.array([0.04992792, 1.        , 0.51233925, 0.28477701, 0.03440712,   1.        , 0.07581077, 0.11870488, 0.01896264, 0.26861602,   0.34273873, 0.48191696, 0.41054221, 0.34488057, 0.28908982,   0.31639538, 0.38464168, 0.41509174]),
                 'H': np.array([0.4618312 , 0.19886317, 0.30983878, 0.30507281, 0.50361723,   0.51990378, 0.17156301, 1.        , 0.06678427, 0.19297404,   0.58458185, 1.        , 0.46137396, 0.54576485, 0.41662311,   0.29119017, 0.54664478, 0.43914538]),
                 'I': np.array([1.        , 0.57918301, 0.55629667, 0.55237683, 0.31192376,   0.36601286, 0.68766443, 0.39531477, 0.        , 0.48074054,   0.18243504, 0.54991781, 0.        , 0.3891144 , 0.64481485,   0.37398143, 0.42624042, 0.39188417]),
                 'L': np.array([0.92898278, 0.50964206, 0.83550329, 0.63365923, 0.38230297,   0.58655499, 0.41256276, 0.45221146, 0.46430196, 0.72177998,   1.        , 0.33163167, 0.48346956, 0.48763699, 0.29647275,   0.        , 0.53675047, 0.32697873]),
                 'K': np.array([0.17063637, 0.06333692, 0.65320857, 0.49067661, 0.13867663,   0.36509276, 0.15181399, 0.32084392, 0.41427255, 1.        ,   0.19207545, 0.61785307, 0.29813143, 0.71848447, 0.14427065,   0.65617354, 0.27602278, 0.41153379]),
                 'M': np.array([0.88471493, 0.29486328, 0.56922306, 0.3723753 , 0.815067  ,   0.59049278, 0.        , 0.61854701, 0.18977451, 0.19764329,   0.1417632 , 0.        , 0.24521326, 0.75802178, 0.51181537,   0.3360651 , 0.12371405, 0.42235329]),
                 'F': np.array([0.96840498, 0.40981568, 0.33920965, 0.51465999, 0.39385077,   0.71258301, 0.47552357, 0.55084468, 0.22501349, 0.30586776,   0.71046636, 0.31210344, 0.37863512, 0.09096827, 0.        ,   1.        , 0.42260121, 0.56215869]),
                 'P': np.array([0.03937641, 0.72243771, 0.        , 1.        , 0.84064282,   0.26605426, 0.20380892, 0.39949457, 0.26094524, 0.43783898,   0.47166396, 0.47647664, 0.35037728, 0.47553892, 0.53745361,   0.45312356, 0.3995222 , 0.39936694]),
                 'S': np.array([0.1460926 , 0.63800788, 0.5109339 , 0.3392021 , 0.33864113,   0.31672056, 0.54976535, 0.61634059, 0.85013649, 0.40269299,   0.53244673, 0.32868787, 0.50579641, 0.59399459, 0.64126657,   0.42060128, 0.31732053, 1.        ]),
                 'T': np.array([0.34393518, 0.58310154, 0.47317091, 0.39814004, 0.3162428 ,   0.1271053 , 0.69498667, 0.57130038, 0.92253682, 0.07998976,   0.29095721, 0.47386474, 0.3759638 , 0.36157429, 0.04234762,   0.37480656, 0.20720859, 0.        ]),
                 'W': np.array([0.92109615, 0.18686246, 0.04482931, 0.4371835 , 0.50166412,   0.9646231 , 0.38330367, 0.        , 1.        , 0.34389313,   0.35571448, 0.67788709, 0.19565441, 0.41993502, 0.62876485,   0.32091188, 0.44642037, 0.48230817]),
                 'Y': np.array([0.6830971 , 0.33403713, 0.05012196, 0.45148907, 0.06960824,   0.58988394, 0.73917426, 0.40684053, 0.10322045, 0.49248518,   0.33845081, 0.30092216, 1.        , 0.7373806 , 0.7550846 ,   0.48399529, 0.36633211, 0.24182714]),
                 'V': np.array([0.87895186, 0.65534862, 0.66412089, 0.4850448 , 0.29168428,   0.118428  , 0.73040279, 0.32477854, 0.10649645, 0.31039278,   0.22434709, 0.68422616, 0.29777741, 0.51823394, 0.60526483,   0.45929967, 0.45333186, 0.61620511])}
elif encoding=='z_score_binned':
    d_encoder = {'A': np.array([0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1.]),
                 'C': np.array([0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 1., 0., 1., 0., 0., 0.]),
                 'D': np.array([1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0.]),
                 'E': np.array([1., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1., 0., 0.]),
                 'F': np.array([0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0.]),
                 'G': np.array([1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0.]),
                 'H': np.array([1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0.]),
                 'I': np.array([0., 0., 0., 1., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.]),
                 'K': np.array([1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0.]),
                 'L': np.array([0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 1., 0.]),
                 'M': np.array([0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0., 0.]),
                 'N': np.array([1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1.]),
                 'P': np.array([0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 1.]),
                 'Q': np.array([0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 0.]),
                 'R': np.array([1., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0.]),
                 'S': np.array([1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0.]),
                 'T': np.array([0., 0., 1., 0., 1., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1., 0., 0.]),
                 'V': np.array([0., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0.]),
                 'W': np.array([0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 1., 0., 0., 0.]),
                 'Y': np.array([0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 1., 0., 0., 1., 0., 0., 0.])}
elif encoding=='z_score':
    d_encoder = {'A': np.array([2., 0., 2., 1., 3.]),
                 'C': np.array([2., 1., 2., 2., 0.]),
                 'D': np.array([0., 2., 2., 0., 2.]),
                 'E': np.array([0., 2., 1., 0., 1.]),
                 'F': np.array([3., 3., 2., 2., 1.]),
                 'G': np.array([0., 0., 2., 1., 1.]),
                 'H': np.array([0., 3., 2., 3., 2.]),
                 'I': np.array([3., 1., 0., 1., 2.]),
                 'K': np.array([0., 2., 0., 2., 2.]),
                 'L': np.array([3., 1., 1., 1., 2.]),
                 'M': np.array([3., 1., 2., 3., 1.]),
                 'N': np.array([0., 2., 2., 1., 3.]),
                 'P': np.array([1., 2., 2., 2., 3.]),
                 'Q': np.array([2., 2., 1., 0., 2.]),
                 'R': np.array([0., 3., 0., 3., 1.]),
                 'S': np.array([0., 1., 2., 0., 2.]),
                 'T': np.array([2., 0., 1., 0., 1.]),
                 'V': np.array([3., 0., 1., 1., 1.]),
                 'W': np.array([3., 3., 2., 3., 0.]),
                 'Y': np.array([3., 3., 2., 1., 0.])}
x = np.ones((m_temp.shape[0],m_temp.shape[1],*list(d_encoder.values())[0].shape))*np.nan
for i,aa in enumerate(np.unique(m_temp)):
    x[m_temp==aa,...]=d_encoder[aa]

x_flatten = x.reshape((df_modified.shape[0],-1))

if ((model == 'body1') | (model == 'body2') | (model == 'body1+body2')):
    x = np.ones((m_temp.shape[0],m_temp.shape[1],len(np.unique(m_temp))))*np.nan
    d_encoder_ohe={}
    for i,aa in enumerate(np.unique(m_temp)):
        temp = np.zeros(len(np.unique(m_temp)))
        temp[i]=1
        d_encoder_ohe[aa]=temp
    del temp
    x_ohe = np.ones((m_temp.shape[0],m_temp.shape[1],*list(d_encoder_ohe.values())[0].shape))*np.nan
    for i,aa in enumerate(np.unique(m_temp)):
        x_ohe[m_temp==aa,...]=d_encoder_ohe[aa]

del m_temp

if (model == 'body1'):# & (encoding=='ohe'):
    df_modified.energy = body1(x_ohe)
elif (model == 'body2'):# & (encoding=='ohe'):
    df_modified.energy = body2(x_ohe)
elif (model == 'body1+body2'):# & (encoding=='ohe'):
    df_modified.energy = body2(x_ohe)+body1(x_ohe)

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
    for i in positions[:1]:temp3.append(f'aa_{i}')
    temp=[]
    for i,p in enumerate(indices):
        if (i==2)&np.isnan(p): temp.append(np.nan)
        elif (i==0)&(p==0.): temp.append(0) 
        else: temp.append(np.round(_real_n_mut_float(len(positions),p,np.unique(df_modified.loc[:,temp3]).size)).astype(int))
    arg_train_min, arg_train_max, arg_valid_min, arg_valid_max, arg_test_min, arg_test_max = temp
    del temp,temp3

#%%
# check if 
filename=export_name(positions,voc_aa,model,encoding,l_distance,randomisation,randomisation_valid_test)
print(filename)

temp=np.cumsum(np.unique(df_modified.n_mutations.values,return_counts=True)[1])
temp2 = []
temp3=[]
for i in positions[:1]:
    temp3.append(f'aa_{i}')
for p in np.arange(len(positions)+1).astype(float):
    temp2.append(_real_n_mut_float(len(positions),p,np.unique(df_modified.loc[:,temp3]).size))
if np.any(temp.astype(float)!=np.array(temp2)):
    raise AttributeError(f'n_mutation in df_modified is inconsistent with ns_train_norm calculation: \n{temp.astype(float)} \nvs \n{np.array(temp2)}')
del temp,temp2

if indices.dtype==np.float64:
    ns_train_norm=np.linspace(indices[0],indices[1],np.round((indices[1]-indices[0])*n_lc).astype(int)+1)
else:
    temp0 = minimize_scalar(lambda p: np.abs(_real_n_mut_float(len(positions),p,np.unique(df_modified.loc[:,temp3]).size)-indices[0]))['x']
    temp1 = minimize_scalar(lambda p: np.abs(_real_n_mut_float(len(positions),p,np.unique(df_modified.loc[:,temp3]).size)-indices[1]))['x']
    ns_train_norm=np.linspace(temp0,temp1,np.round((temp1-temp0)*n_lc).astype(int)+1)
    del temp0,temp1
ns_train=[]
ns_train_float=[]
for i in ns_train_norm:
    ns_train_float.append(_real_n_mut_float(len(positions),i,np.unique(df_modified.loc[:,temp3]).size))
    ns_train.append(np.round(ns_train_float[-1]).astype(int))
ns_train=np.array(ns_train)
ns_train_float=np.array(ns_train_float)                      

if n_lc==0:#if n_lc == 0, then take a linspace span
    ns_train = np.arange(1,np.round(_real_n_mut_float(len(positions),indices[1],np.unique(df_modified.loc[:,temp3]).size))+1).astype(int)
    ns_train_float = ns_train.astype(float)
    ns_train_norm = []
    for i in ns_train:
        print(i)
        ns_train_norm.append(minimize_scalar(lambda p: np.abs(_real_n_mut_float(len(positions),p,np.unique(df_modified.loc[:,temp3]).size)-i))['x'])

del temp3

df_old = df_modified.reset_index().loc[:,['sequence','n_mutations']].copy()

num_time_limit = f_time_limit(time_limit)

res = np.ones((replicates,len(ns_train)))*np.nan
res_tot = np.ones((*res.shape,3))*np.nan
res_tot_mut = np.ones((*res.shape,3,len(WT_seq)+1))*np.nan
l_opt = res.copy()
ls=0.1*2**np.linspace(0,25,250)
idx_seeds = np.ones((res.shape[0],x_flatten.shape[0]))*np.nan
alpha_opt = []
valid_errs = np.ones((*res.shape,len(ls)))*np.nan
test_errs = np.ones((*res.shape,len(ls)))*np.nan

#%% LC loop
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
    
    
    idx_seed = df_new.index.values
    idx_seeds[i_seed,:]=idx_seed
    
    
    d_new = calc_d(x_flatten,idx=idx_seed,idx_train=[arg_train_min,arg_train_max],l=l_distance)
    
    
    
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

# Data to be saved
kwargs = {'initial_parameters':np.array([{'WT_seq':WT_seq,
                                          'positions':positions,
                                          'voc_aa':voc_aa,
                                          'model':model,
                                          'encoding':encoding,
                                          'l_distance':l_distance,
                                          'indices':indices,
                                          'err_kind':err_kind,
                                          'randomisation':randomisation,
                                          'randomisation_valid_test':randomisation_valid_test,
                                          'n_lc':n_lc,
                                          'replicates':replicates,
                                          'cluster':cluster,
                                          'time_limit':time_limit
                                          }],dtype=object),
    'd_encoder':d_encoder,
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
        

