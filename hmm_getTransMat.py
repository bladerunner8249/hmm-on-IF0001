# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 09:53:37 2017

@author: Administrator
"""
charge = 0.0005

import pandas as pd
import numpy as np
from scipy import stats
from hmmlearn.hmm import GaussianHMM,GMMHMM
from matplotlib import pyplot as plt 
#from hmm_getObsv import * 
 
def norm(df):
    frame = (df - df.mean()) / df.std()
    return frame

def getLog(frame):
    log_df = np.log(frame) - np.log(frame.shift())
    log_df = log_df.dropna()
    return log_df

def boxcox(frame):
    for each in frame.columns:
        x = frame[each].values
        boxcox = stats.boxcox(x)[0]
        frame[each] = boxcox
    return frame

def getTMHS(X):
    model = GMMHMM(n_components=3, covariance_type="diag", n_iter=1000).fit(X)
    hidden_states = model.predict(X)
    transmat = model.transmat_
    return hidden_states,transmat
    
#判断隐藏状态的实际意义
#依据：累计收益率
def distinguish(hidden_states,logframe):#list,df,df
    states_dic = {}; ret_list = []; i=0
    #for each in logframe.index:
        #states_dic[each] = hidden_states[i]
        #i += 1
    for i,indexer in enumerate(logframe.index):
        states_dic[indexer] = hidden_states[i] 
    states_series = pd.Series(states_dic)
    for i in range(3):
        pos = (states_series==i)
        x = pos*logframe.close
        ret = sum(x)
        ret_list.append(ret)
    good = ret_list.index(max(ret_list))
    bad =  ret_list.index(min(ret_list))
    return good,bad,max(ret_list),min(ret_list)


def distinguish1(hidden_states,logframe):#list,df,df
    states_dic = {}; ret_list = []; i=0
    for each in logframe.index:
        states_dic[each] = hidden_states[i]
        i += 1
    states_series = pd.Series(states_dic)
    for i in range(3):
        pos = (states_series==i)
        x = pos*logframe.logret
        ret = sum(x)
        ret_list.append(ret)
    good = ret_list.index(max(ret_list))
    bad =  ret_list.index(min(ret_list))
    return good,bad#,max(ret_list)


def nextState(hidden_states,transmat):
    today = hidden_states[-1]
    transition = transmat[today].tolist()
    return transition.index(max(transition))
            
def getTomorrowPos(nextstate,good,bad,retgood,retbad):
    if retgood<0:
        pos = 0
    else: 
        if nextstate==good:# and retgood>0:
            pos = 1
        elif nextstate==bad and retbad<0:
            #pos = -1
            pos = -1
        else:
            pos = 0
    return pos

if __name__ == "__main__":

    frame = pd.read_hdf("IF_abs.h5",'df')
    #frame = frame[['close']]
    #frame = frame[500:]
    #frame = pd.read_pickle('IF_1min.pickle')
    #diff = frame.high-frame.low
    #ret = np.log(frame.close)-np.log(frame.open)
    #X_frame = pd.concat([diff,ret],axis=1)
    #X_frame.columns = ['diff','logret']
    dates = frame.index
    pos_dic = {}
    for i in range(len(frame)-23):
        subdate = dates[i:i+23]
        sub = frame.loc[subdate,:]
        #sub = boxcox(sub)
        sublog = getLog(sub)
        X = sub.values
        hs,tm = getTMHS(X)
        good,bad,retgood,retbad = distinguish(hs,sublog)
        nextstate = nextState(hs,tm)
        pos = getTomorrowPos(nextstate,good,bad,retgood,retbad)
        pos_dic[subdate[-1]] = pos

    
    pos_series = pd.Series(pos_dic)
    framelog = getLog(frame)
    #a= states_series[states_series>0]
    #a = a.fillna(0)
    profit = pos_series.shift(1)*framelog.close # + abs(pos_series.shift(1)-pos_series.shift(2))*np.log(1-charge)
    result = pd.concat([profit,framelog.close],axis=1)
    result.columns = ['GMM-HMM','IF_dayily']
    #result.index = list(range(len(result)))
    result.cumsum().plot(title = 'IF_daily,charge=0.05%,length=69')
    
        

    
    
   

    


    

        
        
        
    
    
    
    


