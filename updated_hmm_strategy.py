# -*- coding: utf-8 -*-
"""
Created on Tue Jan 09 14:06:48 2018
均线配合HMM
@author: Administrator
"""

import pandas as pd
import numpy as np
from scipy import stats
from hmmlearn.hmm import GaussianHMM,GMMHMM
from matplotlib import pyplot as plt 

def meanRoll(pds,window):
    pass

def getLog(frame):
    log_df = np.log(frame) - np.log(frame.shift())
    log_df = log_df.dropna()
    return log_df

def rsi_roll(price,window=24):#price: series
    ret = np.log(price) - np.log(price.shift())
    up =  (ret>=0)*ret
    down = (ret<0)*ret
    A = up.rolling(window).sum()
    B = down.rolling(window).sum()
    RSI = 100*A/(A-B)
    return RSI

#计算离散系数
def var_coef_roll(price,window):
    mean = price.rolling(window=window,center=False).mean()
    std = price.rolling(window=window,center=False).std()
    return mean/std
    

def getTMHS(X):
    model = GMMHMM(n_components=3, covariance_type="diag", n_iter=1000).fit(X)
    hidden_states = model.predict(X)
    transmat = model.transmat_
    return hidden_states,transmat

#判断隐藏状态的实际意义
#依据：累计收益率
def distinguish(hidden_state_array,logRet_series):#list,df,df
    state_series = pd.Series(hidden_state_array,index = logRet_series.index)
    ret_list = []
    for i in range(3):
        pos = (state_series==i)
        i_state_ret = pos*logRet_series
        ret = i_state_ret.sum()
        ret_list.append(ret)
    max_ret = max(ret_list)
    min_ret = min(ret_list)
    bull = ret_list.index(max_ret)
    bear =  ret_list.index(min_ret)
    return bull,bear,max_ret,min_ret
    
def nextState(hidden_states,transmat):
    today = hidden_states[-1]
    transition = transmat[today].tolist()
    return transition.index(max(transition))


def getTomorrowPos(nextstate,bull,bear,max_ret,min_ret):
    pos = 0
    if nextstate == bull:
        pos = 1
    elif nextstate == bear:
        pos = -1
    return pos


if __name__ == "__main__":
    window = 50
    window2 = 23
    #load data
    #用离散系数检测
    data = pd.read_csv('IF00C1.csv',index_col=0,usecols=['date','open','close','volume'], dtype={'close':float,'volume':float})
    #data['std'] = pd.rolling_std(data.close,window)
    data['rsi'] = rsi_roll(data.close,window=window)
    data['varcof'] = var_coef_roll(data.close,window2)
    #data.describe()
    
    data_sub1 = data[window:1200] 
    #data_sub1 = data[window:]  

    pos_series = pd.Series(index=data.index)
    for i in range(len(data_sub1)-window):
        sub = data_sub1[['close','varcof','rsi']][i:i+window]
        sublog = getLog(sub)
        X = sublog[['rsi','varcof']].values
        hs,tm = getTMHS(X)
        bull,bear,max_ret,min_ret = distinguish(hs,sublog.close)
        #nextstate = nextState(hs,tm)
        nextstate = hs[-1]
        pos = getTomorrowPos(nextstate,bull,bear,max_ret,min_ret)
        pos_series[sub.index[-1]] = pos

    
    window2 = 23
    data_sub2 = data[1200-23:]
    for i in range(len(data_sub2)-window2):
        sub = data_sub2[['close','varcof','rsi']][i:i+window2]
        sublog = getLog(sub)
        X = sublog[['rsi','varcof']].values
        hs,tm = getTMHS(X)
        bull,bear,max_ret,min_ret = distinguish(hs,sublog.close)
        #nextstate = nextState(hs,tm)
        nextstate = hs[-1]
        pos = getTomorrowPos(nextstate,bull,bear,max_ret,min_ret)
        pos_series[sub.index[-1]] = pos


    data['logRet'] = np.log(data.close) - np.log(data.close.shift(1))
    profit = pos_series.shift(1)*data.logRet # + abs(pos_series.shift(1)-pos_series.shift(2))*np.log(1-charge)
    result = pd.concat([profit,data.logRet],axis=1)
    result = result.dropna(how = 'any')
    result.columns = ['HMM','IF_dayily']
    #result.index = list(range(len(result)))
    result.cumsum().plot(title = 'IF_daily,charge=0.05%,window ='+str(window))

    

        
        
