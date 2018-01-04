# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 09:53:37 2017

@author: Administrator
"""

import pandas as pd
import numpy as np
from scipy import stats

def loadExcel(filename,required_columns_list):
    x = pd.ExcelFile('close/'+filename+'.xls')
    df = x.parse('Sheet0')
    df.index = df.date.tolist()
    df = df[required_columns_list]
    return df
	
#获取日期对应的下标
def getLocation(item,mylist):
	return mylist.index(item)

def getLog(frame):
    
    log_df = np.log(frame) - np.log(frame.shift())
    log_df = log_df.dropna()
    return log_df

def boxcox(XDarray):
    pass
 
#打包观察序列
def getObsv(logframe):
    X = logframe.values
    return X
    
if __name__ == "__main__":
    
    close = loadExcel('ag_daybar',['close','volume'])
    #close = close.dropna(how='any')
    
    #close2 = pd.read_hdf('close/IF_symbol2.h5')
    #diff = np.exp(diff)
    longhold = pd.read_hdf('long/ag.h5')
    shorthold = pd.read_hdf('short/ag.h5')
    
    longhold_vol = longhold.apply(lambda x: x.sum(), axis=1)
    shorthold_vol = shorthold.apply(lambda x: x.sum(), axis=1)

    frame = pd.concat([close,longhold_vol,shorthold_vol],axis=1,join='inner')
    frame.columns = [ 'close', 'volume','long', 'short']
    frame = frame.replace(0,1)
    frame.to_hdf('ag_abs.h5','df')
    #logframe = getLog(frame)
    #logframe.to_hdf('IF_log.h5','df')
    '''
    frame = pd.read_pickle('IF_1min.pickle')
    frame['diff'] = frame.high-frame.low
    frame['ret'] = np.log(frame.close)-np.log(frame.open)
    '''
    
    
    
	
	


    

        
        
        
    
    
    
    


