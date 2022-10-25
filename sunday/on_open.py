import pandas as pd
import numpy as np
import datetime as dt
import sys
import calendar

def ovnt_response(ticker, timeframe):
    data  = pd.read_csv(f'C:/NewPycharmProjects/FuturesResearch/DATA/{ticker[:2]}/{ticker}'+'_'+timeframe+'.csv')
    if 'Time.1' in data.columns:
        data.drop('Time.1', axis = 1, inplace = True)


    weekdays = [calendar.day_name[c.weekday()] for c in data['Time'].tolist()]
    data['weekday'] = weekdays
    print(weekdays)
    
    
ovnt_response('NQ_2020-2020', '5min')