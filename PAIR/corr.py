import numpy as np
import pandas as pd
import sys

def correlation(ticker1, ticker2, timeframe):
    data1 = pd.read_csv(f'C:/NewPycharmProjects/FuturesResearch/DATA/{ticker1[:2]}/{ticker1}_{timeframe}.csv')
    data2 = pd.read_csv(f'C:/NewPycharmProjects/FuturesResearch/DATA/{ticker2[:2]}/{ticker2}_{timeframe}.csv')
    
    c1, c2 = data1['Close'], data2['Close']

    
    
correlation('NQ_2020-2020', 'ES_2020-2020','60min')