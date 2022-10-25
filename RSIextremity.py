import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from load_data import _load
from indicators import RSI
import time
import sys


def prep_data(ticker, timeframe, top = 80, bottom = 20, reset_top = 70, reset_bottom = 30):
    data = _load(ticker, timeframe)
    data = data.iloc[-5000:, :]

    closes = data['Close']
    data['RSI'] = RSI(closes)


    extreme = []
    reset = True
    
    for i, item in enumerate(data['RSI']):
        if reset:
            if item >= top:
                extreme.append('top')
                reset = False
            elif item <= bottom:
                extreme.append('bottom')
                reset = False
            else:
                extreme.append(0)
        
        else:
            if reset_bottom < item < reset_top:
                reset = True
                extreme.append(0)
            else:
                extreme.append(0)

    data['extreme'] = extreme
    

    splits = []
    start = 0

    for i, item in enumerate(data['extreme']):
        if item == 0:
            continue

        else:
            splits.append(data.iloc[start:i])
            start = i

    master = pd.DataFrame(columns = ['cross_val', 'time', 'max_win',  'max_loss', 'cross_change', 'volume', 'direction'])

    del splits[0]
    forced_closes = []
    for split in splits:
        ind = split.index[0]
        end_ind = split.index[-1]

        if split.extreme[0] == 'bottom':
            current = {'extreme_val': split.Open[ind],
                        'time' : ind,
                        'max_win' : (np.max(split.High[:2]) - split.Open[0]),
                        'max_loss': (split.Open[ind] - np.min(split.Low[:2])),
                        'extremity_change' : split.Change[ind],
                        'volume' : split.Volume[ind],
                        'direction' : 'bottom'}
            master = master.append(current, ignore_index= True)

        else:
            current = {'extreme_val' : split.Open[ind],
                        'time': ind,
                        'max_win': (split.Open[ind] - np.min(split.Low[:2])),
                        'max_loss': (np.max(split.High[:2]) - split.Open[ind]),
                        'extremity_change': split.Change[ind],
                        'volume': split.Volume[ind],
                        'direction': 'top'}
            master = master.append(current, ignore_index= True)

        forced_closes.append(split.Close[1])


    master['forced_close'] = forced_closes

    return master, splits


def simulate(ticker, timeframe, top = 80, bottom = 20, reset_top = 70, reset_bottom = 30, take_prof = 1):
    master, splits = prep_data(ticker, timeframe, top, bottom, reset_top, reset_bottom)

    pnl = 0
    num = [0, 0]
    buys = []
    sells = []

    for row in master.iterrows():
        ind = row[0]
        row = row[1]

        if row['direction'] == 'top':
            sells.append(row['extreme_val'])
        else:
            buys.append(row['extreme_val'])

        if row['max_win'] >= take_prof:
            pnl += take_prof
            num[0] += 1

            if row['direction'] == 'top':
                buys.append(row['extreme_val'] - take_prof)

            else:
                sells.append(row['extreme_val'] + take_prof)


        else:
            if row['direction'] == 'top':
                pnl += row['extreme_val'] - row['forced_close']
                buys.append(row['forced_close'])

            else:
                pnl += row['forced_close'] - row['extreme_val']
                sells.append(row['forced_close'])

            num[1] += 1


    return pnl, num, buys , sells


pnl, num, sells, byuys = simulate('ESU20', '15min')
