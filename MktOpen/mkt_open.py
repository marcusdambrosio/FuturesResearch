import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from helpers.indicators import EMA
from data_prep import _load
import datetime as dt
import time
import sys

def _load(ticker, timeframe):
    data = pd.read_csv(f'C:/NewPycharmProjects/FuturesResearch/DATA/{ticker[:2]}/{ticker}_{timeframe}.csv')
    data = data.loc[:, :'Volume']
    try:
        data.drop('Time.1', axis = 1, inplace = True)
    except:
        print('Time.1 column not present')
    data.dropna(axis=0, inplace=True)
    data.set_index(pd.DatetimeIndex(data['Time']), drop=True, inplace=True)
    data.drop('Time', axis=1, inplace=True)
    data.rename(columns={'Last': 'Close'}, inplace=True)
    data['Change'] = data['Close'] - data['Open']
    return data

def mktopen_data(ticker, timeframe, filtering_quantiles = False):
    data = _load(ticker, timeframe)
    data['Time'] = data.index.strftime('%H:%M')
    TOD = [c[-5:] for c in data['Time']]
    data['TOD'] = TOD
    data['nexthigh'] = data['High'].shift(-1)
    data['nextlow'] = data['Low'].shift(-1)
    data['nextclose'] = data['Close'].shift(-1)
    data['nextopen'] = data['Open'].shift(-1)

    opp_dir = [0]
    loss = [0]
    drawdown = [0]
    span = [0]
    topwick = [0]
    botwick = [0]
    change_frac = [0]

    for row in data.iterrows():
        ind = row[0]
        row = row[1]

        if row['Change'] >= 0:
            topwick.append(np.abs(row['High'] - row['Close']))
            botwick.append(np.abs(row['Low'] - row['Open']))
            opp_dir.append((row['nextopen'] - row['nextlow'])/row['nextopen'])
            loss.append((row['nextopen'] - row['nextclose'])/row['nextopen'])
            drawdown.append((row['nextopen'] - row['nexthigh'])/row['nextopen'])
        else:
            topwick.append(np.abs(row['High'] - row['Open']))
            botwick.append(np.abs(row['Low'] - row['Close']))
            opp_dir.append((row['nexthigh'] - row['nextopen'])/row['nextopen'])
            loss.append((row['nextclose'] - row['nextopen'])/row['nextopen'])
            drawdown.append((row['nextlow'] - row['nextopen'])/row['nextopen'])


    del opp_dir[-1]
    del loss[-1]
    del drawdown[-1]
    data['opp_dir'] = np.abs(opp_dir)
    data['loss'] = loss
    data['drawdown'] = drawdown


    if filtering_quantiles:
        print(f'Data filtered for quantiles', filtering_quantiles)
        botq = data.opp_dir.quantile(filtering_quantiles[0])
        topq = data.opp_dir.quantile(filtering_quantiles[1])
        filtered_data = data[botq < data.opp_dir]
        filtered_data = filtered_data[filtered_data.opp_dir < topq]

    else:
        filtered_data = data

    time_dict = {}
    minrange = np.arange(30, 56, int(timeframe[0]))
    minrange = minrange[1:]
    fulltimes = ['08:' + str(c) for c in minrange]

    for time in fulltimes:
        min_ind = filtered_data[filtered_data['Time'] == time].index
        time_dict[time] = filtered_data.loc[min_ind, :]

    print('Data prepared.')
    return time_dict


def mktopen_data_scaled(ticker, timeframe, filtering_quantiles=False):
    data = _load(ticker, timeframe)
    data['Time'] = data.index.strftime('%H:%M')

    for col_name in ['High','Low','Close','Change','Open']:
        data[col_name] = data[col_name]/data['Open']

    TOD = [c[-5:] for c in data['Time']]
    data['TOD'] = TOD
    data['nexthigh'] = data['High'].shift(-1)
    data['nextlow'] = data['Low'].shift(-1)
    data['nextclose'] = data['Close'].shift(-1)
    data['nextopen'] = data['Open'].shift(-1)
    data['span'] = np.abs(data['High'] - data['Low'])
    data['change_frac'] = np.abs(data['Change']/(data['High'] - data['Low']))

    opp_dir = [0]
    loss = [0]
    drawdown = [0]
    topwick = [0]
    botwick = [0]
    oppwick = [0]

    for row in data.iterrows():
        ind = row[0]
        row = row[1]

        if row['Change'] >= 0:
            topwick.append(np.abs(row['High'] - row['Close']))
            botwick.append(np.abs(row['Low'] - row['Open']))
            oppwick.append(np.abs(row['Low'] - row['Open']))
            opp_dir.append((row['nextopen'] - row['nextlow']))
            loss.append((row['nextopen'] - row['nextclose']) )
            drawdown.append((row['nextopen'] - row['nexthigh']))
        else:
            topwick.append(np.abs(row['High'] - row['Open']))
            botwick.append(np.abs(row['Low'] - row['Close']))
            oppwick.append(np.abs(row['High'] - row['Open']))
            opp_dir.append((row['nexthigh'] - row['nextopen']))
            loss.append((row['nextclose'] - row['nextopen']))
            drawdown.append((row['nextlow'] - row['nextopen']))

    del opp_dir[-1]
    del loss[-1]
    del drawdown[-1]
    del topwick[-1]
    del botwick[-1]
    del oppwick[-1]
    data['opp_dir'] = np.abs(opp_dir)
    data['loss'] = loss
    data['drawdown'] = drawdown
    data['topwick'] = topwick
    data['botwick'] = botwick
    data['oppwick'] = oppwick

    if filtering_quantiles:
        print(f'Data filtered for quantiles', filtering_quantiles)
        botq = data.opp_dir.quantile(filtering_quantiles[0])
        topq = data.opp_dir.quantile(filtering_quantiles[1])
        filtered_data = data[botq < data.opp_dir]
        filtered_data = filtered_data[filtered_data.opp_dir < topq]

    else:
        filtered_data = data

    time_dict = {}
    minrange = np.arange(30, 56, int(timeframe[0]))
    minrange = minrange[1:]
    fulltimes = ['08:' + str(c) for c in minrange]

    for time in fulltimes:
        min_ind = filtered_data[filtered_data['Time'] == time].index
        time_dict[time] = filtered_data.loc[min_ind, :]

    print('Data prepared.')
    return time_dict


