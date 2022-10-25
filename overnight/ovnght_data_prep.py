import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import time
import datetime as dt
style.use('ggplot')

def _load(ticker, timeframe):

    data = pd.read_csv(f'{ticker}_{timeframe}.csv')
    data = data.sort_index(axis=0, ascending=False)
    data = data.loc[:, :'Volume']

    # print(data.isna().values.sum(), 'NaN values found')
    data.dropna(axis=0, inplace=True)
    data.set_index(data['Time'], inplace=True)

    data.rename(columns={'Last': 'Close'}, inplace=True)

    return data


def make_candles(ticker, desired_timeframe):
    data = pd.read_csv(ticker+'_1min.csv')
    if 'Last' in data.columns:
        data.rename(columns = {'Last':'Close'}, inplace = True)

    candles = pd.DataFrame(columns=data.columns)
    start_ind = 0
    end_ind = desired_timeframe-1

    for split_num in np.arange(1, len(data)/desired_timeframe + 1):
        split = data.iloc[int(start_ind):int(end_ind), :]
        candle_time = split.Time[0]
        candle_time = candle_time[-5:]
        if candle_time[0] == ' ':
            candle_time = candle_time[1:]

        if int(candle_time[-2:])%desired_timeframe != 0:

            start_ind+=desired_timeframe - int(candle_time[-2:])%desired_timeframe
            end_ind+=desired_timeframe - int(candle_time[-2:])%desired_timeframe
            continue
        print(split.Time[0])

        candles = candles.append({'Time': split.Time[0],
                                  'Open':split.Open[0],
                                  'High':split.High.max(),
                                  'Low': split.Low.min(),
                                  'Close': split.Close[-1],
                                  'Change': split.Close[-1] - split.Open[0],
                                  'Volume': split.Volume.sum()}, ignore_index=True)

        start_ind+=desired_timeframe
        end_ind+=desired_timeframe

    # 
    # candles.set_index(candles['Time'], drop=True, inplace=True)

    candles.to_csv(ticker+'_'+str(desired_timeframe)+'min.csv', index = False)


# def prep_data(ticker):
#     min1 = pd.read_csv(f'C:/NewPycharmProjects/FuturesResearch/DATA/{ticker[:2]}/' + ticker+'_1min.csv')
#     min3 = pd.read_csv(f'C:/NewPycharmProjects/FuturesResearch/DATA/{ticker[:2]}/' + ticker+'_3min.csv')
#     min5 = pd.read_csv(f'C:/NewPycharmProjects/FuturesResearch/DATA/{ticker[:2]}/' + ticker+'_5min.csv')
#     min15 = pd.read_csv(f'C:/NewPycharmProjects/FuturesResearch/DATA/{ticker[:2]}/' + ticker + '_15min.csv')
#     min30 = pd.read_csv(f'C:/NewPycharmProjects/FuturesResearch/DATA/{ticker[:2]}/' + ticker + '_30min.csv')
#     min60 = pd.read_csv(f'C:/NewPycharmProjects/FuturesResearch/DATA/{ticker[:2]}/' + ticker + '_60min.csv')
#
#
#     for df in [min1, min3, min5, min15, min30, min60]:
#         df.set_index(df['Time'], inplace = True)
#         df.rename(columns={'Last': 'Close'}, inplace=True)
#
#
#     all_dates = []
#     for day in [c[:10] for c in min1.index]:
#         if day not in all_dates:
#             if f'{day} 14:59' and f'{day} 08:30' in min1.index:
#                 if day[:5] == '12/24':
#                     pass
#                 else:
#                     all_dates.append(day)
#
#     del all_dates[-1]
#     master = pd.DataFrame(columns = ['day', 'prev_close', 'next_open', '1min', '3min', '5min', '15min', '30min', '60min'])
#
#     for i, day in enumerate(all_dates):
#         # if day[-1] == ' ':
#         #     day = day[:-1]
#
#         # if f'{day} 14:59' not in min1.index:
#         #     continue
#
#
#         try:
#             close = min1.loc[f'{day} 14:59', 'Close']
#             open = min1.loc[f'{day} 08:30', 'Open']
#             ovnt_range = np.abs(np.max(min1.loc[f'{day} 14:59': f'{all_dates[i+1]} 08:30', 'High']) - np.min(min1.loc[f'{day} 14:59': f'{all_dates[i+1]} 08:30', 'Low']))
#             candle1 = min1.loc[f'{day} 08:30', :]
#             candle3 = min3.loc[f'{day} 08:30', :]
#             candle5 = min5.loc[f'{day} 08:30', :]
#             candle15 = min15.loc[f'{day} 08:30', :]
#             candle30 = min30.loc[f'{day} 08:30', :]
#             candle60 = min60.loc[f'{day} 08:00', :]
#
#         except:
#             print('Threw out ' , day)
#             ovnt_range = 0
#             continue
#
#         master = master.append({'day' : day,
#                                 'prev_close' : close,
#                                 'next_open' : open,
#                                 '1min' : candle1,
#                                 '3min' : candle3,
#                                 '5min' : candle5,
#                                 '15min' : candle15,
#                                 '30min' : candle30,
#                                 '60min' : candle60}, ignore_index=True)
#
#
#     # master.set_index(all_dates, inplace = True)
#     master['prev_close'] = master['prev_close'].shift(1)
#     master['ovnt_change'] = (master['next_open'] - master['prev_close'])/master['prev_close']*100
#     master = master.iloc[1:, :]
#     master.to_csv(ticker+'all_timeframe.csv')
#     return master

def prep_data(ticker, timeframe):
    min1 = pd.read_csv(f'C:/NewPycharmProjects/FuturesResearch/DATA/{ticker[:2]}/' + ticker+'_1min.csv')
    # min3 = pd.read_csv(f'C:/NewPycharmProjects/FuturesResearch/DATA/{ticker[:2]}/' + ticker+'_3min.csv')
    # min5 = pd.read_csv(f'C:/NewPycharmProjects/FuturesResearch/DATA/{ticker[:2]}/' + ticker+'_5min.csv')
    # min15 = pd.read_csv(f'C:/NewPycharmProjects/FuturesResearch/DATA/{ticker[:2]}/' + ticker + '_15min.csv')
    # min30 = pd.read_csv(f'C:/NewPycharmProjects/FuturesResearch/DATA/{ticker[:2]}/' + ticker + '_30min.csv')
    # min60 = pd.read_csv(f'C:/NewPycharmProjects/FuturesResearch/DATA/{ticker[:2]}/' + ticker + '_60min.csv')
    mindata = pd.read_csv(f'C:/NewPycharmProjects/FuturesResearch/DATA/{ticker[:2]}/' + ticker + f'_{timeframe}.csv')


    mindata.set_index(mindata['Time'], inplace = True)
    mindata.rename(columns={'Last': 'Close'}, inplace=True)
    min1.set_index(min1['Time'], inplace=True)
    min1.rename(columns={'Last': 'Close'}, inplace=True)

    all_dates = []
    for day in [c[:10] for c in min1.index]:
        if day not in all_dates:
            if f'{day} 14:59' and f'{day} 08:30' in min1.index:
                if day[:5] == '12/24':
                    pass
                else:
                    all_dates.append(day)

    del all_dates[-1]

    master = pd.DataFrame(columns = ['day', 'prev_close', 'next_open', '1min', '3min', '5min', '15min', '30min', '60min'])

    for i, day in enumerate(all_dates):
        # if day[-1] == ' ':
        #     day = day[:-1]

        # if f'{day} 14:59' not in min1.index:
        #     continue


        try:
            close = min1.loc[f'{day} 14:59', 'Close']
            open = min1.loc[f'{day} 08:30', 'Open']
            ovnt_range = np.abs(np.max(min1.loc[f'{day} 14:59': f'{all_dates[i+1]} 08:30', 'High']) - np.min(min1.loc[f'{day} 14:59': f'{all_dates[i+1]} 08:30', 'Low']))
            if timeframe != '60min':
                candle = mindata.loc[f'{day} 08:30', :]
            else:
                candle = mindata.loc[f'{day} 08:00', :]

        except:
            print('Threw out ' , day)
            ovnt_range = 0
            continue

        master = master.append({'day' : day,
                                'prev_close' : close,
                                'next_open' : open,
                                f'{timeframe}' : candle}, ignore_index=True)


    # master.set_index(all_dates, inplace = True)
    master['prev_close'] = master['prev_close'].shift(1)
    master['ovnt_change'] = (master['next_open'] - master['prev_close'])/master['prev_close']*100
    master = master.iloc[1:, :]
    master.to_csv(ticker+'all_timeframe.csv')
    return master


def reversal_analysis(ticker, timeframe, graph = False):
    if type(timeframe) != str:
        timeframe = str(timeframe) + 'min'
    # data = pd.read_csv(ticker+'all_timeframe.csv')
    data = prep_data(ticker, timeframe)
    max_reverse = []
    on_close_reverse = []
    max_same = []
    on_close_same = []
    on_close = []
    for row in data.iterrows():
        ind = row[0]
        row = row[1]
        tf_candle = row[timeframe]
        if row['ovnt_change'] > 0:
            max_reverse.append(row['next_open'] - row[timeframe].Low)
            max_same.append(row[timeframe].High - row['next_open'])

            on_close.append(row['next_open'] - row[timeframe].Close)
            # if row[timeframe].Close < row['next_open']:
            #     on_close_reverse.append(row['next_open'] - row[timeframe].Close)
            # else:
            #     on_close_same.append(row[timeframe].Close - row['next_open'])

            
        else:
            max_reverse.append(row[timeframe].High - row['next_open'])
            max_same.append(row['next_open'] - row[timeframe].Low)

            on_close.append(row[timeframe].Close - row['next_open'])
            # if row[timeframe].Close < row['next_open']:
            #     on_close_reverse.append(row['next_open'] - row[timeframe].Close)
            # else:
            #     on_close_same.append(row['next_open'] - row[timeframe].Close)
            #

            
    if graph:
        fig, ax = plt.subplots(2, sharex = True)

        ax[0].scatter(data['ovnt_change'], max_reverse, color = 'green', label = 'MAX REVERSE', alpha = .5)
        ax[0].scatter(data['ovnt_change'], max_same, color = 'red', label = 'MAX SAME', alpha = .5)
        ax[1].scatter(data['ovnt_change'], on_close, label = 'CLOSE', alpha = .5)
        ax[0].set_title('MAX')
        ax[1].set_title('ON CLOSE')
        ax[0].legend()
        ax[1].legend()
        plt.show()

    return [np.mean(max_reverse), np.mean(max_same), np.mean(on_close)]

reversal_analysis('NQ_2016-2020', '15min', graph=True)


def all_timeframe_analysis(ticker, timeframes):
    reverse,same,close = [], [], []
    for timeframe in timeframes:
        r,s,c= reversal_analysis(ticker, timeframe)
        reverse.append(r), same.append(s), close.append(c)

    fig, ax = plt.subplots(2,2)
    ax[0,0].bar(timeframes, reverse, label = 'Reverse')
    ax[0, 1].bar(timeframes, same, label='Same')
    ax[1, 0].bar(timeframes, close, label='Close')

    ax[0, 0].set_title('Reverse')
    ax[0, 1].set_title('Same')
    ax[1, 0].set_title('Close')

    plt.show()

# def prep_sim_data(ticker, timeframe):
#     data = pd.read_csv(ticker+'_'+timeframe+'.csv')
#     data.set_index(df['Time'], inplace=True)
#
#     all_dates = []
#     for day in [c[:9] for c in min1['Time']]:
#         if day not in all_dates:
#             if f'{day} 14:59' and f'{day} 8:30' in min1.index:
#                 all_dates.append(day)
#
#     del all_dates[-1]
#
#     master = pd.DataFrame(columns=['day', 'prev_close', 'next_open', '1min', '3min', '5min', '15min', '30min', '60min'])
#
#     for i, day in enumerate(all_dates):
#         if day[-1] == ' ':
#             day = day[:-1]

    
def simulate(ticker, timeframe, take_prof, stop_loss = False):
    data = prep_data('NQU20').reset_index()
    days = data['day']
    candles = data[timeframe]
    
    master = pd.DataFrame(columns = ['day', 'return', 'drawdown', 'win'])

    for i, candle in enumerate(candles):
        if data.iloc[i, -1]>0:
            if np.abs(candle['Low'] - candle['Open']) > take_prof:
                master = master.append({'day': days[i],
                                        'return': take_prof,
                                        'drawdown': candle['High'] - candle['Open'],
                                        'win': 1}, ignore_index=True)
            else:
                master = master.append({'day': days[i],
                                        'return': candle['Open'] - candle['Close'],
                                        'drawdown': candle['Open'] - candle['High'],
                                        'win': 0}, ignore_index=True)
        else:
            if np.abs(candle['High'] - candle['Open']) > take_prof:
                master = master.append({'day': days[i],
                                        'return': take_prof,
                                        'drawdown':  candle['Open'] - candle['Low'],
                                        'win': 1}, ignore_index=True)
            else:
                master = master.append({'day': days[i],
                                        'return': candle['Close'] - candle['Open'],
                                        'drawdown': candle['Low'] - candle['Open'],
                                        'win': 0}, ignore_index=True)

    return master


def optimize(ticker, timeframe, prof_range = [], stop_rang = []):
    master = pd.DataFrame(columns = ['prof', 'pnl', 'max_drawdown', 'win_pct'])
    for prof in prof_range:
        curr = simulate(ticker, timeframe, prof)

        master = master.append({'prof': prof,
                                'pnl': curr['return'].sum(),
                                'max_drawdown': curr['drawdown'].min(),
                                'win_pct': curr['win'].sum()/len(curr)*100,
                                'num_trades': len(curr)}, ignore_index=True)


    max_pnl = master[master['pnl'] == master['pnl'].max()]

    print(f'BEST STRAT: \n'
          f'TP: {max_pnl.prof} \n'
          f'PNL: {max_pnl.pnl} \n'
          f'Max Drawdown: {max_pnl.max_drawdown} \n'
          f'Win PCT: {max_pnl.win_pct}% \n'
          f'Num Trades: {max_pnl.num_trades}')

    return master


def reversal_times(ticker, graph = True):
    data = prep_data(ticker)
    all_data = pd.read_csv(ticker+'_1min.csv')
    all_data.set_index(all_data['Time'], inplace = True)
    times =[]
    for row in data.iterrows():
        ind = row[0]
        row = row[1]
        day = row['day']
        
        curr = all_data.loc[f'{day} 6:30': f'{day} 8:30', :]

        if row['ovnt_change']>0:
            max_ind = curr[curr['High'] == curr['High'].max()].index
            times.append(curr.loc[max_ind, 'Time'])

        else:
            max_ind = curr[curr['Low'] == curr['Low'].max()].index
            times.append(curr.loc[max_ind, 'Time'])

    if graph:
        graph_times = []
        new_times = []
        for time in times:
            time = time[0]
            time = time[-5:]
            graph_times.append(dt.time(hour = int(time[:2]), minute = int(time[-2:])))

        graph_times = sorted(graph_times)
        graph_times = [c.strftime('%H:%M') for c in graph_times]
        plt.hist(graph_times, bins = 60)
        plt.show()





