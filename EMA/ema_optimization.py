import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import time
import sys
from simulation import simulate
from data_prep import prep_data
from numba import jit, cuda
import numpy as np
from timeit import default_timer as timer


def make_period_pairs(short_range=[9, 14, 20, 30, 50], long_range=[20, 30, 50, 100]):
    pairs = []

    for short in short_range:
        for long in long_range:
            if long <= short:
                continue
            else:
                pairs.append([short, long])

    return pairs


def find_best_pair(ticker, timeframe):
    master = pd.DataFrame(columns=['pair', 'pnl', 'num trades'])
    pairs = make_period_pairs()
    for pair in pairs:
        pnl, num = simulate(ticker, timeframe, pair[0], pair[1])
        master = master.append({'pair': pair, 'pnl': pnl, 'num trades': num}, ignore_index=True)

    max = np.max(master['pnl'])

    ind_of_max = (master[master['pnl'] == max].index)
    print(master.iloc[ind_of_max, :])


def optimize_strategy(ticker, timeframe, short_range=[9, 14, 20, 30, 50], long_range=[20, 30, 50, 100],
                      prof_range=[]):
    master = pd.DataFrame(columns=['short', 'long', 'pnl', 'winning', 'losing', 'take prof'])
    pairs = make_period_pairs(short_range, long_range)

    for take_prof in prof_range:

        for pair in pairs:
            pnl, num, buys, sells = simulate(ticker, timeframe, pair[0], pair[1], take_prof)
            master = master.append(
                {'short': pair[0], 'long': pair[1], 'pnl': pnl, 'winning': num[0], 'losing': num[1],
                 'take prof': take_prof}, ignore_index=True)

    master.to_csv(f'C:/NewPycharmProjects/FuturesResearch/DATA/{ticker[:2]}/{ticker}_{timeframe}_optimizeData.csv')
    maxpnl = np.max(master['pnl'])
    ind_of_max = master[master['pnl'] == maxpnl].index
    max = master.iloc[ind_of_max, :]
    print(f'The optimal strategy for {ticker} {timeframe} is:', max)

    return master


def time_specific_master_optimization(data, time, short_range=[9, 14, 20, 30, 50], long_range=[20, 30, 50, 100],
                                      prof_range=[]):
    example_price = data.Open[0]

    master = pd.DataFrame(columns=['time', 'short', 'long', 'pnl', 'winning', 'losing', 'take prof'])
    pairs = make_period_pairs(short_range, long_range)

    for timeframe in timeframes:

        for take_prof in prof_range:

            for pair in pairs:

                try:
                    pnl, num, buys, sells = time_specific_simulate(data, pair[0], pair[1], take_prof)
                    master = master.append({'time': time, 'short': pair[0], 'long': pair[1], 'pnl': pnl,
                                            'winning': num[0], 'losing': num[1], 'take prof': take_prof},
                                           ignore_index=True)

                except:
                    print(f'{ticker} or {timeframe} data not found.')

    maxpnl = np.max(master['pnl'])
    ind_of_max = master[master['pnl'] == maxpnl].index
    max = master.iloc[ind_of_max, :]
    print(f'The optimal strategy for {time} and {ticker} is:', max)

    return master


def find_maxpnl(ticker):
    data = pd.read_csv(f'C:/NewPycharmProjects/FuturesResearch/DATA/{ticker[:2]}/{ticker}_allCombinations.csv')
    maxpnl = np.max(data['pnl'])
    ind_of_max = data[data['pnl'] == maxpnl].index
    maxinfo = data.iloc[ind_of_max, :]
    print(maxinfo)

    return maxinfo


def configure_prof_range(timeframe, price):
    time_to_pct = {'1min': .0015,
                   '5min': .0025,
                   '15min': .0031,
                   '30min': .0035,
                   '60min': .04}

    if timeframe[-1] != 'n':
        timeframe = timeframe[:4]

    pct_prof = time_to_pct[timeframe]
    prof_mid = price * pct_prof
    prof_start = np.floor(prof_mid / 2)
    prof_end = np.floor(prof_mid * 1.5)
    prof_step = np.floor((prof_end - prof_start) / 8)
    prof_range = np.arange(prof_start, prof_end + 1, 1)

    return prof_range


def master_optimization(ticker, timeframes=['1min', '5min', '15min', '30min', '60min'], short_range=[9, 14, 20, 30, 50],
                        long_range=[20, 30, 50, 100], prof_range=[], stop_range = []):

    data = pd.read_csv(f'C:/NewPycharmProjects/FuturesResearch/DATA/{ticker[:2]}/{ticker}_{timeframes[0]}.csv')
    master = pd.DataFrame(columns=['timeframe', 'short', 'long', 'pnl', 'winning', 'losing', 'take_prof', 'stop_loss'])
    pairs = make_period_pairs(short_range, long_range)

    for timeframe in timeframes:
        for take_prof in prof_range:
            for stop_loss in stop_range:
                for pair in pairs:
                        pnl, num, buys, sells = simulate(ticker, timeframe, pair[0], pair[1], take_prof, stop_loss)
                        master = master.append({'timeframe': timeframe, 'short': pair[0], 'long': pair[1], 'pnl': pnl,
                                                'winning': num[0], 'losing': num[1], 'take_prof': take_prof, 'stop_loss': stop_loss},
                                               ignore_index=True)


    master.to_csv(f'C:/NewPycharmProjects/FuturesResearch/DATA/{ticker[:2]}/{ticker}_allCombinations.csv')
    maxpnl = np.max(master['pnl'])
    ind_of_max = master[master['pnl'] == maxpnl].index
    max = master.iloc[ind_of_max, :]
    print(f'The optimal strategy for {ticker} is:', max)

    return master

master_optimization('NQ_2020-2020', ['5min'], prof_range=[.001], stop_range=[50/10000])

def find_best_pnl(ticker):
    data = pd.read_csv(f'C:/NewPycharmProjects/FuturesResearch/DATA/{ticker[:2]}/{ticker}_allCombinations.csv')
    pairs = []

    for row in data.iterrows():
        ind = row[0]
        row = row[1]
        pairs.append((row['short'], row['long']))

    data['pair'] = pairs

    unique_pairs = pd.Series(pairs).unique()

    master = pd.DataFrame(columns=['pair', 'pnl', 'winpct', 'numtrades', 'take_prof'])
    for pair in unique_pairs:
        pair_data = data[data['pair'] == pair]
        max_pnl = pair_data[pair_data['pnl'] == np.max(pair_data['pnl'])]

        winpct = max_pnl['winning'] / (max_pnl['winning'] + max_pnl['losing'])
        numtrades = max_pnl['winning'] + max_pnl['losing']
        master = master.append({'pair': pair,
                                'pnl': max_pnl['pnl'],
                                'winpct': winpct,
                                'numtrades': numtrades,
                                'take_prof': max_pnl['take prof']}, ignore_index=True)

    master.to_csv(f'C:/NewPycharmProjects/FuturesResearch/DATA/{ticker[:2]}/{ticker}_best_pnl_options.csv')
    return master


def find_returns(master, take_prof, stop_loss):
    returns = pd.DataFrame(columns=['type', 'val'])
    for row in master.iterrows():
        ind = row[0]
        row = row[1]

        if row['max_win'] > row['cross_val'] * take_prof:
            returns = returns.append({'type': 1,
                                      'val': row['cross_val'] * take_prof}, ignore_index=True)

        elif stop_loss != None and row['max_loss'] >= stop_loss:
            returns = returns.append({'type': -1,
                                      'val': -row['cross_val'] * stop_loss}, ignore_index=True)

            if row['direction'] == 'up':
                sells.append((1 - stop_loss) * row['cross_val'])

            else:
                buys.append((1 + stop_loss) * row['cross_val'])

        else:
            r = (row['cross_val'] - row['next_cross']) if row['direction'] == 'down' else (
                    row['next_cross'] - row['cross_val'])

            returns = returns.append({'type': 0,
                                      'val': r}, ignore_index=True)


    master['returns_type'] = returns['type']
    master['returns_val'] = returns['val']

    return master


def TOD_optimization(ticker, timeframe, TODfreq='30min', short_range=[9, 14, 20, 30, 50], long_range=[20, 30, 50, 100],
                     prof_range=[], stop_range = []):
    data = pd.read_csv(f'C:/NewPycharmProjects/FuturesResearch/DATA/{ticker[:2]}/{ticker}_{timeframe}.csv')
    example_price = data.Open[0]
    # master = pd.DataFrame(columns=['time_start', 'short', 'long', 'pnl', 'winning', 'losing', 'take prof'])
    pairs = make_period_pairs(short_range, long_range)

    all_times = pd.date_range('00:00:00', '23:59:59', freq=TODfreq)
    all_times = [str(time) for time in all_times]
    all_times = [time[11:] for time in all_times]
    all_timesdt = [dt.time(int(c[:2]), int(c[3:5])) for c in all_times]

    master_df = pd.DataFrame(columns=all_times)
    master_df_index = []
    timeframe_dict = {}

    for t in all_times:
        timeframe_dict[t] = pd.DataFrame()

    for pair in pairs:
        pdata, splits = prep_data(ticker, timeframe, pair[0], pair[1])

        TOD = []
        for t in pdata['time']:
            print(t)
            print(type(t))
            t = t[-5:]

            if t[0] == '0':
                t = dt.time(int(t[1]), int(t[3:]))
            else:
                t = dt.time(int(t[:2]), int(t[3:]))

            if t >= all_timesdt[-1]:
                t = all_timesdt[-1]

            else:
                for i, times in enumerate(all_timesdt):
                    if times < t < all_timesdt[i + 1]:
                        t = times

            TOD.append(t)

        pdata['TOD'] = TOD

        for take_prof in prof_range:
            for stop_loss in stop_range:

                print(f'started {pair}, {take_prof}...')

                pdata = find_returns(pdata, take_prof)

                for tod in all_timesdt:

                    curr_tod = pdata[pdata['TOD'] == tod]

                    if not len(curr_tod):
                        wins, losses, numtrades, winpct, max_drawdown = [0], [0], 0, 0, 0

                    else:
                        wins = curr_tod[curr_tod['returns_type'] == 1].returns_val.tolist()
                        stop_losses = curr_tod[curr_tod['returns_type'] == -1].returns_val.tolist()
                        stop_crosses = curr_tod[curr_tod['returns_type'] == 0].returns_val.tolist()

                        numtrades = len(curr_tod)
                        winpct = len(wins) / len(curr_tod)
                        stoppct = len(stop_losses) / (len(stop_losses) + len(stop_crosses))
                        max_drawdown = np.min(curr_tod['max_loss'])

                    avgwin = np.mean(wins) if len(wins) else 0
                    avgloss = np.mean(stop_losses + stop_crosses) if len(stop_losses) else 0
                    timeframe_dict[str(tod)] = timeframe_dict[str(tod)].append({'pnl': np.sum(curr_tod['returns_val']),
                                                                                'short': pair[0],
                                                                                'long': pair[1],
                                                                                'take_prof': take_prof,
                                                                                'avgwin': avgwin,
                                                                                'avgloss': avgloss,
                                                                                'numtrades': numtrades,
                                                                                'winpct': winpct,
                                                                                'stoppct': stoppct,
                                                                                'drawdown': max_drawdown},
                                                                               ignore_index=True)

    master_df['labels'] = master_df_index
    opti_df = pd.DataFrame(
        columns=['TOD', 'pnl', 'short', 'long', 'take_prof', 'avgwin', 'avgloss', 'numtrades', 'winpct', 'stoppct', 'drawdown',
                 'expected'])

    for tod in all_times:
        curr_df = timeframe_dict[tod]
        maxpnl = np.max(curr_df['pnl'])

        if maxpnl == 0:
            print(f'{tod} not valid')
            continue

        maxind = curr_df[curr_df['pnl'] == maxpnl].index

        if len(maxind) > 1:
            newcurr_df = curr_df.loc[maxind, :].sort_values('drawdown', ascending=True)
            maxind = newcurr_df.index[0]

        short = curr_df.loc[maxind, 'short']
        long = curr_df.loc[maxind, 'long']
        take_prof = curr_df.loc[maxind, 'take_prof']
        avgwin = curr_df.loc[maxind, 'avgwin']
        avgloss = curr_df.loc[maxind, 'avgloss']
        numtrades = curr_df.loc[maxind, 'numtrades']
        winpct = curr_df.loc[maxind, 'winpct']
        stoppct = curr_df.loc[maxind, 'stoppct']
        drawdown = curr_df.loc[maxind, 'drawdown']
        expected = avgwin * winpct + avgloss * (1 - winpct)

        opti_df = opti_df.append({'TOD': str(tod),
                                  'pnl': float(maxpnl),
                                  'short': float(short),
                                  'long': float(long),
                                  'take_prof': float(take_prof),
                                  'avgwin': float(avgwin),
                                  'avgloss': float(avgloss),
                                  'numtrades': float(numtrades),
                                  'winpct': float(winpct),
                                  'stoppct': float(stoppct),
                                  'drawdown': float(drawdown),
                                  'expected': float(expected)}, ignore_index=True)

    if len(short_range) == 1 and len(long_range) == 1:
        opti_df.to_csv(f'C:/NewPycharmProjects/FuturesResearch/DATA/{ticker[:2]}/{ticker}_{timeframe}_{short_range[0]}_{long_range[0]}_{TODfreq}TODopti.csv')

    else:
        opti_df.to_csv(f'C:/NewPycharmProjects/FuturesResearch/DATA/{ticker[:2]}/{ticker}_{timeframe}_{TODfreq}TODopti.csv')


