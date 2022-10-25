import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mkt_open import mktopen_data, mktopen_data_scaled
from helpers.indicators import EMA
import datetime as dt
import time
import sys
import calendar
from matplotlib import style
style.use('ggplot')

def mktopen_avganalysis(ticker, timeframe, filtering_quantiles = [0, .9], graph = False):

    time_dict = mktopen_data(ticker, timeframe, filtering_quantiles)
    allmins_data = pd.DataFrame(columns = ['time', 'avghigh', 'avglow', 'opportunity', 'loss', 'drawdown', 'avgvol'])
    minrange = np.arange(30, 56, int(timeframe[0]))
    minrange = minrange[1:]
    fulltimes = ['08:' + str(c) for c in minrange]

    for time in fulltimes:
        currdata =  time_dict[time]
        allmins_data = allmins_data.append({'time' : time,
                                            'avghigh' : np.mean((currdata['nexthigh'] - currdata['nextopen'])/currdata['nextopen']),
                                            'avglow' : np.mean((currdata['nextlow'] - currdata['nextopen'])/currdata['nextopen']),
                                            'opportunity' : np.mean(currdata['opp_dir']),
                                            'loss' : np.mean(currdata['loss']),
                                            'drawdown' : np.mean(currdata['drawdown']),
                                            'Volume' : np.mean(currdata['Volume'])}, ignore_index = True)

    if graph:

        fig, ax = plt.subplots(2, 2, sharex = True)
        ax[0, 0].bar(fulltimes, allmins_data['avghigh'], color = 'green', label = 'Avg Next High')
        ax[0, 0].bar(fulltimes, allmins_data['avglow'], color = 'red' , label = 'Avg Next Low')
        ax[0, 0].grid(axis = 'y')
        ax[0, 0].legend()
        ax[0, 0].set_title('Next Min Extrema')

        ax[1, 0].bar(fulltimes, allmins_data['opportunity'], color = 'green', label = 'Opportunity')
        ax[1, 0].bar(fulltimes, allmins_data['loss'], color = 'red', label = 'Loss')
        ax[1, 0].grid(axis = 'y')
        ax[1, 0].set_title('Avg Opportunity and Loss')

        ax[0, 1].bar(fulltimes, allmins_data['Volume'])
        ax[0, 1].grid(axis='y')
        ax[0, 1].set_title('Avg Volume')

        ax[1, 1].bar(fulltimes, allmins_data['drawdown'], color = 'red')
        ax[1, 1].grid(axis='y')
        ax[1, 1].set_title('Avg Max Drawdown')
        plt.show()

    return allmins_data

# mktopen_avganalysis('ES_2017-2020', '5min',filtering_quantiles=[0,1], graph=True)

def mktopen_sim(ticker, timeframe, take_prof, stop_loss = False, graph = False):
    time_dict = mktopen_data(ticker, timeframe)
    times = time_dict.keys()
    full_sim_dict = {}

    for key in times:
        full_sim_dict[key] = pd.DataFrame(columns = ['outcome', 'drawdown', 'Volume'])

    for key in times:
        currdata = time_dict[key]

        for row in currdata.iterrows():
            ind = row[0]
            row = row[1]

            if row['opp_dir'] >= take_prof:
                outc = take_prof * row['nextopen']
            elif stop_loss and row['drawdown'] <= -stop_loss:
                outc = -stop_loss * row['nextopen'] - .5
            else:
                outc = row['loss'] * row['nextopen']

            full_sim_dict[key] = full_sim_dict[key].append({'outcome' : outc,
                                                  'drawdown' : row['drawdown'],
                                                  'Volume' : row['Volume']}, ignore_index = True)

    master = pd.DataFrame(columns = ['pnl', 'winpct', 'windrawdown', 'losedrawdown'])

    for key in times:
        currdata = full_sim_dict[key]
        winners = currdata[currdata['outcome'] == take_prof]
        losers = currdata[currdata['outcome'] != take_prof]
        win_drawdown = np.mean(winners['drawdown'])
        loss_drawdown = np.mean(losers['drawdown'])

        pnl = np.sum(currdata['outcome'])
        winpct = len(winners) / len(currdata)

        master = master.append({'pnl' : pnl,
                                'winpct' : winpct,
                                'windrawdown' : win_drawdown,
                                'lossdrawdown' : loss_drawdown}, ignore_index=True)

    if graph:
        fig, ax = plt.subplots(2, 2, sharex = True)
        ax[0, 0].bar(times, master['pnl'], color = 'green')
        ax[0, 0].grid(axis = 'y')
        ax[0, 0].set_title('PNL')

        ax[1, 0].bar(times, master['windrawdown'], color = 'green', width = .5, label = 'Winners')
        ax[1, 0].bar(times, master['lossdrawdown'], color = 'red', width = .25, label = 'Losers')
        ax[1, 0].grid(axis = 'y')
        ax[1, 0].legend()
        ax[1, 0].set_title('Drawdown')

        ax[0, 1].bar(times, master['winpct'])
        ax[0, 1].grid(axis='y')
        ax[0, 1].set_title('Win Pct')

        plt.suptitle(f'Take prof at {take_prof} and stop loss at {stop_loss}')
        plt.show()

    return master


def day_of_week(ticker, timeframe, filtering_quantiles = False):
    time_dict = mktopen_data(ticker, timeframe)
    minrange = np.arange(30, 56, int(timeframe[0]))
    minrange = minrange[1:]
    fulltimes = ['08:' + str(c) for c in minrange]
    wdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    
    for time in fulltimes:
        currdata = time_dict[time]
        weekdays = [calendar.day_name[c.weekday()] for c in currdata.index.tolist()]
        currdata['weekday'] = weekdays
        time_dict[time] = currdata

    master = pd.DataFrame(columns = ['time', 'weekday', 'avghigh', 'avglow', 'opportunity', 'loss', 'drawdown', 'avgvol'])
    for time in fulltimes:
        currdata = time_dict[time]
        for wday in wdays:
            wdaydata = currdata[currdata['weekday'] == wday]
            master = master.append({'time': time,
                                        'weekday': wday,
                                        'avghigh' : np.mean((wdaydata['nexthigh'] - wdaydata['nextopen'])/wdaydata['nextopen']),
                                        'avglow' : np.mean((wdaydata['nextlow'] - wdaydata['nextopen'])/wdaydata['nextopen']),
                                        'opportunity': np.mean((wdaydata['opp_dir'])),
                                        'loss': np.mean(wdaydata['loss']),
                                        'drawdown': np.mean(wdaydata['drawdown']),
                                        'Volume': np.mean(wdaydata['Volume'])}, ignore_index=True)

    fig, ax = plt.subplots(3,4)
    for i, wday in enumerate(wdays):
        if i <= 2:
            plotdata = master[master['weekday'] == wday]
            ax[i,0].bar(fulltimes, plotdata['avghigh'], color='green', label='Avg Next High')
            ax[i,0].bar(fulltimes, plotdata['avglow'], color='red', label='Avg Next Low')
            ax[i,0].grid(axis='y')
            ax[i,0].legend()
            ax[i,0].set_title(wday)

        else:
            plotdata = master[master['weekday'] == wday]
            ax[i-3, 1].bar(fulltimes, plotdata['avghigh'], color='green', label='Avg Next High')
            ax[i-3, 1].bar(fulltimes, plotdata['avglow'], color='red', label='Avg Next Low')
            ax[i-3, 1].grid(axis='y')
            ax[i-3, 1].legend()
            ax[i-3, 1].set_title(wday)

    for i, wday in enumerate(wdays):
        if i == 0:
            plotdata = master[master['weekday'] == wday]
            ax[i+2,1].bar(fulltimes, plotdata['opportunity'], color='blue', label='opportunity')
            # ax[i+2,1].bar(fulltimes, plotdata['loss'], color='orange', label='loss')
            ax[i+2,1].grid(axis='y')
            ax[i+2,1].legend()
            ax[i+2,1].set_title(wday)

        elif 0< i <=3:
            plotdata = master[master['weekday'] == wday]
            ax[i-1, 2].bar(fulltimes, plotdata['opportunity'], color='blue', label='opportunity')
            # ax[i-1, 2].bar(fulltimes, plotdata['loss'], color='orange', label='loss')
            ax[i-1, 2].grid(axis='y')
            ax[i-1, 2].legend()
            ax[i-1, 2].set_title(wday)

        elif i > 3:
            plotdata = master[master['weekday'] == wday]
            ax[i-4, 3].bar(fulltimes, plotdata['opportunity'], color='blue', label='opportunity')
            # ax[i - 4, 3].bar(fulltimes, plotdata['loss'], color='orange', label='loss')
            ax[i-4, 3].grid(axis='y')
            ax[i-4, 3].legend()
            ax[i-4, 3].set_title(wday)

    plt.show()


def wick_vis(ticker, timeframe):
    time_dict = mktopen_data_scaled(ticker, timeframe)

    master = pd.DataFrame()
    for d in time_dict.values():
        master = master.append(d)

    fig,ax = plt.subplots(3)

    for i, wick in enumerate(['topwick', 'botwick', 'oppwick']):
        m,b = np.polyfit(master[wick], master['opp_dir'], 1)
        ax[i].scatter(master[wick], master['opp_dir'], label = 'data')
        ax[i].plot(master[wick], m*master[wick] + b, label = f'fit line, slope = {m}')
        ax[i].set_title(wick)
        ax[i].legend()
    plt.show()

def change_frac_vis(ticker, timeframe):
    time_dict = mktopen_data_scaled(ticker, timeframe)
    master = pd.DataFrame()
    for d in time_dict.values():
        master = master.append(d)

    m,b = np.polyfit(master['change_frac'], master['opp_dir'], 1)
    plt.scatter(master['change_frac'], master['opp_dir'], label = 'data')
    plt.plot(master['change_frac'], master['change_frac']*m + b, label = m)
    plt.title('Change Fraction')
    plt.legend()
    plt.show()
    
change_frac_vis('nqu20','1min')