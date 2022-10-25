import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np


def first_thirty_structure(ticker, timeframe, candleSize):
    filepath = f'C:/NewPycharmProjects/FuturesResearch/DATA/{ticker}/{ticker}_{timeframe}_{candleSize}min.csv'
    data = pd.read_csv(filepath, index_col= 'Time')
    openIndex = [c for c in data.index if '08:30' in c]
    endIndex = [c for c in data.index if '09:00' in c]
    indexPairs = []

    for i , item in enumerate(openIndex):
        date = item[:10]
        end = [c for c in endIndex if date in c][0]
        indexPairs.append([item, end])

    fig, ax = plt.subplots(2,2)

    index = [f'08:{c}' for c in np.arange(30,60, candleSize)]
    index.append('09:30')
    for pair in indexPairs:
        col = 'Close' if 'Close' in data.columns else 'Last'
        plotData = data.loc[pair[0]:pair[1], col] / data.loc[pair[0], 'Open']
        if len(plotData)!=len(index):
            continue
        ax[0,0].plot(index, plotData, alpha = .25, color = 'orange')
        ax[0,0].set_title('all days')

        if plotData[-1] > 1:
            ax[1,0].plot(index, plotData, alpha = .25, color = 'green')
            ax[1,0].set_title('green days')
        else:
            ax[1, 1].plot(index, plotData, alpha=.25, color='red')
            ax[1, 1].set_title('red days')
    plt.show()


def response_to_openvol(ticker, timeframe):
    filepath = f'C:/NewPycharmProjects/FuturesResearch/DATA/{ticker}/{ticker}_{timeframe}_5min.csv'
    data = pd.read_csv(filepath, index_col= 'Time')
    openIndex = [c for c in data.index if '08:30' in c]
    endIndex = [c for c in data.index if '09:00' in c]
    indexPairs = []

    for i , item in enumerate(openIndex):
        date = item[:10]
        end = [c for c in endIndex if date in c][0]
        indexPairs.append([item, end])

    fig, ax = plt.subplots(2)

    index = [f'08:{c}' for c in np.arange(35,60, 5)]
    index.append('09:00')
    for pair in indexPairs:
        col = 'Close' if 'Close' in data.columns else 'Last'
        plotData = data.loc[pair[0]:pair[1], col] / data.loc[pair[0], 'Open']
        plotData.drop(pair[0], axis = 0, inplace = True)

        if len(plotData)!=len(index):
            continue
        if data.loc[pair[0], 'Change'] > .005:
            ax[0].plot(index, plotData, alpha = .25, color = 'blue')
            ax[0].set_title('Green open')
        elif data.loc[pair[0], 'Change'] < -.005:
            ax[1].plot(index, plotData, alpha=.25, color='blue')
            ax[1].set_title('Red open')

    plt.show()

response_to_openvol('NQ', '2020-2020')