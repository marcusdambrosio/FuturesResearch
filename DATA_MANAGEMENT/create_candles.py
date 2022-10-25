import pandas as pd
import numpy as np
import time
import sys
def makeCandles(ticker, desired_timeframe):
    data = pd.read_csv('C:/NewPycharmProjects/FuturesResearch/DATA/'+ticker[:2]+'/'+ticker + '_1min.csv')
    if 'Last' in data.columns:
        data.rename(columns={'Last': 'Close'}, inplace=True)

    candles = pd.DataFrame(columns=data.columns)
    start_ind = 0
    end_ind = desired_timeframe
    m25 = np.floor(.25*(len(data)/desired_timeframe))
    m10 = np.floor(m25/2.5)
    m50 = m25*2
    m75 = m25*3

    counter = 0
    start = time.time()
    for split_num in np.arange(1, len(data) / desired_timeframe + 1):
        counter+=1
        if counter == m10:
            print('10% completed...')
        if counter == m25:
            print('25% completed...')
        if counter == m50:
            print('50% completed...')
        if counter == m75:
            print('75% completed...')

        split = data.iloc[int(start_ind):int(end_ind), :]
        candle_time = split.Time[start_ind]
        candle_time = candle_time[-5:]
        if candle_time[0] == ' ':
            candle_time = candle_time[1:]

        if int(candle_time[-2:]) % desired_timeframe != 0:
            start_ind += desired_timeframe - int(candle_time[-2:]) % desired_timeframe
            end_ind += desired_timeframe - int(candle_time[-2:]) % desired_timeframe
            continue

        candles = candles.append({'Time': split.Time[start_ind],
                                  'Open': split.Open[start_ind],
                                  'High': split.High.max(),
                                  'Low': split.Low.min(),
                                  'Close': split.Close[end_ind-1],
                                  'Change': split.Close[end_ind-1] - split.Open[start_ind],
                                  'Volume': split.Volume.sum()}, ignore_index=True)

        start_ind += desired_timeframe
        end_ind += desired_timeframe

    # 
    # candles.set_index(candles['Time'], drop=True, inplace=True)
    end = time.time()
    print(f'Made candles for {desired_timeframe} min')
    print(f'The process took {(end-start)/60} minutes.')

    candles.to_csv('C:/NewPycharmProjects/FuturesResearch/DATA/'+ticker[:2]+'/'+ticker + '_' + str(desired_timeframe) + 'min.csv', index=False)

# for tf in [5,15,30,60]:
#     makeCandles('NQ_2010-2020', tf)
makeCandles('NQ_2010-2020', 60)