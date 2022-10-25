import pandas as pd
import sys
import datetime as dt


order = ['H', 'M', 'U', 'Z']

def roll_data(ticker, years):
    master_list = []
    int_years = sorted([int(c) for c in years])
    years = [str(c) for c in int_years]
    
    counter = 0
    for year in years:
        print('on year ', year)
        for o in order:
            path = ticker.lower() + o.lower() + year[-2:]
            try:
                data = pd.read_csv(f'C:/NewPycharmProjects/FuturesResearch/DATA/{ticker}/{path}_1min.csv')
            except:
                print(f'NO DATA FOR {path}')
                continue

            master_list.append(data)


    master_df = pd.concat(master_list)
    master_df = master_df[~master_df.Time.str.contains("Downloaded")]
    master_df.drop_duplicates('Time', keep = 'first', inplace=True)

    time = master_df['Time'].tolist()
    dt_time = []
    for t in time:
        year = int(t[6:10])
        mo = int(t[:2])
        day = int(t[3:5])
        hr = int(t[-5:-3])
        min = int(t[-2:])
        dt_time.append(dt.datetime(year = year, month = mo, day = day, hour = hr, minute = min))

    master_df['dt_time'] = dt_time
    master_df.sort_values('dt_time', axis = 0,inplace = True)
    master_df.drop('dt_time', axis = 1, inplace = True)
    master_df = master_df.set_index(master_df['Time'], drop = False)
    master_df.to_csv(f'C:/NewPycharmProjects/FuturesResearch/DATA/{ticker}/{ticker}_{years[0]}-{years[-1]}_1min.csv')
    return master_df

roll_data('NQ', ['2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020'])