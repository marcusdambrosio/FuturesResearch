from binance.client import Client
from binance.websockets import BinanceSocketManager
from binance.enums import *
import urllib3
import datetime as dt
from algo_helpers import get_all_times, get_ema_params
import pandas as pd
import numpy as np
import config
import sys
import time
from functools import partial
urllib3.disable_warnings()


# binance_api_key = '6UYqWk1iWMizbL6b1RA5dQhDkLApulvHq87ekh8bSJO0eo2NB16Mtu9ozTceEIFI'    #Enter your own API-key here
# binance_api_secret = 'RkJDlBnhuqymUuZeWWP9SQGd2zVD1A33GcMUspi1WqGk1QCyb3OTnt0UnMAhiJSI' #Enter your own API-secret here

client = Client(config.API_KEY, config.API_SECRET, tld = 'us') # {'verify' : False, 'timeout' : 20}
# server_time = client.get_server_time()
# sys_status = client.get_system_status()
x = client.get_ticker()
max_vol = {'volume' : '0'}
for ticker in x:
    if float(ticker['volume']) > float(max_vol['volume']):
        max_vol = ticker


# trades = client.get_recent_trades(symbol = 'BNBBTC')
# orders = client.get_all_orders(symbol = 'BNBBTC', limit = 10)


class MyAlgo(Client):

    def __init__(self, tickers_timeframes, TOD_freq):
        Client.__init__(self, config.API_KEY, config.API_SECRET, tld = 'us') # {'verify' : False, 'timeout' : 20}
        self.tickers = []
        self.timeframes = {}
        self.current_price = {}
        self.all_klines = {}
        self.all_ema_params = {}
        self.curr_ema_params = {}
        self.TOD = dt.datetime.today().strftime('%H:%M')
        self.all_timesstr, self.all_timesdt = get_all_times(TOD_freq)
        self.master_ema_df = pd.DataFrame(columns = ['ticker', 'timeframe', 'short', 'long'])
        self.side = {}
        self.positions = {}
        self.orderId_positions = {}
        self.dir = {}
        self.position_scalers = {}
        self.base_position = {}


        for pair in tickers_timeframes:
            if pair[0] not in self.tickers:
                self.tickers.append(pair[0])
            if pair[0] not in self.timeframes.keys():
                self.timeframes[pair[0]] = [pair[1]]
            else:
                self.timeframes[pair[0]].append(pair[1])

        for ticker in self.tickers:
            for timeframe in self.timeframes[ticker]:
                self.positions[ticker + '_' + timeframe] = 0
                curr_ema_dict = get_ema_params(ticker,timeframe)
                # curr_pos_weights = get_position_sizing(ticker, timeframe)
                for TOD in self.all_timesstr:
                    self.all_ema_params[f'{ticker}_{timeframe}_{TOD}'] = curr_ema_dict[TOD]

                    # self.position_scalers[f'{ticker}_{timeframe}_{TOD}'] = curr_pos_weights[TOD]


    def store_price(self, msg, ticker):
        self.current_price[ticker] = msg['p']


    def start_trade_websocket(self, tickers):
        myTradeSocket = BinanceSocketManager(self)
        for ticker in tickers:
            conn_key = myTradeSocket.start_trade_socket(ticker, partial(self.store_price, ticker = ticker))
            myTradeSocket.start()


    def create_order(self, symbol, side, type = 'MKT', TIF = 'GTC', quantity = 5, lim_price = False, margin =  True):
        if margin:
            if type == 'LMT':
                order = self.create_margin_order(symbol=symbol,
                                                    side=side,
                                                    type=type,
                                                    timeInForce=TIF,
                                                    quantity=quantity,
                                                    price = lim_price)
            elif type == 'MKT':
                order = self.create_margin_order(symbol=symbol,
                                                    side=side,
                                                    type=type,
                                                    timeInForce=TIF,
                                                    quantity=quantity)

        else:
            if type == 'LMT':
                order = self.create_margin_order(symbol=symbol,
                                                   side=side,
                                                   type=type,
                                                   timeInForce=TIF,
                                                   quantity=quantity,
                                                   price=lim_price)
            elif type == 'MKT':
                order = self.create_margin_order(symbol=symbol,
                                                   side=side,
                                                   type=type,
                                                   timeInForce=TIF,
                                                   quantity=quantity)
        return order



    def cancel_all_orders(self):
        my_open_orders = self.get_open_orders()
        cancellations = []
        for order in my_open_orders:
            cancellations.append(self.cancel_order(my_open_orders['symbol'], my_open_orders['orderId']))
        return cancellations



    def get_hist_klines(self, ticker, timeframe, periods):
        interval_dict = {'min' : 'm',
                         'hr' : 'h'}
        period_multiplier = ''
        interval_type = ''
        for element in timeframe:
            try:
                period_multiplier += str(int(element))
            except:
                interval_type += element

        start = (dt.datetime.today() - dt.timedelta(minutes = periods * int(period_multiplier))).strftime('%Y/%m/%d %H:%M:%S')


        klines = self.get_historical_klines(ticker, period_multiplier + interval_dict[interval_type], start_str = start, end_str = dt.datetime.today().strftime('%Y/%m/%d %H:%M:%S'))
        kline_data = pd.DataFrame(klines,
                            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av',
                                     'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
        closes = kline_data['close']
        return closes



    def current_ema_params(self):
        for i, t in enumerate(self.all_timesdt):
            if dt.datetime.now().time() >= self.all_timesdt[-1]:
                newTOD = self.all_timesstr[-1]
                break

            elif t <= dt.datetime.now().time() < self.all_timesdt[i + 1]:
                newTOD = self.all_timesstr[i]
                break

        if newTOD != self.TOD:
            self.TOD = newTOD

            for ticker in self.tickers:
                for timeframe in self.timeframes[ticker]:
                    self.curr_ema_params[f'{ticker}_{timeframe}'] = self.all_ema_params[f'{ticker}_{timeframe}_{self.TOD}']




    def update_emas(self):
        self.master_ema_df = pd.DataFrame(columns=['ticker', 'timeframe', 'short', 'long'])
        if not len(self.curr_ema_params):
            self.current_ema_params()

        for ticker in self.tickers:
            for timeframe in self.timeframes[ticker]:
                short, long, take_prof, pos_weight = self.curr_ema_params[f'{ticker}_{timeframe}']
                closes = self.get_hist_klines(ticker, timeframe, periods = max(short,long)).astype(float)

                s_ema = closes[-int(short):].mean()
                l_ema = closes[-int(long):].mean()
                self.master_ema_df = self.master_ema_df.append({'ticker': ticker,
                                                                'timeframe' : timeframe,
                                                                'short' : s_ema,
                                                                'long' : l_ema,
                                                                'take_prof' : take_prof,
                                                                'pos_weight' : pos_weight}, ignore_index = True)
        self.master_ema_df.set_index(self.master_ema_df['ticker'] + '_' + self.master_ema_df['timeframe'], inplace = True)


    def trade_decisions(self):
        if not len(self.master_ema_df):
            print('EMA df initialized')


        if not len(self.dir.keys()):
            for row in self.master_ema_df.iterrows():
                ind = row[0]
                row = row[1]
                self.dir[row['ticker'] + '_' + row['timeframe']] = 1 if row['short'] >= row['long'] else -1
                print('Direction initialized')

        for row in self.master_ema_df.iterrows():
            ind = row[0]
            row = row[1]
            ticker = row['ticker']
            timeframe = row['timeframe']
            curr_dir = 1 if row['short'] >= row['long'] else -1
            print(row['short'], row['long'])
            if curr_dir != self.dir[row['ticker'] + '_' + row['timeframe']]:
                side = 'BUY' if curr_dir >= 0 else 'SELL'
                if f'{ticker}_{timeframe}' not in self.base_position.keys():
                    quantity = row['pos_weight']
                else:
                    quantity = self.base_position[f'{ticker}_{timeframe}'] * row['pos_weight']
                print(f'{side} ORDER FOR {quantity} EXECUTED')
                sys.exit()
                order = self.create_order(ticker, side, 'MKT', quantity = quantity)
                self.side[row[ticker] + '_' + row[timeframe]] = curr_dir
                self.positions[row[ticker] + '_' + row[timeframe]] = quantity if curr_dir > 0 else  -quantity
                self.avg_cost[row[ticker] + '_' + row[timeframe]] = self.current_price[row[ticker]]
                self.orderId_positions[str(order)] = quantity if curr_dir > 0 else -quantity

    def close(self):
        for ticker in self.tickers:
            for timeframe in self.timeframes[ticker]:

                if self.positions[ticker + '_' + timeframe] < 0:
                    if self.avg_cost[ticker] - self.current_price[ticker] > self.master_ema_df[ticker + '_' + timeframe].take_prof:
                        quantity = -self.position[ticker + '_' + timeframe]
                        order = self.create_order(ticker, 'BUY', 'MKT', quantity = quantity)
                        self.positions[row[ticker] + '_' + row[timeframe]] = 0
                        self.avg_cost[row[ticker] + '_' + row[timeframe]] = 0
                        self.orderId_positions[str(order)] = quantity

                elif self.positions[ticker + '_' + timeframe]  > 0:
                    if self.current_price[ticker] - self.avg_cost[ticker] > self.master_ema_df[ticker + '_' + timeframe].take_prof:
                        quantity = self.position[ticker + '_' + timeframe]
                        order = self.create_order(ticker, 'SELL', 'MKT', quantity=quantity)
                        self.positions[row[ticker] + '_' + row[timeframe]] = 0
                        self.avg_cost[row[ticker] + '_' + row[timeframe]] = 0
                        self.orderId_positions[str(order)] = -quantity


app = MyAlgo([['BTCUSDT', '5min']], '30')
app.start_trade_websocket(app.tickers)

time.sleep(1)
print('Algo started...')

update = True
while True:

    if int(dt.datetime.today().strftime('%M'))%5 ==0:
        if update:
            app.update_emas()
            update = False
        app.trade_decisions()

    else:
        update = True


    app.close()