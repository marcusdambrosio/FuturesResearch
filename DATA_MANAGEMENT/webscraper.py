from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys
import time
from selenium.webdriver.support.select import Select
import datetime as dt
import os
import pandas as pd
import sys

def scrape_data(email, password, ticker, period, year, timeframe = '1'):
    DRIVER_PATH = '/path/to/chromedriver'
    driver = webdriver.Chrome(ChromeDriverManager().install())
    period_start_dict = {'H': f'12/19/{(int(year) - 1)}',
                         'M': f'03/19/{year}',
                         'U': f'06/19/{year}',
                         'Z': f'09/19/{year}'}
    period_end_dict = {'H': f'03/19/{(year)}',
                         'M': f'06/19/{year}',
                         'U': f'09/19/{year}',
                         'Z': f'12/19/{year}'}

    tickerpath = ticker+period+year[-2:]
    driver.get(f'https://www.barchart.com/futures/quotes/{tickerpath}/historical-download')
    loginButton = driver.find_element_by_class_name('bc-user-block__button').click()
    time.sleep(2)

    try:
        usernameInput = driver.find_element_by_name('email')
        usernameInput.send_keys(email)
        passwordInput = driver.find_element_by_name('password')
        passwordInput.send_keys(password)
        passwordInput.send_keys(Keys.ENTER)
    except:
        raise ValueError('PASSWORD OR USERNAME INCORRECT')
        sys.exit()


    time.sleep(3)

    timeframeDropdown = Select(driver.find_element_by_xpath(
        '//*[@id="main-content-column"]/div/div[2]/div/div[1]/div[1]/div[2]/div[1]/select'))
    timeframeDropdown.select_by_visible_text('Intraday')

    time.sleep(1)
    
    minInput = driver.find_element_by_xpath(
        '//*[@id="main-content-column"]/div/div[2]/div/div[1]/div[1]/div[2]/div[2]/div/input')
    minInput.clear()
    minInput.send_keys(timeframe)
    
    startDate = period_start_dict[period]
    startDate = dt.date(year = int(startDate[-4:]), month = int(startDate[:2]), day = int(startDate[3:5]))

    endDate = period_end_dict[period]
    endDate = dt.date(year= int(endDate[-4:]), month= int(endDate[:2]), day= int(endDate[3:5]))

    startdateInput = driver.find_element_by_xpath(
        '//*[@id="main-content-column"]/div/div[2]/div/div[1]/div[2]/div[1]/form/div[1]/input')
    enddateInput = driver.find_element_by_xpath(
        '//*[@id="main-content-column"]/div/div[2]/div/div[1]/div[2]/div[1]/form/div[3]/input')
    downloadButton = driver.find_element_by_xpath(
        '//*[@id="main-content-column"]/div/div[2]/div/div[1]/div[2]/a')

    curr_end = endDate
    curr_start = (endDate - dt.timedelta(days=7))

    while curr_start > startDate:
        if dt.date.today() < curr_end:
            curr_end = dt.date.today()
            curr_start = dt.date.today() - dt.timedelta(days = 7)
        startdateInput.clear()
        startdateInput.send_keys(curr_start.strftime('%m/%d/%Y'))
        enddateInput.clear()
        enddateInput.send_keys(curr_end.strftime('%m/%d/%Y'))
        curr_end = curr_start - dt.timedelta(days = 1)
        curr_start = curr_start - dt.timedelta(days = 7)

        downloadButton.click()
        time.sleep(5)

    startdateInput.clear()
    startdateInput.send_keys(startDate.strftime('%m/%d/%Y'))
    enddateInput.clear()
    enddateInput.send_keys(curr_end.strftime('%m/%d/%Y'))
    downloadButton.click()
    time.sleep(3)


def combine_csvs(ticker, period, year):
    tickerpath = ticker.lower()+period.lower()+year[-2:]
    allDownloads = os.listdir("C:/Users/Marcus/Downloads")
    relevant_files = [c for c in allDownloads if c[:5] == tickerpath and dt.datetime.today().strftime('%m-%d-%Y') in c]
    relevant_files_ordered = []
    curr_num = 0
    order_dict = {}
    for i in range(len(relevant_files)):
        order_dict[f'{i}'] = None

    for f in relevant_files:
        if ')' not in f:
            order_dict['0'] = f
        else:
            if f[-7] == '(':
                key = f[-6]
            else:
                key = f[-7:-5]
            order_dict[f'{key}'] = f

    relevant_files_ordered = [c for c in order_dict.values()]
    combined_csv = pd.concat([pd.read_csv('C:/Users/Marcus/Downloads/'+f) for f in relevant_files_ordered])

    #remove bad lines and reverse order
    combined_csv = combined_csv[~combined_csv.Time.str.contains("Downloaded")]
    combined_csv = combined_csv.iloc[::-1]

    if ticker not in os.listdir('C:/NewPycharmProjects/FuturesResearch/DATA'):
        os.mkdir(f'C:/NewPycharmProjects/FuturesResearch/DATA/{ticker}')
    combined_csv.to_csv(f'C:/NewPycharmProjects/FuturesResearch/DATA/{ticker}/{tickerpath}_1min.csv', index=False, encoding='utf-8-sig')


def full_scrape(email, password, tickers, periods, years):
    for ticker in tickers:
        for period in periods:
            for year in years:
                scrape_data(email, password, ticker, period, year)
                combine_csvs(ticker, period, year)
                print(f'Done with {ticker+period+year}.')


full_scrape('baconboy58@gmail.com','basketball34',['NQ'], ['H','M','U','Z'], ['2020'])