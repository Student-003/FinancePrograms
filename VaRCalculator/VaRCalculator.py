'''

    File Name: VaRCalculator.py
    Author: Connor Creek
    Date Created: 26 March 2019
    Date Modified: 26 March 2019
    Information:

'''
# Environment variables
import configparser
# Execution statistics
import profile
import pstats
# Data manipulation
import numpy as np
import pandas as pd
# Plotting
import matplotlib.pyplot as plt
import seaborn
import matplotlib.mlab as mlab
from matplotlib import interactive
# Statistical calculation
import scipy.stats
# Tabular data output
from tabulate import tabulate
from pprint import pprint
# Import stock price data as CSV
from alpha_vantage.timeseries import TimeSeries

def plot_candles(df, title=None, volumeBars=False, colorFunction=None, overlays=None):
    def default_color(index, open, close, low, high):
        return 'r' if open[index] > close[index] else 'g'
    colorFunction = colorFunction or default_color
    overlays = overlays or []
    open = df['Open']
    close = df['Close']
    low = df['Low']
    high = df['High']
    ocMin = pd.concat([open, close], axis=1).min(axis=1)
    ocMax = pd.concat([open, close], axis=1).max(axis=1)

    subplotCount = 1
    if volumeBars:
        subplotCount = 2

    if subplotCount == 1:
        fig, ax1 = plt.subplots(1, 1)
    else:
        ratios = np.insert(np.full(subplotCount - 1, 1), 0, 3)
        fig, subplots = plt.subplots(subplotCount, 1, sharex=True, gridspec_kw={'height_ratios': ratios})
        ax1 = subplots[0]

    if title:
        ax1.set_title(title)
    x = np.arange(len(df))
    candleColors = [colorFunction(i, open, close, low, high) for i in x]
    candles = ax1.bar(x, ocMax-ocMin, bottom=ocMin, color=candleColors, linewidth=0)
    lines = ax1.vlines(x, low, high, color=candleColors, linewidth=1)
    ax1.xaxis.grid(False)
    ax1.xaxis.set_tick_params(which='major', length=1.5, direction='in', top=False)
    # Assume minute frequency if first two bars are in the same day.
    frequency = 'minute' if (df.index[1] - df.index[0]).days == 0 else 'day'
    timeFormat = '%d-%m'
    if frequency == 'minute':
        timeFormat = '%H:%M'

    # Set X axis tick labels.
    plt.xticks(x, [date.strftime(timeFormat) for date in df.index], rotation='vertical')
    for overlay in overlays:
        ax1.plot(x, overlay)
    # Plot volume bars if needed
    if volumeBars:
        ax2 = subplots[1]
        volume = df['Volume']
        volumeScale = None
        scaledVolume = volume
        if volume.max() > 1000000:
            volumeScale = 'M'
            scaledVolume = volume / 1000000
        elif volume.max() > 1000:
            volumeScale = 'K'
            scaledVolume = volume / 1000
        ax2.bar(x, scaledVolume, color=candleColors)
        volumeTitle = 'Volume'
        if volumeScale:
            volumeTitle = 'Volume (%s)' % volumeScale
        ax2.set_title(volumeTitle)
        ax2.xaxis.grid(False)


def vizualize(tickerSymbol, df, candleDf, mean, sigma, tdf, tmean, tsigma):
    # Vizualize data
    interactive(True)

    # Plot candle chart
    #if (df.index[0] - df.index[-1]).days > 30
    candleTitle = tickerSymbol + " Price Chart"

    if (df.index[-1] - df.index[0]).days < 60:
        plot_candles(candleDf, title=candleTitle, volumeBars=True)
    else:
        plt.figure(1)
        df['Close'].plot()
        plt.title(tickerSymbol + " Price Chart", weight='bold');

    # Plot daily returns
    plt.figure(2)
    df['Returns'].plot()
    plt.title(tickerSymbol + " Daily Returns", weight='bold');

    # Plot histogram & returns distribution (norm)
    plt.figure(3)
    df['Returns'].hist(bins=50, density=True, histtype='stepfilled', alpha=0.5)
    x = np.linspace(df['Returns'].min(), df['Returns'].max(), 100)
    plt.plot(x, scipy.stats.norm.pdf(x, mean, sigma), 'r')
    plt.figtext(0.7, 0.8, "μ = {:.3}".format(mean))
    plt.figtext(0.7, 0.75, "σ = {:.3}".format(sigma))
    plt.title(tickerSymbol + " Daily Return Distribution (norm) ", weight='bold');
    plt.xlabel('Returns')
    plt.ylabel('Frequency')

    # Plot histogram & returns distribution (t)
    plt.figure(4)
    df['Returns'].hist(bins=50, density=True, histtype='stepfilled', alpha=0.5)
    x = np.linspace(df['Returns'].min(), df['Returns'].max(), 100)
    plt.plot(x, scipy.stats.t.pdf(x, loc=tmean, scale=tsigma, df=tdf), 'r')
    # plt.plot(x, scipy.stats.norm.pdf(x, mean, sigma), 'r')
    plt.figtext(0.7, 0.8, "tμ = {:.3}".format(tmean))
    plt.figtext(0.7, 0.75, "tσ = {:.3}".format(tsigma))
    plt.figtext(0.7, 0.7, "df = {:.3}".format(tdf))
    plt.title(tickerSymbol + " Daily Return Distribution (t)", weight='bold');
    plt.xlabel('Returns')
    plt.ylabel('Frequency')


    interactive(False)
    plt.show()

def pull_data(tickerSymbol):
    # Find API key
    print('...Fetching data from Alpha Vantage API...')
    parser = configparser.ConfigParser()
    foundFile = parser.read(['./resources/secret.ini', './resources/template_secret.ini'])
    apiKey = parser.get('API_KEYS','ALPHA_VANTAGE') if './resources/secret.ini' in foundFile else parser.get('API_KEYS_TEMPLATE','ALPHA_VANTAGE')
    if (apiKey == 'guest'):
        print('\n\nNOTE: You are using the demo API Key. Please request an Alpha Vantage API key from https://www.alphavantage.co\n\n')
    # Fetch price data
    # Alpha Vantage documentation and free API key can be found at https://www.alphavantage.co
    ts = TimeSeries(key=apiKey, output_format='pandas')
    data, meta_data = ts.get_daily(symbol=tickerSymbol, outputsize='full')
    outFile = "./resources/" + meta_data['2. Symbol'] + "PriceData.csv"
    data.to_csv(outFile)
    print('\n\nInitial Data:')
    pprint(data.head())
    return data
def checkInDataset(startYear, startMonth, startDay, endYear, endMonth, endDay):
        errorMessage = '\nError: the {:s} of the start date you have entered is after the {:s} of the end date given\n\n'

        # Check that the start date is before the end date
        if (int(startYear) > int(endYear)):
            print(errorMessage.format('year', 'year'))
            exit()
        elif (int(startYear) > int(endYear) and int(startMonth) > int(endMonth)):
            print(errorMessage.format('month', 'month'))
            exit()
        elif (int(startYear) > int(endYear) and int(startMonth) > int(endMonth) and int(startDay) > int(endDay)):
            print(errorMessage.format('day', 'day'))
            exit()

def validateDate (dirtyData, startDate, endDate):
    daysShift = 0
    start = startDate.split('-')
    startYear, startMonth, startDay = start
    end = endDate.split('-')
    endYear, endMonth, endDay = end

    # Check that the start date is before the end date
    checkInDataset(startYear, startMonth, startDay, endYear, endMonth, endDay)
    # Check that the endDate is in the dataframe


    # Check that startDate is not on a weekend/holiday and startDate < endDate
    while (startDate not in dirtyData.index):
        checkInDataset(startYear, startMonth, startDay, endYear, endMonth, endDay)
        # Increment the day value
        startDay = int(startDay)
        startMonth = int(startMonth)
        startYear = int(startYear)
        #print(startDay)
        # Check if month has 31 day
        if (startMonth in [1, 3, 5, 7, 8, 10, 12] and startDay < 31):
            if (startDay < 9):
                startDay += 1
                startDay = "0" + str(startDay)
            else:
                startDay += 1
                startDay = str(startDay)
        elif(startMonth in [1, 3, 5, 7, 8, 10, 12] and startDay == 31):
            if (startMonth == 10):
                startMonth+=1
                startMonth = str(startMonth)
                startDay = "01"
            elif(startMonth == 12):
                startMonth = "01"
                startDay = "01"
                startYear += 1
                startYear = str(startYear)
            else:
                startDay = "01"
                startMonth += 1
                startMonth = "0" + str(startMonth)
        elif (startMonth in [4, 6, 9, 11] and startDay < 30):
            if (startDay < 9):
                startDay += 1
                startDay = "0" + str(startDay)
            else:
                startDay += 1
                startDay = str(startDay)
        elif(startMonth in [4, 6, 9, 11] and startDay == 30):
            if (startMonth == 11):
                startMonth += 1
                startMonth = str(startMonth)
                startDay = "01"
            else:
                startMonth += 1
                startMonth = "0" + str(startMonth)
                startDay = "01"
        # Check if the month is Febuary and if it is a leap year
        elif(startMonth == 2):
            if (startDay < 9):
                startDay += 1
                startDay = "0" + str(startDay)
            elif (startDay < 28):
                startDay += 1
                startDay = str(startDay)
            elif(startDay == 28):
                if (((startYear % 2008) % 4 == 0) or ((2008 % startYear) % 4 == 0)):
                    startDay += 1
                    startDay = str(startDay)
                else:
                    startDay = "01"
                    startMonth += 1
                    startMonth = "0" + str(startMonth)
            else:
                startDay = "01"
                startMonth += 1
                startMonth = "0" + str(startMonth)
        daysShift += 1
        startDate = '{}-{}-{}'.format(startYear, startMonth, startDay)
        #print(startDate)
    return startDate, daysShift


def clean_df(dirtyData, startDate, endDate):
    # Validate the given startDate and endDate
    newStartDate, daysShift = validateDate(dirtyData, startDate, endDate)
    # Format column names
    print('\n...Cleaning data...\n')
    dirtyData.rename(columns={'1. open': 'Open', '2. high': 'High', '3. low': 'Low', '4. close': 'Close', '5. volume': 'Volume'}, inplace=True)
    dirtyData.index = pd.to_datetime(dirtyData.index)
    #pprint(data, '\n')

    # Calculate daily returns
    dfClose = dirtyData[['Close']]
    #pprint(dfClose)
    dfChange = pd.DataFrame({'Returns': dfClose.Close.pct_change()})
    #pprint(dfChange)
    df = pd.concat([dfClose, dfChange], axis=1, sort=False)
    #pprint(df)

    # Trim data set to only include desired observations
    newStartDateIdx = df.index.get_loc(newStartDate)
    endDateIdx = df.index.get_loc(endDate)
    df = df[newStartDateIdx:endDateIdx+1]
    candlePlotData = dirtyData[newStartDateIdx:endDateIdx+1]
    print('\nCleaned dataframe,\nChosen start date: ' + startDate + '\nChosen end date: ' + endDate)
    if (daysShift>0):
        print('\nError: there was an issue with the given start date and has been moved by ' + str(daysShift) + ' day(s)\n\nThe start date is now: ' +  newStartDate)

    print('\nFirst observation:')
    pprint(df.head(1))
    print('\nLast observation:')
    pprint(df.tail(1))

    return df, candlePlotData


def single_stock_VaR(tickerSymbol, startDate, endDate, vizData=False):

    # Fetch price data
    dirtyData = pull_data(tickerSymbol)

    df, candleDf = clean_df(dirtyData, startDate, endDate)


    ############################################################################
    ### Variance-Covariance Method #############################################
    ############################################################################

    # Calculate the mean and standard deviation of the daily returns
    tdf, tmean, tsigma = scipy.stats.t.fit(df['Returns'])
    mean, sigma = scipy.stats.norm.fit(df['Returns'])

    # plot data
    if vizData:
        vizualize(tickerSymbol, df, candleDf, mean, sigma, tdf, tmean, tsigma)

    print('\nCalculating VaR for: ' + tickerSymbol + '\n')

    # Calculate the VaR (norm)
    Var_90_1 = "{:.4%}".format(scipy.stats.norm.ppf(0.1, mean, sigma))
    Var_95_1 = "{:.4%}".format(scipy.stats.norm.ppf(0.05, mean, sigma))
    Var_99_1 = "{:.4%}".format(scipy.stats.norm.ppf(0.01, mean, sigma))

    # Calculate the VaR (t)
    Var_90_2 = "{:.4%}".format(scipy.stats.t.ppf(0.1, tdf, tmean, tsigma))
    Var_95_2 = "{:.4%}".format(scipy.stats.t.ppf(0.05, tdf, tmean, tsigma))
    Var_99_2 = "{:.4%}".format(scipy.stats.t.ppf(0.01, tdf, tmean, tsigma))

    ############################################################################
    ### Historical Method ######################################################
    ############################################################################

    histDf = df[['Returns']].dropna()
    #print(histDf)
    histDf.sort_values('Returns', inplace=True, ascending=True)

    Var_90_3 = "{:.4%}".format(histDf['Returns'].quantile(0.1))
    Var_95_3 = "{:.4%}".format(histDf['Returns'].quantile(0.05))
    Var_99_3 = "{:.4%}".format(histDf['Returns'].quantile(0.01))


    # Print the tabulated VaR
    print (tabulate([['90%', Var_90_1, Var_90_2, Var_90_3], ['95%', Var_95_1, Var_95_2, Var_95_3], ['99%', Var_99_1, Var_99_2, Var_99_3]], headers=["Confidence Level", "Value at Risk (VCV-norm)", "Value at Risk (VCV-t)", "Value at Risk (Hist)"]))
    print("\n")

def portfolio_VaR():
    startDate = '2018-04-04'
    endDate = '2019-04-04'
    positions = pd.read_csv("./resources/portfolio.csv", delimiter=',')
    #print(positions)

    for index, row in positions.iterrows():
        single_stock_VaR(row['Symbol'], startDate, endDate, vizData=True)


if __name__ == '__main__':
    print('\n')
    profile.run('portfolio_VaR()', 'resources/VaR_Stats')
    #profile.run("single_stock_VaR('AAPL', '2018-07-20', '2019-04-02', vizData=True)", "VaR_Stats")
    p = pstats.Stats('resources/VaR_Stats')
    p.strip_dirs().sort_stats('time', 'ncalls').print_stats(0)
