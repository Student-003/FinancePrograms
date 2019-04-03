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
# Import stock price data as CSV
from alpha_vantage.timeseries import TimeSeries


def pull_data(ticker_symbol):
    # Find API key
    parser = configparser.ConfigParser()
    foundFile = parser.read(['./secret1.ini', './template_secret.ini'])
    apiKey = parser.get('API_KEYS','ALPHA_VANTAGE') if './secret.ini' in foundFile else parser.get('API_KEYS_TEMPLATE','ALPHA_VANTAGE')
    if (apiKey == 'guest'):
        print('\nNOTE: You are using the demo API Key. Please request an Alpha Vantage API key from https://www.alphavantage.co\n')
    # Fetch price data
    # Alpha Vantage documentation and free API key can be found at https://www.alphavantage.co
    ts = TimeSeries(key=apiKey, output_format='pandas')
    data, meta_data = ts.get_daily(symbol=ticker_symbol, outputsize='full')

    return data


def clean_df(dirtyData, startDate, endDate):
    # Format column names
    dirtyData.rename(columns={'1. open': 'Open', '2. high': 'High', '3. low': 'Low', '4. close': 'Close', '5. volume': 'Volume'}, inplace=True)
    #print(data, '\n')

    # Calculate daily returns
    dfClose = dirtyData[['Close']]
    #print(dfClose)
    dfChange = pd.DataFrame({'Returns': dfClose.Close.pct_change()})
    #print(dfChange)
    df = pd.concat([dfClose, dfChange], axis=1, sort=False)
    #print(df)

    # Trim data set to only include desired observations
    startDateIdx = df.index.get_loc(startDate)
    endDateIdx = df.index.get_loc(endDate)
    df = df[startDateIdx:endDateIdx+1]
    print(df.head())

    return df


def VaR(ticker_symbol, startDate, endDate):

    # Fetch price data
    dirtyData = pull_data(ticker_symbol)
    df = clean_df(dirtyData, startDate, endDate)

    ############################################################################
    ### Variance-Covariance Method #############################################
    ############################################################################
    # Calculate the mean and standard deviation of the daily returns

    mean = np.mean(df['Returns'])
    std_dev = np.std(df['Returns'])

    # Plot data
    interactive(True)
    # Plot stock chart
    plt.figure(1).set_size_inches(10,3)
    df['Close'].plot()
    plt.title(ticker_symbol + " Price Chart", weight='bold');
    # Plot daily returns
    plt.figure(2).set_size_inches(10,3)
    df['Returns'].plot()
    plt.title(ticker_symbol + " Daily Returns", weight='bold');
    # Plot histogram of returns distribution
    plt.figure(3).set_size_inches(10,3)
    df['Returns'].hist(bins=50, density=True, histtype='stepfilled', alpha=0.5)
    x = np.linspace(df['Returns'].min(), df['Returns'].max(), 100)
    tdf, tmean, tsigma = scipy.stats.t.fit(df['Returns'])
    plt.plot(x, scipy.stats.t.pdf(x, loc=tmean, scale=tsigma, df=tdf), 'r')
    #plt.plot(x, scipy.stats.norm.pdf(x, mean, std_dev), 'r')
    plt.figtext(0.6, 0.7, "tμ = {:.3}".format(tmean))
    plt.figtext(0.6, 0.65, "tσ = {:.3}".format(tsigma))
    plt.figtext(0.6, 0.6, "df = {:.3}".format(tdf))
    plt.title(ticker_symbol + " Daily Return Distribution", weight='bold');
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    interactive(False)
    plt.show()
    #print("Calculating VaR\n")

    # Calculate the VaR
    Var_90_1 = "{:.4%}".format(scipy.stats.norm.ppf(0.1, mean, std_dev))
    Var_95_1 = "{:.4%}".format(scipy.stats.norm.ppf(0.05, mean, std_dev))
    Var_99_1 = "{:.4%}".format(scipy.stats.norm.ppf(0.01, mean, std_dev))

    ############################################################################
    ### Historical Method ######################################################
    ############################################################################
    histDf = df[['Returns']].dropna()
    #print(histDf)
    histDf.sort_values('Returns', inplace=True, ascending=True)

    Var_90_2 = "{:.4%}".format(histDf['Returns'].quantile(0.1))
    Var_95_2 = "{:.4%}".format(histDf['Returns'].quantile(0.05))
    Var_99_2 = "{:.4%}".format(histDf['Returns'].quantile(0.01))


    # Print the tabulated VaR
    print (tabulate([['90%', Var_90_1, Var_90_2], ['95%', Var_95_1, Var_95_2], ['99%', Var_99_1, Var_99_2]], headers=["Confidence Level", "Value at Risk (VCV)", "Value at Risk (Hist)"]))
    print("\n")

if __name__ == '__main__':
    print('\n')
    profile.run("VaR('TLRY', '2018-07-20', '2019-01-02')", "VaR_Stats")
    p = pstats.Stats('VaR_Stats')
    p.strip_dirs().sort_stats('time', 'ncalls').print_stats(10)
