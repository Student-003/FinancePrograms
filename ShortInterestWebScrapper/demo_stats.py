#%%
# install the required packages:

# for data handling/analysis
import pandas as pd
import numpy as np

# for plotting
import matplotlib as mp
import matplotlib.pyplot as plt


# read data into Pandas Data Frame from the CSV file we've created

data = pd.read_csv("./shortinterest.csv", delimiter=',')

# take a look at first few records and check if we have what we want:

data.head()

# re-index to make searching much easier:

data.set_index("Symbol", inplace = True)

data.loc['DDD']

def short_lookup():
    ticker = str(input("Please enter a ticker: "))
    info = data.loc[ticker]
    print(info)

short_lookup()

# check how much data we have:
data.shape

# get some summary stats:
data.describe()

# find highly-shorted companies:

data['Company'][data['Percent_Float'] >= 50.0]

# some basic charting
histogram = data.hist(column = "Percent_Float")
histogram = data.hist(column = "DaystoCover")
