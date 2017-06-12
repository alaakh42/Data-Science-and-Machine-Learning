# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
#from pandas import DataFrame

#import datetime
#import pandas.io.data

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#sp500 = pd.io.data.get_data_yahoo('%5EGSPC',
#                                start = datetime.datetime(2000, 10, 1),
#                                  end = datetime.datetime(2014, 6, 11))
#print sp500.head()
#sp500.to_csv('sp500_ohlc.csv')
titanic = pd.read_csv('train.csv')
df = pd.read_csv('sp500_ohlc.csv', index_col = 'Date', parse_dates = True)
#print titanic.head()
#print df.head()

df2 = df['Open']
#print df2.head()

df3 = df[['Open','Close']]
#print df3.head()

df4 = df3[(df3['Close'] > 1400) ]
#print df4.head()

df['H-L'] = df['High'] - df['Low']
#print df.head()

df['100MA'] = pd.rolling_mean(df['Close'], 100)
#print df[200:210]

df['difference'] = df['Close'].diff()
#print df.head()

#df.plot(df['100MA'])
#df['Close'].plot()
#df[['Close', 'High', 'Low','Open', '100MA']].plot()
#plt.show()

## 3D plotting
threedee = plt.figure().gca(projection = '3d')
threedee.scatter( df['difference'], df['H-L'], df['Close'])
threedee.set_xlabel('difference')
threedee.set_ylabel('H-L')
threedee.set_zlabel('Close')
plt.draw()
