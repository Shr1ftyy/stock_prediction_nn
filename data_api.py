# import plotly.graph_objects as go
import datetime
import os

import matplotlib.pyplot as plt
import pandas as pd
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.timeseries import TimeSeries
from matplotlib.pyplot import figure

###############################################

#	This Module is used to parse stock prices
#	The user can either use AlphaVantage's 
#	JSON-based API or parse data from a .csv file
#       
#       @author Syeam_Bin_Abdullah
###############################################

online = True

try:
	os.system('cls')
except Exception as e:
	os.system('clear')


print('---- DATA API FOR STOCK PREDICTION ----\n')

def show_option():
	global option	
	option = int(input('''Would you like to connect to AlphaVantage
for Data or select your own csv file?

1. Use AlphaVantage's API (requires an internet connection)
2. Select a .csv file
0. Exit

Insert reference number to continue: '''))

show_option()
print("www")

if option == 1:
	online = True
elif option == 2:
	online = False
elif option == 0:
	exit()
else:
	print('You did not select the displayed options, please try again')
	show_option()

if online:
	try:
		selection = input("Stock symbol: ")
		print('Establishing Connection with API...')
	# Your key here
		key = '072HERMNM5POYD6B'

		# Choose your output format, or default to JSON (python dict)
		ts = TimeSeries(key, output_format='pandas')
		ti = TechIndicators(key)
		# Get the stat, returns a tuple
		# stat is a pandas statframe, meta_stat is a dict
		stat, meta_stat = ts.get_daily(symbol=selection, outputsize= 'full')
		# sma is a dict, meta_sma also a dict
		sma, meta_sma = ti.get_sma(symbol=selection)
		
		print('--- Connection Established ---')
		print("\n\nShowing Prices for: {}".format(selection))
		print(str(stat))

		stat = stat.iloc[::-1]
		stock_open=stat['open']
		high=stat['high']
		low=stat['low']
		close=stat['close']
		volume=stat['volume']
	except Exception as e:
		stock_open=stat['1. open']
		high=stat['2. high']
		low=stat['3. low']
		close=stat['4. close']
		volume=stat['5. volume']
else:
	selection = input('Insert .csv file path/name: ')
	selection = selection.split(' ')
	selection = selection[0]
	selection = str(selection)

	stat = pd.read_csv(selection)

	stat.index = stat['Date']
	print("\n\nShowing Prices from csv file: {}".format(selection))
	print(stat)
	stock_open=stat['Open']
	high=stat['High']
	low=stat['Low']
	close=stat['Close']
	volume=stat['Volume']


print(f"latest volume: {volume[0]}")	

# Visualization

def visualize():

	figure(num=None, figsize=(15, 6), dpi=80, facecolor='w', edgecolor='k')
	close.plot()
	plt.title(f"{selection} Stock Price")
	plt.ylabel(f'Price')
	plt.tight_layout()
	plt.grid()
	plt.show()

 
