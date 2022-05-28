# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # LSTM - 6 WEEK LATER SALE FORECASTING

# %%
# Importing Libraries
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import ticker
from statsmodels.tsa.stattools import adfuller, acf, pacf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import mlflow
#import local libraries
#Adding scripts path
sys.path.append(os.path.abspath(os.path.join('..')))
from scripts.data_loader import load_df_from_csv
from scripts.ML_modelling_utils import *
from scripts.data_information import DataInfo
from scripts.data_manipulation import DataManipulator


# %%
# Notebook settings
pd.set_option('max_column', None)
pd.set_option('display.float_format', '{:.2f}'.format)
get_ipython().run_line_magic('matplotlib', 'inline')

# %% [markdown]
# ## Loading Unclean Data

# %%
unclean_data = load_df_from_csv('../data/train.csv')
info = DataInfo(unclean_data, deep=True)


# %%
# Creating manipulator class for new data creation, cleaning and scaling data
manipulator = DataManipulator(info.df)


# %%
# Creating Date Column for later use
manipulator.create_date()
# Hold the column
date_column = manipulator.df['Date'].values.tolist()
# Drop the column
manipulator.df.drop('Date', axis=1, inplace=True)


# %%
info.get_dispersion_params()


# %%
info.get_column_based_missing_percentage()


# %%
info.get_column_based_missing_values()

# %% [markdown]
# ### Adding more data columns by extraction

# %%


# %% [markdown]
# ### Cleaning Data

# %%
# Filling None values of numeric columns with the columns max value
manipulator.fill_columns_with_max(['Promo2SinceWeek', 'Promo2SinceYear', 'CompetitionDistance'])


# %%
# Filling None values of categorical columns with the columns most frequent value
manipulator.fill_columns_with_most_frequent(['PromoInterval', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear'])


# %%
# See if all missing values are cleaned
info.get_columns_with_missing_values()

# %% [markdown]
# ### Labeling Data

# %%
# Labelling object type columns with Label Encoder
labelers_dict = manipulator.label_columns(info.get_object_columns())


# %%
# See if any object type columns reamin
info.get_object_columns()

# %% [markdown]
# Scaling Data

# %%
# Before Scaling
info.get_dispersion_params()


# %%
# Scaling all numeric values
manipulator.minmax_scale_columns(info.get_numeric_columns(), range_tup=(-1,1))


# %%
# After Scaling
info.get_dispersion_params()

# %% [markdown]
# ## Time Series Analysis

# %%
# Actual Data
data = unclean_data
unclean_manipulator = DataManipulator(data)
unclean_manipulator.create_date()
data = data[['Date', 'Sales']]


# %%
# Add Date Data
scaled_data = info.df.copy(deep=True)
scaled_data['Date'] = date_column
scaled_data = scaled_data[['Date','Sales']]


# %%
# data.to_csv('../data/lstmcleandata.csv', index=False)
# data = pd.read_csv('../data/lstmcleandata.csv')

# %% [markdown]
# ### Isolate the Rossmann Store Sales dataset into time series data

# %%
# Group Unscaled Data based on Date
data.groupby('Date').agg({'Sales': 'mean'})


# %%
# Group Scaled Data based on Date
scaled_data.groupby('Date').agg({'Sales':'mean'})


# %%
fig = plt.figure()
gs = GridSpec(2, 1, figure=fig)

fig.set_figheight(20)
fig.set_figwidth(30)
fig.tight_layout(pad=15)

M = 100
xticks = ticker.MaxNLocator(M)

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(data.index, data.Sales, 'b-')
ax1.xaxis.set_major_locator(xticks)
ax1.tick_params(labelrotation=90)
ax1.set_xlabel('Date')
ax1.set_ylabel('Thousands of Units')
ax1.title.set_text('Time Series Plot of Rossmann Pharmaceuticals Store Sales')
ax1.grid(True)

ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(scaled_data.index, scaled_data.Sales, 'g-')
ax2.xaxis.set_major_locator(xticks)
ax2.tick_params(labelrotation=90)
ax2.set_xlabel('Date')
ax2.set_ylabel('Scaled Units')
ax2.title.set_text(
    'Time Series Plot of Min Max Scaled Rossmann Pharmaceuticals Store Sales')
ax2.grid(True)
plt.show()


# %%
fig = plt.figure()
gs = GridSpec(2, 1, figure=fig)

fig.set_figheight(10)
fig.set_figwidth(30)
fig.tight_layout(pad=6)

ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(data.Sales, density=True, bins=60)
ax1.title.set_text('Histogram Rossmann Pharmaceuticals Store Sales')
ax1.grid(True)

ax2 = fig.add_subplot(gs[1, 0])
ax2.hist(scaled_data.Sales, density=True, bins=60)
ax2.title.set_text('Histogram of the of Min Max Scaled Rossmann Pharmaceuticals Store Sales')
ax2.grid(True)
plt.show()

# %% [markdown]
# ### Check whether your time Series Data is Stationary

# %%
# hecking stationarity using unit root tests
adfResult = adfuller(scaled_data.Sales.values, autolag='AIC')
print(f'ADF Statistic: {adfResult[0]}')
print(f'p-value: {adfResult[1]}')

# %% [markdown]
# ### Depending on your conclusion from 2 above difference your time series data

# %%
# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return pd.Series(diff)


# %%
sales_difference = difference(scaled_data.Sales.values)

# %% [markdown]
# ### Check for autocorrelation and partial autocorrelation of your data

# %%
def corrPlots(array: np.array, prefix: str):
    plt.figure(figsize=(30, 5))
    plt.title(f"{prefix}  Autocorrelations of House Sales Min Max Scaled")
    plt.bar(range(len(array)), array)
    plt.grid(True)
    plt.show()


# %%
# AutoCorrelation
acfSalesScaled = acf(houseSales.HouseSalesScaled.values, fft=True, nlags=40)
acfSalesScaledNp = np.array(acfSalesScaled)


# %%
corrPlots(acfSalesScaledNp, '')


# %%
# Partial Correlation
pacfSalesScaled = pacf(houseSales.HouseSalesScaled.values, nlags=40)
pacfSalesScaledNp = np.array(pacfSalesScaled)


# %%
corrPlots(pacfSalesScaledNp, "Partial")

# %% [markdown]
# 
# %% [markdown]
# ## Model Data Preparation

# %%
# Set window of past points for LSTM Model
# 6 Weeks is 45 Days (We have daily data of stores)
window = 45

# Spliting Data to Train and Test (80/20)

# %% [markdown]
# ## Creating Model

# %%


# %% [markdown]
# ## Training Model

