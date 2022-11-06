#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd, datetime
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from matplotlib.pylab import rcParams


# In[4]:


#!pip install pandas 1.2.0
get_ipython().system('pip install --upgrade xlrd')


# In[5]:


df=pd.read_excel("D:\DATA SCIENCE\LMS\PROJECT\oil price forcast\RBRTEd.xls", sheet_name='Data 1',header=None)
df


# In[6]:


df2 = df.iloc[3:8904]
df2.head(5)


# In[7]:


df2.columns =['Date', 'Oil_Prices']
df2.head(5)


# In[8]:


df2.reset_index(drop=True)


# In[9]:


print("Data Set:"% df2.columns, df2.shape)
print("Data Types:", df2.dtypes)


# In the above dataset, we have noticed that the series does not contain the values for Saturday and Sunday, if you look closely  11th and 12th June 2022 (weekends data) are seem to be missing and this is because the market is closed on weekends.
#                                    ![image.png](attachment:image.png)
# Hence these missing values were needed to be filled. To fill in weekends, first we used date as index (for resample method), then used forward fill, which will assign the weekend values with Friday values. Resample method is used for frequency conversion and resampling of time series. Object must have a datetime-like index (DatetimeIndex, PeriodIndex, or TimedeltaIndex), or pass datetime-like values to the on or level keyword.

# In[10]:


df2.set_index('Date', inplace=True)
oilPrices = df2.resample('D').ffill().reset_index()
oilPrices.tail(10)


# In[11]:


oilPrices.isnull().values.any()


# In[12]:


oilPrices['year']=oilPrices['Date'].dt.year
oilPrices['month']=oilPrices['Date'].dt.month
oilPrices['week']=oilPrices['Date'].dt.week


# In[13]:


oilPrices.tail(40)


# # EDA

# In[14]:


oilPrices.info()


# In[15]:


plt.ylabel("Crude Oil Prices trend: Brent - Europe")
plt.xlabel("Year")
sns.lineplot(x='Date',y='Oil_Prices',data = oilPrices)


# In[16]:


plt.figure(figsize=(10,8))
oilPrices.groupby('year')['Oil_Prices'].mean().plot(kind='bar',color = 'purple')
plt.show()


# From both the above line and bar graphs, it can be seen that the Oil prices tend to show an upward trend from 1987 till 2011 before dropping dramatically towards 2015. From 2015 onwards, the prices seem to fluctuate with some peaks and falls but is showing a growing trend from 2020 upto present.

# In[17]:


sns.boxplot("Oil_Prices",data=oilPrices)


# There seems be to be no outliers present in our prices data. The series seem to be positively skewed showing no normal distribution of the data points.

# In[18]:


plt.figure(figsize=(30,15))
heatmap_y_month = pd.pivot_table(data=oilPrices,values="Oil_Prices",index="year",columns="month",aggfunc="mean",fill_value=0)
sns.heatmap(heatmap_y_month,annot=True,fmt="g") #fmt is format of the grid values


# In[19]:


heatmap_y_month.plot(figsize=(20,15))
plt.show()


# In[20]:


heatmap_y_month.plot(kind='box',figsize=(8,6))
plt.show()


# ### Checking the Normality of the series

# In[21]:


#Distribution Plot
sns.distplot(oilPrices["Oil_Prices"])


# In[22]:


#Jarque Bera Stastical Test for Normality
from scipy.stats import jarque_bera as jb
is_norm=jb(oilPrices["Oil_Prices"])[1]
print(f"p value:{is_norm.round(4)}", ", Series is Normal" if is_norm >0.05 else ", Series is not Normal")


# ### Checking the Stationarity of the Series using ADFuller test and KPSS Test

# #### Augmented Dickey-Fuller Test (ADFuller Test)
# ADF test is conducted with the following assumptions :
# 
# 1. Null Hypothesis (HO): Series is non-stationary.
# 2. Alternate Hypothesis(HA): Series is stationary.
# 
# If the null hypothesis is failed to be rejected, this test may provide evidence that the series is non-stationary.
# 
# Conditions to Reject Null Hypothesis(HO):
# If Test statistic < Critical Value and p-value < 0.05 – Reject Null Hypothesis(HO) i.e., time series does not have a unit root, meaning it is stationary. It does not have a time-dependent structure.

# In[23]:


#Augmented Dickey-Fuller Test (ADFuller Test)
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.Series(timeseries).rolling(window=365).mean()
    rolstd = pd.Series(timeseries).rolling(window=365).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)


# In[24]:


plt.figure(figsize=(20,15))
test_stationarity(oilPrices["Oil_Prices"])


# The p-value obtained is greater than significance level of 0.05 and the ADF test statistic is higher than any of the critical values. Clearly, there is no reason to reject the null hypothesis. So, the time series is in fact non-stationary.
# 
# #### Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test
# Let us try using the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test and check if we get the same results i.e. the given series is not stationary. 
# 
# KPSS test is conducted with the following assumptions: 
# 
# 1. Null Hypothesis (HO): Series is trend stationary.
# 
# 2. Alternate Hypothesis(HA): Series is non-stationary. 
# 
# ###### Note: Hypothesis is reversed in KPSS test compared to ADF Test.
# 
# If the null hypothesis is failed to be rejected, this test may provide evidence that the series is trend stationary.
# 
# Conditions to Fail to Reject Null Hypothesis(HO)-
# If Test statistic < Critical Value and p-value < 0.05 – Fail to Reject Null Hypothesis(HO) i.e., time series does not have a unit root, meaning it is trend stationary.
# 

# In[25]:


#Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test
from statsmodels.tsa.stattools import kpss
def kpss_test(timeseries):
    print ('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','#Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    print (kpss_output)


# Here Test Statistic > Critical Value and p-value < 0.05. As a result, we reject the Null hypothesis in favor of an Alternative.
# Hence we conclude series is non-stationary. 
# 
# Therefore we can confirm from both the tests that our data series is not stationary and will require some transformations as pre-processing part before building our prediction models. 

# In[26]:


kpss_test(oilPrices["Oil_Prices"])


# ### Time-Series Decomposition to understand the trend and seasonality

# In[27]:


from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse

# Multiplicative Decomposition 
multiplicative_decomposition = seasonal_decompose(oilPrices["Oil_Prices"], model='multiplicative', period=365)

# Additive Decomposition
additive_decomposition = seasonal_decompose(oilPrices["Oil_Prices"], model='additive', period=365)

# Plot
plt.rcParams.update({'figure.figsize': (16,12)})
multiplicative_decomposition.plot().suptitle('Multiplicative Decomposition', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

additive_decomposition.plot().suptitle('Additive Decomposition', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.show()


# In[ ]:





# In[28]:


#train = oilPrices[(oilPrices['Date' ] > '1987-05-20') & (oilPrices['Date' ] <= '2019-12-31')]
latest_price = oilPrices[oilPrices['Date' ] >= '2020-01-01']


# In[ ]:





# In[29]:


df = latest_price["Oil_Prices"]

fig,axes = plt.subplots(2,2)
fig.set_figheight(10)
fig.set_figwidth(15)
axes[0][0].plot(latest_price['Date' ],df,label='Actual')
axes[0][0].plot(latest_price['Date' ],df.rolling(window=4).mean(),label='4 months rolling mean')
axes[0][0].set_xlabel('Year')
axes[0][0].set_ylabel('Oil Prices')
axes[0][0].set_title('4 Months Rolling Mean')
axes[0][0].legend(loc='best')


axes[0][1].plot(latest_price['Date' ],df,label='Actual')
axes[0][1].plot(latest_price['Date' ],df.rolling(window=6).mean(),label='6 months rolling mean')
axes[0][1].set_xlabel('Year')
axes[0][1].set_ylabel('Oil Prices')
axes[0][1].set_title('6 Months Rolling Mean')
axes[0][1].legend(loc='best')

axes[1][0].plot(latest_price['Date' ],df,label='Actual')
axes[1][0].plot(latest_price['Date' ],df.rolling(window=8).mean(),label='8 months rolling mean')
axes[1][0].set_xlabel('Year')
axes[1][0].set_ylabel('Oil Prices')
axes[1][0].set_title('8 Months Rolling Mean')
axes[1][0].legend(loc='best')


axes[1][1].plot(latest_price['Date' ],df,label='Actual')
axes[1][1].plot(latest_price['Date' ],df.rolling(window=12).mean(),label='12 months rolling mean')
axes[1][1].set_xlabel('Year')
axes[1][1].set_ylabel('Oil Prices')
axes[1][1].set_title('12 Months Rolling Mean')
axes[1][1].legend(loc='best')

plt.tight_layout()
plt.show()


# The 12 months moving average produces a good wrinkle free curve as required

# ## Transformation to make the series stationary
# The basic idea is to model the trend and seasonality in this series, so we can remove it and make the series stationary. Then we can go ahead and apply statistical forecasting to the stationary series. And finally we can convert the forecasted values into original by applying the trend and seasonality constrains back to those that we previously separated.
# 
# ### Trend
# The first step is to reduce the trend using transformation, as we can see here that there is a strong positive trend. These transformation can be log, sq-rt, cube root etc . Basically it penalizes larger values more than the smaller. 

# ### 1. Log Transformation 

# In[30]:


transformed_prices = oilPrices["Oil_Prices"].astype(float)
transformed_prices.info()


# In[31]:


prices_log = np.log(transformed_prices)


# In[32]:


plt.plot(prices_log)


# In[33]:


test_stationarity(prices_log)


# In[ ]:





# ### 2. Square-Root Transformation

# In[34]:


prices_sqrt= np.sqrt(transformed_prices)
plt.plot(prices_sqrt)


# In[35]:


test_stationarity(prices_sqrt)


# ### 3. Differentiating by 1 and 2 on the Log Transformation

# In[36]:


log_diff1=prices_log.diff(1).dropna()
log_diff2=prices_log.diff(2).dropna()


# In[37]:


test_stationarity(log_diff1)


# In[38]:


test_stationarity(log_diff2)


# ###  Seasonality (along with trend)
# Previously we saw just trend part of the time series, now we will see both trend and seasonality. Our series has trend along with seasonality. So there are two common methods to remove trend and seasonality as follows:
# 
# • Differencing: by taking difference using time lag also known as Time-shift transformation
# 
# • Decomposition: modeling both trend and seasonality, then removing them

# ### 1. Time-Shift transformation / Differencing

# In[39]:


prices_log_diff = prices_log- prices_log.shift()
plt.plot(prices_log_diff)


# In[40]:


prices_log_diff.dropna(inplace=True)


# In[41]:


test_stationarity(prices_log_diff)


# ### 2. Seasonal Decomposing using the logarithmic transformed values of the prices

# In[42]:


from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse


# Multiplicative Decomposition 
multiplicative_decomposition = seasonal_decompose(prices_log, model='multiplicative', period=365)

# Additive Decomposition
additive_decomposition = seasonal_decompose(prices_log, model='additive', period=365)

# Plot
plt.rcParams.update({'figure.figsize': (16,12)})
multiplicative_decomposition.plot().suptitle('Multiplicative Decomposition', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

additive_decomposition.plot().suptitle('Additive Decomposition', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.show()


# From the above graphs, it is observed that the additive decompositon model seems to be a good fit for series and we can see an increasing trend as well as a seasonal pattern with a cycle of 12 months. So removing the trend and seasonality from the Time series by using the residual values of the additive decomposition model and checking stationarity as follows.

# In[43]:


residuals = additive_decomposition.resid
prices_log_decompose = residuals
prices_log_decompose.dropna(inplace=True)
test_stationarity(prices_log_decompose)


# In[1]:


#!pip install pmdarima


# ## Model Building 

# ### 1. Model cross-validation using the pmdarima's train_test_split cross-validation technique
# Like scikit-learn, the Pyramid Python library provides several different strategies for cross-validating the time series models. The interface was designed to behave as similarly as possible to that of scikit to make its usage as simple as possible.

# In[45]:


# Load the data and split it into separate pieces
import pmdarima as pm
from pmdarima.datasets import load_sunspots
from pmdarima.model_selection import train_test_split
print(f"Using pmdarima {pm.__version__}")
# Using pmdarima 1.5.2

y = load_sunspots(True)
train_len = 8000
train, test = train_test_split(oilPrices, train_size=train_len)
train.head()


# In[46]:


print("Training size is : ", train.shape)
print("Testing size is : ", test.shape)


# In[47]:


train['Oil_Prices'].plot()
test['Oil_Prices'].plot(figsize=(20, 6), 
            title= 'Daily Traffic Counts',
            fontsize=14);


# In[48]:


import statsmodels.api as sm
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
#plot the ACF
fig = sm.graphics.tsa.plot_acf(\test['Oil_Prices'], lags=18, ax=ax1)
ax2 = fig.add_subplot(212)
#plot the PACF
fig = sm.graphics.tsa.plot_pacf(test['Oil_Prices'], lags=18, ax=ax2)


# In[49]:


def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    acf1 = acf(fc-test)[1]                      # ACF1
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 'acf1':acf1, 
            'corr':corr, 'minmax':minmax})



# In[50]:


import pmdarima as pm
model = pm.auto_arima(log_diff1, start_p=0, start_q=0, seasonal=True)
model.summary()


# In[51]:


model.plot_diagnostics()


# In[52]:


# Forecast
n_periods = len(test.Oil_Prices)
fitted, confint = model.predict(n_periods=n_periods, return_conf_int=True)
test['SARIMAX'] = fitted
plt.figure(figsize=(16,8))
plt.plot(train['Oil_Prices'], label='Train')
plt.plot(test['Oil_Prices'], label='Test')
plt.plot(test['SARIMAX'], label='SARIMAX')
plt.legend(loc='best')
plt.show()


# In[53]:


def MAPE(pred,org):
    temp = np.abs((pred-org)/org)*100
    return np.mean(temp)


# In[54]:


MAPE(test['SARIMAX'],test['Oil_Prices'])


# In[55]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing
fit1 = ExponentialSmoothing(np.asarray(train['Oil_Prices']) ,seasonal_periods=12 ,trend='add', seasonal='add').fit(optimized=True, remove_bias=False)
test['Holt_Winter'] = fit1.predict(start=test.index[0], end=test.index[-1])
plt.figure(figsize=(16,8))
plt.plot( train['Oil_Prices'], label='Train')
plt.plot(test['Oil_Prices'], label='Test')
plt.plot(test['Holt_Winter'], label='Holt_Winter')
plt.legend(loc='best')
plt.show()


# In[56]:


def MAPE(pred,org):
    temp = np.abs((pred-org)/org)*100
    return np.mean(temp)


# In[57]:


MAPE(test['Holt_Winter'],test['Oil_Prices'])


# ## Deep Learning model (LSTM)

# In[58]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2s
from math import sqrt
import os


# In[59]:


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
df1=oilPrices['Oil_Prices'].values.astype(float)
df1 = df1.reshape(-1,1)
norm_prices = sc.fit_transform(df1)


# In[60]:


#y = df['Price'].resample('MS').mean()


# In[61]:


# split into train and test sets
train_size = int(len(norm_prices) * 0.70)
test_size = len(norm_prices) - train_size
train, test = norm_prices[0:train_size, :], norm_prices[train_size:len(norm_prices), :]


# In[62]:


# convert an array of values into a data_set matrix def
def create_data_set(_data_set, _look_back=1):
    data_x, data_y = [], []
    for i in range(len(_data_set) - _look_back - 1):
        a = _data_set[i:(i + _look_back), 0]
        data_x.append(a)
        data_y.append(_data_set[i + _look_back, 0])
    return np.array(data_x), np.array(data_y)


# In[63]:


# reshape into X=t and Y=t+1
look_back =90
X_train,Y_train,X_test,Ytest = [],[],[],[]
X_train,Y_train=create_data_set(train,look_back)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test,Y_test=create_data_set(test,look_back)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# In[68]:


from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
# create and fit the LSTM network regressor = Sequential() 
regressor = Sequential()

regressor.add(LSTM(units = 60, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.1))

regressor.add(LSTM(units = 60, return_sequences = True))
regressor.add(Dropout(0.1))

regressor.add(LSTM(units = 60))
regressor.add(Dropout(0.1))

regressor.add(Dense(units = 1))


regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
reduce_lr = ReduceLROnPlateau(monitor='val_loss',patience=5)
history =regressor.fit(X_train, Y_train, epochs = 20, batch_size =50,validation_data=(X_test, Y_test), callbacks=[reduce_lr],shuffle=False)


# In[ ]:


train_predict = regressor.predict(X_train)
test_predict = regressor.predict(X_test)


# In[ ]:


# invert predictions
train_predict = sc.inverse_transform(train_predict)
Y_train = sc.inverse_transform([Y_train])
test_predict = sc.inverse_transform(test_predict)
Y_test = sc.inverse_transform([Y_test])


# In[ ]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
print('Train Mean Absolute Error:', mean_absolute_error(Y_train[0], train_predict[:,0]))
print('Train Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_train[0], train_predict[:,0])))
print('Test Mean Absolute Error:', mean_absolute_error(Y_test[0], test_predict[:,0]))
print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_test[0], test_predict[:,0])))
plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.show();


# In[ ]:


#Compare Actual vs. Prediction
aa=[x for x in range(180)]
plt.figure(figsize=(8,4))
plt.plot(aa, Y_test[0][:180], marker='.', label="actual")
plt.plot(aa, test_predict[:,0][:180], 'r', label="prediction")
plt.tight_layout()
sns.despine(top=True)
plt.subplots_adjust(left=0.07)
plt.ylabel('Price', size=15)
plt.xlabel('Time step', size=15)
plt.legend(fontsize=15)
plt.show();


# In[69]:


from prophet import Prophet
d={'ds':oilPrices['Date'],'y':oilPrices['Oil_Prices']}
df_pred=pd.DataFrame(data=d)
modelp = Prophet(daily_seasonality=False)
modelp.fit(df_pred)


# In[70]:


#df.columns = ["ds","y"]

#model = Prophet()
#model.fit(df)

#future = model.make_future_dataframe(periods= 60, freq='m')


# In[71]:


models_scores = []
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse})


# In[72]:


future = modelp.make_future_dataframe(periods= 60, freq='m')
future.tail()


# In[73]:


forecast = modelp.predict(future)
forecast.tail()


# In[74]:


forecast[["ds","yhat","yhat_lower","yhat_upper"]].head()


# In[75]:


import pickle


# In[76]:


filename= 'trained_model_s.sav'
pickle.dump(modelp,open(filename,'wb'))


# In[80]:


m=forecast[["ds","yhat"]].head()
m.columns=['ds','y']


# In[81]:


m


# In[83]:


import pickle
import csv
m.to_csv('Data_Prophet.csv')

