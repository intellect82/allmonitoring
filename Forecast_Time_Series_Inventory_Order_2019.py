
# coding: utf-8

# In[64]:


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
get_ipython().magic('matplotlib inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 30,12


# In[65]:


train = pd.read_csv("Train1982.csv")
test= pd.read_csv("Test.csv")
submit = pd.read_csv("Sample_Submission.csv")


# In[66]:


print("Train shape: " + str(train.shape))
print("Test shape: " + str(test.shape))
train1.head()


# In[68]:


from math import floor
#split the train data into training set and valid set
train = train.loc[:floor(2*train.shape[0]/3)]
valid = train.loc[floor(2*train.shape[0]/3):]
train.set_index('Datetime', inplace = True)
valid.set_index('Datetime', inplace = True)
test.set_index('Datetime', inplace = True)


# In[69]:


print (train.shape)
print (valid.shape)
train.head()


# In[70]:


#parsing the datetime data 
dataparse = lambda dates: pd.datetime.strptime(dates, "%d-%m-%Y %H:%M")
train.index = train.index.map(dataparse)
valid.index = valid.index.map(dataparse)
test.index = test.index.map(dataparse)
train.head()


# In[178]:


ts=train['Count']


# In[179]:


ts


# In[73]:


ts.head(10)

plt.plot(ts)


# In[ ]:


import numpy as np


# In[74]:


ts_log = np.log(ts)


# In[75]:


moving_avg = ts_log.rolling(24).mean()


# In[79]:


moving_avg.fillna(0, inplace=True)


# In[81]:


from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    ts_log = np.log(ts)
    
    #Determing rolling statistics
    rolmean = ts_log.rolling(24).mean()
    rolstd = ts_log.rolling(24).std()

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


# In[82]:


test_stationarity(ts)


# In[89]:


moving_avg = ts_log.rolling(24).mean()
plt.plot(ts_log)
plt.plot(moving_avg, color = 'red')


# In[90]:


train_log_moving_avg_diff = ts_log - moving_avg


# In[91]:


# Since we are taking the average of 24 values, rolling mean is not defined for the first 23 values. (NaN)
train_log_moving_avg_diff.dropna(inplace = True)
test_stationarity(train_log_moving_avg_diff)


# In[98]:


expwighted_avg = ts_log.ewm(24).mean()
plt.plot(ts_log)
plt.plot(expwighted_avg, color='red')


# In[99]:


train_log_moving_avg_diff = ts - moving_avg


# In[96]:


# Since we are taking the average of 24 values, rolling mean is not defined for the first 23 values. (NaN)
train_log_moving_avg_diff.dropna(inplace = True)
test_stationarity(train_log_moving_avg_diff)


# In[101]:


moving_avg = ts_log.rolling(24).mean() 
plt.plot(ts_log)
plt.plot(moving_avg, color = 'red')


# In[102]:


train_log_moving_avg_diff = ts_log - moving_avg


# In[103]:


# Since we are taking the average of 24 values, rolling mean is not defined for the first 23 values. (NaN)
train_log_moving_avg_diff.dropna(inplace = True)
test_stationarity(train_log_moving_avg_diff)


# In[108]:


expwighted_avg =  ts_log.ewm(halflife=24).mean() 
plt.plot(train_log)
plt.plot(expwighted_avg, color='red')


# In[109]:


ts_log.ewm(halflife=24).mean()


# In[114]:


expwighted_avg =  ts_log.ewm(halflife=24).mean()
plt.plot(train_log)
plt.plot(expwighted_avg, color='red')


# In[ ]:


train_log_moving_avg_diff = ts_log - moving_avg


# In[118]:


#removing the trend of increasing
train_log_ewma_diff = ts_log - expwighted_avg
test_stationarity(train_log_ewma_diff)


# In[133]:


ts_log_diff = ts_log - ts_log.shift()
test_stationarity(ts_log_diff.dropna())


# In[134]:


ts_log_diff.head()


# In[135]:


from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(pd.DataFrame(ts_log).Count.values, freq = 24)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(train_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()


# In[136]:


#A closer look at the seasonality 
plt.plot(seasonal[:200],label='Sub-Seasonality')
plt.legend(loc='best')


# In[137]:


train_log_decompose = pd.DataFrame(residual)
train_log_decompose['date'] = ts_log.index
train_log_decompose.set_index('date', inplace = True)
train_log_decompose.dropna(inplace=True)
test_stationarity(train_log_decompose[0])


# In[138]:


#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(ts_log_diff.dropna(), nlags=25)
lag_pacf = pacf(ts_log_diff.dropna(), nlags=25, method='ols')
#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(train_log_diff.dropna())),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(train_log_diff.dropna())),linestyle='--',color='gray')
plt.title('Autocorrelation Function')
#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(train_log_diff.dropna())),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(train_log_diff.dropna())),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()


# In[139]:


ts_log_diff.head()


# In[140]:


from statsmodels.tsa.arima_model import ARIMA


# In[141]:


model = ARIMA(ts_log, order=(1, 1, 0))  
results_AR = model.fit(disp=-1)  
plt.plot(train_log_diff.dropna())
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-train_log_diff)**2))


# In[142]:


model = ARIMA(ts_log, order=(0, 1, 1))  
results_ARIMA = model.fit(disp=-1)  
plt.plot(train_log_diff.dropna())
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-train_log_diff)**2))


# In[143]:


model = ARIMA(ts_log, order=(1, 1, 1))  
results_MA = model.fit(disp=-1)  
plt.plot(train_log_diff.dropna())
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-train_log_diff)**2))


# In[157]:


#bring the differencing back to the original scale
def check_prediction_diff(predict_diff, given_set):
    predict_diff= predict_diff.cumsum().shift().fillna(0)
    predict_base = pd.Series(np.ones(given_set.shape[0]) * np.log(given_set['Count'])[0], index = given_set.index)
    predict_log = predict_base.add(predict_diff,fill_value=0)
    predict = np.exp(predict_log)
    
    plt.plot(given_set['Count'], label = "Given set")
    plt.plot(predict, color = 'red', label = "Predict")
    plt.legend(loc= 'best')
    plt.title('RMSE: %.4f'% (np.sqrt(np.dot(predict, given_set['Count']))/given_set.shape[0]))
    


# In[160]:


def check_prediction_log(predict_log, given_set):
    predict = np.exp(predict_log)
    
    plt.plot(given_set['Count'], label = "Given set")
    plt.plot(predict, color = 'red', label = "Predict")
    plt.legend(loc= 'best')
    plt.title('RMSE: %.4f'% (np.sqrt(np.dot(predict, given_set['Count']))/given_set.shape[0]))
    


# In[172]:


# Tried on MA model 
model = ARIMA(train_log.dropna(), order=(0, 1, 1))  
results_ARIMA = model.fit(disp=-1)  
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print (predictions_ARIMA_diff.head())


# In[173]:


check_prediction_diff(predictions_ARIMA_diff, train)


# In[174]:


def check_prediction(predict_diff, given_set):
    predict_diff= predict_diff.cumsum().shift().fillna(0)
    predict_base = pd.Series(np.ones(given_set.shape[0]) * np.log(given_set['Count'])[0], index = given_set.index)
    predict_log = predict_base.add(predict_diff,fill_value=0)
    predict = np.exp(predict_log)
    
    plt.plot(given_set['Count'], label = "Given set")
    plt.plot(predict, color = 'red', label = "Predict")
    plt.legend(loc= 'best')
    plt.title('RMSE: %.4f'% (np.sqrt(np.dot(predict, given_set['Count']))/given_set.shape[0]))
    


# In[175]:


check_prediction(predictions_ARIMA_diff, train)


# In[176]:



start = train.shape[0]
end = start + valid.shape[0]
valid_predict_diff = results_ARIMA.predict(start = start-1, end = end-2, typ = 'levels')
print (valid_predict_diff.head())
print (valid_predict_diff.tail())


# In[177]:


check_prediction_log(valid_predict_diff, valid)

