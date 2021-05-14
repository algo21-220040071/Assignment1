# -*- coding: utf-8 -*-
"""
@author: Liran CHEN 220040071
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from math import sqrt
import tushare as ts

class LSTM_stock:
    def __init__(self, token, file):
        self.token = token
        self.file = file
    
    #get the stock data from tushare
    def getdata(self):
        #use own tushare token
        TOKEN = self.token   
        ts.set_token(TOKEN)
        pro = ts.pro_api()
        #get the daily price of stock 600309.SH
        daily = pro.daily(ts_code = '600309.SH', start_date = '20190101',
                          end_date = '20210228', 
                          fields = 'trade_date,open,high,low,close')
        #get the daily basic info of stock 600309.SH
        basic = pro.daily_basic(ts_code = '600309.SH', start_date = '20190101',
                                end_date = '20210228', 
                                fields = 'trade_date,turnover_rate,pe')
        
        #data process
        daily = daily.sort_values(by ='trade_date').reset_index(drop=True)
        basic = basic.sort_values(by ='trade_date').reset_index(drop=True)
        daily.index = daily['trade_date']
        basic.index = basic['trade_date']
        daily = daily.iloc[:,1:]
        basic = basic.iloc[:,1:]
        
        stock_data = pd.concat([daily, basic], axis = 1)
        
        stock_data['return'] = stock_data['close'].pct_change()
        
        #write into excel file
        stock_data.to_excel(r'stock_data.xlsx')

    #read excel file
    def file_read(self):
        data = pd.read_excel(self.file,index_col = 0)
        data = data.dropna()
        data.index = data.index.astype('str')
        return data
    
    #Normalization
    def normalization(self,data):
        values = data.values
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled = scaler.fit_transform(values)
        return scaler, scaled
    #Transform series to supervised data
    def series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):
    	n_vars = 1 if type(data) is list else data.shape[1]
    	df = pd.DataFrame(data)
    	cols, names = list(), list()
    	# input sequence (t-n, ... t-1)
    	for i in range(n_in, 0, -1):
    		cols.append(df.shift(i))
    		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    	# forecast sequence (t, t+1, ... t+n)
    	for i in range(0, n_out):
    		cols.append(df.shift(-i))
    		if i == 0:
    			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
    		else:
    			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    	# put it all together
    	agg = pd.concat(cols, axis=1)
    	agg.columns = names
    	# drop rows with NaN values
    	if dropnan:
    		agg.dropna(inplace=True)
    	return agg
    

    #model building and prediction
    def model(self):
        #read the data
        stock_data = self.file_read()
        #normalization
        scaler, scaled = self.normalization(stock_data)
        #transform the time  series to supervised data
        reframed = self.series_to_supervised(scaled,1,1)
        reframed.drop(reframed.columns[[7,8,9,10,11,12]], axis = 1, 
                      inplace = True)
        
        #split the data into train and test data
        values = reframed.values
        num = [i for i in stock_data.index if i[0:4] == '2021']
        test_cnt = np.size(num)
        train_cnt = np.size(stock_data.index) - test_cnt
        train = values[:train_cnt,:]
        test = values[train_cnt:,:]
        
        #split the input and output of train data and test data
        train_x, train_y = train[:, :-1], train[:, -1]
        test_x, test_y = test[:, :-1], test[:, -1]
        
        #reshape input to be 3D data like [sample, timesteps, features]
        train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
        test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
        model = Sequential()
        model.add(LSTM(64, input_shape=(train_x.shape[1], train_x.shape[2])))
        model.add(Dropout(0.5))
        model.add(Dense(1,activation='relu'))
        model.compile(loss='mae', optimizer='adam')
        #fit network
        history = model.fit(train_x, train_y, epochs=50, batch_size=100, 
                            validation_data=(test_x, test_y), 
                            verbose=2,shuffle=False)
         
        #draw the loss graph
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.title('LSTM_600000.SH', fontsize='12')
        plt.ylabel('loss', fontsize='10')
        plt.xlabel('epoch', fontsize='10')
        plt.legend()
        plt.show()
        
        #use the model to predict the return
        y_predict = model.predict(test_x)
        test_x = test_x.reshape((test_x.shape[0], test_x.shape[2]))
         
        #invert scaling for forecast
        inv_y_test = np.concatenate((test_x[:, :6],y_predict), axis=1)
        inv_y_test = scaler.inverse_transform(inv_y_test)
        inv_y_predict=inv_y_test[:,-1]
         
        #invert scaling for actual
        test_y = test_y.reshape((len(test_y), 1))
        inv_y_train = np.concatenate((test_x[:, :6],test_y), axis=1)
        inv_y_train = scaler.inverse_transform(inv_y_train)
        inv_y = inv_y_train[:, -1]
        print('反归一化后的预测结果：',inv_y_predict)
        print('反归一化后的真实结果：',inv_y)

        plt.plot(inv_y,color='red',label='Original')
        plt.plot(inv_y_predict,color='green',label='Predict')
        plt.xlabel('the number of test data')
        plt.ylabel('earn_rate')
        plt.title('2019.1—2020.12')
        plt.legend()
        plt.show()
        
        #regression evaluation index
        # calculate MSE
        mse=mean_squared_error(inv_y,inv_y_predict)
        # calculate RMSE
        rmse = sqrt(mean_squared_error(inv_y, inv_y_predict))
        #calculate MAE
        mae=mean_absolute_error(inv_y,inv_y_predict)
        #calculate R square
        r_square=r2_score(inv_y,inv_y_predict)
        print('均方误差: %.6f' % mse)
        print('均方根误差: %.6f' % rmse)
        print('平均绝对误差: %.6f' % mae)
        print('R_square: %.6f' % r_square)
