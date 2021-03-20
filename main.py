# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 23:56:49 2021

@author: Muffler
"""
from LSTM import LSTM_stock

def main():
    
    token = ''
    #use your own tushare token
    file = r'stock_data.xlsx'
    
    predict = LSTM_stock(token, file)
    predict.model()


if __name__ == "__main__":
    main()