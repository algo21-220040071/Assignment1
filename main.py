# -*- coding: utf-8 -*-
"""
@author: Liran CHEN 220040071
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
