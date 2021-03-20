#### LSTM Method in Stock Return

This project uses LTSM method to predict the stock return.

- data: stock price , pe and turnover of 600309.SH
- period: 20190101-20210228
- train period: 20190101-20201231
- test period: 20210101-20210228

##### Dependency package

- tensorflow
- keras
- sklearn

##### Instruction
Download all the files at the same place and run the main.py

- LSTM.py: this python file contains functions of data processing and LSTM model, and I package it as a class named LTSM_stock
- main.py: the main function to get the final outcome
- stock_data.xlsx: this excel file is the stock data getting from tushare that uses the getdata function in the LSTM.py

##### Contributions and thoughts

- The reference blog uses the period from 20160301 to 20171231, but the market in this period performs stable or even good. In order to test a different period, I choose the recent years and 2019 had a small stock crush and the covid-19 affected the world, so this period do not have a good prediction, which shows that this model in the blog is not always right.
- Also, I put the code into a package, making it easier for others to use.
- This model can be optimized in many ways. For example, we can change the period, parameters in the LSTM to get different outcome.
