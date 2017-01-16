import pandas as pd
import datetime as dt
from sklearn.linear_model import LinearRegression

stock = pd.read_csv("YAHOO-INDEX_GSPC.csv", error_bad_lines=False)
print(stock.head(5))
stock["Date"] = pd.to_datetime(stock["Date"])
stock.sort("Date", inplace = True)

stock["Av_5day"] = pd.rolling_mean(stock["Close"], window = 5).shift(1)
stock["Av_1yr"] = pd.rolling_mean(stock["Close"], window = 365).shift(1)
stock["ratio_dy"] = stock["Av_5day"]/stock["Av_1yr"]
stock["AvVol_5day"] = pd.rolling_mean(stock["Volume"], window = 5).shift(1)
stock["AvVol_1yr"] = pd.rolling_mean(stock["Volume"], window = 365).shift(1)

stock = stock[stock["Date"] > dt.datetime(year = 1990, month = 1, day = 2)]
stock.dropna(axis = 0, inplace = True)

train = stock[stock["Date"] < dt.datetime(year = 2013, month = 1, day = 1)]
test = stock[stock["Date"] >= dt.datetime(year = 2013, month = 1, day = 1)]

columns_use = ["Av_5day", "Av_1yr", "ratio_dy", "AvVol_5day", "AvVol_1yr"]
y_train = train["Close"]
y_test = test["Close"]
lr = LinearRegression()

lr.fit(train[columns_use], y_train)
train_fitted = lr.predict(train[columns_use])
predictions = lr.predict(test[columns_use])


#using standard error for model evaluation
import numpy as np
se = np.mean((predictions - y_test)**2)
print(se)