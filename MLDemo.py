#Data Imports
import numpy as np
import pandas as pd
import yfinance as yf

#ML Imports
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#Data Cleaing
def processData(stock):
    stock = yf.download(stock, start = "2024-09-24", end = "2025-09-24")
    stock.columns = ["Open", "High", "Low", "Close", "Volume"]
    stock.dropna(inplace = True)
    print(stock.head())
    return stock

data = processData("TSLA")

#Data Features
data["Return"] = data["Close"].pct_change()
data["MA5"] = data["Close"].rolling(5).mean()
data["MA10"] = data["Close"].rolling(10).mean()
data["EMA10"] = data["Close"].ewm(span=10, adjust=False).mean()
data["Spread"] = data["High"] - data["Low"]
data["STD10"] = data["Close"].rolling(10).std()
data["Upper_BB"] = data["MA10"] + (data["STD10"]*2)
data["Lower_BB"] = data["MA10"] - (data["STD10"]*2)
data["Target"] = (data["Close"].shift(-1)>data["Close"]).astype(int)
data.dropna(inplace=True)

#Create Features/Target Variables
X = data[["Return", "MA5", "MA10", "EMA10", "Spread", "Upper_BB", "Lower_BB"]]
Y = data["Target"]

#Split Data
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.25, random_state=42)

#Scaling
scaler = MinMaxScaler()
trainX_scaled = scaler.fit_transform(trainX)
testX_scaled = scaler.transform(testX)

#Hyperparameter Tuning of KNN Model
parameterGrid = {
    "metric": ["euclidean", "manhattan", "minkowski"],
    "weights": ["uniform", "distance"]
}

model = KNeighborsClassifier()
gridSearch = GridSearchCV(model, parameterGrid, cv=5, n_jobs=-1)
gridSearch.fit(trainX_scaled, trainY)
bestModel = gridSearch.best_estimator_

#Predictions and Evaluate
predY = bestModel.predict(testX_scaled)
print(accuracy_score(testY, predY))
