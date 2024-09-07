# stock_prediction.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import yfinance as yf

def fetch_data(ticker):
    # Fetch historical stock data from Yahoo Finance
    data = yf.download(ticker, start="2020-01-01", end="2024-01-01")
    data = data[['Close']]
    return data

def prepare_data(data):
    # Prepare data for training and testing
    data['Lag1'] = data['Close'].shift(1)
    data = data.dropna()
    X = data[['Lag1']]
    y = data['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, data):
    # Evaluate the model and return MSE and prediction results
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    predicted_dates = data.index[-len(predictions):]
    prediction_results = list(zip(predicted_dates, predictions))
    return mse, prediction_results

def plot_predictions(data, predictions, dates):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'], label='Actual Prices', color='blue')
    plt.plot(dates, predictions, label='Predicted Prices', color='red')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Stock Price Prediction')
    plt.legend()
    plt.tight_layout()  # Adjust plot layout for better fit
