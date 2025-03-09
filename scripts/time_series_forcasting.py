import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error

class TimeSeriesForecaster:
    def __init__(self, asset_name, prices, returns):
        self.asset_name = asset_name
        self.prices = prices[asset_name]
        self.returns = returns[asset_name]
        self.model = None
        self.model_fit = None
    
    def check_stationarity(self, timeseries):
        result = adfuller(timeseries.dropna())
        print(f'ADF Statistic: {result[0]:.4f}')
        print(f'p-value: {result[1]:.4f}')
        print('Critical Values:')
        for key, value in result[4].items():
            print(f'\t{key}: {value:.4f}')
        if result[1] <= 0.05:
            print("Result: The series is stationary (reject H0)")
        else:
            print("Result: The series is non-stationary (fail to reject H0)")
    
    def decompose_series(self):
        decomposition = seasonal_decompose(self.prices, model='multiplicative', period=252)
        fig = decomposition.plot()
        plt.tight_layout()
        plt.show()
    
    def fit_arima(self, p, d, q):
        train_size = int(len(self.returns) * 0.9)
        train_data = self.returns.iloc[:train_size]
        
        self.model = ARIMA(train_data, order=(p, d, q))
        self.model_fit = self.model.fit()
        print(self.model_fit.summary())
    
    def forecast_and_evaluate(self):
        train_size = int(len(self.returns) * 0.9)
        test_data = self.returns.iloc[train_size:]
        forecast_steps = len(test_data)
        forecast = self.model_fit.forecast(steps=forecast_steps)
        
        mae = mean_absolute_error(test_data, forecast)
        rmse = np.sqrt(mean_squared_error(test_data, forecast))
        print(f"\nModel Evaluation on Test Data:")
        print(f"Mean Absolute Error (MAE): {mae:.6f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
        
        plt.figure(figsize=(15, 8))
        plt.plot(test_data.index, test_data, label='Actual Returns', color='blue')
        plt.plot(test_data.index, forecast, label='Forecasted Returns', color='red', linestyle='--')
        plt.title(f'{self.asset_name} - Actual vs Forecasted Returns')
        plt.xlabel('Date')
        plt.ylabel('Return')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def future_forecast(self, future_steps=30):
        future_forecast = self.model_fit.forecast(steps=future_steps)
        future_dates = pd.date_range(start=self.returns.index[-1], periods=future_steps)
        
        plt.figure(figsize=(15, 8))
        plt.plot(self.returns.index[-90:], self.returns.iloc[-90:], label='Historical Returns', color='blue')
        plt.plot(future_dates, future_forecast, label='Future Returns Forecast', color='red', linestyle='--')
        plt.title(f'{self.asset_name} - Returns Forecast for Next {future_steps} Trading Days')
        plt.xlabel('Date')
        plt.ylabel('Return')
        plt.axvline(x=self.returns.index[-1], color='green', linestyle='-', label='Forecast Start')
        plt.legend()
        plt.tight_layout()
        plt.show()
