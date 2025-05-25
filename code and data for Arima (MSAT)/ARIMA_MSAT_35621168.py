'''
Description: This file applies ARIMA modeling for MSTA data set.
Step1: Data Preparation and Visualization
Step2: Stationarity Test (ADF Test)
Step3: Differencing (if data is non-stationary)
Step4: Determine ARIMA Parameters (p, d, q)
Step5: Model Fitting
Step6: Model Diagnostics
Step7: Model Prediction and Comparison
'''
import numpy as np
# 1. Data Preparation and Visualization
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

# Load MSTA data
# Example data path, should be replaced with the actual path
msta = pd.read_csv(
    r'C:\Users\mac\Desktop\Forcasting CW\HadCRUT.5.0.2.0.analysis.summary_series.global.monthly.csv',
    parse_dates=['Time'],
    index_col='Time'
)['Anomaly (deg C)']
msta = msta.asfreq('MS')
# Visualize the raw data
plt.figure(figsize=(12, 6))
plt.plot(msta)
plt.title('Global Average Surface Temperature Anomaly (MSTA) Time Series')
plt.xlabel('Date')
plt.ylabel('Temperature Anomaly (°C)')
plt.grid(True)
plt.show()

# 2. Stationarity Test (ADF Test)
# Define ADF test function
def adf_test(timeseries):
    result = adfuller(timeseries.dropna())
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value}')

# Perform ADF test on the original data
print("ADF test result for the original data:")
adf_test(msta)

# 3. Differencing (if data is non-stationary)
# First-order differencing
msta_diff1 = msta.diff().dropna()

# Perform ADF test again on the differenced data
print("ADF test result for the first-order differenced data:")
adf_test(msta_diff1)

# Visualize the differenced data
plt.figure(figsize=(12, 6))
plt.plot(msta_diff1)
plt.title('First-order Differenced MSTA Series')
plt.xlabel('Date')
plt.ylabel('Differenced Values')
plt.grid(True)
plt.show()

# 4. Determine ARIMA Parameters (p, d, q)
# Method 1: Observing ACF and PACF plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(msta_diff1, lags=40, ax=ax1)
plot_pacf(msta_diff1, lags=40, ax=ax2, method='ywm')
plt.tight_layout()
plt.show()

# Method 2: Automatic Order Selection (Recommended)
auto_model = auto_arima(
    msta,
    seasonal=False,  # Non-seasonal model
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True
)
print("Automatic order selection result:")
print(auto_model.summary())

# 5. Model Fitting
# Assume the automatic order selection result is ARIMA(1,1,2)
model = ARIMA(msta, order=(1, 1, 2))
model_fit = model.fit()
print(model_fit.summary())

# 6. Model Diagnostics
# Residual Analysis
residuals = model_fit.resid

# Plot ACF of residuals and QQ plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(residuals, lags=40, ax=ax1)
sm.qqplot(residuals, line='s', ax=ax2)
ax1.set_title('Residual ACF Plot')
ax2.set_title('Residual QQ Plot')
plt.show()

# Ljung-Box Test (Check if residuals are white noise)
from statsmodels.stats.diagnostic import acorr_ljungbox

lb_test = acorr_ljungbox(residuals, lags=20)
print("Ljung-Box test p-values:\n", lb_test['lb_pvalue'])
# If all p-values > 0.05, residuals are white noise

# 7. Model Prediction and Comparison
# Forecast the next 12 months (January to December 2025)
forecast_steps = 12
forecast = model_fit.get_forecast(steps=forecast_steps)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# Visualize the forecast results
plt.figure(figsize=(12, 6))
plt.plot(msta, label='Actual Values')
plt.plot(forecast_mean, label='Forecasted Values', color='red')
plt.fill_between(
    forecast_ci.index,
    forecast_ci.iloc[:, 0],
    forecast_ci.iloc[:, 1],
    color='pink',
    alpha=0.3
)
plt.title('ARIMA Model Forecast Results')
plt.xlabel('Date')
plt.ylabel('Temperature Anomaly (°C)')
plt.legend()
plt.grid(True)
plt.show()


# 生成样本内预测值（历史数据的拟合值）
fitted_values = model_fit.get_prediction().predicted_mean

# 对齐实际值（因为ARIMA(p,d,q)需要丢弃前d个观测值）
actual = msta.iloc[1:]  # 假设d=1，丢弃第一个观测值
predicted = fitted_values[1:]  # 保持相同长度

# 计算误差指标
mse = mean_squared_error(actual, predicted)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actual, predicted)
mape = np.mean(np.abs((actual - predicted) / actual)) * 100

print("\nModel Evaluation Metrics:")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"MAPE : {mape:.2f}%")
