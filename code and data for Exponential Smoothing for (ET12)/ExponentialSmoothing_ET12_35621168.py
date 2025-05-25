''''
This code uses the Exponential Smoothing method to make time series forecasts for ET12 data.

Steps:
1. Import necessary libraries
2. Load Data
3. Data Preprocessing
4. Plot Time-Series Data & ACF
5. Apply Holt-Winters Triple Exponential Smoothing
6. Forecast Until December 2025
7. Plot the smoothed data and forecast
8. Model Evaluation
'''




#Step 1: Import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import mean_absolute_error, mean_squared_error

#Step 2: Load Data
file_path = r'C:\Users\mac\Desktop\Forcasting CW\ET12.xlsx'  # 修改为 ET12 文件路径
# 加载数据时不使用 'date_parser'，直接使用 parse_dates 来指定日期列
series = pd.read_excel(file_path, header=0, index_col=0, parse_dates=[0])
# 显式转换为日期格式，避免警告
series.index = pd.to_datetime(series.index)
series.columns = ["Consumption"]  # 规范列名
series = series.asfreq("MS")
print(series.info())
print(series.head())

#Step 3: Data Preprocessing
# Check NA
if series.isnull().values.any():
    print("Warning: There are missing values in the data, please process!！")
    print(series[series["Consumption"].isna()])
else:
    print("No missing values!")

# Step 4: Plot Time-Series Data & ACF
plt.figure(figsize=(12, 5))
#Time-series
plt.plot(series, label="ET12 Consumption", color="blue", linewidth=1)
plt.title("ET12 Time-Series Data")
plt.xlabel("Time")
plt.ylabel("Consumption")
plt.legend()
plt.grid()
plt.show()

#Ploting ACF
plot_acf(series["Consumption"], lags=30)
plt.title("Autocorrelation Plot of ET12 Consumption")
plt.show()

# Step 5: Apply Holt-Winters Triple Exponential Smoothing
# Choose an additive model
model = ExponentialSmoothing(series["Consumption"],
                             trend="add",      # trend
                             seasonal="add",   # seasonal
                             seasonal_periods=12)  # period=12

fit = model.fit(optimized=True)  # Automatic optimization of smoothing parameters
y_smooth = fit.fittedvalues
print(fit.params)
#Step 6: Forecast Until December 2025
# Calculate the number of forecast steps to 2025-12
forecast_steps = (2025 - series.index[-1].year) * 12 + (12 - series.index[-1].month)
forecast = fit.forecast(forecast_steps)

# Step 7: Plot the smoothed data and forecast
plt.figure(figsize=(12, 6))

# Orighnal data
plt.plot(series, label="Actual Data", color="blue", linewidth=1)

# Holt-Winters smoothing
plt.plot(y_smooth, label="Holt-Winters Smoothing", color="red", linestyle="--", linewidth=2)

# Future predictions
forecast_index = pd.date_range(start=series.index[-1] + pd.DateOffset(months=1), periods=forecast_steps, freq="MS")
plt.plot(forecast_index, forecast, label="Forecast (2025)", color="green", linestyle="dotted", linewidth=2)

# Calculate 95% confidence interval
ci_std = np.std(series["Consumption"] - y_smooth)
ci_upper = forecast + 1.96 * ci_std
ci_lower = forecast - 1.96 * ci_std
plt.fill_between(forecast_index, ci_lower, ci_upper, color="gray", alpha=0.2, label="95% Confidence Interval")

plt.title("Holt-Winters Forecasting (Additive)")
plt.xlabel("Time")
plt.ylabel("Consumption")
plt.legend()
plt.grid()

plt.show()

#Step 8: Model Evaluation
mse = mean_squared_error(series["Consumption"], y_smooth)
rmse = np.sqrt(mse)
mae = mean_absolute_error(series["Consumption"], y_smooth)
mape = np.mean(np.abs((series["Consumption"] - y_smooth) / series["Consumption"])) * 100

# print
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.2f}%")
