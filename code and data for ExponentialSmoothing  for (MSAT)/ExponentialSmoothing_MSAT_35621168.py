'''
Description: This file will apply the Exponential smoothing for MSAT
step1: import data and package
step2: Loading data
step3: Plotting the original time series
step4: Handling missing values
step5: Calculate and plot the ACF
step6: Holt exponential smoothing
step7: Forecast data until December 2025
step8: Draw the prediction result curve
step9: Error calculation
step10: Comparison of forecast and actual data
'''

# 1.import data and package

# Import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf

# Define data path
file_path = r'C:\Users\mac\Desktop\Forcasting CW\HadCRUT.5.0.2.0.analysis.summary_series.global.monthly.csv'

#2. Loading data

# Load data
series = pd.read_csv(file_path, header=0, index_col=0, parse_dates=True)

# Set the frequency to monthly (Month Start frequency)
series = series.asfreq('MS')  # Set the frequency explicitly to monthly start

#3. Plotting the original time series

# Plotting the original time series
series["Anomaly (deg C)"].plot(title="MSAT_Time-Anomaly", xlabel="Time", ylabel="Anomaly")
plt.show()

#4. Handling missing values

# Check for missing values
missing_rows = series[series["Anomaly (deg C)"].isna()]
if not missing_rows.empty:
    print(missing_rows)
else:
    print("No any NA values")

#5. Calculate and plot the ACF

# Plotting ACF
y = series["Anomaly (deg C)"]
plot_acf(y, lags=55, title="ACF of MSAT Anomaly")
plt.show()

#6. Holt exponential smoothing

# Exponential smoothing method selection (with seasonal component)
model = ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=12)
fit = model.fit()  # Fit the model
y_smooth = fit.fittedvalues
print(fit.params)
#7. Forecast data until December 2025

# Forecast future data (extended to December 2025)
forecast_steps = (2025 - y.index[-1].year) * 12 + (12 - y.index[-1].month)  # Calculate the number of months to predict
forecast = fit.forecast(forecast_steps)  # Forecast future data

#8. Draw the prediction result curve

# Plotting forecasting result
plt.figure(figsize=(12, 6))
forecast_range = forecast.loc["2025-01":"2025-12"]
plt.plot(forecast_range, label="Forecast to 2025-12", color="green", linestyle="dotted", linewidth=2)
plt.title("Forecast (2025-01 to 2025-12)")
plt.xlabel("Time")
plt.ylabel("Anomaly (deg C)")
plt.legend()
plt.tight_layout()
plt.show()

#9. Error calculation

# Calculate error metrics
mse = mean_squared_error(y, y_smooth)  # MSE
rmse = np.sqrt(mse)  # RMSE
mae = mean_absolute_error(y, y_smooth)  # MAE
mape = np.mean(np.abs((y - y_smooth) / y)) * 100  # MAPE

# Print error metrics
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.2f}%")

#10. Comparison of forecast and actual data

# Plot actual data vs. predicted
plt.figure(figsize=(12, 6))
plt.plot(y, label="Actual Data", color="blue", linewidth=1)
plt.plot(y_smooth, label="Holt-Winters Smoothing", color="red", linestyle="--", linewidth=2)
plt.plot(forecast, label="Forecast (2025)", color="green", linestyle="dotted", linewidth=2)

# Calculate the 95% confidence interval
ci_std = np.std(y - y_smooth)  # Residual standard deviation
ci_upper = forecast + 1.96 * ci_std  # Upper confidence bound
ci_lower = forecast - 1.96 * ci_std  # Lower confidence bound
plt.fill_between(forecast.index, ci_lower, ci_upper, color="gray", alpha=0.2, label="95% Confidence Interval")

plt.title("Actual vs Predicted MSAT Anomaly")
plt.xlabel("Time")
plt.ylabel("Anomaly (deg C)")
plt.legend()
plt.grid(True)
plt.show()
