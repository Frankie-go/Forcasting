'''
Description: This file will apply the Exponential Smoothing for UK Visits Abroad.
Step1: Load Data
Step2: Data Inspection
Step3: Plot Time Series & Autocorrelation
Step4: Apply Holt-Winters Exponential Smoothing
Step5: Forecast Until December 2025
Step6: Plot Forecast Results
Step7: Compute Error Metrics
Step8: Compare Actual and Predicted
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf

# ========== 1. Load Data ==========
file_path = r"C:\Users\mac\Desktop\Forcasting CW\ott.csv"  # Modify with your data file path
df = pd.read_csv(file_path)

# Parse date format
df["Date"] = pd.to_datetime(df["Date"], format="%Y %b")
df.set_index("Date", inplace=True)

df = df.asfreq("MS")

# ========== 2. Data Check ==========
print(df.info())
print(df.head())
print("Missing values:", df.isna().sum())

# ========== 3. Plot Time-Series & ACF ==========
plt.figure(figsize=(12, 5))
plt.plot(df["GMAF"], label="UK Visits (Thousands)", color="blue")
plt.title("UK Visits Abroad Time Series")
plt.xlabel("Time")
plt.ylabel("Visits (Thousands)")
plt.legend()
plt.grid()
plt.show()

# Plot ACF
plot_acf(df["GMAF"], lags=30)
plt.title("Autocorrelation Plot of UK Visits Abroad")
plt.show()

# ========== 4. Holt-Winters Exponential Smoothing ==========
y = df["GMAF"]

model = ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=12)
fit = model.fit(optimized=True)  # Automatically find the best smoothing_level and smoothing_trend
y_smooth = fit.fittedvalues
print(fit.params)

# ========== 5. Forecast Until December 2025 ==========
forecast_steps = (2025 - y.index[-1].year) * 12 + (12 - y.index[-1].month)
forecast = fit.forecast(forecast_steps)

# ========== 6. Plot the Smoothed Data and Forecast ==========
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(y, label="Original Data", color="blue", linewidth=1)
plt.plot(y_smooth, label="Holt-Winters Smoothing", color="red", linestyle="--", linewidth=2)
plt.title("Holt-Winters Exponential Smoothing for UK Visits Abroad")
plt.xlabel("Time")
plt.ylabel("Visits (Thousands)")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(forecast, label="Forecast (2025)", color="green", linestyle="dotted", linewidth=2)
plt.title("Forecast of UK Visits Abroad (until Dec 2025)")
plt.xlabel("Time")
plt.ylabel("Visits (Thousands)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# ========== 7. Calculate Errors ==========
mse = mean_squared_error(y, y_smooth)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_smooth)
mape = np.mean(np.abs((y - y_smooth) / y)) * 100

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.2f}%")

# ========== 8. Actual vs Predicted ==========
plt.figure(figsize=(12, 6))

plt.plot(y, label="Actual Data", color="blue", linewidth=1)
plt.plot(y_smooth, label="Holt-Winters Smoothed", color="red", linestyle="--", linewidth=2)
plt.plot(forecast, label="Forecast (2025)", color="green", linestyle="dotted", linewidth=2)

# Calculate confidence interval
ci_std = np.std(y - y_smooth)  # Calculate residual standard deviation
ci_upper = forecast + 1.96 * ci_std  # 95% confidence interval upper bound
ci_lower = forecast - 1.96 * ci_std  # 95% confidence interval lower bound

plt.fill_between(forecast.index, ci_lower, ci_upper, color="gray", alpha=0.2, label="95% Confidence Interval")

plt.title("Actual vs Predicted UK Visits Abroad")
plt.xlabel("Time")
plt.ylabel("Visits (Thousands)")
plt.legend()
plt.grid()

plt.show()
