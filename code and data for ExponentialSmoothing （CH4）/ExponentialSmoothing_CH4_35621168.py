'''
Description: This file will apply the Exponential Smoothing for UK Visits Abroad.
Step1:Load Data
Step2:Data Inspection
Step3:Plot Time Series & Autocorrelation
Step4: Holt-Winters Exponential Smoothing
Step5: Forecast Until December 2025
Step6: Plot Forecast Results
Step7:Compute Error Metrics
Step8: Comparing Actual and Predicted
'''
pip install tk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf

# Step1:Load Data
file_path = r"C:\Users\mac\Desktop\Forcasting CW\ch4_NOAA CH4.csv"
df = pd.read_csv(file_path)
# Parse date format
df["Date"] = pd.to_datetime(df[["Year", "Month"]].assign(day=1))
df.set_index("Date", inplace=True)
df = df.drop(columns=["Year", "Month"])
df = df.asfreq("MS")

# Step2:Data Inspection
print(df.info())
print(df.head())
print("Missing values:", df.isna().sum())

# Step3:Plot Time Series & Autocorrelation
plt.figure(figsize=(12, 5))
#Time Series
plt.plot(df, label="NOAA CH4 (ppb)", color="blue")
plt.title("NOAA CH4 Time Series")
plt.xlabel("Time")
plt.ylabel("CH4 (ppb)")
plt.legend()
plt.grid()
plt.show()
#Autocorrelation
plot_acf(df["NOAA CH4 (ppb)"], lags=30)
plt.title("Autocorrelation Plot of NOAA CH4")
plt.show()

# Step4 Holt-Winters Exponential Smoothing
y = df["NOAA CH4 (ppb)"]

model = ExponentialSmoothing(y,
                             trend="add",
                             seasonal="add",
                             seasonal_periods=12)
fit = model.fit(optimized=True)  # find the optimal smoothing_level 和 smoothing_trend
y_smooth = fit.fittedvalues
print(fit.params)


# Step5: Forecast Until December 2025
forecast_steps = (2025 - y.index[-1].year) * 12 + (12 - y.index[-1].month)
forecast = fit.forecast(forecast_steps)

# Step6: Plot Forecast Results
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(y, label="Original Data", color="blue", linewidth=1)
plt.plot(y_smooth, label="Holt-Winters Smoothing", color="red", linestyle="--", linewidth=2)
plt.title("Holt-Winters Exponential Smoothing for NOAA CH4")
plt.xlabel("Time")
plt.ylabel("CH4 (ppb)")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(forecast, label="Forecast (2025)", color="green", linestyle="dotted", linewidth=2)
plt.title("Forecast of NOAA CH4 (until Dec 2025)")
plt.xlabel("Time")
plt.ylabel("CH4 (ppb)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# Step7:Compute Error Metrics
mse = mean_squared_error(y, y_smooth)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_smooth)
mape = np.mean(np.abs((y - y_smooth) / y)) * 100

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.2f}%")



# Step8: Comparing Actual and Predicted

plt.figure(figsize=(12, 6))

# Plotting original and predicted data
plt.plot(y, label="Actual Data", color="blue", linewidth=1)
plt.plot(y_smooth, label="Holt-Winters Smoothed", color="red", linestyle="--", linewidth=2)
plt.plot(forecast, label="Forecast (2025)", color="green", linestyle="dotted", linewidth=2)

# Compute Confidence Interval
ci_std = np.std(y - y_smooth)
ci_upper = forecast + 1.96 * ci_std
ci_lower = forecast - 1.96 * ci_std
plt.fill_between(forecast.index, ci_lower, ci_upper, color="gray", alpha=0.2, label="95% Confidence Interval")

plt.title("Actual vs Predicted NOAA CH4 Levels")
plt.xlabel("Time")
plt.ylabel("CH4 (ppb)")
plt.legend()
plt.grid()

plt.show()