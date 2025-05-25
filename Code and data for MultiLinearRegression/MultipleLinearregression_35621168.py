'''
Description: This file performs Multiple Linear Regression to forecast Global Mean Surface Temperature Anomaly (MSTA).
Step1: Load and Preprocess Data
Step2: Build Regression Model
Step3: Forecast Independent Variables
Step4: Predict Future MSTA
Step5: Model Evaluation
Step6: Visualize Forecast
Step7: Output Results
'''



# import packages
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

print(matplotlib.matplotlib_fname())
# 1. Data Loading and Preprocessing (Original code by user)
def load_and_preprocess():
    # (1) Load MSTA data
    MSAT = pd.read_csv(
        r'C:\Users\mac\Desktop\Forcasting CW\HadCRUT.5.0.2.0.analysis.summary_series.global.monthly.csv',
        parse_dates=['Time'], index_col='Time')
    MSAT.index.name = 'Date'

    # (2) Load CH4 data
    CH4 = pd.read_csv(r"C:\Users\mac\Desktop\Forcasting CW\ch4_NOAA CH4.csv")
    CH4['Date'] = pd.to_datetime(CH4['Year'].astype(str) + '-' + CH4['Month'].astype(str).str.zfill(2))
    CH4.set_index('Date', inplace=True)
    CH4 = CH4[['NOAA CH4 (ppb)']]
    CH4.columns = ['CH4']

    # (3) Load GMAF data
    GMAF = pd.read_csv(r"C:\Users\mac\Desktop\Forcasting CW\ott.csv")
    GMAF['Date'] = pd.to_datetime(GMAF['Date'], format='%Y %b')
    GMAF.set_index('Date', inplace=True)
    GMAF = GMAF[['GMAF']]

    # (4) Load ET12 data
    ET12 = pd.read_excel(
        r'C:\Users\mac\Desktop\Forcasting CW\ET12.xlsx',
        parse_dates=['Date'], index_col='Date', engine='openpyxl')
    ET12 = ET12[['Unadjusted total Consumption']]
    ET12.columns = ['ET12']

    # Merge all data
    df = pd.concat([MSAT, CH4, GMAF, ET12], axis=1, join='outer')
    df.columns = ['MSTA', 'CH4', 'GMAF', 'ET12']
    df = df.loc['1995-01':'2024-12']
    df = df.interpolate(method='time')
    return df


# 2. Build Multiple Linear Regression Model (Original code by user)
def build_multiple_regression(df):
    # Extract features (X) and target (y)
    X = sm.add_constant(df[['CH4', 'GMAF', 'ET12']])
    y = df['MSTA']

    # Before fitting the model, calculate VIF to check multicollinearity
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print("\nVariance Inflation Factor (VIF) Check Results:")
    print(vif_data)

    # Fit the regression model
    model = sm.OLS(y, X).fit()
    print("\nMultiple Linear Regression Model Summary:")
    print(model.summary())

    return model

# 3. Use Simple Linear Regression to Forecast Independent Variables
def forecast_features_with_linear(df, features):
    # Generate time index (starting from 0 for months)
    time_index = np.arange(len(df)).reshape(-1, 1)  # X is a 2D array

    forecasts = {}
    for col in features:
        # Build simple linear regression model
        X = sm.add_constant(time_index)
        y = df[col]
        model = sm.OLS(y, X).fit()

        # Forecast the next 12 months
        future_time = np.arange(len(df), len(df) + 12).reshape(-1, 1)
        X_future = sm.add_constant(future_time)
        forecast = model.predict(X_future)
        forecasts[col] = forecast

    # Generate future date index
    future_dates = pd.date_range(
        start=df.index[-1] + pd.DateOffset(months=1),
        periods=12,
        freq='MS'
    )
    return pd.DataFrame(forecasts, index=future_dates)


# 4. Model Evaluation (added model evaluation and diagnostics)
def evaluate_model(model, X, y):
    import pandas as pd
    import numpy as np
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error, r2_score

# 2. Variance Inflation Factor (VIF) to check multicollinearity
    X = sm.add_constant(df[['CH4', 'GMAF', 'ET12']])
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print("\nVariance Inflation Factor (VIF) Check Results:")
    print(vif_data)

# 3. Multiple Linear Regression Model
# Define the dependent and independent variables
    y = df['MSTA']
    X = sm.add_constant(df[['CH4', 'GMAF', 'ET12']])

 # Fit the regression model
    model = sm.OLS(y, X).fit()

 # Print the model summary
    print("\nMultiple Linear Regression Model Summary:")
    print(model.summary())

    # 4. Model Evaluation (Evaluation of Regression Model)
    # Predict y values based on the model
    y_pred = model.predict(X)
    # Calculate model evaluation metrics
    mse = mean_squared_error(y, y_pred)  # Mean Squared Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    mae = np.mean(np.abs(y - y_pred))  # Mean Absolute Error
    mape = np.mean(np.abs((y - y_pred) / y)) * 100  # Mean Absolute Percentage Error (MAPE)
    # Display evaluation metrics
    print("\nModel Evaluation Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")  # Print Mean Absolute Percentage Error
    # 5. Residuals and QQ Plot
    # Calculate residuals
    residuals = y - y_pred
    # Plot the residuals
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(residuals)
    plt.title('Residuals Plot')
    plt.xlabel('Time')
    plt.ylabel('Residuals')

    # QQ plot of residuals
    plt.subplot(1, 2, 2)
    sm.qqplot(residuals, line='s', ax=plt.gca())  # Ensure QQ plot uses the correct axis
    plt.title('QQ Plot of Residuals')

    plt.tight_layout()
    plt.show()
    # 6. Forecast MSTA for 2025 (using the regression model)
    # Forecasting the independent variables for the next 12 months
    future_features = pd.DataFrame({
        'CH4': np.linspace(df['CH4'].iloc[-1], df['CH4'].iloc[-1] + 10, 12),  # Simulating values for example
        'GMAF': np.linspace(df['GMAF'].iloc[-1], df['GMAF'].iloc[-1] + 10, 12),
        'ET12': np.linspace(df['ET12'].iloc[-1], df['ET12'].iloc[-1] + 10, 12),
    }, index=pd.date_range(start='2025-01-01', periods=12, freq='MS'))
    # Adding a constant term for intercept
    future_features = sm.add_constant(future_features)
    # Use the trained regression model to forecast MSTA
    msta_forecast = model.predict(future_features)
    # Display forecasted MSTA values
    forecast_df = pd.DataFrame({
        'Date': future_features.index.strftime('%Y-%m'),
        'MSTA Forecast': msta_forecast.round(4)
    })

    print("\n2025 MSTA Forecast Results:")
    print(forecast_df.to_string(index=False))

# 5. Main Program Flow

if __name__ == "__main__":
    # Data loading and preprocessing
    df = load_and_preprocess()

    # Build the multiple linear regression model
    regression_model = build_multiple_regression(df)

    # Forecast future values for independent variables
    future_features = forecast_features_with_linear(df, ['CH4', 'GMAF', 'ET12'])
    print("\nForecasted Future Independent Variables:")
    print(future_features.round(4))

    # Forecast MSTA
    X_future = sm.add_constant(future_features)
    msta_forecast = regression_model.predict(X_future)

    # Output forecast results
    forecast_df = pd.DataFrame({
        'Date': future_features.index.strftime('%Y-%m'),
        'MSTA Forecast': msta_forecast.round(4)
    })
    print("\n2025 MSTA Forecast Results:")
    print(forecast_df.to_string(index=False))

    # Model evaluation (evaluation of regression model)
    X = sm.add_constant(df[['CH4', 'GMAF', 'ET12']])
    y = df['MSTA']
    evaluate_model(regression_model, X, y)

    # Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(df['MSTA'], label='Historical Data', linewidth=1)
    plt.plot(future_features.index, msta_forecast,
             label='2025 Forecast', linestyle='--',color="red",linewidth=2)
    plt.title('MSTA Forecast Based on Linear Regression')
    plt.xlabel('Date')
    plt.ylabel('Temperature Anomaly (°C)')
    plt.legend()
    plt.grid(True)
    plt.show()
