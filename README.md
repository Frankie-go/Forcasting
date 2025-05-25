## 📊 Time Series Forecasting Analysis

This project analyzes multiple real-world time series using statistical forecasting models including **Exponential Smoothing**, **ARIMA**, and **Multiple Linear Regression**.

### 🧾 Datasets
The analysis focuses on the following datasets:
- **MSTA**: Global mean surface temperature anomaly
- **CH4**: Atmospheric methane concentration
- **GMAF**: UK outbound tourism (from the International Passenger Survey)
- **ET12**: UK inland energy consumption

### 📈 Methods
1. **Exponential Smoothing (Holt-Winters)**:
   - Applied to all datasets to capture trend and seasonality.
   - CH4 and ET12 showed accurate forecasting results with low MAPE.
   - GMAF and MSTA had poor results due to high short-term fluctuations.

2. **ARIMA (AutoRegressive Integrated Moving Average)**:
   - Applied to MSTA after ADF test and first-order differencing.
   - Final model: ARIMA(1,1,2)
   - Residuals passed Ljung-Box test (p=0.91), indicating a good fit.

3. **Multiple Linear Regression**:
   - Modeled MSTA as a function of CH4, GMAF, and ET12.
   - CH4 and GMAF had significant impact; ET12 was statistically insignificant.
   - R² = 0.678 and predicted temperature anomaly for December 2025 is 0.9936°C.

### 📌 Results Summary
| Model      | Best Fit Dataset | Key Metric         | Observations |
|------------|------------------|--------------------|--------------|
| Exponential Smoothing | CH4 / ET12        | MAPE < 4%          | Performs well on stable data |
| ARIMA      | MSTA             | AIC = -3010.8       | Captures trend, good residuals |
| Regression | MSTA             | R² = 0.678          | CH4 & GMAF significant predictors |

### 📁 Project Files
- Jupyter Notebook with code for all models
- Visual output: forecast plots, residual plots, QQ plots
- Excel sheets for preprocessing
- Final report in DOCX format

### ✅ Conclusion
Exponential smoothing is well-suited for stable trend-seasonal series, while ARIMA offers better modeling of non-stationary data. Multiple regression provides interpretable results, highlighting the environmental and behavioral drivers of temperature anomalies.

