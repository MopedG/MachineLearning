import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import os

# Streamlit app
st.set_page_config(page_title="ARIMA/SARIMA Time Series Analysis Dashboard", layout="wide")

# Header
st.title("ARIMA/SARIMA Time Series Analysis Dashboard")

# Upload dataset files
uploaded_files = {
    "AirPassengers.csv": os.path.abspath("C:\\Entw\\MachineLearning\\Ticketverkäufe\\AirPassengers.csv"),
    "monthly_sales.csv": "monthly_sales.csv",
    "gold_monthly.csv": "gold_monthly.csv"
}

# Sidebar selection
st.sidebar.header("Dataset Selection and Stationarity")

data_source = st.sidebar.selectbox("Select Dataset", list(uploaded_files.keys()), key="Data_Source_Selection")
make_stationary = st.sidebar.checkbox("Difference the time series to make it stationary")

# Load selected dataset
df = pd.read_csv(uploaded_files[data_source], parse_dates=['Month'], index_col='Month')
y = df.iloc[:, 0]

# Area 1: Plotting Time Series with ADF Test
st.header("Area 1: Time Series Plot with ADF Test")

if make_stationary:
    y_diff = y.diff().dropna()
    adf_result = sm.tsa.adfuller(y_diff)
    y_to_plot = y_diff
else:
    adf_result = sm.tsa.adfuller(y)
    y_to_plot = y

# Plotting time series
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(y_to_plot, label="Time Series")
plt.title("Time Series Plot")
plt.xlabel("Date")
plt.ylabel("Values")
plt.legend([f"ADF-Statistik: {adf_result[0]:.2f}, p-Wert: {adf_result[1]:.4f}"])

st.pyplot(fig1)

# Area 2: Autocorrelation and Partial Autocorrelation Plots
st.header("Area 2: Autocorrelation and Partial Autocorrelation")

fig2, ax2 = plt.subplots(1, 2, figsize=(16, 6))

# Lollipop diagrams for ACF and PACF
plot_acf(y_to_plot, ax=ax2[0], title="Autocorrelation")
plot_pacf(y_to_plot, ax=ax2[1], title="Partial Autocorrelation")

st.pyplot(fig2)

# Area 3: ARIMA/SARIMA Model Fitting and Prediction
st.header("Area 3: ARIMA/SARIMA Model Prediction")

# Model selection
model_type = st.selectbox("Select Model Type", ["ARIMA", "SARIMA"])

# Fields to enter p, d, q, and n
p = st.number_input("Enter p (autoregressive order)", min_value=0, value=1, step=1)
d = st.number_input("Enter d (differencing order)", min_value=0, value=1, step=1)
q = st.number_input("Enter q (moving average order)", min_value=0, value=1, step=1)
n = st.number_input("Enter number of prediction steps (n)", min_value=1, value=6, step=1)

# Additional fields for SARIMA
if model_type == "SARIMA":
    P = st.number_input("Enter P (seasonal autoregressive order)", min_value=0, value=1, step=1)
    D = st.number_input("Enter D (seasonal differencing order)", min_value=0, value=1, step=1)
    Q = st.number_input("Enter Q (seasonal moving average order)", min_value=0, value=1, step=1)
    s = st.number_input("Enter s (seasonal period)", min_value=1, value=12, step=1)

if st.button("Submit", key="Determined_p_d_q_values_and_number_of_prediction_steps"):
    if model_type == "ARIMA":
        model = ARIMA(y, order=(p, d, q))
    elif model_type == "SARIMA":
        model = SARIMAX(y, order=(p, d, q), seasonal_order=(P, D, Q, s))
    
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=n)

    # Create forecast index
    forecast_index = pd.date_range(y.index[-1] + pd.DateOffset(months=1), periods=n, freq='M')

    # Plotting actual data, forecast, and model fit
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(y, label="Actual Data")
    ax3.plot(forecast_index, forecast, label="Forecast", color="red")
    ax3.plot(y.index, model_fit.fittedvalues, label=f"{model_type} Fit", color="purple")
    plt.title("Actual vs Forecasted Data")
    plt.xlabel("Date")
    plt.ylabel("Values")
    plt.legend()

    st.pyplot(fig3)

    # Calculate RMSE
    if len(y) >= n:
        y_true = y[-n:]
        y_pred = model_fit.predict(start=len(y)-n, end=len(y)-1)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        st.write(f"Root Mean Square Error (RMSE): {rmse:.2f}")
