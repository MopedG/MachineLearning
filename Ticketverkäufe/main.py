import os

import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def getCsvData():
    script_dir = os.path.dirname(__file__)
    csv_path = os.path.join(script_dir, 'Ticketsales.csv')
    df = pd.read_csv(csv_path)
    df['Event_Year'] = df['Event Name'].apply(lambda x: int(x.split()[-1]))
    df = df.sort_values(by=['Event_Year', 'Relative show day'])
    return df

def plotTimeSeries(timeSeriesSelected, year):
    plt.figure(figsize=(10, 6))
    plt.plot(timeSeriesSelected['Relative show day'], timeSeriesSelected['Sum Tickets sold'], marker='o', label=f'ART SHOW YEAR {year}')
    plt.xlabel('Relative show day')
    plt.ylabel('Sum Tickets sold')
    plt.title(f'Ticketverkäufe für ART SHOW YEAR {year}')
    plt.grid(True)
    plt.legend()
    return plt


def plotEachWeek(timeSeriesSelected, year):
    plt.figure(figsize=(10, 6))
    weeks = timeSeriesSelected['Relative show day'] // 7
    for week in weeks.unique():
        week_data = timeSeriesSelected[weeks == week]
        plt.plot(week_data['Relative show day'] % 7, week_data['Sum Tickets sold'], marker='o',
                 label=f'Week {week + 1}')

    plt.xlabel('Day of the week')
    plt.ylabel('Sum Tickets sold')
    plt.title(f'Ticket verkauf, pro Woche ART SHOW YEAR {year}')
    plt.grid(True)
    plt.legend()
    return plt




def plotEachWeekday(timeSeriesSelected, saleWeekDays, year):
    marketing_campaigns = {
        1: [-38, -30, -24],  # Campaign days for year 1
        2: [-37, -22],  # Campaign days for year 2
        3: [-43, -38, -16]  # Campaign days for year 3
    }

    plt.figure(figsize=(10, 6))
    days = saleWeekDays.split(', ')
    for day in days:
        weekday_data = timeSeriesSelected[timeSeriesSelected['SaleWeekDay'] == day]
        plt.plot(weekday_data['Relative show day'], weekday_data['Sum Tickets sold'], marker='o', label=f'{day}')

    if year in marketing_campaigns:
        for campaign_day in marketing_campaigns[year]:
            plt.axvline(x=campaign_day, color='red', linestyle='--',
                        label='Marketing Campaign' if campaign_day == marketing_campaigns[year][0] else "")

    plt.xlabel('Relative show day')
    plt.ylabel('Sum Tickets sold')
    plt.title(f'Ticket sales for {saleWeekDays} in ART SHOW YEAR {year}')
    plt.grid(True)
    plt.legend()
    return plt


def dickeyFullerTest(timeSeriesSelected):
    time_series = timeSeriesSelected['Sum Tickets sold']
    result = adfuller(time_series)
    return result

def makeStationary(timeSeriesSelected):
    timeSeriesSelected = timeSeriesSelected.copy()
    timeSeriesSelected['Sum Tickets sold'] = np.log(timeSeriesSelected['Sum Tickets sold'])
    timeSeriesSelected['Sum Tickets sold'] = timeSeriesSelected['Sum Tickets sold'].diff().dropna()
    timeSeriesSelected = timeSeriesSelected.dropna().reset_index(drop=True)
    return timeSeriesSelected

def plot_acf_pacf(stationarySeries, year):
    plt.figure(figsize=(12, 6))
    plot_acf(stationarySeries['Sum Tickets sold'], lags=20, ax=plt.subplot(121))
    plt.title(f"ACF für ART SHOW YEAR {year}")

    plot_pacf(stationarySeries['Sum Tickets sold'], lags=20, ax=plt.subplot(122))
    plt.title(f"PACF für ART SHOW YEAR {year}")
    plt.tight_layout()
    return plt


def plotTimeSeriesWithSARIMA(timeSeriesSelected, sarimaResults):
    plt.figure(figsize=(12, 6))
    plt.plot(timeSeriesSelected['Relative show day'],
             timeSeriesSelected['Sum Tickets sold'],
             marker='o', label=f'Tatsächliche Ticketverkäufe', color='blue')

    fittedValues = sarimaResults.fittedvalues
    rmse = np.sqrt(mean_squared_error(timeSeriesSelected['Sum Tickets sold'], fittedValues))
    plt.plot(timeSeriesSelected['Relative show day'],
             fittedValues,
             linestyle='--', label='SARIMA-Modellierte Werte', color='orange')

    plt.xlabel('Relative show day')
    plt.ylabel('Sum Tickets sold')
    plt.title(f'Ticketverkäufe vs. SARIMA-Modell \nRMSE: {rmse:.2f}')
    plt.grid(True)
    plt.legend()
    return plt

def doSarima(stationarySeries):
    model = ARIMA(
        stationarySeries['Sum Tickets sold'],
        order=(2, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit()
    return results



'''

LSTM Model Training

'''

#Prepare Data for LSTM Model

def prepare_data_lstm(time_series, time_steps=3):
    X, y = [], []
    for i in range(len(time_series) - time_steps):
        X.append(time_series[i:(i + time_steps)])
        y.append(time_series[i + time_steps])
    return np.array(X), np.array(y)

def normalize_series(timeSeriesSelected):
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(np.array(timeSeriesSelected).reshape(-1, 1)), scaler


# Build and train the LSTM model

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1))  # Only one output for the next prediction
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model



# Predictions for Art SHow 4

def predict_future_sales(model, scaler, time_series, time_steps=3, future_steps=30):
    predictions = []
    current_input = time_series[-time_steps:]
    for _ in range(future_steps):
        current_input = np.reshape(current_input, (1, time_steps, 1))
        next_prediction = model.predict(current_input)
        predictions.append(next_prediction[0, 0])
        current_input = np.append(current_input[:, 1:, :], np.reshape(next_prediction, (1, 1, 1)), axis=1)
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))


def LSTMTraining():

    # Load the data
    df = getCsvData()
    # Combine the data from all three art shows
    combined_data = df[df['Event_Year'].isin([1, 2, 3])]

    # Normalize the combined data
    ticket_sales, scaler = normalize_series(combined_data['Sum Tickets sold'].values)

    # Prepare the data for LSTM
    time_steps = 3
    X, y = prepare_data_lstm(ticket_sales, time_steps)

    # Reshape the data for LSTM
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Build the LSTM model
    model = build_lstm_model((X.shape[1], 1))

    # Train the model
    model.fit(X, y, epochs=50, batch_size=32)
    # Predict future sales for Art Show 4
    future_steps = 60  # Number of days to predict
    predicted_sales = predict_future_sales(model, scaler, ticket_sales, time_steps, future_steps)

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(combined_data)), combined_data['Sum Tickets sold'], label='Historical Sales')
    plt.plot(range(len(combined_data), len(combined_data) + future_steps), predicted_sales,
             label='Predicted Sales for Art Show 4', color='red')
    plt.xlabel('Relative show day')
    plt.ylabel('Sum Tickets sold')
    plt.title('Predicted Ticket Sales for Art Show 4')
    plt.legend()
    plt.grid(True)
    return plt



