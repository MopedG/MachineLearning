import os

import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
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

def revertStationary(original_series, stationary_series):
    # Step 1: Reverse Differencing
    recovered_log = original_series['Sum Tickets sold'].iloc[0] + stationary_series['Sum Tickets sold'].cumsum()
    # Step 2: Reverse Log Transform
    recovered_series = np.exp(recovered_log)
    return recovered_series

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

