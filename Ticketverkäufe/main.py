import os

import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
import matplotlib.pyplot as plt


# Holt CSV Daten der Ticketverkäufe und gibt diese Als Dataframe zurück
def getCsvData():
    script_dir = os.path.dirname(__file__)
    csv_path = os.path.join(script_dir, 'Ticketsales.csv')
    df = pd.read_csv(csv_path)
    df['Event_Year'] = df['Event Name'].apply(lambda x: int(x.split()[-1]))
    df = df.sort_values(by=['Event_Year', 'Relative show day'])
    return df

# Plottet die Ticketverläufe eines Jahrs der angegebenen Zeitreihe. Anwendbar auf stationaere und nicht stationaere Zeitreihen
def plotTimeSeries(timeSeriesSelected, year):
    plt.figure(figsize=(10, 6))
    plt.plot(timeSeriesSelected['Relative show day'], timeSeriesSelected['Sum Tickets sold'], marker='o', label=f'ART SHOW YEAR {year}')
    plt.xlabel('Relative show day')
    plt.ylabel('Sum Tickets sold')
    plt.title(f'Ticketverkäufe für ART SHOW YEAR {year}')
    plt.grid(True)
    plt.legend()
    return plt

# Plottet jede einzelne Woche der Zeitreihe in einem Diagramm um graphisch nach Mustern (z.B. Saisonalität) zu suchen
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

# Plotted die Ticketverläufe für die angegebenen Wochentage.
# z.B. plotEachWeekday(X, "Monday", X): Plotted alle Montage der Zeitreihe X im Jahr X aufeinanderfolgend
def plotEachWeekday(timeSeriesSelected, saleWeekDays, year):
    # Gibt marketing Kampagnen an, die in den Jahren 1, 2 und 3 stattgefunden Haben
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

# Führt den dickeyFuller Test für eine Zeitreihe durch
def dickeyFullerTest(timeSeriesSelected):
    time_series = timeSeriesSelected['Sum Tickets sold']
    result = adfuller(time_series)
    return result

# Wandelt die angegebene Zeitreihe als stationäre Zeitreihe zurück, indem sie logarithmiert und differenziert wird.
def makeStationary(timeSeriesSelected):
    timeSeriesSelected = timeSeriesSelected.copy()
    # Logarithmierung
    timeSeriesSelected['Sum Tickets sold'] = np.log(timeSeriesSelected['Sum Tickets sold'])
    # Differenzierung
    timeSeriesSelected['Sum Tickets sold'] = timeSeriesSelected['Sum Tickets sold'].diff().dropna()
    timeSeriesSelected = timeSeriesSelected.dropna().reset_index(drop=True)
    return timeSeriesSelected

# Wandelt eine stationäre Zeitreihe zurück in eine nicht stationäre Zeitreihe und gibt diese zurück.
def revertStationary(original_series, stationary_series):
    # Differenzierung rückgänging machen
    #   - Kumulative Summe der differenzierten Werte berechnen
    #   - Ersten Wert der "originalen" Zeitreihe hinzufügen. (Es kann hier eine beliebige Zeitreihe genommen werden)
    recovered_log = original_series['Sum Tickets sold'].iloc[0] + stationary_series['Sum Tickets sold'].cumsum()
    # Logarithmierung rückgängig machen
    recovered_series = np.exp(recovered_log)
    return recovered_series

# Plotted die ACF und PACF Tests für eine stationäre Zeitreihe in einem Bestimmten Jahr.
def plot_acf_pacf(stationarySeries, year):
    #ACF Test
    plt.figure(figsize=(12, 6))
    plot_acf(stationarySeries['Sum Tickets sold'], lags=20, ax=plt.subplot(121))
    plt.title(f"ACF für ART SHOW YEAR {year}")
    #PACF Test
    plot_pacf(stationarySeries['Sum Tickets sold'], lags=20, ax=plt.subplot(122))
    plt.title(f"PACF für ART SHOW YEAR {year}")
    plt.tight_layout()
    return plt

# Plotted eine Zeitreihe sowie die Sarima modellierten Werte in einem Diagramm
def plotTimeSeriesWithSARIMA(timeSeriesSelected, sarimaResults):
    # Plotten der Zeitreihe
    plt.figure(figsize=(12, 6))
    plt.plot(timeSeriesSelected['Relative show day'],
             timeSeriesSelected['Sum Tickets sold'],
             marker='o', label=f'Tatsächliche Ticketverkäufe', color='blue')

    # Plotten der Sarima modellierten Werte
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

# Wegen vorgegebener saisonaler Komponente machen wir SARIMA (Arima für Saisonalität).
# Die Werte für order haben wir aus den ACF und PACF Plots abgelesen.
def doSarima(stationarySeries):
    model = SARIMAX(
        stationarySeries['Sum Tickets sold'],
        order=(2, 0, 1), # d=0, da bereits zuvor schon differenziert wurde
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit()
    return results

