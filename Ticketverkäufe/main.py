import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def getCsvData():
    script_dir = os.path.dirname(__file__)
    csv_path = os.path.join(script_dir, 'Ticketsales.csv')
    df = pd.read_csv(csv_path)
    df['Event_Year'] = df['Event Name'].apply(lambda x: int(x.split()[-1]))
    df = df.sort_values(by=['Event_Year', 'Relative show day'])
    return df

def plotTimeSeries(df, year):
    df_year = df[df['Event_Year'] == year]
    plt.figure(figsize=(10, 6))
    plt.plot(df_year['Relative show day'], df_year['Sum Tickets sold'], marker='o', label=f'ART SHOW YEAR {year}')
    plt.xlabel('Relative show day')
    plt.ylabel('Sum Tickets sold')
    plt.title(f'Ticketverkäufe für ART SHOW YEAR {year}')
    plt.grid(True)
    plt.legend()
    return plt

def dickeyFullerTest(df, year):
    df_year = df[df['Event_Year'] == year]
    time_series = df_year['Sum Tickets sold']
    result = adfuller(time_series.dropna())
    return result

def makeStationary(df, year):
    df_year = df[df['Event_Year'] == year]
    df_year['Log Tickets sold'] = np.log(df_year['Sum Tickets sold'])
    stationarySeries = df_year['Sum Tickets sold'].diff().dropna()
    return stationarySeries

def plotStationary(df, year):
    stationarySeries = makeStationary(df, year)
    plt.figure(figsize=(10, 6))
    plt.plot(
        stationarySeries.index,
        stationarySeries.values,
        marker='o',
        label=f'Stationäre Serie für ART SHOW YEAR {year}'
    )
    plt.xlabel('Index')
    plt.ylabel('Log-Differenz der Tickets')
    plt.title(f'Stationäre Serie für ART SHOW YEAR {year}')
    plt.grid(True)
    plt.legend()
    return plt

# CSV-Daten einlesen
def mainIrgendwas():
    df = pd.read_csv("Ticketsales.csv")

    # Show-Jahre in numerische Werte umwandeln (YEAR 1, YEAR 2, YEAR 3)
    df['Event_Year'] = df['Event Name'].apply(lambda x: int(x.split()[-1]))

    # Daten nach Event-Jahr sortieren
    df = df.sort_values(by=['Event_Year', 'Relative show day'])

    # Funktion zur Prüfung der Stationarität und Anwendung von Differenzierung
    # TODO: ACF und PACF requires stationary Series, Machen wir mit dieser Funktion, Trotzdem zeigt der Plot fehler auf!
    def make_stationary(series):
        adf_result = adfuller(series.dropna())
        if adf_result[1] < 0.05:
            print("Serie ist stationär (p-Wert < 0.05)")
            return series, 0  # Keine Differenzierung erforderlich
        else:
            print("Serie ist nicht stationär (p-Wert >= 0.05). Differenzierung wird angewendet.")
            return series.diff().dropna(), 1  # Erste Differenzierung anwenden


    # Für jedes Show-Jahr ein SARIMA-Modell anpassen
    for year in df['Event_Year'].unique():
        # Daten für das aktuelle Jahr filtern
        df_year = df[df['Event_Year'] == year]

        # Logarithmische Transformation, um exponentiellen Trend zu berücksichtigen
        df_year['Log Tickets sold'] = np.log(df_year['Sum Tickets sold'])

        # Stationarität prüfen und Differenzierung anwenden
        stationary_series, d = make_stationary(df_year['Log Tickets sold'])

        # ACF und PACF für die stationäre Serie plotten
        plt.figure(figsize=(12, 6))
        plot_acf(stationary_series, lags=20, ax=plt.subplot(121))
        plt.title(f"ACF für ART SHOW YEAR {year}")

        plot_pacf(stationary_series, lags=20, ax=plt.subplot(122))
        plt.title(f"PACF für ART SHOW YEAR {year}")
        plt.tight_layout()
        plt.show()

        print(f"Angewandte Differenzierung für ART SHOW YEAR {year}: d={d}")

        # SARIMA-Modell anpassen (z.B. SARIMA(1, 1, 1)(1, 1, 1, 7))
        model = SARIMAX(df_year['Log Tickets sold'],
                        # Werte für die Orders haben wir bestimmt, basierend auf den Lolipop Test beim ACF und PACF Plot des ersten Jahres.
                        # TODO: Könnte man auch automatisch machen.
                        order=(1, 1, 2),
                        seasonal_order=(1, 1, 1, 7),
                        enforce_stationarity=False,
                        enforce_invertibility=False)

        results = model.fit()

        # Modellzusammenfassung
        print(f"SARIMA-Modell für ART SHOW YEAR {year}:")
        print(results.summary())

        # Residuen analysieren
        residuals = results.resid

        # Plot der Residuen
        plt.figure(figsize=(10, 6))
        plt.plot(df_year['Relative show day'], residuals)
        plt.xlim(-50, 3)  # X-Achse für alle Jahre von -50 bis 3 festlegen
        plt.title(f'Residuen für ART SHOW YEAR {year}')
        plt.xlabel('Relative show day')
        plt.ylabel('Residuen')
        plt.show()

        # Histogramm der Residuen
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=20)
        plt.title(f'Histogramm der Residuen für ART SHOW YEAR {year}')
        plt.xlabel('Residuen')
        plt.ylabel('Häufigkeit')
        plt.show()


