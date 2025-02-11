import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
import time
import main
import lstm


def plotTimeSeries(timeSeriesSelected, selectedYear):
    plt = main.plotTimeSeries(timeSeriesSelected, selectedYear)
    st.pyplot(plt)

def testStationarity(timeSeriesSelected):
    p_value = main.dickeyFullerTest(timeSeriesSelected)[1]
    adf_statistik = main.dickeyFullerTest(timeSeriesSelected)[0]
    st.write(f"ADF Stationarität p-Wert: {p_value}, ADF-Statistik: {adf_statistik}")
    if p_value < 0.05:
        st.write("Die Zeitreihe ist **stationär**.")
    else:
        st.write("Die Zeitreihe ist **nicht stationär**.")

# Auswahl der Daten
st.title("ARIMA Zeitreihenanalyse")
st.sidebar.header("Event-Auswahl")
selected_event = st.sidebar.selectbox(
    "Wähle ein Event aus:",
    ["ART SHOW YEAR 1", "ART SHOW YEAR 2", "ART SHOW YEAR 3"]
)
selectedYear = int(selected_event.split()[-1])

# Stationäre und nicht stationäre Zeitreihen
rawTimeSeriesAllYears = main.getCsvData()
rawTimeSeriesSelected = rawTimeSeriesAllYears[rawTimeSeriesAllYears['Event_Year'] == selectedYear]
stationaryTimeSeriesSelected = main.makeStationary(rawTimeSeriesSelected)

# Zeitreihenanalyse
st.write(f"Die Zeitreihenanalyse wird auf **{selected_event}** angewendet.")
plotTimeSeries(rawTimeSeriesSelected, selectedYear)

# Test auf Stationarität der urspünglichen Zeitreihe
testStationarity(rawTimeSeriesSelected)


# Vergleich der Verkäufe pro woche:
st.subheader("Vergleich der Verkäufe pro Woche")
week_plot = main.plotEachWeek(stationaryTimeSeriesSelected, selectedYear)
st.pyplot(week_plot)

# Jeder Montag im Jahr
st.subheader("Einfluss der Marketingkampagnen")
saleWeekDaysInput = st.text_input("Gib die Wochentage auf Englisch ein (getrennt durch Komma, z.B. 'Mo, Tu, Fr'):")
if saleWeekDaysInput != "":
    monday_plot = main.plotEachWeekday(stationaryTimeSeriesSelected, saleWeekDaysInput, selectedYear)
    st.pyplot(monday_plot)

# Zeitreihe differenzieren
st.subheader("Differenzierte Zeitreihe (stationär)")
plotTimeSeries(stationaryTimeSeriesSelected, selectedYear)

# Test auf Stationarität der differenzierten Zeitreihe
testStationarity(stationaryTimeSeriesSelected)

# ACF und PACF Test (Lollipop) p=PACF und q=ACF
st.subheader("ACF und PACF Plots für die differenzierte Zeitreihe:")
acf_plot = main.plot_acf_pacf(stationaryTimeSeriesSelected, selectedYear)
st.pyplot(acf_plot)
st.write("Die PACF zeigt, dass die Zeitreihe eine Autoregressive Ordnung von **2** hat. Die ACF zeigt, dass die Zeitreihe eine Moving Average Ordnung von **1** hat. **Das ergibt: p=2, q=1.**")

# SARIMA Test
st.subheader("SARIMA Modellierung")
results = main.doSarima(rawTimeSeriesSelected)
st.write(f"Das SARIMA-Modell wurde erfolgreich erstellt. Die Modellparameter sind: (p=2, d=0, q=1), (1, 1, 1, 7). [d=0, da bereits differenziert]")
st.pyplot(main.plotTimeSeriesWithSARIMA(rawTimeSeriesSelected, results))

st.subheader("Vorhersage der Ticketverkäufe mit LSTM")
st.write("Input für das LSTM: Logarithmierte Ticketverkäufe der Art Show Year 2 und 3")
st.write("Art Show Year 1 wurde nicht verwendet, da es sich zu sehr von den anderen Jahren unterscheidet.")
st.pyplot(lstm.show_train_set())
st.subheader("Vorhersage der Ticketverkäufe für das Jahr 4 (dauert ein bisschen)")

# Button hinzufügen
if st.button("Jetzt beginnen"):
    st.write("1. Modell wird trainiert (50 Epochen, 20 Look Back Werte, 1 Feature, 0.4 Dropout)")
    st.write("2. Vorhersage wird erstellt")
    st.pyplot(lstm.make_prediction_year4())
    st.write("**Der anfängliche Abschwung kommt von dem gelernten Muster durch das Aneinanderhängen der Art Shows 2 und 3. Diesen Verlauf versucht es abzubilden.**")