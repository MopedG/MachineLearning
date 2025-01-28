import numpy as np
import streamlit as st
import main

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
saleWeekDaysInput = st.text_input("Gib die Wochentage ein (getrennt durch Komma, z.B. 'Mo, Tu, Fr'):")
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
st.write(f"Das SARIMA-Modell wurde erfolgreich erstellt. Die Modellparameter sind: (2, 1, 1), (1, 1, 1, 7).")
st.pyplot(main.plotTimeSeriesWithSARIMA(rawTimeSeriesSelected, results))

#st.subheader("Vorhersage der Ticketverkäufe mit LSTM")

# Plot in Streamlit anzeigen
#st.pyplot(main.LSTMTraining())