import os

import streamlit as st
from statsmodels.tsa.stattools import adfuller

import main

def showRawTimeSeries(data):
    plt = main.plotTimeSeries(data, selected_year)
    st.pyplot(plt)

def showStationaryTimeSeries(data):
    plt = main.plotStationary(data, selected_year)
    st.pyplot(plt)

st.title("ARIMA Zeitreihenanalyse")
st.sidebar.header("Event-Auswahl")

selected_event = st.sidebar.selectbox(
    "Wähle ein Event aus:",
    ["ART SHOW YEAR 1", "ART SHOW YEAR 2", "ART SHOW YEAR 3"]
)
selected_year = int(selected_event.split()[-1])
data = main.getCsvData()

st.write(f"Die Zeitreihenanalyse wird auf **{selected_event}** angewendet.")
showRawTimeSeries(data)

p_value = main.dickeyFullerTest(data, selected_year)[1]
adf_statistik = main.dickeyFullerTest(data, selected_year)[0]

# Testen auf Stationarität
st.write(f"ADF Stationarität p-Wert: {p_value}, ADF-Statistik: {adf_statistik}")
if p_value < 0.05:
    st.write("Die Zeitreihe ist **stationär**.")
else:
    st.write("Die Zeitreihe ist **nicht stationär**.")

st.subheader("Differenzierte Zeitreihe (stationär)")
showStationaryTimeSeries(data)
stationarySeries = main.makeStationary(data, selected_year)
p_value = adfuller(stationarySeries.dropna())[1]  # Direkt den p-Wert berechnen
adf_statistik = adfuller(stationarySeries.dropna())[0]

if p_value < 0.05:
    st.write(f"Die Zeitreihe ist **stationär**. (p-Wert: {p_value}, ADF-Statistik: {adf_statistik})")
else:
    st.write("Die Zeitreihe ist nicht **stationär**.")

