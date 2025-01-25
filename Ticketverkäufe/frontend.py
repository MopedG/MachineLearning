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

# Zeitreihe differenzieren
st.subheader("Differenzierte Zeitreihe (stationär)")
plotTimeSeries(stationaryTimeSeriesSelected, selectedYear)

# Test auf Stationarität der differenzierten Zeitreihe
testStationarity(stationaryTimeSeriesSelected)

# ACF und PACF Test (Lollipop) p=PACF und q=ACF
st.subheader("ACF und PACF Plots für die differenzierte Zeitreihe:")
acf_plot = main.plot_acf_pacf(stationaryTimeSeriesSelected, selectedYear)
st.pyplot(acf_plot)

