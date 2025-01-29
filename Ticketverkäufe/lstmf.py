import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Hyperparameter als Variablen für einfache Anpassung
SEQ_LENGTH = 50         # Anzahl der Tage, die als Eingabe für das Modell verwendet werden
LSTM_UNITS_1 = 128       # Anzahl der LSTM-Einheiten in der ersten LSTM-Schicht
LSTM_UNITS_2 = 64       # Anzahl der LSTM-Einheiten in der zweiten LSTM-Schicht
DROPOUT_RATE = 0.2      # Dropout-Rate zur Vermeidung von Overfitting
EPOCHS = 60            # Anzahl der Trainings-Epochen
BATCH_SIZE = 64         # Batch-Größe beim Training
SCALE_TICKETS = True    # Wenn True, werden die Ticketverkäufe skaliert

# Daten einlesen und Vorverarbeiten
def getCsvData():
    script_dir = os.path.dirname(__file__)
    csv_path = os.path.join(script_dir, 'Ticketsales.csv')
    df = pd.read_csv(csv_path)
    df['Event_Year'] = df['Event Name'].apply(lambda x: int(x.split()[-1]))
    df = df.sort_values(by=['Event_Year', 'Relative show day'])
    return df

# Daten vorbereiten
df = getCsvData()

# Nur die Jahre 2 und 3 verwenden, Jahr 1 wird ausgeschlossen
df = df[df['Event_Year'] != 1]

# Skalierung vorbereiten
day_scaler = StandardScaler()
ticket_scaler = MinMaxScaler(feature_range=(0, 1)) if SCALE_TICKETS else None

df["Relative show day scaled"] = day_scaler.fit_transform(df[["Relative show day"]])
if SCALE_TICKETS:
    df["Tickets scaled"] = ticket_scaler.fit_transform(df[["Sum Tickets sold"]])
else:
    df["Tickets scaled"] = df["Sum Tickets sold"]

# Daten für das Modell vorbereiten
X, y = [], []
for year in [2, 3]:  # Nur Jahr 2 und 3
    year_data = df[df["Event_Year"] == year]
    X_scaled = year_data[["Relative show day scaled"]].values
    y_scaled = year_data["Tickets scaled"].values

    for i in range(len(X_scaled) - SEQ_LENGTH):
        X.append(X_scaled[i:i + SEQ_LENGTH])
        y.append(y_scaled[i + SEQ_LENGTH])

X_train = np.array(X)
y_train = np.array(y)

# Modell erstellen
model = Sequential([
    LSTM(LSTM_UNITS_1, activation='relu', return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
    Dropout(DROPOUT_RATE),
    LSTM(LSTM_UNITS_2, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Modell trainieren
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

# **Neue Vorhersagefunktion mit Iteration**
def predict_year4():
    year4_days = np.arange(-50, 4)
    relative_day_scaled_year4 = day_scaler.transform(year4_days.reshape(-1, 1))

    predictions = []
    current_sequence = X_train[-1]  # Letzte bekannte Sequenz aus den Trainingsdaten

    for i in range(len(year4_days)):
        pred = model.predict(current_sequence.reshape(1, SEQ_LENGTH, 1), verbose=0)
        predictions.append(pred[0, 0])  # Vorhersage speichern

        # Update der Sequenz
        new_value = relative_day_scaled_year4[i].reshape(1, 1)
        current_sequence = np.vstack([current_sequence[1:], new_value])

    # Rücktransformation der Vorhersagen
    return ticket_scaler.inverse_transform(np.array(predictions).reshape(-1, 1)) if SCALE_TICKETS else np.array(predictions)

# Vorhersage und Plot
year4_predictions = predict_year4()

plt.figure(figsize=(12, 6))
plt.plot(range(-50, 4), year4_predictions, color='red', label='Vorhersage Jahr 4')
plt.ylim(0, max(df["Sum Tickets sold"]) * 1.1)  # Dynamische Y-Achse
plt.title('Ticketverkäufe Prognose für ART SHOW YEAR 4')
plt.xlabel('Ralative Tage zur Show')
plt.ylabel('Tickets verkauft')
plt.legend()
plt.grid(True)
plt.show()
