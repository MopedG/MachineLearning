import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.src.initializers import GlorotUniform
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf
import random

def show_train_set():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'TicketsalesCleaner.csv')
    
    df = pd.read_csv(csv_path, index_col='Relative show day', parse_dates=False)
    df.index.freq = 'D'
    df.drop(['Event Name', 'SaleWeekDay', 'TicketSalesRevenue', 'Show Started'], axis=1, inplace=True)

    # Normalerweise braucht man auch eine Testmenge, allerdings haben wir dafür nicht genug Daten.
    # Wir nutzen also Jahr 3, um die predicteten Werte mit Jahr 4 zu vergleichen.
    # Vor der Skalierung: logarithmische Transformation der Datensätze
    train = np.log(df.iloc[:106])
    return plot_train_set(train)

def plot_train_set(train):
    plt.figure(figsize=(12, 8))
    plt.plot(train.index, train.values)
    plt.xlabel("Relative show day")
    plt.ylabel("Logarithmierte Tickets")
    plt.title("Stationierte Zeitreihe - Train")
    return plt


def make_prediction_year4():
    global df
    # Hyperparameter
    # Hier könnte man einen Hyperparameter Tuner verwenden, um die besten Werte zu finden.
    EPOCHS = 50
    LOOK_BACK_VALUES = 20
    FEATURES = 1
    DROPOUT = 0.4

    np.random.seed(42)  # Seed für Numpy, damit die Ergebnisse reproduzierbar sind
    tf.random.set_seed(42)  # Seed für TensorFlow, damit die Ergebnisse reproduzierbar sind
    random.seed(42)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'TicketsalesCleaner.csv')
    df = pd.read_csv(csv_path, index_col='Relative show day', parse_dates=False)

    df.index.freq = 'D'
    df.drop(['Event Name', 'SaleWeekDay', 'TicketSalesRevenue', 'Show Started'], axis=1, inplace=True)

    # Normalerweise braucht man auch eine Testmenge, allerdings haben wir dafür nicht genug Daten.
    # Wir nutzen also Jahr 3, um die predicteten Werte mit Jahr 4 zu vergleichen.
    # Vor der Skalierung: logarithmische Transformation der Datensätze
    train = np.log(df.iloc[:106])
    test = np.log(df.iloc[53:])

    # Wir benutzen einen MinMaxScaler, um die Daten zwischen 0 und 1 zu skalieren, damit das Modell besser trainiert werden kann.
    scaler = MinMaxScaler()
    scaler.fit(train)
    scaled_train = scaler.transform(train)
    scaled_test = scaler.transform(test)

    generator = TimeseriesGenerator(scaled_train, scaled_train, length=LOOK_BACK_VALUES, batch_size=1)
    # Initialisiert ein seququentielles Modell (sequentille = Schichten werden nacheinander durchlaufen)
    model = Sequential()
    model.add(LSTM(200, activation='relu', input_shape=(LOOK_BACK_VALUES, FEATURES),
                   return_sequences=True, kernel_initializer=GlorotUniform(seed=42)))
    # Zusätzliche Schicht, die mit jedem Update zufällig 20% der Neuronen abschaltet. Vermindert Overfitting.
    model.add(Dropout(DROPOUT))
    model.add(LSTM(100, activation='relu', kernel_initializer=GlorotUniform(seed=42)))
    model.add(Dense(1))
    # Kompilieren des Modells. Adam ist der Optimizer mit dem die Parameter des Modells angepasst werden.
    # MSE = Mean Squared Error, also der Fehler, der minimiert werden soll.
    model.compile(optimizer='adam', loss='mse')
    model.fit(generator, epochs=EPOCHS)
    # Hier können wir einen Chart anzeigen lassen, der den Loss über die Epochen hinweg anzeigt
    # -> Hilft uns die optimale Anzahl an Epochen zu finden
    loss_per_epoch = model.history.history['loss']
    # plt.plot(range(len(loss_per_epoch)), loss_per_epoch)
    # plt.show()
    test_predictions = []
    first_eval_batch = scaled_train[-LOOK_BACK_VALUES:]
    current_batch = first_eval_batch.reshape((1, LOOK_BACK_VALUES, FEATURES))
    # Test ist 53 Tage lang, wir wollen auch für Jahr 4 53 Tage predicten
    for i in range(len(test)):
        current_pred = model.predict(current_batch)[0]
        test_predictions.append(current_pred)
        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)
    # Rücktransformieren (MinMaxScaling und Logarithmische Transformation rückgängig machen)
    log_predictions = scaler.inverse_transform(test_predictions)
    true_predictions = np.exp(log_predictions)
    # Plot der vom Modell gemachten predicitons für Jahr 4
    plt.figure(figsize=(12, 8))
    x_values = list(range(-50, 0)) + list(range(1, 4))
    plt.plot(x_values, true_predictions, label='True Predictions')
    plt.xlabel('Relative show day')
    plt.ylabel('Sum Tickets sold')
    plt.title(f'True Predictions - Look Back Values: {LOOK_BACK_VALUES}, Epochs: {EPOCHS}')
    plt.legend()
    return plt

