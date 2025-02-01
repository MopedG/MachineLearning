import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.src.initializers import GlorotUniform
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf

def plot_train_set():
    plt.figure(figsize=(12, 8))
    plt.plot(train.index, train.values)
    plt.xlabel("Relative show day")
    plt.ylabel("Logarithmierte Tickets")
    plt.title("Stationierte Zeitreihe - Train")
    plt.show()

# Hyperparameter
EPOCHS = 50
LOOK_BACK_VALUES = 20
FEATURES = 1
DROPOUT = 0.1

np.random.seed(42)  # Seed für Numpy
tf.random.set_seed(42)  # Seed für TensorFlow

df = pd.read_csv('TicketsalesCleaner.csv', index_col='Relative show day', parse_dates=False)
df.index.freq = 'D'
df.drop(['Event Name', 'SaleWeekDay', 'TicketSalesRevenue', 'Show Started'], axis=1, inplace=True)

# Normalerweise braucht man auch eine Testmenge, allerdings haben wir dafür nicht genug Daten.
# Wir nutzen also Jahr 3, um die predicteten werte mit Jahr 2 zu vergleichen.
# Vor der Skalierung: logarithmische Transformation der Datensätze

# train = np.log(df.iloc[:106])
train = np.log(df.iloc[:106])
test = np.log(df.iloc[53:])

# plot_train_set()

scaler = MinMaxScaler()
scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)
# print(scaled_train[:10])


generator = TimeseriesGenerator(scaled_train, scaled_train, length=LOOK_BACK_VALUES, batch_size = 1)

#Initialisiert ein seququentielles Modell (sequentille = Schichten werden nacheinander durchlaufen)
model = Sequential()
# Initialisiert eine LSTM Schicht mit 200 Neuronen, die als Aktivierungsfunktion die ReLu Funktion nutzen.
model.add(LSTM(200, activation='relu', input_shape=(LOOK_BACK_VALUES, FEATURES),
               return_sequences=True, kernel_initializer=GlorotUniform(seed=42)))
# Zusätzliche Schicht, die mit jedem Update zufällig 20% der Neuronen abschaltet. Verhindert Overfitting.
model.add(Dropout(DROPOUT))  # Regularisierung
# Weitere LSTM Schicht mit 100 Neuronen, Relu Aktivierungsfunktion
model.add(LSTM(100, activation='relu', kernel_initializer=GlorotUniform(seed=42)))
# Output Layer mit einem Neuronen. Sagt den nächsten Wert vorher. Dense = voll vernetzte Schicht.
model.add(Dense(1))
# Kompilieren des Modells. Adam ist der Optimizer mit dem die Parameter des Modells angepasst werden.
# MSE = Mean Squared Error, also der Fehler, der minimiert werden soll.
# Kompilerung verwenden MSE als Loss Funktion.
model.compile(optimizer='adam', loss='mse')

model.fit(generator, epochs=EPOCHS)

loss_per_epoch = model.history.history['loss']
#plt.plot(range(len(loss_per_epoch)), loss_per_epoch)
#plt.show()

last_train_batch = scaled_train[-LOOK_BACK_VALUES:]
last_train_batch = last_train_batch.reshape((1,LOOK_BACK_VALUES, FEATURES))
print(model.predict(last_train_batch))

test_predictions = []
first_eval_batch = scaled_train[-LOOK_BACK_VALUES:]
current_batch = first_eval_batch.reshape((1, LOOK_BACK_VALUES, FEATURES))

for i in range(len(test)):
    current_pred = model.predict(current_batch)[0]
    test_predictions.append(current_pred)
    current_batch = np.append(current_batch[:,1:,:], [[current_pred]], axis=1)

# true_predictions = scaler.inverse_transform(test_predictions)
# test.loc['Predictions'] =
log_predictions = scaler.inverse_transform(test_predictions)
true_predictions = np.exp(log_predictions)

#print(true_predictions)
print("\n---------------------------\n")


# Plot the true predictions
plt.figure(figsize=(12, 8))
x_values = list(range(-50, 0)) + list(range(1, 4))
plt.plot(x_values, true_predictions, label='True Predictions')
plt.xlabel('Relative show day')
plt.ylabel('Sum Tickets sold')
plt.title(f'True Predictions - Look Back Values: {LOOK_BACK_VALUES}, Epochs: {EPOCHS}')
plt.legend()
plt.show()

# test.plot(figsize=(12,8))