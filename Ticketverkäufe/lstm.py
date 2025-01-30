import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# Hyperparameter
EPOCHS = 50
LOOK_BACK_VALUES = 7
FEATURES = 1

df = pd.read_csv('TicketsalesCleaner.csv', index_col='Relative show day', parse_dates=False)
df.index.freq = 'D'
df.drop(['Event Name', 'SaleWeekDay', 'TicketSalesRevenue', 'Show Started'], axis=1, inplace=True)

# Normalerweise braucht man auch eine Testmenge, allerdings haben wir dafür nicht genug Daten.
# Wir nutzen also Jahr 3, um die predicteten werte mit Jahr 2 zu vergleichen.
train = df.iloc[:106]
test = df.iloc[53:]

scaler = MinMaxScaler()
scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)
# print(scaled_train[:10])


generator = TimeseriesGenerator(scaled_train, scaled_train, length=LOOK_BACK_VALUES, batch_size = 1)

# X,y = generator[1]
# print(f'Given the Array: \n{X.flatten()}')
# print(f'Predict this y: \n {y}')
# print(X.shape)

model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(LOOK_BACK_VALUES, FEATURES)))
model.add(Dense(1))
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

true_predictions = scaler.inverse_transform(test_predictions)
# test.loc['Predictions'] = true_predictions

#print(true_predictions)
print("\n---------------------------\n")
#p

# Plot the true predictions
plt.figure(figsize=(12, 8))
plt.plot(test.index, true_predictions, label='True Predictions')
plt.xlabel('Relative show day')
plt.ylabel('Sum Tickets sold')
plt.legend()
plt.show()

# test.plot(figsize=(12,8))