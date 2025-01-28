import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam


def load_and_preprocess(filename):
    data = pd.read_csv(filename)

    # Drop irrelevant column
    data = data.drop('TicketSalesRevenue', axis=1)

    # Create event name dummies first (but keep original column)
    event_dummies = pd.get_dummies(data['Event Name'], prefix='event')
    data = pd.concat([data, event_dummies], axis=1)

    # One-hot encode weekdays
    weekday_dummies = pd.get_dummies(data['SaleWeekDay'], prefix='weekday')
    data = pd.concat([data, weekday_dummies], axis=1)

    # Sort each event's data chronologically
    events = data.groupby('Event Name', sort=False)
    processed_data = []
    for name, group in events:
        processed_data.append(group.sort_values('Relative show day'))
    data_sorted = pd.concat(processed_data)

    # Drop original categorical columns after sorting
    data_sorted = data_sorted.drop(['SaleWeekDay'], axis=1)

    return data_sorted

# Create sequences for LSTM
def create_sequences(data, look_back=10):
    sequences = []
    targets = []

    feature_columns = ['Relative show day', 'Sum Tickets sold'] + \
                      [col for col in data.columns if col.startswith('weekday_')]

    # Ensure numeric data types
    data[feature_columns] = data[feature_columns].apply(pd.to_numeric, errors='coerce')

    # Handle missing values (you might want to customize this)
    data = data.dropna(subset=feature_columns)

    scaler = MinMaxScaler()
    numerical_features = ['Relative show day', 'Sum Tickets sold']
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    event_groups = data.groupby('Event Name', sort=False)
    for name, group in event_groups:
        # Explicit type conversion to float32
        features = group[feature_columns].values.astype(np.float32)
        event_targets = group['Sum Tickets sold'].shift(-1).values[:-1]

        for i in range(len(event_targets) - look_back):
            sequences.append(features[i:i + look_back])
            targets.append(event_targets[i + look_back])

    return np.array(sequences), np.array(targets), scaler, ["ART SHOW YEAR 1", "ART SHOW YEAR 2", "ART SHOW YEAR 3"]

    # Build LSTM model
def build_model(input_shape):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Progression')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_predictions(y_true, y_pred, sample_size=100):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true[:sample_size], label='Actual Tickets Sold', marker='o')
    plt.plot(y_pred[:sample_size], label='Predicted Tickets Sold', marker='x')
    plt.title('Actual vs Predicted Ticket Sales')
    plt.ylabel('Tickets Sold')
    plt.xlabel('Time Steps')
    plt.legend()
    plt.grid(True)
    plt.show()


# ... (keep all previous imports and functions until main block)

if __name__ == "__main__":
    # Parameters
    look_back = 10
    test_size = 0.2
    epochs = 100
    batch_size = 32

    # Load and preprocess data
    data = load_and_preprocess('TicketSales.csv')

    # Create sequences with event tracking
    X, y, scaler, event_names = create_sequences(data, look_back)

    # Split data while maintaining event groups
    event_df = pd.DataFrame({'index': range(len(event_names)), 'event': event_names})
    unique_events = event_df['event'].unique()

    # Split events into train/test groups
    train_events, test_events = train_test_split(unique_events, test_size=test_size, random_state=42)

    # Get indices for splits
    train_indices = event_df[event_df['event'].isin(train_events)]['index'].values
    test_indices = event_df[event_df['event'].isin(test_events)]['index'].values

    # Split the data
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    # Build and train model
    model = build_model((look_back, X.shape[2]))
    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_test, y_test),
                        verbose=1)


    # Plot training history (keep previous code)

    # New: Plot predictions for a complete event
    def plot_event_predictions(event_name, model, data, scaler, look_back):
        # Filter original data for the event
        event_mask = (data['Event Name'] == event_name)
        event_data = data[event_mask].sort_values('Relative show day')

        # Create sequences for this event
        X_event, y_event, _, _ = create_sequences(event_data, look_back)

        # Make predictions
        y_pred = model.predict(X_event)

        # Inverse scaling
        data_min = scaler.data_min_[1]
        data_max = scaler.data_max_[1]
        y_true_actual = y_event * (data_max - data_min) + data_min
        y_pred_actual = y_pred.flatten() * (data_max - data_min) + data_min

        # Get relative days for predictions
        relative_days = []
        for seq in X_event:
            last_day = scaler.inverse_transform([[seq[-1, 0], 0]])[0][0]
            relative_days.append(last_day + 1)

        # Get full timeline
        full_days = event_data['Relative show day'].values
        full_sales = event_data['Sum Tickets sold'].values

        plt.figure(figsize=(15, 6))
        plt.plot(full_days, full_sales, label='Actual Sales', marker='o')
        plt.plot(relative_days, y_pred_actual, label='Predicted Sales',
                 marker='x', linestyle='--', color='orange')
        plt.title(f'Ticket Sales Predictions for {event_name}')
        plt.xlabel('Relative Show Day')
        plt.ylabel('Tickets Sold')
        plt.legend()
        plt.grid(True)
        plt.show()


    # Example usage: Plot predictions for ART SHOW YEAR 1
    plot_event_predictions('ART SHOW YEAR 2', model, data, scaler, look_back)