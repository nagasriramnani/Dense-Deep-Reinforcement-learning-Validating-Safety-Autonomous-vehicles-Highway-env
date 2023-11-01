import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from sklearn.model_selection import train_test_split

ACTIONS_ALL = {
    0: 'LANE_LEFT',
    1: 'IDLE',
    2: 'LANE_RIGHT',
    3: 'FASTER',
    4: 'SLOWER'
}

# Convert action string to index
def action_to_index(action_str):
    for key, value in ACTIONS_ALL.items():
        if value == action_str:
            return key
    return -1  # If not found

def load_processed_data(dataset_path):
    dataset = pd.read_csv(dataset_path)
    
    # Convert observations from string to numpy arrays
    dataset['Observations'] = dataset['Observations'].apply(eval)
    
    # Convert action strings to indices
    dataset['Actions'] = dataset['Actions'].apply(action_to_index)
    
    return dataset

def train_behavior_cloning_model(dataset):
    # Split data into training and testing sets
    X = np.stack(dataset['Observations'].values)
    y = dataset['Actions'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define a simple neural network model for behavior cloning
    model = Sequential([
        Flatten(input_shape=(5, 5)),
        Dense(128, activation='relu'),
        Dense(5, activation='softmax')  # 5 possible actions
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10000, validation_data=(X_test, y_test))
    model.save("behavior_cloning_model.keras")  # Save the trained model

if __name__ == "__main__":
    dataset_path = "collected_data_ppo.csv"
    dataset = load_processed_data(dataset_path)
    train_behavior_cloning_model(dataset)
