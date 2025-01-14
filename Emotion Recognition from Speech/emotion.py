import os
import librosa
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Function to extract features from audio
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast', duration=2.5, sr=22050)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

# Load dataset
def load_data(data_folder):
    features = []
    labels = []
    for emotion in os.listdir(data_folder):
        emotion_folder = os.path.join(data_folder, emotion)
        if os.path.isdir(emotion_folder):
            print(f"Processing emotion: {emotion}")  # Debugging statement
            for file in os.listdir(emotion_folder):
                if file.endswith('.wav'):
                    file_path = os.path.join(emotion_folder, file)
                    try:
                        mfccs = extract_features(file_path)
                        features.append(mfccs)
                        labels.append(emotion)
                        print(f"Loaded file: {file_path}")  # Debugging statement
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")  # Error handling
    print(f"Total features extracted: {len(features)}")  # Debugging statement
    print(f"Total labels extracted: {len(labels)}")      # Debugging statement
    return np.array(features), np.array(labels)

# Load data
data_folder = 'c:/Users/MY PC/OneDrive/Desktop/CodeAlpha/emotion detection/wav'  # Folder containing emotion subfolders
X, y = load_data(data_folder)

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reshape X for LSTM input
X = X.reshape(X.shape[0], X.shape[1], 1)  # Reshape to (samples, time steps, features)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.5))
model.add(Dense(y_categorical.shape[1], activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')

# Save the model
model.save('emotion_recognition_model.h5')
print("Model saved as 'emotion_recognition_model.h5'.")

# Make predictions on a new audio file
def predict_emotion(file_path):
    try:
        mfccs = extract_features(file_path)
        mfccs = mfccs.reshape(1, mfccs.shape[0], 1)  # Reshape for LSTM input
        prediction = model.predict(mfccs)
        predicted_emotion = encoder.inverse_transform([np.argmax(prediction)])
        return predicted_emotion[0]
    except Exception as e:
        print(f"Error predicting emotion for {file_path}: {e}")
        return None

# Example usage
new_file_path = 'c:/Users/MY PC/OneDrive/Desktop/CodeAlpha/emotion detection/wav/harvard.wav'  # Replace with your audio file
predicted_emotion = predict_emotion(new_file_path)
if predicted_emotion:
    print(f'The predicted emotion is: {predicted_emotion}')