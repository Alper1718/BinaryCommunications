import tensorflow as tf
import numpy as np
import librosa
import os
from sklearn.model_selection import train_test_split

# Paths
DATASET_PATH = "dataset/"
MODEL_PATH = "binary_audio_model.keras"

# Feature Extraction
def extract_features(file_path):
    """Extracts MFCC features from an audio file and normalizes them."""
    y, sr = librosa.load(file_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs, axis=1)  # Normalize across time axis

# Dataset Loading
def load_dataset():
    """Loads dataset and prepares it for training."""
    X, Y = [], []
    for file in os.listdir(DATASET_PATH):
        if file.endswith(".wav"):
            label = int(file.split("_")[0])  # Extract label (0 or 1) from filename
            features = extract_features(os.path.join(DATASET_PATH, file))
            X.append(features)
            Y.append(label)

    return train_test_split(np.array(X), np.array(Y), test_size=0.2, random_state=42)

# Model Definition
def build_model():
    """Creates a simple neural network for binary audio classification."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(13,)),  # 13 MFCC features
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")  # Binary classification
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# Model Training
def train_model():
    """Trains and saves the audio classification model."""
    X_train, X_test, Y_train, Y_test = load_dataset()
    model = build_model()
    model.fit(X_train, Y_train, epochs=20, batch_size=4, validation_data=(X_test, Y_test))
    model.save(MODEL_PATH)
    print("✅ Model trained and saved.")

# Audio Prediction (Efficient model loading)
def predict_audio(file_path, model):
    """Predicts whether an audio sample represents '0' or '1'."""
    features = extract_features(file_path).reshape(1, -1)
    prediction = model.predict(features, verbose=0)[0][0]
    return "1" if prediction > 0.5 else "0"

# Binary to Text Conversion
def binary_to_text(binary_string):
    """Decodes a binary sequence into text based on 9-bit encoding."""
    decoded_text = ""
    
    for i in range(0, len(binary_string), 9):
        if i + 9 <= len(binary_string):
            is_int = int(binary_string[i])  # First bit: 0 = char, 1 = int
            value = int(binary_string[i+1:i+9], 2)  # Next 8 bits

            if is_int:
                # Decoding as integer
                decoded_text += f"{value} "
            else:
                # Decoding as character
                decoded_text += chr(value)

    return decoded_text.strip()

# Text Reconstruction
def reconstruct_text(audio_folder):
    """Processes a folder of sequential audio bits and reconstructs the original text."""
    if not os.path.exists(MODEL_PATH):
        print("❌ Model not found! Train it first using train_model().")
        return
    
    # Load model once
    model = tf.keras.models.load_model(MODEL_PATH)

    binary_string = ""
    files = sorted(os.listdir(audio_folder))  # Ensure proper order
    
    print("\n📢 Predicting bits for each audio file:")
    for file in files:
        if file.endswith(".wav"):
            bit = predict_audio(os.path.join(audio_folder, file), model)
            print(f"File: {file} -> Predicted Bit: {bit}")
            binary_string += bit

    print(f"\n🔢 Binary sequence: {binary_string}")

    # Convert binary to text
    decoded_text = binary_to_text(binary_string)
    print(f"\n📝 Decoded Output: {decoded_text}")

if __name__ == "__main__":
    # Uncomment below to train the model
    # train_model()

    # Example usage (assuming bit audio files exist)
    reconstruct_text("received_audio/")
