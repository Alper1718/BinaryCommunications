import tensorflow as tf
import numpy as np
import librosa
import soundfile as sf
import os
from sklearn.model_selection import train_test_split

DATASET_PATH = "dataset/"
MODEL_PATH = "binary_audio_model.h5"

def extract_features(file_path):
    """ Extracts features from an audio file using MFCCs """
    y, sr = librosa.load(file_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs, axis=1)

def load_dataset():
    """ Loads audio dataset and prepares training data """
    X, Y = [], []
    for file in os.listdir(DATASET_PATH):
        if file.endswith(".wav"):
            label = int(file.split("_")[0])  # Extract label from filename
            features = extract_features(os.path.join(DATASET_PATH, file))
            X.append(features)
            Y.append(label)

    X, Y = np.array(X), np.array(Y)
    return train_test_split(X, Y, test_size=0.2, random_state=42)

def build_model():
    """ Builds a simple neural network for classification """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(13,)),  # 13 MFCC features
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")  # Binary classification
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def train_model():
    """ Trains and saves the audio classification model """
    X_train, X_test, Y_train, Y_test = load_dataset()
    model = build_model()
    model.fit(X_train, Y_train, epochs=20, batch_size=4, validation_data=(X_test, Y_test))
    model.save(MODEL_PATH)
    print("Model trained and saved.")

def predict_audio(file_path):
    """ Predicts whether the given audio sample represents '0' or '1' """
    if not os.path.exists(MODEL_PATH):
        print("Model not found! Train it first using train_model().")
        return

    model = tf.keras.models.load_model(MODEL_PATH)
    features = extract_features(file_path).reshape(1, -1)
    prediction = model.predict(features)[0][0]
    return "1" if prediction > 0.5 else "0"

def reconstruct_text(audio_folder):
    """ Processes a folder of sequential audio bits and reconstructs the original text """
    binary_string = ""
    files = sorted(os.listdir(audio_folder))  # Ensure proper order
    for file in files:
        if file.endswith(".wav"):
            bit = predict_audio(os.path.join(audio_folder, file))
            binary_string += bit

    print(f"Binary sequence: {binary_string}")

    # Convert binary to text
    decoded_text = ""
    for i in range(0, len(binary_string), 9):  # 9-bit chunks
        if i + 9 <= len(binary_string):
            is_int = int(binary_string[i])  # First bit (0 = char, 1 = int)
            value = int(binary_string[i+1:i+9], 2)  # Next 8 bits

            if is_int:
                decoded_text += f"{value} "
            else:
                decoded_text += chr(value)

    print(f"Decoded Output: {decoded_text.strip()}")

if __name__ == "__main__":
    # Uncomment below to train the model
    train_model()

    # Example usage (assuming you have received bit audio files)
    reconstruct_text("received_audio/")
