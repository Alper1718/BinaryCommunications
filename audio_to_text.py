import tensorflow as tf
import numpy as np
import librosa
import os
from sklearn.model_selection import train_test_split

DATASET_PATH = "dataset/"
MODEL_PATH = "binary_audio_model.keras"

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs, axis=1)

def load_dataset():
    X, Y = [], []
    for file in os.listdir(DATASET_PATH):
        if file.endswith(".wav"):
            label = int(file.split("_")[0])
            features = extract_features(os.path.join(DATASET_PATH, file))
            X.append(features)
            Y.append(label)

    return train_test_split(np.array(X), np.array(Y), test_size=0.2, random_state=42)

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(13,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def train_model():
    X_train, X_test, Y_train, Y_test = load_dataset()
    model = build_model()
    model.fit(X_train, Y_train, epochs=20, batch_size=4, validation_data=(X_test, Y_test))
    model.save(MODEL_PATH)
    print("✅ Model trained and saved.")

def predict_audio(file_path, model):
    features = extract_features(file_path).reshape(1, -1)
    prediction = model.predict(features, verbose=0)[0][0]
    return "1" if prediction > 0.5 else "0"

def binary_to_text(binary_string):
    decoded_text = ""
    
    for i in range(0, len(binary_string), 9):
        if i + 9 <= len(binary_string):
            is_int = int(binary_string[i])
            value = int(binary_string[i+1:i+9], 2)

            if is_int:
                decoded_text += f"{value} "
            else:
                decoded_text += chr(value)

    return decoded_text.strip()

def reconstruct_text(audio_folder):
    if not os.path.exists(MODEL_PATH):
        print("Model not found! Train it first using train_model().")
        return
    
    model = tf.keras.models.load_model(MODEL_PATH)

    binary_string = ""
    files = sorted(os.listdir(audio_folder))
    
    print("\nPredicting bits for each audio file:")
    for file in files:
        if file.endswith(".wav"):
            bit = predict_audio(os.path.join(audio_folder, file), model)
            print(f"File: {file} -> Predicted Bit: {bit}")
            binary_string += bit

    print(f"\nBinary sequence: {binary_string}")

    decoded_text = binary_to_text(binary_string)
    print(f"\nDecoded Output: {decoded_text}")

if __name__ == "__main__":
    # Uncomment below to train the model
    # train_model()

    reconstruct_text("received_audio/")
