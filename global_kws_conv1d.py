import os
import json
import numpy as np
import tensorflow as tf
import librosa
from sklearn.model_selection import train_test_split

# --- CONFIG ---
SR = 16000
DURATION = 1.0
N_MFCC = 13
N_MELS = 40
N_FFT = 512
HOP_LENGTH = 160
CLASSES = ["heytm", "unknown", "background"]

def extract_raw_mfcc(path):
    """Extracts MFCC without any normalization."""
    try:
        y, _ = librosa.load(path, sr=SR, duration=DURATION)
        # Pad to exactly 1 second
        if len(y) < SR:
            y = np.pad(y, (0, SR - len(y)))
        else:
            y = y[:SR]
        
        mfcc = librosa.feature.mfcc(
            y=y, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, 
            hop_length=HOP_LENGTH, n_mels=N_MELS
        )
        return mfcc.T # Shape (Time, MFCC)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def main():
    dataset_path = "dataset" # Ensure folders: heytm, unknown, _background_noise_
    X_raw, y_labels = [], []

    print("Step 1: Extracting raw features to calculate Global Stats...")
    for idx, label in enumerate(CLASSES):
        folder = "_background_noise_" if label == "background" else label
        dir_path = os.path.join(dataset_path, folder)
        for fn in os.listdir(dir_path):
            if fn.endswith((".wav", ".mp3")):
                feat = extract_raw_mfcc(os.path.join(dir_path, fn))
                if feat is not None:
                    X_raw.append(feat)
                    y_labels.append(idx)

    X_raw = np.array(X_raw)
    y_labels = np.array(y_labels)

    # --- CALCULATE GLOBAL STATS ---
    global_mean = np.mean(X_raw)
    global_std = np.std(X_raw)

    print("\n" + "="*50)
    print("COPY THESE TO FLUTTER")
    print(f"MEAN: {global_mean:.10f}")
    print(f"STD:  {global_std:.10f}")
    print("="*50 + "\n")

    # Step 2: Normalize with Global Stats
    X = (X_raw - global_mean) / (global_std + 1e-6)
    Y = tf.keras.utils.to_categorical(y_labels, len(CLASSES))

    # Step 3: Train Model
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X.shape[1], X.shape[2])),
        tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same'),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(len(CLASSES), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=32)

    # Step 4: Export TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open("global_kws_conv1d_float.tflite", "wb") as f:
        f.write(tflite_model)
    print("Successfully exported: global_kws_conv1d_float.tflite")

if __name__ == "__main__":
    main()