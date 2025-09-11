# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import tensorflow as tf
import librosa
from scipy.io.wavfile import write

# --- Define Preprocessing Constants ---
# These must match the training script's settings
SAMPLE_RATE = 16000
CLIP_DURATION_MS = 1000
WINDOW_SIZE_MS = 30
WINDOW_STRIDE_MS = 20
FEATURE_BIN_COUNT = 40

# --- Function to load and preprocess a single audio file ---
def preprocess_audio(file_path):
    """
    Loads an audio file and computes its MFCC features using the same
    parameters as the training script.
    """
    try:
        # Load audio file and resample
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # Pad or trim the audio to exactly 1 second (16000 samples)
        target_length = int(SAMPLE_RATE * CLIP_DURATION_MS / 1000)
        if len(audio) > target_length:
            audio = audio[:target_length]
        elif len(audio) < target_length:
            padding = np.zeros(target_length - len(audio), dtype=np.float32)
            audio = np.concatenate((audio, padding))

        # Compute MFCCs
        hop_length = int(SAMPLE_RATE * WINDOW_STRIDE_MS / 1000)
        n_fft = int(SAMPLE_RATE * WINDOW_SIZE_MS / 1000)
        
        mfccs = librosa.feature.mfcc(
            y=audio, 
            sr=SAMPLE_RATE, 
            n_mfcc=FEATURE_BIN_COUNT,
            hop_length=hop_length,
            n_fft=n_fft
        )

        # Flatten the features to match the model's expected shape
        # Expected shape is (frames, features)
        flattened_mfccs = mfccs.T.flatten()

        return flattened_mfccs.astype(np.float32)

    except Exception as e:
        print(f"❌ Error processing audio file: {e}")
        return None

# --- Main script ---
def main():
    # ---- Setup paths and config ----
    MODELS_DIR = "models"
    MODEL_TFLITE = os.path.join(MODELS_DIR, "KWS_custom.tflite")
    
    WANTED_WORDS = "heytm"
    labels = ["_silence_", "_unknown_"] + WANTED_WORDS.split(",")

    # --- Check for model file ---
    if not os.path.exists(MODEL_TFLITE):
        print(f"❌ Error: TFLite model not found at '{MODEL_TFLITE}'.")
        print("Please run your training script first to generate the model.")
        sys.exit(1)
    
    # --- Prepare a sample audio file for testing ---
    sample_wav_path = "test_audio.wav"
    if not os.path.exists(sample_wav_path):
        print(f"Creating a dummy WAV file for testing at '{sample_wav_path}'.")
        dummy_audio = np.random.normal(0, 0.1, 16000).astype(np.float32)
        write(sample_wav_path, 16000, dummy_audio)

    # --- Load the TFLite model and allocate tensors ---
    print(f"\n🔄 Loading TFLite model from: {MODEL_TFLITE}")
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_TFLITE)
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"❌ Error loading TFLite model: {e}")
        sys.exit(1)

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']

    # Get the number of expected features from the shape
    num_features = input_shape[1]

    print(f"✅ Model loaded. Expected input shape: {input_shape}")
    print(f"   Output details: {output_details}")

    # --- Preprocess the audio file ---
    print(f"🔄 Preprocessing audio file: {sample_wav_path}")
    processed_audio = preprocess_audio(sample_wav_path)

    if processed_audio is None:
        sys.exit(1)
    
    # Check if the processed data shape matches the model's input shape
    if processed_audio.shape[0] != num_features:
        print(f"❌ Input shape mismatch.")
        print(f"   Model expects a flattened array of size {num_features}")
        print(f"   Audio has a flattened array of size {processed_audio.shape[0]}")
        sys.exit(1)

    # --- Run inference ---
    print("🚀 Running inference...")

    # The model expects a batch size of 1.
    input_data = np.expand_dims(processed_audio, axis=0)

    # Ensure data type matches
    input_type = input_details[0]['dtype']
    if input_type == np.int8:
        # Quantized model requires int8
        input_scale, input_zero_point = input_details[0]['quantization']
        input_data = (input_data / input_scale + input_zero_point).astype(input_type)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get the output tensor and dequantize if necessary
    output_data = interpreter.get_tensor(output_details[0]['index'])
    if output_details[0]['dtype'] == np.int8:
        output_scale, output_zero_point = output_details[0]['quantization']
        output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale

    # --- Interpret the results ---
    prediction = np.argmax(output_data)
    confidence = np.max(tf.nn.softmax(output_data))
    
    print(f"\n🎉 Inference completed!")
    print(f"   Prediction Index: {prediction}")
    print(f"   Predicted Label: {labels[prediction]}")
    print(f"   Confidence: {confidence:.2f}")

    print("\n💡 Note: A good confidence level for a correct prediction is typically > 0.9. If the confidence is low, the model may be unsure of its prediction.")

if __name__ == "__main__":
    main()