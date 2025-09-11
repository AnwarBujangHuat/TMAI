import os
import numpy as np
import tensorflow as tf
from glob import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- CONFIGURATION ---
TFLITE_MODEL_PATH = "hey_tm_modelv2.tflite"
DATASET_DIR = "dataset"
CLASSES = ["positive", "negative", "silence"]
SAMPLE_RATE = 16000
DURATION = 1  # seconds
AUDIO_LEN = SAMPLE_RATE * DURATION

# --- DATA PREPROCESSING FUNCTIONS (Copied from your training script) ---

def decode_audio(audio_binary):
    """
    Decodes a WAV audio binary to a Tensor, handles padding/trimming.
    """
    audio, _ = tf.audio.decode_wav(audio_binary, desired_channels=1)
    audio = tf.squeeze(audio, axis=-1)
    zero_padding = tf.zeros([AUDIO_LEN] - tf.shape(audio), dtype=tf.float32)
    audio = tf.concat([audio, zero_padding], 0)
    audio = audio[:AUDIO_LEN]
    return audio

def get_spectrogram(audio):
    """
    Converts audio to a Mel-spectrogram.
    """
    stft = tf.signal.stft(audio, frame_length=640, frame_step=320, fft_length=1024, window_fn=tf.signal.hann_window)
    spectrogram = tf.abs(stft)
    mel_spectrogram = tf.tensordot(
        spectrogram,
        tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=40,
            num_spectrogram_bins=stft.shape[-1],
            sample_rate=SAMPLE_RATE
        ),
        1
    )
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    return log_mel_spectrogram

def normalize_with_global_values(spectrogram, mean, std):
    """
    Normalizes a spectrogram using pre-calculated global mean and std.
    NOTE: You must replace `global_mean` and `global_std` with the values
    printed at the end of your training script.
    """
    return (spectrogram - mean) / (std + 1e-6)

def preprocess_test_data(file_path, mean, std):
    """
    Prepares a single audio file for inference, including normalization.
    """
    audio_binary = tf.io.read_file(file_path)
    audio = decode_audio(audio_binary)
    spectrogram = get_spectrogram(audio)
    spectrogram = tf.expand_dims(spectrogram, -1)
    normalized_spec = normalize_with_global_values(spectrogram, mean, std)
    return normalized_spec

# --- EVALUATION SCRIPT ---

def evaluate_model(model, file_paths, labels, mean, std):
    """
    Evaluates the model and calculates FRR and FAR.
    """
    positive_files = [f for f, l in zip(file_paths, labels) if l == 0] # Assuming "positive" is index 0
    negative_files = [f for f, l in zip(file_paths, labels) if l == 1] # Assuming "negative" is index 1

    true_positives = 0
    false_negatives = 0
    false_positives = 0
    true_negatives = 0
    
    # Get quantization parameters from the model's input details
    input_details = interpreter.get_input_details()
    input_quantization = input_details[0]['quantization']
    input_scale = input_quantization[0]
    input_zero_point = input_quantization[1]

    # 1. Test for False Rejection Rate (FRR)
    print("\n--- Evaluating False Rejection Rate (FRR) ---")
    for file in tqdm(positive_files):
        try:
            spectrogram = preprocess_test_data(file, mean, std)
            
            # Correctly quantize the input data before inference
            spectrogram_int8 = ((spectrogram / input_scale) + input_zero_point).numpy().astype(np.int8)
            input_tensor = np.expand_dims(spectrogram_int8, 0)
            
            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            interpreter.invoke()
            output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
            
            # Get the highest confidence class
            predicted_class = np.argmax(output_data[0])
            
            if predicted_class == 0:
                true_positives += 1
            else:
                false_negatives += 1
        except Exception as e:
            print(f"Error processing {file}: {e}")
            
    # 2. Test for False Acceptance Rate (FAR)
    print("\n--- Evaluating False Acceptance Rate (FAR) ---")
    for file in tqdm(negative_files):
        try:
            spectrogram = preprocess_test_data(file, mean, std)
            
            # Correctly quantize the input data before inference
            spectrogram_int8 = ((spectrogram / input_scale) + input_zero_point).numpy().astype(np.int8)
            input_tensor = np.expand_dims(spectrogram_int8, 0)

            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            interpreter.invoke()
            output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

            predicted_class = np.argmax(output_data[0])
            
            if predicted_class == 0: # If the model incorrectly predicts "positive"
                false_positives += 1
            else:
                true_negatives += 1
        except Exception as e:
            print(f"Error processing {file}: {e}")

    frr = false_negatives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    far = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
    
    print("\n--- EVALUATION RESULTS ---")
    print(f"Total Positive Samples: {len(positive_files)}")
    print(f"Total Negative Samples: {len(negative_files)}")
    print(f"False Rejection Rate (FRR): {frr:.4f} ({false_negatives}/{len(positive_files)})")
    print(f"False Acceptance Rate (FAR): {far:.4f} ({false_positives}/{len(negative_files)})")
    print("----------------------------\n")

# --- MAIN SCRIPT ---

if __name__ == "__main__":
    # Load the TFLite model and allocate tensors
    try:
        interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("TFLite model loaded successfully.")
    except Exception as e:
        print(f"Error loading TFLite model: {e}")
        exit()
        
    # IMPORTANT: Replace these with the actual mean and std values from your training script.
    GLOBAL_MEAN = -0.214911
    GLOBAL_STD = 1.919348

    # Prepare a test dataset (this is a simple split, for a real test, use new data!)
    file_paths = []
    labels = []
    for idx, cls in enumerate(CLASSES):
        files = glob(os.path.join(DATASET_DIR, cls, "*.wav"))
        file_paths.extend(files)
        labels.extend([idx] * len(files))
    
    # Stratified split to ensure class balance in the test set
    X_train, X_test, y_train, y_test = train_test_split(
        file_paths, labels, test_size=0.8, stratify=labels, random_state=42
    )
    
    evaluate_model(interpreter, X_test, y_test, GLOBAL_MEAN, GLOBAL_STD)
