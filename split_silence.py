import os
import numpy as np
import tensorflow as tf
from glob import glob
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
DATASET_DIR = "dataset"
CLASSES = ["positive", "negative", "silence"]
SAMPLE_RATE = 16000
DURATION = 1  # seconds
AUDIO_LEN = SAMPLE_RATE * DURATION
N_MELS = 40
BATCH_SIZE = 32
EPOCHS = 30
MODEL_PATH = "hey_tm_model.h5"
TFLITE_PATH = "hey_tm_model.tflite"

# --- DATA LOADING & PREPROCESSING ---

def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary, desired_channels=1)
    audio = tf.squeeze(audio, axis=-1)
    zero_padding = tf.zeros([AUDIO_LEN] - tf.shape(audio), dtype=tf.float32)
    audio = tf.concat([audio, zero_padding], 0)
    audio = audio[:AUDIO_LEN]
    return audio

def get_spectrogram(audio):
    stft = tf.signal.stft(audio, frame_length=640, frame_step=320, fft_length=1024)
    spectrogram = tf.abs(stft)
    mel_spectrogram = tf.tensordot(
        spectrogram,
        tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=N_MELS,
            num_spectrogram_bins=stft.shape[-1],
            sample_rate=SAMPLE_RATE
        ),
        1
    )
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    return log_mel_spectrogram

def normalize(spectrogram):
    mean = tf.math.reduce_mean(spectrogram)
    std = tf.math.reduce_std(spectrogram)
    return (spectrogram - mean) / (std + 1e-6)

def random_time_shift(audio):
    shift = tf.random.uniform([], minval=-1600, maxval=1600, dtype=tf.int32)
    return tf.roll(audio, shift, axis=0)

def add_background_noise(audio, noises):
    if noises:
        noise = noises[np.random.randint(len(noises))]
        noise = tf.convert_to_tensor(noise, dtype=tf.float32)
        noise = noise[:AUDIO_LEN]
        scale = tf.random.uniform([], 0.0, 0.2)
        audio = audio + scale * noise
    return audio

def preprocess(file_path, label, noises, augment=False):
    audio_binary = tf.io.read_file(file_path)
    audio = decode_audio(audio_binary)
    if augment:
        audio = random_time_shift(audio)
        audio = add_background_noise(audio, noises)
    spectrogram = get_spectrogram(audio)
    spectrogram = normalize(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, -1)
    return spectrogram, label

def load_noises(noise_dir):
    noise_files = glob(os.path.join(noise_dir, "*.wav"))
    noises = []
    for nf in noise_files:
        audio_binary = tf.io.read_file(nf)
        audio, _ = tf.audio.decode_wav(audio_binary, desired_channels=1)
        audio = tf.squeeze(audio, axis=-1)
        noises.append(audio.numpy())
    return noises

def get_dataset(file_paths, labels, noises, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds = ds.map(lambda x, y: preprocess(x, y, noises, augment), num_parallel_calls=tf.data.AUTOTUNE)
    return ds

# --- DATASET PREPARATION ---

def prepare_data():
    file_paths = []
    labels = []
    for idx, cls in enumerate(CLASSES):
        files = glob(os.path.join(DATASET_DIR, cls, "*.wav"))
        file_paths.extend(files)
        labels.extend([idx] * len(files))
    file_paths = np.array(file_paths)
    labels = np.array(labels)
    y = tf.keras.utils.to_categorical(labels, num_classes=len(CLASSES))
    X_train, X_val, y_train, y_val = train_test_split(file_paths, y, test_size=0.2, stratify=labels, random_state=42)
    return X_train, X_val, y_train, y_val

# --- MODEL DEFINITION (DS-CNN) ---

def ds_cnn_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(64, (3,3), padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    for _ in range(3):
        x = tf.keras.layers.DepthwiseConv2D((3,3), padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(64, (1,1), padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs)

# --- MAIN TRAINING SCRIPT ---

def main():
    X_train, X_val, y_train, y_val = prepare_data()
    noises = load_noises(os.path.join(DATASET_DIR, "silence"))

    train_ds = get_dataset(X_train, y_train, noises, augment=True)
    val_ds = get_dataset(X_val, y_val, noises, augment=False)
    train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    example_spec, _ = next(iter(train_ds))
    input_shape = example_spec.shape[1:]

    model = ds_cnn_model(input_shape, len(CLASSES))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[checkpoint]
    )

    # Print final accuracy
    val_acc = history.history['val_accuracy'][-1]
    train_acc = history.history['accuracy'][-1]
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")

    # Reload best model
    model = tf.keras.models.load_model(MODEL_PATH)

    # TFLite conversion with int8 quantization
    def representative_data_gen():
        for spec, _ in val_ds.take(100):
            yield [spec]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    with open(TFLITE_PATH, "wb") as f:
        f.write(tflite_model)

    print("✅ Model training complete and exported to TFLite")

if __name__ == "__main__":
    main()
