# -*- coding: utf-8 -*-
"""
Transfer-learning KWS model for "HEY TM" with separate external test evaluation.
- No internal train/test split.
- Trains from datasets/ (fallback: dataset/).
- Exports float + quantized TFLite.
"""

import json
import logging
import os
import sys
from datetime import datetime

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training_log_transfer.txt"), logging.StreamHandler()],
)


def resolve_dataset_dir(candidates):
    """Pick the first existing dataset directory from preferred names."""
    for path in candidates:
        if os.path.exists(path):
            logging.info(f"Using training dataset directory: {path}")
            return path

    logging.error(
        "❌ Training dataset folder not found. Expected one of: " + ", ".join(candidates)
    )
    sys.exit(1)


class TransferFeatureExtractor:
    """Extract log-mel + MFCC hybrid features for transfer learning."""

    def __init__(self, sample_rate=16000, duration=1.0, n_mels=64, n_mfcc=13):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = int(sample_rate * duration)
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc

    def _load_and_fix_length(self, audio_path):
        audio, _ = librosa.load(audio_path, sr=self.sample_rate)
        if len(audio) > self.n_samples:
            return audio[: self.n_samples]
        return np.pad(audio, (0, self.n_samples - len(audio)), mode="constant")

    def extract_features(self, audio_path):
        """Return [time, 77] = [log-mel(64) + MFCC(13)]"""
        try:
            audio = self._load_and_fix_length(audio_path)

            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_fft=400,
                hop_length=160,
                n_mels=self.n_mels,
                fmin=125,
                fmax=7500,
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-6)

            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=512,
                hop_length=160,
                n_mels=40,
            )
            mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)

            mel_t = mel_db.T
            mfcc_t = mfcc.T

            min_t = min(mel_t.shape[0], mfcc_t.shape[0])
            combined = np.concatenate([mel_t[:min_t], mfcc_t[:min_t]], axis=1)
            return combined.astype(np.float32)
        except Exception as exc:
            logging.warning(f"Error processing {audio_path}: {exc}")
            return None

    def load_dataset(self, dataset_path, class_names):
        features, labels, filenames = [], [], []

        for class_idx, class_name in enumerate(class_names):
            folder_name = "_background_noise_" if class_name == "background" else class_name
            class_path = os.path.join(dataset_path, folder_name)

            if not os.path.exists(class_path):
                logging.warning(f"Warning: {class_path} not found")
                continue

            class_samples = 0
            logging.info(f"Loading {class_name} samples from {class_path}...")

            for filename in sorted(os.listdir(class_path)):
                if not filename.endswith((".wav", ".mp3", ".m4a")):
                    continue

                file_path = os.path.join(class_path, filename)
                feat = self.extract_features(file_path)
                if feat is None:
                    continue

                features.append(feat)
                labels.append(class_idx)
                filenames.append(filename)
                class_samples += 1

            logging.info(f"Loaded {class_samples} {class_name} samples")

        X = np.array(features, dtype=np.float32)
        y = np.array(labels, dtype=np.int64)

        logging.info(f"Dataset shape: {X.shape}")
        logging.info(f"Label shape: {y.shape}")
        return X, y, filenames


class TransferTester:
    def __init__(self, model, class_names, extractor):
        self.model = model
        self.class_names = class_names
        self.extractor = extractor

    def test_on_folder(self, test_folder_path):
        """Evaluate test/<class_subfolder>/... structure with labels."""
        X_test, y_test, filenames = self.extractor.load_dataset(test_folder_path, self.class_names)

        if len(X_test) == 0:
            logging.error(f"❌ No test samples found in {test_folder_path}")
            return None

        y_test_cat = tf.keras.utils.to_categorical(y_test, len(self.class_names))
        test_loss, test_acc = self.model.evaluate(X_test, y_test_cat, verbose=0)

        y_pred_probs = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)

        report = classification_report(y_test, y_pred, target_names=self.class_names, digits=4)
        cm = confusion_matrix(y_test, y_pred)

        logging.info(f"Test Loss: {test_loss:.4f}")
        logging.info(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        logging.info("\n" + report)

        results = {
            "timestamp": datetime.now().isoformat(),
            "test_folder": test_folder_path,
            "total_samples": int(len(X_test)),
            "test_accuracy": float(test_acc),
            "test_loss": float(test_loss),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "samples": [],
        }

        for i, name in enumerate(filenames):
            probs = y_pred_probs[i]
            results["samples"].append(
                {
                    "filename": name,
                    "true_label": self.class_names[int(y_test[i])],
                    "predicted_label": self.class_names[int(y_pred[i])],
                    "probabilities": {
                        self.class_names[k]: float(probs[k]) for k in range(len(self.class_names))
                    },
                }
            )

        with open("test_results_transfer_detailed.json", "w") as f:
            json.dump(results, f, indent=2)

        self.plot_confusion_matrix(cm, test_folder_path)
        return results

    def plot_confusion_matrix(self, cm, test_folder_path):
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={"label": "Count"},
        )
        plt.title(f"Confusion Matrix - Transfer\n{test_folder_path}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig("confusion_matrix_transfer.png", dpi=300, bbox_inches="tight")
        plt.close()


def create_transfer_model(input_shape, num_classes):
    """
    Build a transfer model with MobileNetV2 backbone.
    Input: [time, features] where features = 77 (64 mel + 13 mfcc)
    """
    time_steps, n_features = input_shape[1], input_shape[2]

    inputs = tf.keras.layers.Input(shape=(time_steps, n_features), name="audio_features")
    x = tf.keras.layers.Reshape((time_steps, n_features, 1))(inputs)
    x = tf.keras.layers.Concatenate()([x, x, x])
    x = tf.keras.layers.Resizing(96, 96)(x)

    base_model = None
    try:
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(96, 96, 3), include_top=False, weights="imagenet", pooling="avg"
        )
        logging.info("Loaded MobileNetV2 ImageNet pretrained weights.")
    except Exception as exc:
        logging.warning(
            "Could not load ImageNet weights (likely offline). Falling back to random init. "
            f"Error: {exc}"
        )
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(96, 96, 3), include_top=False, weights=None, pooling="avg"
        )

    base_model.trainable = False

    x = base_model(x, training=False)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="kws_transfer_mobilenet")
    return model, base_model


DATASET_DIR = resolve_dataset_dir(["datasets", "dataset"])
TEST_DIR = "test"
MODELS_DIR = "models_transfer"
LOGS_DIR = "logs_transfer"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

WANTED_WORDS = ["heytm", "unknown", "background"]
NUM_CLASSES = len(WANTED_WORDS)

HEAD_EPOCHS = 20
FINETUNE_EPOCHS = 8
BATCH_SIZE = 16

MODEL_TAG = "mobilenet_transfer"
MODEL_PREFIX = f"KWS_{MODEL_TAG}"
MODEL_KERAS = os.path.join(MODELS_DIR, f"{MODEL_PREFIX}.keras")
MODEL_TFLITE = os.path.join(MODELS_DIR, f"{MODEL_PREFIX}.tflite")
MODEL_FLOAT_TFLITE = os.path.join(MODELS_DIR, f"{MODEL_PREFIX}_float.tflite")


logging.info("\n" + "=" * 80)
logging.info("🔄 Loading transfer-learning dataset")
logging.info("=" * 80)

extractor = TransferFeatureExtractor()
X, y, _ = extractor.load_dataset(DATASET_DIR, WANTED_WORDS)

if len(X) == 0:
    logging.error("❌ No valid training samples found.")
    sys.exit(1)

y_cat = tf.keras.utils.to_categorical(y, NUM_CLASSES)
X_train, y_train = X, y_cat

logging.info(f"Training samples: {len(X_train)}")
logging.info(f"Input shape: {X_train.shape[1:]}")


model, base_model = create_transfer_model(X_train.shape, NUM_CLASSES)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary(print_fn=logging.info)

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.5, patience=4, min_lr=1e-6, monitor="loss", verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_KERAS, save_best_only=True, monitor="loss", mode="min", verbose=1
    ),
    tf.keras.callbacks.TensorBoard(log_dir=LOGS_DIR, histogram_freq=0),
]

logging.info("\n" + "=" * 80)
logging.info("🚀 Stage 1: train transfer head (frozen backbone)")
logging.info("=" * 80)

model.fit(
    X_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=HEAD_EPOCHS,
    callbacks=callbacks,
    verbose=1,
)


logging.info("\n" + "=" * 80)
logging.info("🔧 Stage 2: fine-tune top MobileNetV2 layers")
logging.info("=" * 80)

base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    X_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=HEAD_EPOCHS + FINETUNE_EPOCHS,
    initial_epoch=HEAD_EPOCHS,
    callbacks=callbacks,
    verbose=1,
)

train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
logging.info(f"\nTraining Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
logging.info(f"Training Loss: {train_loss:.4f}")


if os.path.exists(TEST_DIR):
    audio_files = [f for f in sorted(os.listdir(TEST_DIR)) if f.endswith((".wav", ".mp3", ".m4a"))]

    if audio_files:
        logging.info("\n" + "=" * 80)
        logging.info("🧪 Testing transfer model on flat test folder")
        logging.info("=" * 80)

        print(
            f"\n{'Filename':<50} {'Prediction':<15} {'Positive':<12} {'Negative':<12} {'Silence':<12}"
        )
        print("-" * 110)

        results = []
        for audio_file in audio_files:
            fp = os.path.join(TEST_DIR, audio_file)
            feat = extractor.extract_features(fp)
            if feat is None:
                print(f"{audio_file:<50} {'ERROR':<15} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
                continue

            pred = model.predict(np.expand_dims(feat, axis=0), verbose=0)[0]

            positive_prob = float(pred[0] * 100.0)
            negative_prob = float(pred[1] * 100.0)
            silence_prob = float(pred[2] * 100.0)
            pred_label = WANTED_WORDS[int(np.argmax(pred))]

            print(
                f"{audio_file:<50} {pred_label:<15} {positive_prob:>10.2f}%  {negative_prob:>10.2f}%  {silence_prob:>10.2f}%"
            )

            results.append(
                {
                    "filename": audio_file,
                    "prediction": pred_label,
                    "probabilities": {
                        "positive": positive_prob,
                        "negative": negative_prob,
                        "silence": silence_prob,
                    },
                }
            )

        print("-" * 110)

        with open("test_predictions_transfer.json", "w") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "test_folder": TEST_DIR,
                    "total_files": len(results),
                    "results": results,
                },
                f,
                indent=2,
            )

        logging.info("💾 Saved transfer predictions to test_predictions_transfer.json")
    else:
        logging.info("Detected subfolder test structure. Running detailed labeled evaluation.")
        tester = TransferTester(model, WANTED_WORDS, extractor)
        tester.test_on_folder(TEST_DIR)
else:
    logging.warning(f"⚠️ Test folder '{TEST_DIR}' not found. Skipping external evaluation.")


logging.info("\n" + "=" * 80)
logging.info("🔄 Converting transfer model to TFLite")
logging.info("=" * 80)

try:
    float_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    float_tflite = float_converter.convert()
    with open(MODEL_FLOAT_TFLITE, "wb") as f:
        f.write(float_tflite)
    logging.info(
        f"✅ Float TFLite saved: {MODEL_FLOAT_TFLITE} ({len(float_tflite)/(1024*1024):.2f} MB)"
    )
except Exception as exc:
    logging.error(f"❌ Float TFLite conversion failed: {exc}")

try:
    quant_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    quant_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quant_converter.target_spec.supported_types = [tf.float16]
    quant_tflite = quant_converter.convert()

    with open(MODEL_TFLITE, "wb") as f:
        f.write(quant_tflite)

    logging.info(
        f"✅ Quantized TFLite saved: {MODEL_TFLITE} ({len(quant_tflite)/(1024*1024):.2f} MB)"
    )
except Exception as exc:
    logging.error(f"❌ Quantized TFLite conversion failed: {exc}")

logging.info("\n" + "🎉" * 40)
logging.info("✅ TRANSFER TRAINING + EVALUATION COMPLETED")
logging.info("🎉" * 40)
logging.info(f"\nArtifacts directory: {MODELS_DIR}")
logging.info("\n" + "=" * 80 + "\n")
