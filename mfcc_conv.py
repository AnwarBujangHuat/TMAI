# -*- coding: utf-8 -*-
"""
kws_conv1d_industry.py

Industry-standard-ish Keyword Spotting (KWS) training script for wakeword "hey tm"
using MFCC features + a Conv1D (temporal) CNN.

Highlights vs a quick prototype:
- Reproducible seeds
- Optional stratified train/val split (recommended)
- Class-imbalance handling (class weights)
- tf.data pipelines with on-the-fly feature augmentation (SpecAugment-style)
- Separate test-folder evaluation (supports flat or subfolder structure)
- Exports Keras + TFLite (float + float16; optional full-int8)
- Generates: logs, JSON metrics, confusion matrix PNG

Folder structure expected (same as your current scripts):
dataset/
  heytm/
  unknown/
  _background_noise_/
test/
  (either flat wav/mp3/m4a) OR (same subfolders as dataset/)

Run:
  python kws_conv1d_industry.py --dataset dataset --test test

Tip:
  Print the model input shape after training, and ensure your Flutter MFCC pipeline
  matches:
    sr=16000, duration=1s, n_fft=512, hop=160, n_mels=40, n_mfcc=13
"""

from __future__ import annotations

import os
import sys
import json
import math
import random
import logging
import argparse
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple, Optional, Dict

import numpy as np
import tensorflow as tf
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Optional plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------
# Reproducibility helpers
# ---------------------------
def set_global_determinism(seed: int = 42) -> None:
    """Best-effort reproducibility (won't be perfectly deterministic on all GPUs)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # TF deterministic ops (may reduce performance; ignore if unavailable)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


# ---------------------------
# Config
# ---------------------------
@dataclass
class Config:
    dataset_dir: str = "dataset"
    test_dir: str = "test"

    # Audio / MFCC
    sample_rate: int = 16000
    duration_sec: float = 1.0
    n_samples: int = 16000  # derived but we keep explicit for clarity
    n_mels: int = 40
    n_mfcc: int = 13
    n_fft: int = 512
    hop_length: int = 160  # 10 ms

    # Classes
    class_names: Tuple[str, ...] = ("heytm", "unknown", "background")

    # Training
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 5e-4
    val_split: float = 0.15  # set 0 to disable
    seed: int = 42

    # Augmentations on MFCC (simple + effective)
    aug_enable: bool = True
    aug_time_shift_frames: int = 8  # ~8*10ms = 80ms
    aug_time_mask_max: int = 12     # max masked frames
    aug_freq_mask_max: int = 4      # max masked MFCC bins
    aug_noise_std: float = 0.05     # Gaussian noise on MFCC values
    aug_prob: float = 0.7           # apply augmentation with this probability

    # Output
    out_dir: str = "kws_conv1d_out"
    models_dir: str = "models_conv1d"
    logs_dir: str = "logs_conv1d"

    export_int8: bool = False  # full int8 needs int8 input/output handling on device


# ---------------------------
# Logging
# ---------------------------
def setup_logger(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "training_log_conv1d.txt")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    logging.info("Logging to %s", log_path)


# ---------------------------
# Feature extraction
# ---------------------------
class MFCCExtractor:
    """MFCC extractor aligned to your Flutter / previous Python settings."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.n_samples = int(cfg.sample_rate * cfg.duration_sec)

    def extract(self, audio_path: str) -> Optional[np.ndarray]:
        """
        Returns MFCC with shape (time_steps, n_mfcc) float32.
        """
        try:
            audio, _ = librosa.load(audio_path, sr=self.cfg.sample_rate)

            if len(audio) > self.n_samples:
                audio = audio[: self.n_samples]
            else:
                audio = np.pad(audio, (0, self.n_samples - len(audio)), mode="constant")

            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=self.cfg.sample_rate,
                n_mfcc=self.cfg.n_mfcc,
                n_fft=self.cfg.n_fft,
                hop_length=self.cfg.hop_length,
                n_mels=self.cfg.n_mels,
            )

            # Normalize per-sample (matches your current approach)
            mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)
            mfcc = mfcc.T.astype(np.float32)  # (T, n_mfcc)
            return mfcc
        except Exception as e:
            logging.warning("Failed MFCC for %s: %s", audio_path, e)
            return None


def list_audio_files(root: str, class_names: Tuple[str, ...]) -> Tuple[List[str], List[int]]:
    """
    Returns (file_paths, labels) for subfolder dataset structure.
    background class maps to _background_noise_ folder.
    """
    paths: List[str] = []
    labels: List[int] = []

    for idx, cname in enumerate(class_names):
        folder = "_background_noise_" if cname == "background" else cname
        cpath = os.path.join(root, folder)
        if not os.path.exists(cpath):
            logging.warning("Missing folder: %s", cpath)
            continue

        for fn in os.listdir(cpath):
            if fn.lower().endswith((".wav", ".mp3", ".m4a")):
                paths.append(os.path.join(cpath, fn))
                labels.append(idx)

    return paths, labels


def load_features_into_memory(cfg: Config, extractor: MFCCExtractor, root: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Loads the entire dataset into RAM as (X, y, filenames).
    X shape: (N, T, n_mfcc)
    y shape: (N,)
    """
    file_paths, labels = list_audio_files(root, cfg.class_names)

    features: List[np.ndarray] = []
    good_labels: List[int] = []
    filenames: List[str] = []

    for p, y in zip(file_paths, labels):
        feat = extractor.extract(p)
        if feat is not None:
            features.append(feat)
            good_labels.append(y)
            filenames.append(os.path.basename(p))

    if not features:
        return np.empty((0, 0, 0), dtype=np.float32), np.empty((0,), dtype=np.int32), []

    X = np.stack(features, axis=0).astype(np.float32)
    y = np.asarray(good_labels, dtype=np.int32)
    return X, y, filenames


# ---------------------------
# Augmentations (MFCC-level)
# ---------------------------
def _augment_mfcc_np(mfcc: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Simple SpecAugment-style augmentations on MFCC matrix (T, C).
    This is cheap and commonly used in KWS pipelines.
    """
    x = mfcc.copy()

    # Random time shift
    if cfg.aug_time_shift_frames > 0:
        shift = np.random.randint(-cfg.aug_time_shift_frames, cfg.aug_time_shift_frames + 1)
        if shift != 0:
            x = np.roll(x, shift=shift, axis=0)

    # Add small Gaussian noise
    if cfg.aug_noise_std > 0:
        x = x + np.random.normal(0.0, cfg.aug_noise_std, size=x.shape).astype(np.float32)

    T, C = x.shape

    # Time masking
    if cfg.aug_time_mask_max > 0 and T > 1:
        t = np.random.randint(0, cfg.aug_time_mask_max + 1)
        if t > 0:
            t0 = np.random.randint(0, max(1, T - t))
            x[t0 : t0 + t, :] = 0.0

    # Frequency masking
    if cfg.aug_freq_mask_max > 0 and C > 1:
        f = np.random.randint(0, cfg.aug_freq_mask_max + 1)
        if f > 0:
            f0 = np.random.randint(0, max(1, C - f))
            x[:, f0 : f0 + f] = 0.0

    return x.astype(np.float32)


def make_tf_augment_fn(cfg: Config):
    """
    Wrap numpy augmentation in tf.numpy_function for tf.data.
    """
    def _tf_aug(x: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        if not cfg.aug_enable:
            return x, y

        # Apply augmentation with probability
        p = tf.random.uniform([], 0.0, 1.0, seed=cfg.seed)
        def do_aug():
            out = tf.numpy_function(lambda z: _augment_mfcc_np(z, cfg), [x], tf.float32)
            out.set_shape(x.shape)
            return out

        x_aug = tf.cond(p < cfg.aug_prob, do_aug, lambda: x)
        return x_aug, y

    return _tf_aug


# ---------------------------
# Conv1D model (industry-friendly)
# ---------------------------
def residual_conv1d_block(x: tf.Tensor,
                          filters: int,
                          kernel_size: int = 5,
                          dilation: int = 1,
                          dropout: float = 0.1,
                          name: str = "res") -> tf.Tensor:
    """
    A small residual block:
      Conv1D -> BN -> ReLU -> Dropout -> Conv1D -> BN
      + shortcut (1x1 if channel mismatch)
      -> ReLU
    """
    shortcut = x

    x = tf.keras.layers.Conv1D(filters, kernel_size, padding="same",
                               dilation_rate=dilation, use_bias=False,
                               name=f"{name}_conv1")(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = tf.keras.layers.ReLU(name=f"{name}_relu1")(x)
    x = tf.keras.layers.Dropout(dropout, name=f"{name}_drop1")(x)

    x = tf.keras.layers.Conv1D(filters, kernel_size, padding="same",
                               dilation_rate=dilation, use_bias=False,
                               name=f"{name}_conv2")(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn2")(x)

    # Match channels if needed
    if shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv1D(filters, 1, padding="same", use_bias=False,
                                          name=f"{name}_proj")(shortcut)
        shortcut = tf.keras.layers.BatchNormalization(name=f"{name}_proj_bn")(shortcut)

    x = tf.keras.layers.Add(name=f"{name}_add")([x, shortcut])
    x = tf.keras.layers.ReLU(name=f"{name}_out_relu")(x)
    return x


def build_conv1d_kws_model(input_shape: Tuple[int, int], num_classes: int) -> tf.keras.Model:
    """
    Input: (time_steps, n_mfcc)
    Output: softmax over classes.
    """
    inp = tf.keras.layers.Input(shape=input_shape, name="mfcc")

    # Light front-end
    x = tf.keras.layers.Conv1D(32, 5, padding="same", use_bias=False, name="stem_conv")(inp)
    x = tf.keras.layers.BatchNormalization(name="stem_bn")(x)
    x = tf.keras.layers.ReLU(name="stem_relu")(x)

    # Residual temporal blocks (dilations help context without extra pooling)
    x = residual_conv1d_block(x, 32, kernel_size=5, dilation=1, dropout=0.1, name="b1")
    x = residual_conv1d_block(x, 64, kernel_size=5, dilation=2, dropout=0.15, name="b2")
    x = residual_conv1d_block(x, 64, kernel_size=3, dilation=4, dropout=0.15, name="b3")
    x = residual_conv1d_block(x, 128, kernel_size=3, dilation=8, dropout=0.2, name="b4")

    # Global pooling -> classifier head
    x = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="fc1")(x)
    x = tf.keras.layers.Dropout(0.4, name="fc1_drop")(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax", name="out")(x)

    return tf.keras.Model(inp, out, name="kws_conv1d_resnet")


# ---------------------------
# Evaluation utilities
# ---------------------------
def save_confusion_matrix(cm: np.ndarray, class_names: Tuple[str, ...], out_path: str, title: str) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, cbar_kws={"label": "Count"})
    plt.title(title)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def evaluate_and_dump(model: tf.keras.Model,
                      X: np.ndarray,
                      y: np.ndarray,
                      class_names: Tuple[str, ...],
                      out_dir: str,
                      tag: str) -> Dict:
    """
    Evaluates on labeled arrays and writes:
      - classification report (txt/json)
      - confusion matrix (png)
    """
    y_cat = tf.keras.utils.to_categorical(y, len(class_names))
    loss, acc = model.evaluate(X, y_cat, verbose=0)

    probs = model.predict(X, verbose=0)
    pred = np.argmax(probs, axis=1)

    report_txt = classification_report(y, pred, target_names=list(class_names), digits=4)
    cm = confusion_matrix(y, pred)

    # Save artifacts
    report_path = os.path.join(out_dir, f"classification_report_{tag}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_txt)

    cm_path = os.path.join(out_dir, f"confusion_matrix_{tag}.png")
    save_confusion_matrix(cm, class_names, cm_path, title=f"Confusion Matrix ({tag})")

    results = {
        "timestamp": datetime.now().isoformat(),
        "tag": tag,
        "loss": float(loss),
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report_txt,
    }
    json_path = os.path.join(out_dir, f"metrics_{tag}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logging.info("[%s] loss=%.4f acc=%.4f", tag, loss, acc)
    logging.info("Saved: %s | %s | %s", report_path, cm_path, json_path)
    return results


# ---------------------------
# Test folder evaluation (flat or subfolders)
# ---------------------------
def predict_flat_folder(model: tf.keras.Model, extractor: MFCCExtractor, cfg: Config, folder: str, out_dir: str) -> None:
    audio_files = [f for f in os.listdir(folder) if f.lower().endswith((".wav", ".mp3", ".m4a"))]
    if not audio_files:
        logging.warning("No audio files in %s", folder)
        return

    results = []
    print(f"\n{'Filename':<50} {'Prediction':<12} {'heytm':>8} {'unknown':>8} {'background':>12}")
    print("-" * 100)

    for fn in sorted(audio_files):
        path = os.path.join(folder, fn)
        mfcc = extractor.extract(path)
        if mfcc is None:
            print(f"{fn:<50} {'ERROR':<12}")
            continue

        probs = model.predict(np.expand_dims(mfcc, axis=0), verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        pred_label = cfg.class_names[pred_idx]

        print(f"{fn:<50} {pred_label:<12} {probs[0]*100:8.2f} {probs[1]*100:8.2f} {probs[2]*100:12.2f}")

        results.append({
            "filename": fn,
            "prediction": pred_label,
            "probs": {cfg.class_names[i]: float(probs[i]) for i in range(len(cfg.class_names))}
        })

    print("-" * 100)

    out_json = os.path.join(out_dir, "test_predictions_flat.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "folder": folder, "results": results}, f, indent=2)
    logging.info("Saved flat test predictions: %s", out_json)


def evaluate_subfolders(model: tf.keras.Model, extractor: MFCCExtractor, cfg: Config, folder: str, out_dir: str) -> None:
    X_test, y_test, _ = load_features_into_memory(cfg, extractor, folder)
    if len(X_test) == 0:
        logging.warning("No labeled test samples found in %s", folder)
        return
    evaluate_and_dump(model, X_test, y_test, cfg.class_names, out_dir, tag="test")


# ---------------------------
# TFLite export
# ---------------------------
def export_tflite_models(cfg: Config,
                         model: tf.keras.Model,
                         X_train: np.ndarray,
                         out_models_dir: str) -> Dict[str, str]:
    os.makedirs(out_models_dir, exist_ok=True)

    paths = {}

    # 1) Float32
    float_path = os.path.join(out_models_dir, "kws_conv1d_float.tflite")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_float = converter.convert()
    with open(float_path, "wb") as f:
        f.write(tflite_float)
    paths["float32"] = float_path
    logging.info("Saved TFLite float32: %s (%.2f MB)", float_path, len(tflite_float) / (1024 * 1024))

    # 2) Float16 quant (keeps float input; good for mobile/Flutter)
    f16_path = os.path.join(out_models_dir, "kws_conv1d_float16.tflite")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_f16 = converter.convert()
    with open(f16_path, "wb") as f:
        f.write(tflite_f16)
    paths["float16"] = f16_path
    logging.info("Saved TFLite float16: %s (%.2f MB)", f16_path, len(tflite_f16) / (1024 * 1024))

    # 3) Optional full int8 (requires int8 input/output on device)
    if cfg.export_int8:
        int8_path = os.path.join(out_models_dir, "kws_conv1d_int8.tflite")

        def rep_ds():
            # Representative dataset: yield small batches of float32 inputs
            for i in range(min(200, len(X_train))):
                yield [X_train[i:i+1].astype(np.float32)]

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = rep_ds
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        tflite_int8 = converter.convert()
        with open(int8_path, "wb") as f:
            f.write(tflite_int8)
        paths["int8"] = int8_path
        logging.info("Saved TFLite int8: %s (%.2f MB)", int8_path, len(tflite_int8) / (1024 * 1024))

    # Write small metadata JSON with input shape hints for Flutter
    meta = {
        "timestamp": datetime.now().isoformat(),
        "class_names": list(cfg.class_names),
        "sample_rate": cfg.sample_rate,
        "duration_sec": cfg.duration_sec,
        "mfcc": {
            "n_mfcc": cfg.n_mfcc,
            "n_mels": cfg.n_mels,
            "n_fft": cfg.n_fft,
            "hop_length": cfg.hop_length,
            "normalization": "per-sample (x-mean)/(std+1e-6)",
        },
        "keras_input_shape": model.input_shape,   # (None, T, n_mfcc)
        "tflite_input_shape": [1, model.input_shape[1], model.input_shape[2]],
        "tflite_outputs": len(cfg.class_names),
        "exported": paths,
    }
    meta_path = os.path.join(out_models_dir, "kws_conv1d_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logging.info("Saved metadata: %s", meta_path)

    return paths


# ---------------------------
# Main
# ---------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Conv1D KWS training (MFCC -> Conv1D).")
    p.add_argument("--dataset", type=str, default="dataset", help="Training dataset root folder")
    p.add_argument("--test", type=str, default="test", help="Test dataset root folder")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--val_split", type=float, default=0.15, help="0 disables validation split")
    p.add_argument("--no_aug", action="store_true", help="Disable MFCC augmentations")
    p.add_argument("--export_int8", action="store_true", help="Also export full int8 TFLite (needs int8 inputs)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="kws_conv1d_out")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = Config(
        dataset_dir=args.dataset,
        test_dir=args.test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_split=args.val_split,
        aug_enable=(not args.no_aug),
        export_int8=args.export_int8,
        seed=args.seed,
        out_dir=args.out_dir,
        models_dir=os.path.join(args.out_dir, "models"),
        logs_dir=os.path.join(args.out_dir, "logs"),
    )

    set_global_determinism(cfg.seed)
    setup_logger(cfg.out_dir)

    os.makedirs(cfg.models_dir, exist_ok=True)
    os.makedirs(cfg.logs_dir, exist_ok=True)

    logging.info("=" * 80)
    logging.info("KWS Conv1D Training")
    logging.info("=" * 80)
    logging.info("Dataset: %s", cfg.dataset_dir)
    logging.info("Test:    %s", cfg.test_dir)
    logging.info("Classes: %s", cfg.class_names)
    logging.info("MFCC: sr=%d dur=%.2fs n_mfcc=%d n_mels=%d n_fft=%d hop=%d",
                 cfg.sample_rate, cfg.duration_sec, cfg.n_mfcc, cfg.n_mels, cfg.n_fft, cfg.hop_length)
    logging.info("Train: epochs=%d batch=%d lr=%.6f val_split=%.2f aug=%s",
                 cfg.epochs, cfg.batch_size, cfg.learning_rate, cfg.val_split, cfg.aug_enable)

    extractor = MFCCExtractor(cfg)

    # Load training features
    logging.info("Loading training features into memory ...")
    X, y, _ = load_features_into_memory(cfg, extractor, cfg.dataset_dir)
    if len(X) == 0:
        logging.error("No training samples found in %s", cfg.dataset_dir)
        sys.exit(1)

    logging.info("Training array shape: X=%s y=%s", X.shape, y.shape)
    time_steps, n_mfcc = X.shape[1], X.shape[2]
    logging.info("Model input will be: (time_steps=%d, n_mfcc=%d)", time_steps, n_mfcc)

    # Split train/val (industry standard)
    if cfg.val_split and cfg.val_split > 0.0:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=cfg.val_split, random_state=cfg.seed, stratify=y
        )
        logging.info("Split: train=%d val=%d", len(X_train), len(X_val))
    else:
        X_train, y_train = X, y
        X_val, y_val = None, None
        logging.info("No validation split (val_split=0). Training on full dataset.")

    # Class weights (helps imbalance)
    classes = np.unique(y_train)
    class_weights_arr = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, class_weights_arr)}
    logging.info("Class weights: %s", class_weight)

    # tf.data pipelines
    AUTOTUNE = tf.data.AUTOTUNE

    y_train_cat = tf.keras.utils.to_categorical(y_train, len(cfg.class_names))
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train_cat))
    train_ds = train_ds.shuffle(min(len(X_train), 20000), seed=cfg.seed, reshuffle_each_iteration=True)

    aug_fn = make_tf_augment_fn(cfg)
    train_ds = train_ds.map(aug_fn, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.batch(cfg.batch_size).prefetch(AUTOTUNE)

    val_ds = None
    if X_val is not None:
        y_val_cat = tf.keras.utils.to_categorical(y_val, len(cfg.class_names))
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val_cat)).batch(cfg.batch_size).prefetch(AUTOTUNE)

    # Build model
    model = build_conv1d_kws_model((time_steps, n_mfcc), num_classes=len(cfg.class_names))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary(print_fn=logging.info)

    # Callbacks
    ckpt_path = os.path.join(cfg.models_dir, "kws_conv1d_best.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_loss" if val_ds is not None else "loss",
            save_best_only=True, mode="min", verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss" if val_ds is not None else "loss",
            factor=0.5, patience=5, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss" if val_ds is not None else "loss",
            patience=10, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.TensorBoard(log_dir=cfg.logs_dir),
        tf.keras.callbacks.CSVLogger(os.path.join(cfg.out_dir, "training_metrics.csv"), append=True),
    ]

    logging.info("=" * 80)
    logging.info("TRAINING START")
    logging.info("=" * 80)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    logging.info("Training finished.")
    # Save final model too
    final_path = os.path.join(cfg.models_dir, "kws_conv1d_final.keras")
    model.save(final_path)
    logging.info("Saved final Keras model: %s", final_path)

    # Evaluate on train/val arrays (more detailed than Keras history)
    evaluate_and_dump(model, X_train, y_train, cfg.class_names, cfg.out_dir, tag="train")
    if X_val is not None:
        evaluate_and_dump(model, X_val, y_val, cfg.class_names, cfg.out_dir, tag="val")

    # Test folder
    if os.path.exists(cfg.test_dir):
        # Decide flat vs subfolder
        flat_audio = [f for f in os.listdir(cfg.test_dir) if f.lower().endswith((".wav", ".mp3", ".m4a"))]
        if flat_audio:
            logging.info("Detected FLAT test folder with %d audio files.", len(flat_audio))
            predict_flat_folder(model, extractor, cfg, cfg.test_dir, cfg.out_dir)
        else:
            logging.info("Detected subfolder test structure. Running labeled evaluation.")
            evaluate_subfolders(model, extractor, cfg, cfg.test_dir, cfg.out_dir)
    else:
        logging.warning("Test folder not found: %s", cfg.test_dir)

    # Export TFLite models
    exported = export_tflite_models(cfg, model, X_train=X_train, out_models_dir=cfg.models_dir)
    logging.info("Exported TFLite: %s", exported)

    # Small final summary
    logging.info("=" * 80)
    logging.info("DONE. Artifacts in: %s", cfg.out_dir)
    logging.info("Models in:        %s", cfg.models_dir)
    logging.info("TensorBoard:      tensorboard --logdir=%s", cfg.logs_dir)
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
