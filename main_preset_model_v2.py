# -*- coding: utf-8 -*-
"""
Transfer learning for custom keyword spotting (KWS) model for "heytm"
using Google's pre-trained Speech Commands model (TensorFlow 2.x).
Recommended approach: higher accuracy, faster training, smaller data needed.
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
import numpy as np
import tensorflow as tf
import logging

# Setup logging (cleaner than print statements)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Check for required modules from TensorFlow speech_commands example
try:
    import input_data
    import models
except ImportError:
    logging.error("Error: `input_data.py` and `models.py` are required.")
    logging.error("They are part of TensorFlow's speech_commands example.")
    logging.error("Make sure the directory containing them is in your PYTHONPATH.")
    sys.exit(1)

def run_command(command, description):
    """Run a shell command and log success/failure."""
    logging.info(f"🔄 {description}")
    logging.debug(f"Command: {command}")

    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        logging.error(f"❌ {description} failed!")
        logging.error(f"STDERR: {result.stderr.strip()}")
        logging.error(f"STDOUT: {result.stdout.strip()}")
        return False
    else:
        logging.info(f"✅ {description} completed successfully!")
        return True

# ----------------------------- Configuration -----------------------------
DATASET_DIR = "dataset"                  # Your dataset root (with folders like heytm/, _background_noise_/, etc.)
TRAIN_DIR = "train_transfer"
LOGS_DIR = "logs_transfer"
MODELS_DIR = "models_transfer"
PRETRAINED_DIR = "pretrained"

for d in [TRAIN_DIR, LOGS_DIR, MODELS_DIR, PRETRAINED_DIR]:
    os.makedirs(d, exist_ok=True)

# Training hyperparameters (fine-tuning needs fewer steps and lower LR)
WANTED_WORDS = "heytm"
TRAINING_STEPS = "3000,500"              # Total ~3500 steps
LEARNING_RATE = "0.0005,0.0001"           # Lower to preserve pre-trained features
MODEL_ARCHITECTURE = "tiny_conv"

TOTAL_STEPS = str(sum(int(x) for x in TRAINING_STEPS.split(",")))

# Audio / data settings
SILENT_PERCENTAGE = 10
UNKNOWN_PERCENTAGE = 10
PREPROCESS = "micro"
WINDOW_STRIDE = 20
SAMPLE_RATE = 16000
CLIP_DURATION_MS = 1000
WINDOW_SIZE_MS = 30.0
FEATURE_BIN_COUNT = 40
BACKGROUND_FREQUENCY = 0.8
BACKGROUND_VOLUME_RANGE = 0.1
TIME_SHIFT_MS = 100.0
VALIDATION_PERCENTAGE = 0
TESTING_PERCENTAGE = 0

# Output files
MODEL_TFLITE = os.path.join(MODELS_DIR, "KWS_transfer.tflite")
FLOAT_MODEL_TFLITE = os.path.join(MODELS_DIR, "KWS_transfer_float.tflite")
SAVED_MODEL = os.path.join(MODELS_DIR, "KWS_transfer_saved_model")

# ----------------------------- Find speech_commands scripts -----------------------------
def find_speech_commands_path():
    tf_path = tf.__file__
    tf_dir = os.path.dirname(tf_path)
    possible_paths = [
        os.path.join(tf_dir, "examples", "speech_commands"),
        os.path.join(os.path.dirname(tf_dir), "tensorflow", "examples", "speech_commands"),
        "tensorflow/tensorflow/examples/speech_commands",
        "../tensorflow/examples/speech_commands",
    ]
    for path in possible_paths:
        if os.path.exists(os.path.join(path, "train.py")) and os.path.exists(os.path.join(path, "freeze.py")):
            return path
    return None

speech_commands_path = find_speech_commands_path()
if speech_commands_path is None:
    logging.error("❌ Cannot find TensorFlow speech_commands examples (train.py and freeze.py).")
    logging.error("Clone TensorFlow repo or add the directory to your path.")
    sys.exit(1)

logging.info(f"✅ Found speech_commands at: {speech_commands_path}")

# ----------------------------- Download pre-trained model -----------------------------
PRETRAINED_URL = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/speech_micro_pretrained_model_2020_04_28.zip"
pretrained_zip = os.path.join(PRETRAINED_DIR, "pretrained.zip")
pretrained_ckpt_prefix = os.path.join(PRETRAINED_DIR, "tiny_conv.ckpt")

if not os.path.exists(pretrained_ckpt_prefix + ".index"):
    logging.info("🔄 Downloading pre-trained tiny_conv model...")
    urllib.request.urlretrieve(PRETRAINED_URL, pretrained_zip)
    logging.info("Extracting...")
    with zipfile.ZipFile(pretrained_zip, 'r') as zip_ref:
        zip_ref.extractall(PRETRAINED_DIR)
    os.remove(pretrained_zip)
    logging.info("✅ Pre-trained model ready.")
else:
    logging.info("✅ Pre-trained model already exists.")

# ----------------------------- Fine-tuning -----------------------------
logging.info(f"Fine-tuning on keyword: {WANTED_WORDS}")

train_command = (
    f"python \"{os.path.join(speech_commands_path, 'train.py')}\" "
    f"--data_dir=\"{DATASET_DIR}\" "
    f"--wanted_words={WANTED_WORDS} "
    f"--silence_percentage={SILENT_PERCENTAGE} "
    f"--unknown_percentage={UNKNOWN_PERCENTAGE} "
    f"--validation_percentage={VALIDATION_PERCENTAGE} "
    f"--testing_percentage={TESTING_PERCENTAGE} "
    f"--preprocess={PREPROCESS} "
    f"--window_stride={WINDOW_STRIDE} "
    f"--model_architecture={MODEL_ARCHITECTURE} "
    f"--how_many_training_steps={TRAINING_STEPS} "
    f"--learning_rate={LEARNING_RATE} "
    f"--train_dir=\"{TRAIN_DIR}\" "
    f"--summaries_dir=\"{LOGS_DIR}\" "
    f"--verbosity=INFO "
    f"--eval_step_interval=500 "
    f"--save_step_interval=500 "
    f"--start_checkpoint=\"{pretrained_ckpt_prefix}\""
)

if not run_command(train_command, "Fine-tuning the model"):
    sys.exit(1)

# ----------------------------- Find final checkpoint -----------------------------
checkpoint_path = os.path.join(TRAIN_DIR, f"{MODEL_ARCHITECTURE}.ckpt-{TOTAL_STEPS}")
if not os.path.exists(checkpoint_path + ".index"):
    logging.warning("Exact final checkpoint not found. Looking for latest...")
    checkpoint_state = tf.train.get_checkpoint_state(TRAIN_DIR)
    if checkpoint_state and checkpoint_state.model_checkpoint_path:
        checkpoint_path = checkpoint_state.model_checkpoint_path
        logging.info(f"Using latest checkpoint: {checkpoint_path}")
    else:
        logging.error("❌ No checkpoint found. Training may have failed.")
        sys.exit(1)
else:
    logging.info(f"✅ Using checkpoint: {checkpoint_path}")

# ----------------------------- Freeze to SavedModel -----------------------------
freeze_command = (
    f"python \"{os.path.join(speech_commands_path, 'freeze.py')}\" "
    f"--wanted_words={WANTED_WORDS} "
    f"--window_stride_ms={WINDOW_STRIDE} "
    f"--preprocess={PREPROCESS} "
    f"--model_architecture={MODEL_ARCHITECTURE} "
    f"--start_checkpoint=\"{checkpoint_path}\" "
    f"--save_format=saved_model "
    f"--output_file=\"{SAVED_MODEL}\""
)

if not run_command(freeze_command, "Freezing to SavedModel"):
    sys.exit(1)

if not os.path.exists(os.path.join(SAVED_MODEL, "saved_model.pb")):
    logging.error("❌ SavedModel not created properly.")
    sys.exit(1)

logging.info("✅ SavedModel created successfully.")

# ----------------------------- Convert to TFLite -----------------------------
logging.info("🔄 Converting to TFLite models...")

try:
    model_settings = models.prepare_model_settings(
        len(input_data.prepare_words_list(WANTED_WORDS.split(","))),
        SAMPLE_RATE, CLIP_DURATION_MS, WINDOW_SIZE_MS,
        WINDOW_STRIDE, FEATURE_BIN_COUNT, PREPROCESS,
    )

    audio_processor = input_data.AudioProcessor(
        "", DATASET_DIR,
        SILENT_PERCENTAGE, UNKNOWN_PERCENTAGE,
        WANTED_WORDS.split(","), VALIDATION_PERCENTAGE,
        TESTING_PERCENTAGE, model_settings, LOGS_DIR
    )

    # Float TFLite
    logging.info("Converting to float TFLite...")
    float_converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL)
    float_tflite = float_converter.convert()
    with open(FLOAT_MODEL_TFLITE, "wb") as f:
        f.write(float_tflite)
    logging.info(f"✅ Float model saved: {FLOAT_MODEL_TFLITE}")

    # Quantized int8 TFLite
    logging.info("Converting to quantized int8 TFLite...")
    with tf.compat.v1.Session() as sess:
        converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        REP_DATA_SIZE = 100
        def representative_dataset_gen():
            logging.info(f"Generating {REP_DATA_SIZE} representative samples for quantization...")
            for i in range(REP_DATA_SIZE):
                try:
                    data, _ = audio_processor.get_data(
                        1, i, model_settings,
                        BACKGROUND_FREQUENCY, BACKGROUND_VOLUME_RANGE,
                        TIME_SHIFT_MS, "training", sess
                    )
                    flattened = np.array(data.flatten(), dtype=np.float32).reshape(1, -1)
                    yield [flattened]
                except Exception as e:
                    logging.warning(f"Sample {i} failed: {e}")

        converter.representative_dataset = representative_dataset_gen
        quantized_tflite = converter.convert()

        with open(MODEL_TFLITE, "wb") as f:
            f.write(quantized_tflite)
        logging.info(f"✅ Quantized model saved: {MODEL_TFLITE}")

except Exception as e:
    logging.error(f"❌ TFLite conversion failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ----------------------------- Final Summary -----------------------------
logging.info("\n🎉 Transfer learning and conversion completed successfully!")
logging.info(f"Models saved in: {MODELS_DIR}")
for path in [FLOAT_MODEL_TFLITE, MODEL_TFLITE]:
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        logging.info(f"  • {os.path.basename(path)} ({size_mb:.3f} MB)")

logging.info("\nNext steps:")
logging.info("  1. Test the model with the official test scripts or your own code.")
logging.info("  2. Deploy KWS_transfer.tflite (quantized) to your microcontroller.")
logging.info("  3. Enjoy your highly accurate 'hey tm' wake word!")
