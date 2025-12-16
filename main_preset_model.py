# -*- coding: utf-8 -*-
"""
Transfer learning for custom keyword spotting (KWS) model for "hey tm"
using a pre-trained Google Speech Commands model in TensorFlow 2.x.
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
import numpy as np
import tensorflow as tf
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Check for required modules
try:
    import input_data
    import models
except ImportError:
    logging.error("`input_data.py` and `models.py` are required from TensorFlow's speech_commands example.")
    sys.exit(1)

def run_command(command, description):
    """Run a command and check if it succeeded."""
    logging.info(f"🔄 {description}")
    logging.info(f"Command: {command}")
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        logging.error(f"❌ {description} failed!")
        logging.error(f"Error: {result.stderr}")
        logging.error(f"Output: {result.stdout}")
        return False
    else:
        logging.info(f"✅ {description} completed successfully!")
        return True

# ---- Setup paths ----
DATASET_DIR = "dataset"  # Your local dataset root
TRAIN_DIR = "train_transfer"
LOGS_DIR = "logs_transfer"
MODELS_DIR = "models_transfer"
PRETRAINED_DIR = "pretrained"

for d in [TRAIN_DIR, LOGS_DIR, MODELS_DIR, PRETRAINED_DIR]:
    os.makedirs(d, exist_ok=True)

# ---- Config ----
WANTED_WORDS = "heytm"
TRAINING_STEPS = "3000,500"  # Fewer steps for fine-tuning
LEARNING_RATE = "0.0005,0.0001"  # Lower LR to preserve pre-trained weights
MODEL_ARCHITECTURE = "tiny_conv"

TOTAL_STEPS = str(sum(map(int, TRAINING_STEPS.split(","))))

SILENT_PERCENTAGE = 10
UNKNOWN_PERCENTAGE = 10
VERBOSITY = "INFO"
EVAL_STEP_INTERVAL = "500"
SAVE_STEP_INTERVAL = "500"

MODEL_TFLITE = os.path.join(MODELS_DIR, "KWS_transfer.tflite")
FLOAT_MODEL_TFLITE = os.path.join(MODELS_DIR, "KWS_transfer_float.tflite")
SAVED_MODEL = os.path.join(MODELS_DIR, "KWS_transfer_saved_model")

# ---- Audio preprocessing constants ----
PREPROCESS = "micro"
WINDOW_STRIDE = 20
SAMPLE_RATE = 16000
CLIP_DURATION_MS = 1000
WINDOW_SIZE_MS = 30.0
FEATURE_BIN_COUNT = 40
BACKGROUND_FREQUENCY = 0.8
BACKGROUND_VOLUME_RANGE = 0.1
TIME_SHIFT_MS = 100.0
VALIDATION_PERCENTAGE = 10
TESTING_PERCENTAGE = 10

# ---- Find speech_commands path ----
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
        train_script = os.path.join(path, "train.py")
        freeze_script = os.path.join(path, "freeze.py")
        if os.path.exists(train_script) and os.path.exists(freeze_script):
            return path
    return None

speech_commands_path = find_speech_commands_path()
if speech_commands_path is None:
    logging.error("Cannot find TensorFlow speech_commands examples. Please ensure they are available.")
    sys.exit(1)

logging.info(f"✅ Found speech_commands at: {speech_commands_path}")

# ---- Download pre-trained model ----
PRETRAINED_URL = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/speech_micro_pretrained_model_2020_04_28.zip"
pretrained_zip = os.path.join(PRETRAINED_DIR, "pretrained.zip")
pretrained_ckpt = os.path.join(PRETRAINED_DIR, "tiny_conv.ckpt")

if not os.path.exists(pretrained_ckpt + ".index"):  # Check for extracted checkpoint
    logging.info("🔄 Downloading pre-trained tiny_conv model...")
    urllib.request.urlretrieve(PRETRAINED_URL, pretrained_zip)
    with zipfile.ZipFile(pretrained_zip, 'r') as zip_ref:
        zip_ref.extractall(PRETRAINED_DIR)
    os.remove(pretrained_zip)  # Cleanup zip
    logging.info("✅ Pre-trained model downloaded and extracted.")

# ---- Run fine-tuning ----
logging.info(f"Fine-tuning words: {WANTED_WORDS}")
logging.info(f"Steps: {TRAINING_STEPS}, LR: {LEARNING_RATE}")

train_command = (
    f"python {os.path.join(speech_commands_path, 'train.py')} "
    f"--data_dir={DATASET_DIR} "
    f"--data_url= "
    f"--wanted_words={WANTED_WORDS} "
    f"--silence_percentage={SILENT_PERCENTAGE} "
    f"--unknown_percentage={UNKNOWN_PERCENTAGE} "
    f"--preprocess={PREPROCESS} "
    f"--window_stride={WINDOW_STRIDE} "
    f"--model_architecture={MODEL_ARCHITECTURE} "
    f"--how_many_training_steps={TRAINING_STEPS} "
    f"--learning_rate={LEARNING_RATE} "
    f"--train_dir={TRAIN_DIR} "
    f"--summaries_dir={LOGS_DIR} "
    f"--verbosity={VERBOSITY} "
    f"--eval_step_interval={EVAL_STEP_INTERVAL} "
    f"--save_step_interval={SAVE_STEP_INTERVAL} "
    f"--start_checkpoint={pretrained_ckpt}"
)

if not run_command(train_command, "Model fine-tuning"):
    logging.error("❌ Fine-tuning failed. Cannot proceed.")
    sys.exit(1)

# ---- Check checkpoint ----
checkpoint_path = os.path.join(TRAIN_DIR, f"{MODEL_ARCHITECTURE}.ckpt-{TOTAL_STEPS}")
checkpoint_files = [
    f"{checkpoint_path}.data-00000-of-00001",
    f"{checkpoint_path}.index",
    f"{checkpoint_path}.meta"
]

logging.info(f"🔍 Checking for checkpoint files at: {checkpoint_path}")
missing_files = [file for file in checkpoint_files if not os.path.exists(file)]

if missing_files:
    logging.warning("Missing some checkpoint files. Looking for latest...")
    checkpoint_state = tf.train.get_checkpoint_state(TRAIN_DIR)
    if checkpoint_state and checkpoint_state.model_checkpoint_path:
        checkpoint_path = checkpoint_state.model_checkpoint_path
        logging.info(f"✅ Found latest checkpoint: {checkpoint_path}")
    else:
        logging.error("❌ No valid checkpoint found.")
        sys.exit(1)

# ---- Freeze to SavedModel ----
freeze_command = (
    f"python {os.path.join(speech_commands_path, 'freeze.py')} "
    f"--wanted_words={WANTED_WORDS} "
    f"--window_stride_ms={WINDOW_STRIDE} "
    f"--preprocess={PREPROCESS} "
    f"--model_architecture={MODEL_ARCHITECTURE} "
    f"--start_checkpoint={checkpoint_path} "
    f"--save_format=saved_model "
    f"--output_file={SAVED_MODEL}"
)

if not run_command(freeze_command, "Model freezing to SavedModel"):
    logging.error("❌ Freezing failed.")
    sys.exit(1)

# ---- Verify SavedModel ----
saved_model_pb = os.path.join(SAVED_MODEL, "saved_model.pb")
if not os.path.exists(saved_model_pb):
    logging.error(f"❌ SavedModel not found at: {saved_model_pb}")
    sys.exit(1)

logging.info(f"✅ SavedModel verified at: {saved_model_pb}")

# ---- Convert to TFLite ----
logging.info("🔄 Converting to TFLite...")

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

    # Float model
    logging.info("🔄 Converting to float TFLite model...")
    float_converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL)
    float_tflite_model = float_converter.convert()
    with open(FLOAT_MODEL_TFLITE, "wb") as f:
        f.write(float_tflite_model)
    logging.info(f"✅ Float TFLite model saved to: {FLOAT_MODEL_TFLITE}")

    # Quantized model
    logging.info("🔄 Converting to quantized TFLite model...")
    with tf.compat.v1.Session() as sess:
        converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        REP_DATA_SIZE = 100  # Increased for better calibration
        def representative_dataset_gen():
            logging.info(f"Generating {REP_DATA_SIZE} representative samples...")
            for i in range(REP_DATA_SIZE):
                try:
                    data, _ = audio_processor.get_data(
                        1, i * 1, model_settings,
                        BACKGROUND_FREQUENCY, BACKGROUND_VOLUME_RANGE,
                        TIME_SHIFT_MS, "testing", sess
                    )
                    yield [np.array(data.flatten(), dtype=np.float32).reshape(1, -1)]
                except Exception as e:
                    logging.warning(f"Failed to generate sample {i}: {e}")

        converter.representative_dataset = representative_dataset_gen
        tflite_model = converter.convert()

        with open(MODEL_TFLITE, "wb") as f:
            f.write(tflite_model)
        logging.info(f"✅ Quantized TFLite model saved to: {MODEL_TFLITE}")

except Exception as e:
    logging.error(f"❌ TFLite conversion failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ---- Summary ----
logging.info("🎉 Fine-tuning and conversion finished!")
logging.info(f"📁 Models saved in: {MODELS_DIR}")
for model_file in [FLOAT_MODEL_TFLITE, MODEL_TFLITE]:
    if os.path.exists(model_file):
        size_mb = os.path.getsize(model_file) / (1024 * 1024)
        logging.info(f"  - {model_file} ({size_mb:.2f} MB)")

logging.info("\n💡 Next steps: Test and deploy the model.")