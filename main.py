# -*- coding: utf-8 -*-
"""
Train and convert a custom keyword spotting (KWS) model for "hey tm"
using a TensorFlow 2.x workflow with proper error handling.
"""

import os
import sys
import subprocess
import numpy as np
import tensorflow as tf

# Check for required modules
try:
    import input_data
    import models
except ImportError:
    print("Error: `input_data.py` and `models.py` are required.")
    print("These scripts are part of the TensorFlow repository's `speech_commands` example.")
    print("Please ensure your Python path includes the `tensorflow/tensorflow/examples/speech_commands/` directory.")
    sys.exit(1)

def run_command(command, description):
    """Run a command and check if it succeeded."""
    print(f"\n🔄 {description}")
    print(f"Command: {command}")
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ {description} failed!")
        print(f"Error: {result.stderr}")
        print(f"Output: {result.stdout}")
        return False
    else:
        print(f"✅ {description} completed successfully!")
        return True

# ---- Setup paths ----
DATASET_DIR = "dataset"  # Your local dataset root
TRAIN_DIR = "train"
LOGS_DIR = "logs"
MODELS_DIR = "models"

for d in [TRAIN_DIR, LOGS_DIR, MODELS_DIR]:
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# ---- Config ----
WANTED_WORDS = "heytm"
TRAINING_STEPS = "6000,1000"
LEARNING_RATE = "0.001,0.0001"
MODEL_ARCHITECTURE = "tiny_conv"

TOTAL_STEPS = str(sum(map(int, TRAINING_STEPS.split(","))))

SILENT_PERCENTAGE = 10
UNKNOWN_PERCENTAGE = 10
VERBOSITY = "DEBUG"
EVAL_STEP_INTERVAL = "1000"
SAVE_STEP_INTERVAL = "1000"

MODEL_TF = os.path.join(MODELS_DIR, "KWS_custom.pb")
MODEL_TFLITE = os.path.join(MODELS_DIR, "KWS_custom.tflite")
FLOAT_MODEL_TFLITE = os.path.join(MODELS_DIR, "KWS_custom_float.tflite")
SAVED_MODEL = os.path.join(MODELS_DIR, "KWS_custom_saved_model")

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

# ---- Check if TensorFlow speech_commands examples are available ----
def find_speech_commands_path():
    """Find the TensorFlow speech_commands examples directory."""
    import tensorflow as tf
    tf_path = tf.__file__
    tf_dir = os.path.dirname(tf_path)
    
    # Common locations for speech_commands
    possible_paths = [
        os.path.join(tf_dir, "examples", "speech_commands"),
        os.path.join(os.path.dirname(tf_dir), "tensorflow", "examples", "speech_commands"),
        "tensorflow/tensorflow/examples/speech_commands",  # If running from TensorFlow repo
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
    print("❌ Error: Cannot find TensorFlow speech_commands examples.")
    print("Please clone the TensorFlow repository and ensure the examples are available.")
    print("You can get them from: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/speech_commands")
    sys.exit(1)

print(f"✅ Found speech_commands at: {speech_commands_path}")

# ---- Run training ----
print("Training words:", WANTED_WORDS)
print("Steps:", TRAINING_STEPS, "LR:", LEARNING_RATE)

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
    f"--save_step_interval={SAVE_STEP_INTERVAL}"
)

if not run_command(train_command, "Model training"):
    print("❌ Training failed. Cannot proceed with conversion.")
    sys.exit(1)

# ---- Check if checkpoint exists ----
checkpoint_path = os.path.join(TRAIN_DIR, f"{MODEL_ARCHITECTURE}.ckpt-{TOTAL_STEPS}")
checkpoint_files = [
    f"{checkpoint_path}.data-00000-of-00001",
    f"{checkpoint_path}.index",
    f"{checkpoint_path}.meta"
]

print(f"\n🔍 Checking for checkpoint files at: {checkpoint_path}")
missing_files = []
for file in checkpoint_files:
    if not os.path.exists(file):
        missing_files.append(file)
    else:
        print(f"✅ Found: {file}")

if missing_files:
    print("❌ Missing checkpoint files:")
    for file in missing_files:
        print(f"  - {file}")
    
    # Look for alternative checkpoint
    print("\n🔍 Looking for latest checkpoint...")
    checkpoint_state = tf.train.get_checkpoint_state(TRAIN_DIR)
    if checkpoint_state and checkpoint_state.model_checkpoint_path:
        latest_checkpoint = checkpoint_state.model_checkpoint_path
        print(f"✅ Found latest checkpoint: {latest_checkpoint}")
        checkpoint_path = latest_checkpoint
    else:
        print("❌ No valid checkpoint found. Training may have failed.")
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
    print("❌ Freezing failed. Cannot proceed with TFLite conversion.")
    sys.exit(1)

# ---- Verify SavedModel exists ----
saved_model_pb = os.path.join(SAVED_MODEL, "saved_model.pb")
if not os.path.exists(saved_model_pb):
    print(f"❌ SavedModel not found at: {saved_model_pb}")
    print("Available files in models directory:")
    if os.path.exists(MODELS_DIR):
        for root, dirs, files in os.walk(MODELS_DIR):
            for file in files:
                print(f"  {os.path.join(root, file)}")
    sys.exit(1)

print(f"✅ SavedModel verified at: {saved_model_pb}")

# ---- Convert to TFLite using TF2 APIs ----
print("\n🔄 Converting to TFLite using TensorFlow 2.x APIs...")

try:
    # Prepare the data processor for the representative dataset
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

    # ---- Float model conversion ----
    print("🔄 Converting to float TFLite model...")
    float_converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL)
    float_tflite_model = float_converter.convert()
    with open(FLOAT_MODEL_TFLITE, "wb") as f:
        f.write(float_tflite_model)
    print(f"✅ Float TFLite model saved to: {FLOAT_MODEL_TFLITE}")

    # ---- Quantized model conversion ----
    print("🔄 Converting to quantized TFLite model...")
    
    # Create a new session for data generation
    with tf.compat.v1.Session() as sess:
        converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL)

        # Enable full integer quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        # Define the representative dataset generator
        REP_DATA_SIZE = 50
        def representative_dataset_gen():
            """Generates representative dataset for quantization."""
            print(f"Generating {REP_DATA_SIZE} representative samples...")
            for i in range(REP_DATA_SIZE):
                try:
                    data, _ = audio_processor.get_data(
                        1, i * 1, model_settings,
                        BACKGROUND_FREQUENCY, BACKGROUND_VOLUME_RANGE,
                        TIME_SHIFT_MS, "testing", sess
                    )
                    yield [np.array(data.flatten(), dtype=np.float32).reshape(1, -1)]
                    if (i + 1) % 10 == 0:
                        print(f"Generated {i + 1}/{REP_DATA_SIZE} samples")
                except Exception as e:
                    print(f"Warning: Failed to generate sample {i}: {e}")
                    continue

        # Attach the representative dataset to the converter
        converter.representative_dataset = representative_dataset_gen
        tflite_model = converter.convert()

        with open(MODEL_TFLITE, "wb") as f:
            f.write(tflite_model)
        print(f"✅ Quantized TFLite model saved to: {MODEL_TFLITE}")

except Exception as e:
    print(f"❌ TFLite conversion failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ---- Summary ----
print(f"\n🎉 Training and conversion finished successfully!")
print(f"📁 Models are saved in: {MODELS_DIR}")
print(f"📄 Available models:")
for model_file in [FLOAT_MODEL_TFLITE, MODEL_TFLITE]:
    if os.path.exists(model_file):
        size_mb = os.path.getsize(model_file) / (1024 * 1024)
        print(f"  - {model_file} ({size_mb:.2f} MB)")

print(f"\n💡 Next steps:")
print(f"  1. Test your models with: python test_model.py")
print(f"  2. Deploy {MODEL_TFLITE} to your target device")
print(f"  3. Integrate with your application")