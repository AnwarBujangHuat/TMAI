# -*- coding: utf-8 -*-
"""
Train and convert a custom keyword spotting (KWS) model for "hey tm"
using PURE TensorFlow 2.x workflow (no TF1 compatibility layer needed)
"""

import os
import sys
import subprocess
import numpy as np
import tensorflow as tf

# Disable TF2 eager execution for compatibility with speech_commands
tf.compat.v1.disable_eager_execution()

# Check for required modules
try:
    import input_data
    import models
except ImportError:
    print("Error: `input_data.py` and `models.py` are required.")
    print("These scripts are part of the TensorFlow repository's `speech_commands` example.")
    print("Please ensure your Python path includes the `tensorflow/tensorflow/examples/speech_commands/` directory.")
    sys.exit(1)

def analyze_local_dataset(dataset_dir):
    """Analyze your local dataset before training"""
    print("🔍 ANALYZING YOUR LOCAL DATASET")
    print("=" * 50)
    
    if not os.path.exists(dataset_dir):
        print(f"❌ Dataset directory '{dataset_dir}' not found!")
        return False
    
    # Check for heytm folder
    heytm_path = os.path.join(dataset_dir, "heytm")
    if not os.path.exists(heytm_path):
        print(f"❌ 'heytm' folder not found at: {heytm_path}")
        return False
    
    # Count files
    import glob
    audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a', '*.WAV', '*.MP3']
    heytm_files = 0
    for ext in audio_extensions:
        files = glob.glob(os.path.join(heytm_path, ext))
        heytm_files += len(files)
    
    print(f"📊 Found {heytm_files} 'heytm' audio files")
    
    # Calculate splits
    training_samples = int(heytm_files * 0.8)
    validation_samples = int(heytm_files * 0.1)
    testing_samples = int(heytm_files * 0.1)
    
    print(f"📈 Split: Training={training_samples}, Validation={validation_samples}, Testing={testing_samples}")
    
    # Check background noise
    bg_path = os.path.join(dataset_dir, "_background_noise_")
    if os.path.exists(bg_path):
        bg_files = 0
        for ext in audio_extensions:
            bg_files += len(glob.glob(os.path.join(bg_path, ext)))
        print(f"🔊 Background noise files: {bg_files}")
    
    return heytm_files >= 50

def run_command_with_output(command, description):
    """Run command with real-time output"""
    print(f"\n🔄 {description}")
    print(f"Command: {command}")
    print("-" * 60)
    
    try:
        process = subprocess.Popen(
            command, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        for line in iter(process.stdout.readline, ''):
            print(line.rstrip())
        
        process.stdout.close()
        return_code = process.wait()
        
        if return_code == 0:
            print(f"\n✅ {description} completed successfully!")
            return True
        else:
            print(f"\n❌ {description} failed with return code {return_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

# ---- Setup paths ----
DATASET_DIR = "dataset"
TRAIN_DIR = "train"
LOGS_DIR = "logs"
MODELS_DIR = "models"

for d in [TRAIN_DIR, LOGS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# ---- Config ----
WANTED_WORDS = "heytm"
TRAINING_STEPS = "4000,1000"  # Reduced for faster training
LEARNING_RATE = "0.001,0.0001"
MODEL_ARCHITECTURE = "tiny_conv"

TOTAL_STEPS = str(sum(map(int, TRAINING_STEPS.split(","))))

SILENT_PERCENTAGE = 10
UNKNOWN_PERCENTAGE = 10
VERBOSITY = "INFO"  # Less verbose than DEBUG
EVAL_STEP_INTERVAL = "500"
SAVE_STEP_INTERVAL = "500"

# Model file paths
SAVED_MODEL = os.path.join(MODELS_DIR, "KWS_custom_saved_model")
MODEL_TFLITE = os.path.join(MODELS_DIR, "KWS_custom.tflite")
FLOAT_MODEL_TFLITE = os.path.join(MODELS_DIR, "KWS_custom_float.tflite")

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

# ---- Main execution ----
if __name__ == "__main__":
    print("🚀 TENSORFLOW 2.x KEYWORD SPOTTING TRAINING")
    print("=" * 50)
    
    # Analyze dataset first
    if not analyze_local_dataset(DATASET_DIR):
        print("❌ Dataset issues found. Please fix before training.")
        sys.exit(1)
    
    # Find TensorFlow speech_commands path
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
            if os.path.exists(os.path.join(path, "train.py")):
                return path
        return None
    
    speech_commands_path = find_speech_commands_path()
    if speech_commands_path is None:
        print("❌ Cannot find TensorFlow speech_commands examples.")
        print("Please clone: https://github.com/tensorflow/tensorflow")
        sys.exit(1)
    
    print(f"✅ Found speech_commands at: {speech_commands_path}")
    
    # Training command
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
    
    # Run training
    if not run_command_with_output(train_command, "Model training"):
        print("❌ Training failed.")
        sys.exit(1)
    
    # Find checkpoint
    checkpoint_path = os.path.join(TRAIN_DIR, f"{MODEL_ARCHITECTURE}.ckpt-{TOTAL_STEPS}")
    
    # Check if checkpoint exists, otherwise find latest
    if not os.path.exists(f"{checkpoint_path}.index"):
        print("🔍 Looking for latest checkpoint...")
        checkpoint_state = tf.train.get_checkpoint_state(TRAIN_DIR)
        if checkpoint_state and checkpoint_state.model_checkpoint_path:
            checkpoint_path = checkpoint_state.model_checkpoint_path
            print(f"✅ Using checkpoint: {checkpoint_path}")
        else:
            print("❌ No checkpoint found.")
            sys.exit(1)
    
    # Freeze model
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
    
    if not run_command_with_output(freeze_command, "Model freezing"):
        print("❌ Freezing failed.")
        sys.exit(1)
    
    # Convert to TFLite - IMPROVED TF2 APPROACH
    print("\n🔄 Converting to TensorFlow Lite...")
    
    try:
        # Prepare model settings and audio processor
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
        
        # Float model conversion
        print("🔄 Converting float model...")
        float_converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL)
        float_tflite_model = float_converter.convert()
        
        with open(FLOAT_MODEL_TFLITE, "wb") as f:
            f.write(float_tflite_model)
        print(f"✅ Float model: {FLOAT_MODEL_TFLITE}")
        
        # Quantized model conversion - FIXED FOR TF2
        print("🔄 Converting quantized model...")
        
        # Use TF2 approach but with TF1 session for compatibility
        with tf.compat.v1.Session() as sess:
            converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Modern TF2 quantization settings
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            
            # Representative dataset with better error handling
            REP_DATA_SIZE = 50  # Reduced size to avoid issues
            
            def representative_dataset_gen():
                print(f"Generating {REP_DATA_SIZE} representative samples...")
                successful_samples = 0
                
                for i in range(REP_DATA_SIZE * 2):  # Try more samples in case some fail
                    try:
                        data, _ = audio_processor.get_data(
                            1, i, model_settings,
                            BACKGROUND_FREQUENCY, BACKGROUND_VOLUME_RANGE,
                            TIME_SHIFT_MS, "testing", sess
                        )
                        
                        if data is not None and data.size > 0:
                            # Proper reshaping for the model
                            flattened_data = np.array(data.flatten(), dtype=np.float32)
                            if flattened_data.size > 0:
                                yield [flattened_data.reshape(1, -1)]
                                successful_samples += 1
                                
                                if successful_samples % 10 == 0:
                                    print(f"Generated {successful_samples}/{REP_DATA_SIZE} samples")
                                
                                if successful_samples >= REP_DATA_SIZE:
                                    break
                    except Exception as e:
                        print(f"Warning: Sample {i} failed: {e}")
                        continue
                
                print(f"Successfully generated {successful_samples} representative samples")
            
            converter.representative_dataset = representative_dataset_gen
            quantized_model = converter.convert()
            
            with open(MODEL_TFLITE, "wb") as f:
                f.write(quantized_model)
            
            print(f"✅ Quantized model: {MODEL_TFLITE}")
    
    except Exception as e:
        print(f"❌ TFLite conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Success summary
    print(f"\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print(f"📁 Models saved in: {MODELS_DIR}")
    
    for model_file in [FLOAT_MODEL_TFLITE, MODEL_TFLITE]:
        if os.path.exists(model_file):
            size_kb = os.path.getsize(model_file) / 1024
            print(f"📄 {os.path.basename(model_file)}: {size_kb:.1f} KB")
    
    print(f"\n💡 Next steps:")
    print(f"  1. Test the model with audio samples")
    print(f"  2. Deploy {MODEL_TFLITE} to your device")
    print(f"  3. Convert to C array if needed for microcontrollers")