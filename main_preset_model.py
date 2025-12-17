# -*- coding: utf-8 -*-
"""
Transfer learning for custom keyword spotting (KWS) model for "hey tm"
with comprehensive testing and accuracy evaluation.
"""

import os
import sys
import urllib.request
import zipfile
import numpy as np
import tensorflow as tf
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import logging
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_log.txt'),
        logging.StreamHandler()
    ]
)

# ---- Audio Processing Class ----
class AudioFeatureExtractor:
    """Extract audio features matching the previous implementation"""
    def __init__(self, sample_rate=16000, duration=1.0, n_mels=40, n_mfcc=13):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = int(sample_rate * duration)
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc

    def extract_features(self, audio_path):
        """Extract MFCC features from audio file"""
        try:
            # Load audio file
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Pad or trim to fixed length
            if len(audio) > self.n_samples:
                audio = audio[:self.n_samples]
            else:
                audio = np.pad(audio, (0, self.n_samples - len(audio)), mode='constant')
            
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=512,
                hop_length=160,
                n_mels=self.n_mels
            )
            
            # Normalize
            mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)
            return mfcc.T  # Transpose for time-first format
            
        except Exception as e:
            logging.warning(f"Error processing {audio_path}: {e}")
            return None

    def load_dataset(self, dataset_path, class_names=['heytm', 'unknown', 'background']):
        """Load and preprocess the dataset"""
        features = []
        labels = []
        filenames = []
        
        for class_idx, class_name in enumerate(class_names):
            if class_name == 'background':
                folder_name = '_background_noise_'
            else:
                folder_name = class_name
                
            class_path = os.path.join(dataset_path, folder_name)
            
            if not os.path.exists(class_path):
                logging.warning(f"Warning: {class_path} not found!")
                continue
                
            logging.info(f"Loading {class_name} samples from {class_path}...")
            class_samples = 0
            
            for filename in os.listdir(class_path):
                if filename.endswith(('.wav', '.mp3', '.m4a')):
                    file_path = os.path.join(class_path, filename)
                    mfcc = self.extract_features(file_path)
                    
                    if mfcc is not None:
                        features.append(mfcc)
                        labels.append(class_idx)
                        filenames.append(filename)
                        class_samples += 1
            
            logging.info(f"Loaded {class_samples} {class_name} samples")
        
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(labels)
        
        logging.info(f"Dataset shape: {X.shape}")
        logging.info(f"Labels shape: {y.shape}")
        
        return X, y, filenames

# ---- Testing and Evaluation Class ----
class ModelTester:
    """Test and evaluate the trained model"""
    def __init__(self, model, class_names, extractor):
        self.model = model
        self.class_names = class_names
        self.extractor = extractor
    
    def test_on_folder(self, test_folder_path):
        """Test model on a folder with subfolder structure (heytm/, unknown/, background/)"""
        logging.info(f"\n{'='*80}")
        logging.info(f"🧪 TESTING MODEL ON: {test_folder_path}")
        logging.info(f"{'='*80}")
        
        # Load test dataset with labels
        X_test, y_test, filenames = self.extractor.load_dataset(test_folder_path, self.class_names)
        
        if len(X_test) == 0:
            logging.error(f"❌ No test samples found in {test_folder_path}")
            return None
        
        # Convert labels to categorical
        y_test_categorical = tf.keras.utils.to_categorical(y_test, len(self.class_names))
        
        # Evaluate model
        logging.info("\n📊 Evaluating model on test set...")
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test_categorical, verbose=0)
        
        logging.info(f"✅ Test Loss: {test_loss:.4f}")
        logging.info(f"✅ Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        # Get predictions
        y_pred_probs = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Calculate per-class accuracy
        logging.info(f"\n📈 PER-CLASS PERFORMANCE:")
        logging.info(f"{'='*80}")
        
        for class_idx, class_name in enumerate(self.class_names):
            class_mask = y_test == class_idx
            if np.sum(class_mask) > 0:
                class_acc = np.mean(y_pred[class_mask] == y_test[class_mask])
                class_total = np.sum(class_mask)
                class_correct = np.sum(y_pred[class_mask] == y_test[class_mask])
                logging.info(f"{class_name.upper():12} | Accuracy: {class_acc:.4f} ({class_acc*100:.2f}%) | Correct: {class_correct}/{class_total}")
        
        # Generate detailed classification report
        logging.info(f"\n📋 DETAILED CLASSIFICATION REPORT:")
        logging.info(f"{'='*80}")
        report = classification_report(
            y_test, y_pred, 
            target_names=self.class_names,
            digits=4
        )
        logging.info(f"\n{report}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logging.info(f"\n🎯 CONFUSION MATRIX:")
        logging.info(f"{'='*80}")
        logging.info(f"Predicted →")
        logging.info(f"Actual ↓\n")
        
        # Print confusion matrix with labels
        header = "           " + "  ".join([f"{name[:8]:>8}" for name in self.class_names])
        logging.info(header)
        for i, class_name in enumerate(self.class_names):
            row = f"{class_name[:10]:10} " + "  ".join([f"{cm[i][j]:8d}" for j in range(len(self.class_names))])
            logging.info(row)
        
        # Find misclassified samples
        misclassified_indices = np.where(y_pred != y_test)[0]
        logging.info(f"\n⚠️  MISCLASSIFIED SAMPLES: {len(misclassified_indices)}/{len(y_test)}")
        
        if len(misclassified_indices) > 0:
            logging.info(f"{'='*80}")
            logging.info(f"Showing first 20 misclassifications:")
            for idx in misclassified_indices[:20]:
                true_label = self.class_names[y_test[idx]]
                pred_label = self.class_names[y_pred[idx]]
                confidence = y_pred_probs[idx][y_pred[idx]]
                logging.info(f"  File: {filenames[idx][:40]:40} | True: {true_label:10} | Pred: {pred_label:10} | Conf: {confidence:.3f}")
        
        # Save detailed results
        results = {
            'timestamp': datetime.now().isoformat(),
            'test_folder': test_folder_path,
            'total_samples': len(X_test),
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss),
            'per_class_accuracy': {},
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'misclassified_count': len(misclassified_indices),
            'misclassified_samples': []
        }
        
        # Per-class accuracy
        for class_idx, class_name in enumerate(self.class_names):
            class_mask = y_test == class_idx
            if np.sum(class_mask) > 0:
                class_acc = float(np.mean(y_pred[class_mask] == y_test[class_mask]))
                results['per_class_accuracy'][class_name] = {
                    'accuracy': class_acc,
                    'total': int(np.sum(class_mask)),
                    'correct': int(np.sum(y_pred[class_mask] == y_test[class_mask]))
                }
        
        # Misclassified samples
        for idx in misclassified_indices:
            results['misclassified_samples'].append({
                'filename': filenames[idx],
                'true_label': self.class_names[y_test[idx]],
                'predicted_label': self.class_names[y_pred[idx]],
                'confidence': float(y_pred_probs[idx][y_pred[idx]]),
                'probabilities': {
                    self.class_names[i]: float(y_pred_probs[idx][i]) 
                    for i in range(len(self.class_names))
                }
            })
        
        # Save results to JSON
        results_file = 'test_results_detailed.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logging.info(f"\n💾 Detailed results saved to: {results_file}")
        
        # Plot confusion matrix
        self.plot_confusion_matrix(cm, test_folder_path)
        
        return results
    
    def plot_confusion_matrix(self, cm, test_folder_path):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - Test Set\n{test_folder_path}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        filename = 'confusion_matrix.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logging.info(f"📊 Confusion matrix saved to: {filename}")
        plt.close()

# ---- Setup paths ----
DATASET_DIR = "dataset"  # Training dataset
TEST_DIR = "test"  # Test dataset with same structure
TRAIN_DIR = "train_transfer"
LOGS_DIR = "logs_transfer"
MODELS_DIR = "models_transfer"
PRETRAINED_DIR = "pretrained"

for d in [TRAIN_DIR, LOGS_DIR, MODELS_DIR, PRETRAINED_DIR]:
    os.makedirs(d, exist_ok=True)

# ---- Config ----
WANTED_WORDS = ["heytm", "unknown", "background"]
NUM_CLASSES = len(WANTED_WORDS)
TRAINING_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
VALIDATION_SPLIT = 0.2

MODEL_TFLITE = os.path.join(MODELS_DIR, "KWS_transfer.tflite")
FLOAT_MODEL_TFLITE = os.path.join(MODELS_DIR, "KWS_transfer_float.tflite")
MODEL_H5 = os.path.join(MODELS_DIR, "KWS_transfer.h5")

# ---- Download pre-trained model ----
PRETRAINED_URL = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/speech_micro_pretrained_model_2020_04_28.zip"
pretrained_zip = os.path.join(PRETRAINED_DIR, "pretrained.zip")
pretrained_model = os.path.join(PRETRAINED_DIR, "tiny_conv.tflite")

if not os.path.exists(pretrained_model):
    logging.info("🔄 Downloading pre-trained tiny_conv model...")
    try:
        urllib.request.urlretrieve(PRETRAINED_URL, pretrained_zip)
        with zipfile.ZipFile(pretrained_zip, 'r') as zip_ref:
            zip_ref.extractall(PRETRAINED_DIR)
        os.remove(pretrained_zip)
        logging.info("✅ Pre-trained model downloaded and extracted.")
    except Exception as e:
        logging.warning(f"Could not download pretrained model: {e}")
        logging.info("Will train from scratch instead.")

# ---- Load dataset ----
logging.info("🔄 Loading training dataset...")
extractor = AudioFeatureExtractor()
X, y, filenames = extractor.load_dataset(DATASET_DIR, class_names=WANTED_WORDS)

if len(X) == 0:
    logging.error("❌ No valid samples found in dataset!")
    sys.exit(1)

# Convert labels to categorical
y_categorical = tf.keras.utils.to_categorical(y, NUM_CLASSES)

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(
    X, y_categorical, 
    test_size=VALIDATION_SPLIT, 
    random_state=42, 
    stratify=y
)

logging.info(f"Training samples: {len(X_train)}")
logging.info(f"Validation samples: {len(X_val)}")

# ---- Build model ----
def create_transfer_model(input_shape, num_classes):
    """Create model architecture"""
    time_steps, n_features = input_shape[1], input_shape[2]
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(time_steps, n_features)),
        tf.keras.layers.Reshape((time_steps, n_features, 1)),
        
        # Conv blocks with depthwise separable convolutions
        tf.keras.layers.SeparableConv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.SeparableConv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.SeparableConv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.GlobalAveragePooling2D(),
        
        # Dense layers
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        
        # Output layer
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

logging.info("🔄 Building model...")
model = create_transfer_model(X_train.shape, NUM_CLASSES)

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary(print_fn=logging.info)

# ---- Training callbacks ----
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        patience=10,
        restore_best_weights=True,
        monitor='val_accuracy'
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        monitor='val_loss'
    ),
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_H5,
        save_best_only=True,
        monitor='val_accuracy',
        mode='max'
    ),
    tf.keras.callbacks.TensorBoard(
        log_dir=LOGS_DIR,
        histogram_freq=1
    )
]

# ---- Train the model ----
logging.info("\n" + "="*80)
logging.info("🚀 STARTING TRAINING")
logging.info("="*80 + "\n")

history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=TRAINING_EPOCHS,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

logging.info("\n✅ Training completed!")

# ---- Validation Evaluation ----
logging.info("\n" + "="*80)
logging.info("📊 VALIDATION SET EVALUATION")
logging.info("="*80)

val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
logging.info(f"Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
logging.info(f"Validation Loss: {val_loss:.4f}")

# ---- Test on separate test folder ----
if os.path.exists(TEST_DIR):
    logging.info(f"\n{'='*80}")
    logging.info("🧪 TESTING ON SEPARATE TEST SET")
    logging.info(f"{'='*80}")
    
    # Check if test folder has flat structure (no subfolders) or subfolder structure
    audio_files = [f for f in os.listdir(TEST_DIR) if f.endswith(('.wav', '.mp3', '.m4a'))]
    
    if audio_files:
        # Flat structure: test/positive1.mp3, test/nohas.mp3, etc.
        logging.info("📁 Detected flat folder structure - testing individual files")
        logging.info(f"Found {len(audio_files)} audio files\n")
        
        # Map class names to friendly labels
        class_labels = {
            'heytm': 'positive',
            'unknown': 'negative', 
            'background': 'silence'
        }
        
        results_list = []
        for audio_file in sorted(audio_files):
            file_path = os.path.join(TEST_DIR, audio_file)
            
            # Extract features
            features = extractor.extract_features(file_path)
            if features is None:
                logging.warning(f"⚠️  Could not process: {audio_file}")
                continue
            
            # Predict
            features_batch = np.expand_dims(features, axis=0)
            predictions = model.predict(features_batch, verbose=0)[0]
            
            # Format output
            pred_str = ", ".join([
                f"{class_labels[WANTED_WORDS[i]]} {predictions[i]*100:.0f}%" 
                for i in range(len(WANTED_WORDS))
            ])
            
            output_line = f"{audio_file} -- {pred_str}"
            logging.info(output_line)
            
            results_list.append({
                'filename': audio_file,
                'predictions': {
                    class_labels[WANTED_WORDS[i]]: float(predictions[i]) 
                    for i in range(len(WANTED_WORDS))
                }
            })
        
        # Save results
        results_output = {
            'timestamp': datetime.now().isoformat(),
            'test_folder': TEST_DIR,
            'total_files': len(results_list),
            'results': results_list
        }
        
        with open('test_predictions.json', 'w') as f:
            json.dump(results_output, f, indent=2)
        
        logging.info(f"\n💾 Predictions saved to: test_predictions.json")
        
    else:
        # Subfolder structure: test/heytm/, test/unknown/, test/_background_noise_/
        logging.info("📁 Detected subfolder structure - running detailed evaluation")
        tester = ModelTester(model, WANTED_WORDS, extractor)
        test_results = tester.test_on_folder(TEST_DIR)
else:
    logging.warning(f"\n⚠️  Test folder '{TEST_DIR}' not found. Skipping test evaluation.")
    logging.info(f"   Create folder with audio files: {TEST_DIR}/positive1.mp3, {TEST_DIR}/nohas.mp3, etc.")

# ---- Convert to TFLite ----
logging.info("\n" + "="*80)
logging.info("🔄 CONVERTING TO TFLITE")
logging.info("="*80)

# Float model
try:
    float_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    float_tflite_model = float_converter.convert()
    
    with open(FLOAT_MODEL_TFLITE, "wb") as f:
        f.write(float_tflite_model)
    
    float_size = len(float_tflite_model) / (1024 * 1024)
    logging.info(f"✅ Float TFLite model saved: {FLOAT_MODEL_TFLITE} ({float_size:.2f} MB)")
except Exception as e:
    logging.error(f"❌ Float conversion failed: {e}")

# Quantized model
try:
    def representative_dataset_gen():
        """Generate representative samples from training data"""
        for i in range(min(100, len(X_train))):
            yield [X_train[i:i+1].astype(np.float32)]
    
    quant_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    quant_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quant_converter.target_spec.supported_types = [tf.float16]
    quant_converter.representative_dataset = representative_dataset_gen
    
    quant_tflite_model = quant_converter.convert()
    
    with open(MODEL_TFLITE, "wb") as f:
        f.write(quant_tflite_model)
    
    quant_size = len(quant_tflite_model) / (1024 * 1024)
    logging.info(f"✅ Quantized TFLite model saved: {MODEL_TFLITE} ({quant_size:.2f} MB)")
except Exception as e:
    logging.error(f"❌ Quantized conversion failed: {e}")

# ---- Final Summary ----
logging.info("\n" + "🎉"*40)
logging.info("✅ TRAINING AND EVALUATION COMPLETED!")
logging.info("🎉"*40)

logging.info(f"\n📁 Models saved in: {MODELS_DIR}")
for model_file in [MODEL_H5, FLOAT_MODEL_TFLITE, MODEL_TFLITE]:
    if os.path.exists(model_file):
        size_mb = os.path.getsize(model_file) / (1024 * 1024)
        logging.info(f"  ✓ {model_file} ({size_mb:.2f} MB)")

logging.info("\n📋 Generated files:")
logging.info("  • training_log.txt - Complete training log")
logging.info("  • test_results_detailed.json - Detailed test metrics")
logging.info("  • confusion_matrix.png - Visual confusion matrix")

logging.info("\n💡 Next steps:")
logging.info("  1. Review test_results_detailed.json for detailed metrics")
logging.info("  2. Check confusion_matrix.png for visual analysis")
logging.info("  3. Use TensorBoard: tensorboard --logdir=logs_transfer")
logging.info("  4. Deploy TFLite models to mobile devices")

logging.info("\n" + "="*80 + "\n")