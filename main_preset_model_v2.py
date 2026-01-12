# -*- coding: utf-8 -*-
"""
Transfer Learning Keyword Spotting (KWS) for "hey tm" using TensorFlow Hub YAMNet
Pure TensorFlow implementation - no PyTorch dependency issues!
Combines pre-trained YAMNet audio embeddings with custom MFCC features
"""

import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from sklearn.metrics import classification_report, confusion_matrix
import logging
import json
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_log_yamnet.txt'),
        logging.StreamHandler()
    ]
)

# ---- YAMNet Feature Extractor ----
class YAMNetFeatureExtractor:
    """Extract features using pre-trained YAMNet model from TensorFlow Hub"""
    def __init__(self):
        logging.info("🔄 Loading pre-trained YAMNet model from TensorFlow Hub...")
        try:
            # Load YAMNet from TensorFlow Hub
            self.model = hub.load('https://tfhub.dev/google/yamnet/1')
            logging.info("✅ YAMNet model loaded successfully!")
            logging.info("   YAMNet was trained on AudioSet (2M+ audio clips)")
        except Exception as e:
            logging.error(f"❌ Failed to load YAMNet: {e}")
            logging.error("   Make sure you have tensorflow-hub installed: pip install tensorflow-hub")
            raise
    
    def extract_yamnet_features(self, audio_array, sample_rate=16000):
        """Extract embeddings from audio using YAMNet"""
        try:
            # YAMNet expects float32 waveform
            waveform = tf.cast(audio_array, tf.float32)
            
            # YAMNet returns (scores, embeddings, log_mel_spectrogram)
            _, embeddings, _ = self.model(waveform)
            
            # Average embeddings across time for a fixed-size representation
            embedding = tf.reduce_mean(embeddings, axis=0).numpy()
            
            return embedding  # Shape: (1024,)
        except Exception as e:
            logging.warning(f"Error in YAMNet extraction: {e}")
            return None

# ---- Combined Audio Feature Extractor ----
class HybridAudioFeatureExtractor:
    """Extract both MFCC and YAMNet features for superior performance"""
    def __init__(self, sample_rate=16000, duration=1.0, n_mels=40, n_mfcc=13, use_yamnet=True):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = int(sample_rate * duration)
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.use_yamnet = use_yamnet
        
        # Initialize YAMNet if enabled
        if self.use_yamnet:
            self.yamnet_extractor = YAMNetFeatureExtractor()
        else:
            self.yamnet_extractor = None

    def extract_mfcc_features(self, audio):
        """Extract MFCC features from audio"""
        try:
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
            logging.warning(f"Error in MFCC extraction: {e}")
            return None

    def extract_features(self, audio_path):
        """Extract hybrid features (MFCC + YAMNet) from audio file"""
        try:
            # Load audio file
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Pad or trim to fixed length
            if len(audio) > self.n_samples:
                audio = audio[:self.n_samples]
            else:
                audio = np.pad(audio, (0, self.n_samples - len(audio)), mode='constant')
            
            # Extract MFCC
            mfcc_features = self.extract_mfcc_features(audio)
            
            if self.use_yamnet and self.yamnet_extractor is not None:
                # Extract YAMNet features
                yamnet_features = self.yamnet_extractor.extract_yamnet_features(audio, self.sample_rate)
                
                if yamnet_features is not None and mfcc_features is not None:
                    # Combine features: append YAMNet as additional context
                    # MFCC shape: (time_steps, n_mfcc)
                    # YAMNet shape: (1024,) - we'll tile it to match time dimension
                    time_steps = mfcc_features.shape[0]
                    yamnet_tiled = np.tile(yamnet_features, (time_steps, 1))
                    
                    # Concatenate along feature dimension
                    combined_features = np.concatenate([mfcc_features, yamnet_tiled], axis=1)
                    return combined_features
            
            return mfcc_features
            
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
                    feature = self.extract_features(file_path)
                    
                    if feature is not None:
                        features.append(feature)
                        labels.append(class_idx)
                        filenames.append(filename)
                        class_samples += 1
            
            logging.info(f"Loaded {class_samples} {class_name} samples")
        
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(labels)
        
        logging.info(f"Dataset shape: {X.shape}")
        logging.info(f"Feature dimensions: MFCC + YAMNet (1024D)" if self.use_yamnet else "MFCC only")
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
        """Test model on a folder with subfolder structure"""
        logging.info(f"\n{'='*80}")
        logging.info(f"🧪 TESTING MODEL ON: {test_folder_path}")
        logging.info(f"{'='*80}")
        
        X_test, y_test, filenames = self.extractor.load_dataset(test_folder_path, self.class_names)
        
        if len(X_test) == 0:
            logging.error(f"❌ No test samples found in {test_folder_path}")
            return None
        
        y_test_categorical = tf.keras.utils.to_categorical(y_test, len(self.class_names))
        
        logging.info("\n📊 Evaluating model on test set...")
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test_categorical, verbose=0)
        
        logging.info(f"✅ Test Loss: {test_loss:.4f}")
        logging.info(f"✅ Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        y_pred_probs = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        logging.info(f"\n📈 PER-CLASS PERFORMANCE:")
        logging.info(f"{'='*80}")
        
        for class_idx, class_name in enumerate(self.class_names):
            class_mask = y_test == class_idx
            if np.sum(class_mask) > 0:
                class_acc = np.mean(y_pred[class_mask] == y_test[class_mask])
                class_total = np.sum(class_mask)
                class_correct = np.sum(y_pred[class_mask] == y_test[class_mask])
                logging.info(f"{class_name.upper():12} | Accuracy: {class_acc:.4f} ({class_acc*100:.2f}%) | Correct: {class_correct}/{class_total}")
        
        logging.info(f"\n📋 DETAILED CLASSIFICATION REPORT:")
        logging.info(f"{'='*80}")
        report = classification_report(y_test, y_pred, target_names=self.class_names, digits=4)
        logging.info(f"\n{report}")
        
        cm = confusion_matrix(y_test, y_pred)
        logging.info(f"\n🎯 CONFUSION MATRIX:")
        logging.info(f"{'='*80}")
        
        header = "           " + "  ".join([f"{name[:8]:>8}" for name in self.class_names])
        logging.info(header)
        for i, class_name in enumerate(self.class_names):
            row = f"{class_name[:10]:10} " + "  ".join([f"{cm[i][j]:8d}" for j in range(len(self.class_names))])
            logging.info(row)
        
        self.plot_confusion_matrix(cm, test_folder_path)
        
        return {'test_accuracy': float(test_accuracy), 'test_loss': float(test_loss)}
    
    def plot_confusion_matrix(self, cm, test_folder_path):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - YAMNet Transfer Learning\n{test_folder_path}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix_yamnet.png', dpi=300, bbox_inches='tight')
        logging.info(f"📊 Confusion matrix saved to: confusion_matrix_yamnet.png")
        plt.close()

# ---- Setup paths ----
DATASET_DIR = "dataset"
TEST_DIR = "test"
MODELS_DIR = "models_yamnet"

os.makedirs(MODELS_DIR, exist_ok=True)

# ---- Config ----
WANTED_WORDS = ["heytm", "unknown", "background"]
NUM_CLASSES = len(WANTED_WORDS)
TRAINING_EPOCHS = 35
BATCH_SIZE = 32
LEARNING_RATE = 0.0003

USE_YAMNET = True  # Set False to use MFCC only

MODEL_TAG = "yamnet_transfer" if USE_YAMNET else "mfcc_only"
MODEL_H5 = os.path.join(MODELS_DIR, f"KWS_{MODEL_TAG}.keras")
MODEL_TFLITE = os.path.join(MODELS_DIR, f"KWS_{MODEL_TAG}.tflite")

# ---- Load dataset ----
logging.info("\n" + "="*80)
logging.info("🔄 LOADING DATASET WITH YAMNET TRANSFER LEARNING")
logging.info("="*80)

extractor = HybridAudioFeatureExtractor(use_yamnet=USE_YAMNET)
X, y, filenames = extractor.load_dataset(DATASET_DIR, class_names=WANTED_WORDS)

if len(X) == 0:
    logging.error("❌ No valid samples found!")
    sys.exit(1)

y_categorical = tf.keras.utils.to_categorical(y, NUM_CLASSES)
X_train, y_train = X, y_categorical

logging.info(f"\nTraining samples: {len(X_train)}")
logging.info(f"Input shape: {X_train.shape[1:]}")

# ---- Build model ----
def create_yamnet_transfer_model(input_shape, num_classes):
    """Model optimized for YAMNet embeddings"""
    time_steps, n_features = input_shape[1], input_shape[2]
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(time_steps, n_features)),
        tf.keras.layers.Reshape((time_steps, n_features, 1)),
        
        # Rich features need deeper network
        tf.keras.layers.SeparableConv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.SeparableConv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.SeparableConv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.GlobalAveragePooling2D(),
        
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

logging.info("\n🔄 Building YAMNet transfer learning model...")
model = create_yamnet_transfer_model(X_train.shape, NUM_CLASSES)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary(print_fn=logging.info)

# ---- Training ----
logging.info("\n" + "="*80)
logging.info("🚀 STARTING YAMNET TRANSFER LEARNING")
logging.info("="*80)

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(MODEL_H5, save_best_only=True, monitor='loss', verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True, verbose=1)
]

history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=TRAINING_EPOCHS, 
                   callbacks=callbacks, verbose=1)

logging.info("\n✅ Training completed!")

# ---- Evaluation ----
train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
logging.info(f"\n📊 Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")

# ---- Testing ----
if os.path.exists(TEST_DIR):
    audio_files = [f for f in os.listdir(TEST_DIR) if f.endswith(('.wav', '.mp3', '.m4a'))]
    
    if audio_files:
        logging.info("\n" + "="*80)
        logging.info("🧪 TESTING ON AUDIO FILES")
        logging.info("="*80 + "\n")
        
        print(f"\n{'Filename':<50} {'Prediction':<15} {'Positive':<12} {'Negative':<12} {'Silence':<12}")
        print("-" * 110)
        
        for audio_file in sorted(audio_files):
            file_path = os.path.join(TEST_DIR, audio_file)
            features = extractor.extract_features(file_path)
            
            if features is None:
                print(f"{audio_file:<50} {'ERROR':<15}")
                continue
            
            predictions = model.predict(np.expand_dims(features, 0), verbose=0)[0]
            
            positive_prob = predictions[0] * 100
            negative_prob = predictions[1] * 100
            silence_prob = predictions[2] * 100
            
            prediction_label = WANTED_WORDS[np.argmax(predictions)]
            
            print(f"{audio_file:<50} {prediction_label:<15} {positive_prob:>10.2f}%  {negative_prob:>10.2f}%  {silence_prob:>10.2f}%")
        
        print("-" * 110)
    else:
        tester = ModelTester(model, WANTED_WORDS, extractor)
        tester.test_on_folder(TEST_DIR)

# ---- TFLite Conversion ----
logging.info("\n🔄 Converting to TFLite...")

try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open(MODEL_TFLITE, "wb") as f:
        f.write(tflite_model)
    
    size_mb = len(tflite_model) / (1024 * 1024)
    logging.info(f"✅ TFLite model: {MODEL_TFLITE} ({size_mb:.2f} MB)")
except Exception as e:
    logging.error(f"❌ TFLite conversion failed: {e}")

# ---- Summary ----
logging.info("\n" + "🎉"*40)
logging.info("✅ YAMNET TRANSFER LEARNING COMPLETED!")
logging.info("🎉"*40)
logging.info(f"\n💡 Using: YAMNet (trained on AudioSet - 2M+ clips)")
logging.info("   • Pure TensorFlow - no PyTorch issues!")
logging.info("   • 1024D audio embeddings + 13D MFCC")
logging.info("   • Production-ready for deployment")
logging.info("\n" + "="*80 + "\n")