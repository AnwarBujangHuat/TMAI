import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import librosa
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class HeyTMKeywordSpotter:
    def __init__(self, sample_rate=16000, duration=1.0, n_mels=40, n_mfcc=13):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = int(sample_rate * duration)
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.model = None
        
        # Class mapping
        self.class_names = ['heytm', 'unknown', 'background']
        self.num_classes = len(self.class_names)
        
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
            print(f"Error processing {audio_path}: {e}")
            return None
    
    def load_dataset(self, dataset_path):
        """Load and preprocess the dataset"""
        features = []
        labels = []
        
        for class_idx, class_name in enumerate(self.class_names):
            if class_name == 'background':
                folder_name = '_background_noise_'
            else:
                folder_name = class_name
                
            class_path = os.path.join(dataset_path, folder_name)
            
            if not os.path.exists(class_path):
                print(f"Warning: {class_path} not found!")
                continue
                
            print(f"Loading {class_name} samples...")
            class_samples = 0
            
            for filename in os.listdir(class_path):
                if filename.endswith(('.wav', '.mp3', '.m4a')):
                    file_path = os.path.join(class_path, filename)
                    mfcc = self.extract_features(file_path)
                    
                    if mfcc is not None:
                        features.append(mfcc)
                        labels.append(class_idx)
                        class_samples += 1
                        
            print(f"Loaded {class_samples} {class_name} samples")
        
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(labels)
        
        print(f"Dataset shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        
        return X, y
    
    def create_model(self, input_shape=None):
        """Create a lightweight CNN model optimized for mobile deployment"""
        
        # Use actual input shape if provided, otherwise calculate
        if input_shape is not None:
            time_steps, n_features = input_shape[1], input_shape[2]
        else:
            time_steps = int(self.n_samples / 160)  # Based on hop_length=160
            n_features = self.n_mfcc
        
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(time_steps, n_features)),
            
            # Reshape for CNN
            layers.Reshape((time_steps, n_features, 1)),
            
            # First Conv2D block - using depthwise separable convolution for efficiency
            layers.SeparableConv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Conv2D block
            layers.SeparableConv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Conv2D block
            layers.SeparableConv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            
            # Global average pooling instead of flatten for efficiency
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def train(self, dataset_path, epochs=50, batch_size=32, validation_split=0.2):
        """Train the keyword spotting model"""
        
        # Load dataset
        X, y = self.load_dataset(dataset_path)
        
        if len(X) == 0:
            raise ValueError("No valid samples found in dataset!")
        
        # Convert labels to categorical
        y_categorical = keras.utils.to_categorical(y, self.num_classes)
        
        # Split dataset
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_categorical, test_size=validation_split, 
            random_state=42, stratify=y
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        # Store input shape for later use
        self._input_shape = X_train.shape
        
        # Create model with actual input shape
        self.model = self.create_model(input_shape=X_train.shape)
        
        # Compile with appropriate optimizer and loss for mobile deployment
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Print model summary
        self.model.summary()
        
        # Callbacks for better training
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=10, restore_best_weights=True, monitor='val_accuracy'
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.5, patience=5, min_lr=1e-6, monitor='val_loss'
            ),
            keras.callbacks.ModelCheckpoint(
                'best_heytm_model.h5', save_best_only=True, 
                monitor='val_accuracy', mode='max'
            )
        ]
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def convert_to_tflite(self, model_path='heytm_model.h5', 
                         tflite_path='heytm_model.tflite'):
        """Convert trained model to TensorFlow Lite for mobile deployment"""
        
        if self.model is None:
            self.model = keras.models.load_model(model_path)
        
        # Convert to TensorFlow Lite with optimization
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Optimize for mobile deployment
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]  # Use float16 for smaller size
        
        # Additional mobile-specific optimizations
        converter.representative_dataset = self._representative_dataset
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS  # Allow some TF ops if needed
        ]
        
        tflite_model = converter.convert()
        
        # Save the model
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"TensorFlow Lite model saved to {tflite_path}")
        
        # Print model info
        model_size = len(tflite_model) / 1024 / 1024
        print(f"Model size: {model_size:.2f} MB")
        
        return tflite_model
    
    def _representative_dataset(self):
        """Representative dataset for quantization"""
        # This should be implemented with actual data samples
        # For now, return dummy data matching input shape
        if hasattr(self, '_input_shape'):
            time_steps, n_features = self._input_shape[1], self._input_shape[2]
        else:
            time_steps = int(self.n_samples / 160)
            n_features = self.n_mfcc
        
        for _ in range(100):
            yield [np.random.random((1, time_steps, n_features)).astype(np.float32)]
    
    def predict(self, audio_path):
        """Predict keyword for a single audio file"""
        if self.model is None:
            raise ValueError("Model not trained or loaded!")
        
        features = self.extract_features(audio_path)
        if features is None:
            return None, 0.0
        
        features = np.expand_dims(features, axis=0)
        predictions = self.model.predict(features)
        
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        return self.class_names[predicted_class], confidence
    
    def evaluate_model(self, test_data_path=None):
        """Evaluate model performance"""
        if test_data_path:
            X_test, y_test = self.load_dataset(test_data_path)
            y_test_categorical = keras.utils.to_categorical(y_test, self.num_classes)
            
            loss, accuracy = self.model.evaluate(X_test, y_test_categorical, verbose=0)
            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"Test Loss: {loss:.4f}")
            
            return accuracy, loss
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        plt.show()

# Usage example
if __name__ == "__main__":
    # Initialize the keyword spotter
    kws = HeyTMKeywordSpotter()
    
    # Train the model
    print("Starting training...")
    history = kws.train('dataset', epochs=50, batch_size=32)
    
    # Plot training history
    kws.plot_training_history(history)
    
    # Save the model
    kws.model.save('heytm_model.h5')
    
    # Convert to TensorFlow Lite
    kws.convert_to_tflite()
    
    # Test prediction on a sample file
    # prediction, confidence = kws.predict('path/to/test/audio.wav')
    # print(f"Prediction: {prediction}, Confidence: {confidence:.4f}")
    
    print("Training completed! Model saved as 'heytm_model.h5' and 'heytm_model.tflite'")