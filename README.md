# Keyword Spotting with Deep Learning

This project implements a Keyword Spotting (KWS) system using a Deep Separable Convolutional Neural Network (DS-CNN) architecture. The model is designed to recognize a specific keyword ("hey_tm") from audio input while distinguishing it from background noise.

## Project Structure

```
kws_force
├── data
│   ├── hey_tm          # Contains positive sample audio files (e.g., "hey_tm" keyword)
│   └── background      # Contains negative or background audio files
├── saved_model         # Directory to save the trained model
├── main.py             # The training script for the KWS model
└── README.md           # Documentation for the project
```

## Requirements

To run this project, you need to install the following Python packages:

- tensorflow-macos
- tensorflow-metal
- librosa
- numpy
- scikit-learn

You can install these dependencies using pip:

```
pip install tensorflow-macos tensorflow-metal librosa numpy scikit-learn
```

## Dataset Preparation

1. Place your audio files in the appropriate directories:
   - Positive samples (keyword "hey_tm") should be placed in `data/hey_tm/`.
   - Negative samples (background noise) should be placed in `data/background/`.

Each audio file should be in WAV format, mono or stereo, and will be trimmed or padded to 1 second during preprocessing.

## Running the Training Script

To train the model, execute the following command:

```
python main.py
```

The trained model will be saved in the `saved_model/` directory, and a TensorFlow Lite model will be generated as `kws_model.tflite`.

## Model Architecture

The model uses a Deep Separable Convolutional Neural Network (DS-CNN) architecture, which is efficient for audio classification tasks. The training script includes data loading, preprocessing, model definition, and evaluation.

## License

This project is licensed under the MIT License.