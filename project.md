# Project: Tiny Keyword Spotting (KWS) for Flutter TFLite

## Objective
Build a Keyword Spotting model that is deployable with a Flutter TFLite package.

## Custom Prompt
Use this prompt when guiding model development:

```text
You are a senior developer in AI and Machine Learning. Your task is to develop a keyword spotting (KWS) model that detects the keyword "HEY TM". You may test multiple model methods and training strategies to get the best external test accuracy. Keep deployment constraints as first-class requirements: the model will run on continuous microphone input using PCM16 audio (typically 16 kHz, or lower sample rate only if accuracy is preserved), must export to TFLite, should stay under 10 MB, and should be optimized for low battery consumption without sacrificing practical accuracy.
```

## Hard Requirements
- Model format must be `.tflite`.
- Model size target: **<= 10 MB** (smaller is better).
- Training data source: `datasets/`.
- Do **not** split training data into testing inside the script.
- Use `test/` as a separate external audio test folder.

## Dataset Layout
Training folder (`datasets/`):
- `datasets/heytm/`
- `datasets/unknown/`
- `datasets/_background_noise_/`

External test folder (`test/`):
- Flat folder with audio samples (already provided).

## Training/Evaluation Workflow
1. Load all audio from `datasets/` for training.
2. Train on full training data (no internal test split).
3. Run inference/evaluation on `test/` after training.
4. Export:
- Float TFLite (reference)
- Quantized TFLite (deployment target)

## Primary Deliverables
- `models_scratch/KWS_scratch.tflite` (deployment model)
- `models_scratch/KWS_scratch_float.tflite` (reference model)
- `test_predictions.json` (external test predictions)

## Acceptance Criteria
- A `.tflite` model is generated successfully.
- Quantized model is at or under 10 MB.
- Model can be consumed by Flutter TFLite runtime.
- Training uses only `datasets/` and testing uses `test/`.
- Development choices consider continuous-mic power efficiency (feature cost, inference cadence, model size).
