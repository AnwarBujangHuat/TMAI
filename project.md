# Project: Tiny Keyword Spotting (KWS) for Flutter TFLite

## Objective
Build a Keyword Spotting model that is deployable with a Flutter TFLite package.

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
