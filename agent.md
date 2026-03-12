# Agent Instructions for This Repository

## Mission
Maintain and improve a compact KWS pipeline that outputs a Flutter-ready TFLite model.

## Custom Prompt
```text
You are a senior developer in AI and Machine Learning. Your task is to develop a keyword spotting (KWS) model that detects the keyword "HEY TM". You may test multiple model methods and training strategies to get the best external test accuracy. Keep deployment constraints as first-class requirements: the model will run on continuous microphone input using PCM16 audio (typically 16 kHz, or lower sample rate only if accuracy is preserved), must export to TFLite, should stay under 10 MB, and should be optimized for low battery consumption without sacrificing practical accuracy.
```

## Non-Negotiable Rules
- Train from `datasets/` (fallback to `dataset/` only if `datasets/` is missing).
- Do not create an internal test split from training data.
- Use `test/` as the only external evaluation folder.
- Keep deployment model size **<= 10 MB**.
- Output format must be `.tflite`.

## Data Contract
Expected classes and order:
1. `heytm`
2. `unknown`
3. `background` (folder name: `_background_noise_`)

Do not change class order unless all downstream consumers are updated.

## Model Contract
- Input pipeline: 1-second audio, 16 kHz, MFCC-based features.
- Produce both:
- Float model for comparison.
- Quantized model for deployment.
- Prefer quantization and lightweight architecture decisions that reduce size while preserving stability.
- Keep continuous-listening power usage low: efficient features, sparse inference cadence, and simple post-processing.

## Standard Run
```bash
python main_preset_model.py
```

## Expected Outputs
- `models_scratch/KWS_scratch.tflite`
- `models_scratch/KWS_scratch_float.tflite`
- `models_scratch/KWS_scratch.h5`
- `test_predictions.json` (when `test/` has flat files)

## When Editing Training Code
- Preserve no-split training behavior.
- Keep external `test/` evaluation enabled.
- Log model file sizes after export.
- Prefer deterministic, minimal changes over broad refactors.
