# Agent Instructions for This Repository

## Mission
Maintain and improve a compact KWS pipeline that outputs a Flutter-ready TFLite model.

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
