# A/B Protocol: Scratch vs Transfer for HEY TM

## Goal
Select the best deployment model for continuous-listening HEY TM detection with these constraints:
- `.tflite` output
- model size target `<= 10 MB`
- stable wake-word recall with low false triggers
- practical CPU/battery behavior for always-on mic

## Shared Conditions
Keep these fixed for both approaches:
- Training data source: `datasets/` (fallback: `dataset/`)
- No internal test split
- External test folder: `test/`
- Class order: `heytm`, `unknown`, `background`

## Variant A
- Script: `main_preset_model.py`
- Method: train from scratch (compact separable CNN)
- Artifacts: `models_scratch/*`

## Variant B
- Script: `transfer_model.py`
- Method: MobileNetV2 transfer learning on hybrid log-mel+MFCC features
- Artifacts: `models_transfer/*`

## Execution
```bash
python main_preset_model.py
python transfer_model.py
python ab_experiment.py
```

Single-command option:
```bash
python ab_experiment.py --run-training
```

## Primary Decision Rule
1. Higher keyword recall on positive samples wins.
2. If recall ties, lower false trigger rate wins.
3. If both tie, prefer smaller quantized `.tflite` for deployment.

## Report Files
- `ab_results.json`
- `ab_summary.md`

## Note on Flat Test Folders
When `test/` is flat (no class subfolders), comparison uses filename heuristic labels:
- `positive*` => `heytm`
- names containing `background|silence|noise` => `background`
- otherwise => `unknown`

For production sign-off, replace this with a manually labeled evaluation set.
