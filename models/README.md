# Model Weights & Architecture: FDCS-Net V4

This directory houses documentation and instructions for obtaining the trained weights file for the **FDCS-Net V4 (Fused Domain Color-Stability Network)** architecture.

## Model File Details

- **File Name:** `fdcsnet_v4_final.keras`
- **File Size:** ~18.9 MB (18,920,935 bytes)
- **Format:** Native Keras 3 SavedModel (`.keras`)
- **Direct Weights URL:** [Download fdcsnet_v4_final.keras](https://raw.githubusercontent.com/ga2495/Real_image_VS_Fake_Image/main/models/fdcsnet_v4_final.keras)

## Benchmark Performance (Held-Out Test Set)

- **Test Accuracy:** 93.78% (2,080 / 2,218 correct)
- **AUC-ROC:** 0.9748
- **Optimal Decision Threshold (Validation-Selected):** 0.80
- **Overall Precision:** 0.94
- **Overall Recall:** 0.94
- **Overall F1-Score:** 0.94

## Loading the Model

```python
from utils import resolve_and_load_model

# Loads 'models/fdcsnet_v4_final.keras' with all custom objects registered
model = resolve_and_load_model()
```

## Automatic Resolution & Remote Fallback

If `models/fdcsnet_v4_final.keras` is not present locally (e.g. on clean cloud deployment environments like Render), the application will automatically download it on startup if you set the `FDCSNET_MODEL_URL` environment variable:

```bash
export FDCSNET_MODEL_URL="https://raw.githubusercontent.com/ga2495/Real_image_VS_Fake_Image/main/models/fdcsnet_v4_final.keras"
```
