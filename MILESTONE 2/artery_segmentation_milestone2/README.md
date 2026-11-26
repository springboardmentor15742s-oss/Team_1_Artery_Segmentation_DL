# Artery Segmentation — Milestone 2 (CPU-friendly 2D starter)

This starter project implements:
- Baseline U-Net and an Attention U-Net variant
- Training with Dice + BCE loss
- Validation and checkpoint saving
- Test-set mask generation
- Dice, IoU, precision, sensitivity, specificity
- Basic 2D post-processing, skeleton/centerline extraction and pixel-width estimates
- CSV metrics and visual overlays

## Important scope limitation
This is a 2D research starter. Retinal vessel datasets can be used to validate the software pipeline, but they do NOT validate coronary CTA segmentation or clinical stenosis measurement. For 3D CTA, use a de-identified dataset with paired 3D artery annotations, preserve image spacing/orientation, and adapt the pipeline to 3D before claiming physical diameters in mm.

## 1. Windows PowerShell setup
Use Python 3.10 or 3.11 (64-bit). From this folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install CPU PyTorch from the official CPU wheel index if normal installation fails:
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 2. Dataset layout
Put paired images and masks in:
```text
data/
  images/
    case_001.png
    case_002.png
  masks/
    case_001.png
    case_002.png
```
The image and mask must share the same filename stem. PNG, JPG, and TIFF are supported by Pillow. Masks must have background black and vessels/artery foreground white (or any nonzero value). Use de-identified/public data only.

Example: for a retinal pipeline demo, use a public dataset such as DRIVE, following the dataset provider's access and citation requirements. DRIVE is retinal, not coronary. For your final coronary project, choose a coronary CTA dataset with ground-truth artery masks.

## 3. Run a smoke test without a real dataset
This creates clearly synthetic tube-like images and masks so you can test code wiring only. It is NOT real medical data and must never be reported as model performance.
```powershell
python make_demo_data.py
python train.py --model unet --epochs 2 --size 128
python predict.py --model unet --checkpoint checkpoints/unet_best.pth
python evaluate.py --model unet --checkpoint checkpoints/unet_best.pth
python vessel_analysis.py --mask outputs/predictions/case_001_mask.png
```
Synthetic images are generated under `data/`; back up/remove them before adding real data so the dataset isn't mixed.

## 4. Train on your dataset
Start small on CPU:
```powershell
python train.py --model unet --epochs 10 --size 256
python train.py --model attention_unet --epochs 10 --size 256
```
Increase epochs only after the first run works. Use `--size 128` if memory/time is a problem. These settings are examples, not guaranteed optimal settings.

## 5. Generate masks and evaluate
```powershell
python predict.py --model unet --checkpoint checkpoints/unet_best.pth
python evaluate.py --model unet --checkpoint checkpoints/unet_best.pth
python predict.py --model attention_unet --checkpoint checkpoints/attention_unet_best.pth
python evaluate.py --model attention_unet --checkpoint checkpoints/attention_unet_best.pth
python compare_models.py
```

Outputs:
- `checkpoints/`: saved model weights
- `outputs/predictions/`: predicted binary masks
- `outputs/overlays/`: image + prediction overlays
- `outputs/metrics/`: per-image CSVs and summary CSV
- `outputs/analysis/`: centerlines and diameter estimates in pixels

## 6. Data split and scientific reporting
The script makes a deterministic image-level split (70% train, 15% validation, 15% test). For medical volumes with multiple slices per patient, this is NOT sufficient: split by patient before extracting slices to prevent leakage. Report the dataset, split, model, threshold, and per-case metrics. Do not tune repeatedly against the test set.

## 7. Limitations
- 2D only; no DICOM series/3D volume handling.
- The simple diameter estimate is in pixels and is only a rough width proxy.
- Skeletonization can produce noisy branches; inspect overlays.
- No stenosis detection or clinical severity classification is implemented here.
- Research prototype only, not for clinical decisions.
