"""Experimental Stenosis Detection and Severity Estimation (Phase 16.1 Refined).

Implements:
1. Candidate narrowing region identification along vessel centerlines
2. Percentage Diameter Stenosis calculation:
   Stenosis (%) = [(Reference Diameter - MLD) / Reference Diameter] * 100
3. Strict Quality & Safety Validation Guards:
   - Reject empty masks (NO_VESSEL_DETECTED) -> stenosis = None, display = 'N/A — Not measurable'
   - Reject short/fragmented centerlines (< 10 px threshold: REJECTED_TOO_SHORT) -> stenosis = None
   - Reject invalid reference diameters (<= 0) -> stenosis = None
   - Handle missing physical spacing metadata without silent hardcoded fallback
   - Safe nullable representation (None) for unavailable numeric measurements (no silent zeroing)
4. Longitudinal diameter profile plot with quality status badge
5. Preserves research prototype disclaimer (NOT approved for clinical diagnosis)
"""
import sys
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from app.ml.config import config
except ImportError:
    try:
        from .config import config
    except ImportError:
        from config import config


def classify_severity(stenosis_pct: Optional[float]) -> str:
    """Classify stenosis percentage according to standard cardiovascular research criteria."""
    if stenosis_pct is None:
        return "N/A — Not measurable"
    for category, (low, high) in config.analysis.severity_bins.items():
        if low <= stenosis_pct <= high:
            return category
    if stenosis_pct >= 99.0:
        return "Total Occlusion"
    return "Normal / Insignificant"


def run_stenosis_detection(
    mask_arr: np.ndarray,
    case_name: str = "sample",
    pixel_spacing_mm: Optional[float] = None,
    min_component_size: int = 0,
    min_centerline_len: int = 10
) -> Tuple[Dict[str, Any], Optional[Path]]:
    """Detect candidate stenosis, compute diameter profile, and estimate severity with safety guards.

    Parameters:
        mask_arr: 2D binary segmentation mask.
        case_name: Case identifier for saving plots.
        pixel_spacing_mm: Patient physical pixel spacing in mm. If None or <=0, mm values are marked UNAVAILABLE.
        min_component_size: Component noise filtering size (default 0 preserves raw mask).
        min_centerline_len: Minimum skeleton pixels required for valid stenosis quantification (default 10).
    """
    bin_mask = (mask_arr > 0).astype(bool)

    # 1. Configurable cleaning
    if min_component_size > 0:
        labels, _ = ndi.label(bin_mask)
        counts = np.bincount(labels.ravel())
        keep = counts >= min_component_size
        keep[0] = False
        cleaned = keep[labels]
    else:
        cleaned = bin_mask.copy()

    skeleton = skeletonize(cleaned)
    num_skel = int(skeleton.sum())

    # Check Physical Spacing
    has_physical_spacing = (pixel_spacing_mm is not None and float(pixel_spacing_mm) > 0.0)
    spacing_val = float(pixel_spacing_mm) if has_physical_spacing else None

    # SAFETY GUARD 1: Empty mask
    if num_skel == 0:
        results = {
            "case_name": case_name,
            "status": "No vessel centerline detected",
            "is_valid": False,
            "stenosis_percentage": None,
            "stenosis_display": "N/A — Not measurable",
            "severity_category": "N/A — Not measurable",
            "reference_diameter_mm": None,
            "mld_mm": None,
            "minimum_lumen_diameter_mm": None,
            "reference_diameter_px": None,
            "mld_px": None,
            "centerline_points": 0,
            "num_candidate_lesions": 0,
            "candidate_narrowing_points": 0,
            "physical_spacing_available": has_physical_spacing,
            "pixel_spacing_mm": spacing_val,
            "rejection_reason": "Empty vessel segmentation mask (0 foreground pixels).",
            "validation_status": "Invalid (No vessel detected)"
        }
        return results, None

    distance_map = ndi.distance_transform_edt(cleaned)
    coords = np.where(skeleton)
    local_diameters_px = distance_map[coords] * 2.0

    ref_diam_px = float(np.percentile(local_diameters_px, config.analysis.reference_percentile))
    mld_px = float(np.min(local_diameters_px))

    # SAFETY GUARD 2: Short / Fragmented Centerline (< min_centerline_len)
    # The 10-pixel threshold is an engineering safeguard, not a clinical standard.
    # Passing it does not prove anatomical accuracy; failing it guarantees high discretization error.
    if num_skel < min_centerline_len:
        results = {
            "case_name": case_name,
            "status": f"REJECTED_TOO_SHORT (Length: {num_skel} px < {min_centerline_len} px threshold)",
            "is_valid": False,
            "stenosis_percentage": None,
            "stenosis_display": "N/A — Not measurable",
            "severity_category": "N/A — Not measurable",
            "reference_diameter_mm": round(ref_diam_px * spacing_val, 2) if has_physical_spacing else None,
            "mld_mm": round(mld_px * spacing_val, 2) if has_physical_spacing else None,
            "minimum_lumen_diameter_mm": round(mld_px * spacing_val, 2) if has_physical_spacing else round(mld_px, 2),
            "reference_diameter_px": round(ref_diam_px, 2),
            "mld_px": round(mld_px, 2),
            "centerline_points": num_skel,
            "num_candidate_lesions": 0,
            "candidate_narrowing_points": 0,
            "physical_spacing_available": has_physical_spacing,
            "pixel_spacing_mm": spacing_val,
            "rejection_reason": f"Centerline length ({num_skel} px) is below engineering reliability threshold ({min_centerline_len} px). Measurement suppressed to prevent false alarm.",
            "validation_status": "Rejected (Unreliable short fragment)"
        }

        # Generate warning plot
        fig, ax = plt.subplots(figsize=(8, 4), dpi=130)
        ax.text(0.5, 0.5,
                f"STENOSIS MEASUREMENT REJECTED\n\n"
                f"Centerline Length: {num_skel} pixels (Minimum Required: {min_centerline_len} px)\n"
                f"Status: N/A — Not measurable\n"
                f"Reason: Fragment too short for reliable diameter percentile estimation.\n"
                f"Engineering Safeguard: Suppressed to prevent false-positive stenosis alarm.",
                ha="center", va="center", fontsize=11, color="darkred",
                bbox=dict(boxstyle="round,pad=1.0", facecolor="#FFEEEE", edgecolor="red"))
        ax.axis("off")
        ax.set_title(f"Stenosis Assessment: {case_name} [N/A — Not measurable]", fontsize=12, fontweight="bold", color="darkred")
        plt.tight_layout()

        out_dir = config.paths.analysis_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_path = out_dir / f"{case_name}_stenosis_profile.png"
        fig.savefig(plot_path, dpi=130, bbox_inches="tight")
        plt.close(fig)

        return results, plot_path

    # SAFETY GUARD 3: Invalid reference diameter
    if ref_diam_px <= 0.0:
        results = {
            "case_name": case_name,
            "status": "INVALID_REFERENCE_DIAMETER",
            "is_valid": False,
            "stenosis_percentage": None,
            "stenosis_display": "N/A — Not measurable",
            "severity_category": "N/A — Not measurable",
            "reference_diameter_mm": None,
            "mld_mm": None,
            "minimum_lumen_diameter_mm": None,
            "reference_diameter_px": round(ref_diam_px, 2),
            "mld_px": round(mld_px, 2),
            "centerline_points": num_skel,
            "num_candidate_lesions": 0,
            "candidate_narrowing_points": 0,
            "physical_spacing_available": has_physical_spacing,
            "pixel_spacing_mm": spacing_val,
            "rejection_reason": "Reference diameter evaluated to zero or negative.",
            "validation_status": "Invalid (Zero reference diameter)"
        }
        return results, None

    # 4. Valid Stenosis Calculation
    # Formula: Stenosis (%) = (1 - MLD / Ref) * 100
    stenosis_percentage = float(max(0.0, min(100.0, (1.0 - (mld_px / ref_diam_px)) * 100.0)))
    severity = classify_severity(stenosis_percentage)

    # Candidate lesions (>30% reduction from reference)
    narrow_idx = np.where(local_diameters_px < (0.70 * ref_diam_px))[0]
    num_lesions = len(narrow_idx)

    ref_diam_mm = round(ref_diam_px * spacing_val, 2) if has_physical_spacing else None
    mld_mm = round(mld_px * spacing_val, 2) if has_physical_spacing else None

    results = {
        "case_name": case_name,
        "status": "Analyzed (Valid Continuous Centerline)",
        "is_valid": True,
        "centerline_points": num_skel,
        "reference_diameter_px": round(ref_diam_px, 2),
        "mld_px": round(mld_px, 2),
        "physical_spacing_available": has_physical_spacing,
        "pixel_spacing_mm": spacing_val,
        "reference_diameter_mm": ref_diam_mm,
        "mld_mm": mld_mm,
        "minimum_lumen_diameter_mm": mld_mm if has_physical_spacing else round(mld_px, 2),
        "stenosis_percentage": round(stenosis_percentage, 1),
        "stenosis_display": f"{round(stenosis_percentage, 1):.1f}%",
        "severity_category": severity,
        "num_candidate_lesions": num_lesions,
        "candidate_narrowing_points": num_lesions,
        "rejection_reason": None,
        "validation_status": "Experimental (No per-lesion clinical percent ground truth provided in ImageCAS)"
    }

    # 5. Longitudinal Profile Plot
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=130)
    if has_physical_spacing:
        y_vals = local_diameters_px * spacing_val
        y_ref = ref_diam_mm
        y_mld = mld_mm
        y_label = "Lumen Diameter (mm)"
        ref_label = f"Reference Diameter ({y_ref:.2f} mm)"
        mld_label = f"MLD ({y_mld:.2f} mm)"
    else:
        y_vals = local_diameters_px
        y_ref = ref_diam_px
        y_mld = mld_px
        y_label = "Lumen Diameter (pixels)"
        ref_label = f"Reference Diameter ({y_ref:.1f} px)"
        mld_label = f"MLD ({y_mld:.1f} px)"

    ax.plot(range(len(y_vals)), y_vals, "b-o", markersize=4, label="Lumen Diameter")
    ax.axhline(y=y_ref, color="green", linestyle="--", label=ref_label)
    ax.axhline(y=y_mld, color="red", linestyle=":", label=mld_label)

    if len(narrow_idx) > 0:
        ax.scatter(narrow_idx, y_vals[narrow_idx], color="crimson", s=40, zorder=5, label="Candidate Narrowing")

    spacing_sub = f"Spacing: {spacing_val:.4f} mm/px" if has_physical_spacing else "Physical spacing unavailable (showing pixels)"
    ax.set_title(f"Longitudinal Lumen Diameter Profile — {case_name}\n"
                 f"Estimated Stenosis: {stenosis_percentage:.1f}% ({severity}) | {spacing_sub}", fontsize=11)
    ax.set_xlabel("Centerline Sample Point Index")
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()

    out_dir = config.paths.analysis_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / f"{case_name}_stenosis_profile.png"
    fig.savefig(plot_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    return results, plot_path


def generate_stenosis_report(findings: Dict[str, Any]):
    """Generate stenosis report markdown summary."""
    reports_dir = config.paths.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "stenosis_analysis_report.md"

    ref_str = f"{findings.get('reference_diameter_mm')} mm" if findings.get('reference_diameter_mm') is not None else (f"{findings.get('reference_diameter_px')} px (Physical UNAVAILABLE)" if findings.get('reference_diameter_px') is not None else "N/A — Not measurable")
    mld_str = f"{findings.get('mld_mm')} mm" if findings.get('mld_mm') is not None else (f"{findings.get('mld_px')} px (Physical UNAVAILABLE)" if findings.get('mld_px') is not None else "N/A — Not measurable")
    sten_display = findings.get('stenosis_display', 'N/A — Not measurable')

    content = f"""# Phase 16.1: Stenosis Detection & Quality Safeguard Report
**Artery Segmentation and Stenosis Detection Using Deep Learning**

> [!CAUTION]
> **RESEARCH & EXPERIMENTAL PROTOTYPE ONLY**:
> This module performs experimental geometric narrowing analysis on 2D segmentation masks.
> It is **NOT approved for clinical decision-making or diagnosis**.

---

## 1. Quality Safeguards and Rejection Criteria
To prevent false-positive stenosis alarms on short or noisy segmentation artifacts:
1. **Empty Mask Guard**: Masks with 0 centerline pixels are rejected (`N/A — Not measurable`).
2. **Minimum Centerline Length Guard**: Skeletons shorter than 10 pixels are marked **REJECTED_TOO_SHORT** and reported as **`N/A — Not measurable`** (never converted to 0.0%).
3. **Physical Spacing Verification**: Physical millimeter measurements are only reported when valid patient metadata is available.

---

## 2. Experimental Analysis Findings (Sample Case: `{findings.get('case_name', 'N/A')}`)

| Parameter | Value | Clinical / Technical Interpretation |
|---|---|---|
| **Analysis Status** | {findings.get('status', 'N/A')} | Quality check result |
| **Measurement Valid?** | {'YES' if findings.get('is_valid') else 'NO (REJECTED)'} | Acceptance status |
| **Centerline Sample Points** | {findings.get('centerline_points', 0)} px | Extracted medial-axis length |
| **Reference Diameter** | {ref_str} | 90th percentile baseline caliber |
| **Minimum Lumen Diameter (MLD)** | {mld_str} | Narrowest cross-section |
| **Estimated Stenosis Percentage** | **{sten_display}** | NASCET formula result |
| **Severity Category** | **{findings.get('severity_category', 'N/A — Not measurable')}** | Standard research severity tier |
| **Candidate Narrowing Points** | {findings.get('candidate_narrowing_points', 0)} | Points with >30% caliber reduction |
| **Physical Spacing Available** | {'YES (' + str(findings.get('pixel_spacing_mm')) + ' mm/px)' if findings.get('physical_spacing_available') else 'NO'} | Metadata ingestion status |

---

## 3. Severity Categorization Criteria
| Category | Percent Stenosis Range | Clinical Research Meaning |
|---|---|---|
| **Normal / Insignificant** | $0\\% - 25\\%$ | Non-obstructive or healthy vessel |
| **Mild Stenosis** | $25\\% - 50\\%$ | Minor plaque accumulation |
| **Moderate Stenosis** | $50\\% - 70\\%$ | Potentially flow-limiting |
| **Severe Stenosis** | $70\\% - 99\\%$ | Obstructive CAD |
| **Total Occlusion** | $100\\%$ | Complete cutoff of vessel lumen |
| **N/A — Not measurable** | N/A | Centerline below reliability threshold (<10 px) or empty |
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser(description="Phase 16.1 Stenosis Detection & Severity Estimation")
    parser.add_argument("--mask", default=None, help="Path to binary mask PNG")
    parser.add_argument("--spacing", type=float, default=None, help="Physical pixel spacing in mm")
    parser.add_argument("--min-size", type=int, default=0, help="Min component size (0=raw default)")
    args = parser.parse_args()

    config.setup_environment()
    out_dir = config.paths.analysis_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.mask:
        pred_masks = list(config.paths.predictions_dir.glob("*_mask.png"))
        if not pred_masks:
            pred_masks = list(config.paths.legacy_outputs_dir.glob("predictions/*_mask.png"))
        if not pred_masks:
            raise FileNotFoundError("No predicted masks found. Run predict.py first.")
        mask_path = pred_masks[0]
    else:
        mask_path = Path(args.mask)

    case_stem = mask_path.stem.replace("_mask", "")
    print(f"Running Phase 16.1 experimental stenosis analysis on {case_stem}...", flush=True)

    mask_arr = np.asarray(Image.open(mask_path).convert("L"))
    stats, plot_path = run_stenosis_detection(
        mask_arr, case_name=case_stem, pixel_spacing_mm=args.spacing, min_component_size=args.min_size
    )

    df = pd.DataFrame([stats])
    csv_path = out_dir / "stenosis_measurements.csv"
    df.to_csv(csv_path, index=False)

    print("\n--- Stenosis Detection Findings ---", flush=True)
    for k, v in stats.items():
        print(f"  {k}: {v}", flush=True)

    if plot_path:
        print(f"Saved stenosis profile plot to {plot_path}", flush=True)

    generate_stenosis_report(stats)


if __name__ == "__main__":
    main()
