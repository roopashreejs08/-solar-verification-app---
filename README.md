#  EcoInnovators Ideathon 2026 – AI Powered Rooftop PV Detection

##  Overview
This project verifies rooftop solar installations using **AI and satellite imagery**.  
It supports the *PM Surya Ghar: Muft Bijli Yojana* scheme by ensuring subsidies reach genuine households.  

Instead of sending inspectors to every house, our pipeline:
- Fetches rooftop images for given coordinates.
- Detects whether solar panels are present.
- Estimates panel area (m²).
- Produces **audit‑friendly overlays** (bounding boxes/polygons).
- Outputs JSON records with confidence scores and QC status.

This makes subsidy distribution **faster, cheaper, and more trustworthy**.

---

##  Real‑World Example
A DISCOM officer uploads a file with 1,000 household coordinates:
- 700 houses → solar panels found.  
- 200 houses → no solar panels.  
- 100 houses → images too blurry/cloudy → NOT_VERIFIABLE.  

Officer downloads JSON + overlay images → submits as audit proof.  
Subsidies go only to verified households.

---

## Repository Structure
pipeline_code/     → Python scripts for inference pipeline  
environment/       → requirements.txt, environment.yml, python_version.txt  
trained_model/     → Saved AI model files (.pt, .pkl, .joblib)  
model_card/        → Transparency document (PDF)  
predictions/       → JSON outputs for sample/training dataset  
artefacts/         → Overlay images with bounding boxes/polygons  
training_logs/     → Metrics (Loss, F1 Score, RMSE) across epochs  
README.md          → Project overview + run instructions  
LICENSE            → OSI‑approved license (MIT/Apache 2.0)  

---

## Setup Instructions
### 1. Clone the Repository
git clone https://github.com/<your-repo>.git  
cd <your-repo>  

### 2. Create Environment
Using pip:
pip install -r environment/requirements.txt  

Using conda:
conda env create -f environment/environment.yml  
conda activate rooftop-ai  

### 3. Verify Python Version
Check `environment/python_version.txt` (e.g., Python 3.10.12).

---

## How to Run
Run the pipeline with an input `.xlsx` file containing sample_id, latitude, longitude:
python pipeline_code/run_pipeline.py input.xlsx output/  

Outputs:
- JSON file per site (with detection results, confidence, PV area, QC status).
- Overlay images in `artefacts/`.

Example JSON:
{
  "sample_id": 1234,
  "lat": 12.9716,
  "lon": 77.5946,
  "has_solar": true,
  "confidence": 0.92,
  "pv_area_sqm_est": 23.5,
  "buffer_radius_sqft": 1200,
  "qc_status": "VERIFIABLE",
  "bbox_or_mask": "<encoded polygon>",
  "image_metadata": {"source": "Google", "capture_date": "2025-11-01"}
}

---

## Evaluation Criteria
- Detection Accuracy (40%) → F1 score on solar presence.  
- Quantification Quality (20%) → RMSE for PV area estimation.  
- Generalization & Robustness (20%) → Works across diverse roof types/states.  
- Usability & Documentation (20%) → Clear repo structure, reproducibility, auditability.  

---

## Model Card (Summary)
- Data Sources: Roboflow datasets + augmentations.  
- Assumptions: Resolution thresholds, buffer zones.  
- Logic: Classification + segmentation.  
- Limitations: Shadows, occlusion, rural imagery gaps.  
- Failure Modes: Low resolution, stale imagery.  
- Retraining Guidance: Add new annotated data for diverse roof types.  

---

## Extra Features
- Solar Health Monitoring: Predicts panel efficiency using weather + visual cues.  
- Digital Certificates: Tamper‑proof verification for households.  
- Citizen Portal: Transparency for households to track subsidy status.  
- Gamification: Solar Points redeemable for eco‑friendly rewards.  

---

## 📜 License
This project is licensed under the **MIT License** – see the LICENSE file for details.
