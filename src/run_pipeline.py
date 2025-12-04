import os
import pandas as pd
import cv2
from ultralytics import YOLO

os.makedirs("outputs/overlays", exist_ok=True)
os.makedirs("outputs/manifests", exist_ok=True)
os.makedirs("outputs/metrics", exist_ok=True)

# Load YOLO model
model = YOLO("models/yolo/best.pt")

# Read input Excel
input_df = pd.read_excel("data/input.xlsx")

for idx, row in input_df.iterrows():
    sample_id = str(row["sample_id"]).split(".")[0]
    image_path = f"data/fetched/{sample_id}.jpg"
    
    # Load image
    img = cv2.imread(image_path)
    
    # Run YOLO inference
    results = model(img)
    
    # Draw boxes on image
    annotated_img = results[0].plot()
    
    # Calculate area (dummy example, replace with actual logic)
    area = 0
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        width = x2 - x1
        height = y2 - y1
        area += width * height
        panel_count = len(results[0].boxes)
        print(f"Processed {sample_id}: {panel_count} panels, area={area:.2f}, QC={qc_pass}")

    # Apply QC logic (example threshold)
    qc_pass = area > 1000
    
    # Save overlay image
    overlay_path = f"outputs/overlays/{sample_id}.jpg"
    cv2.imwrite(overlay_path, annotated_img)
    
    # Save JSON manifest
    manifest = {
    'sample_id': sample_id,
    'panel_count': panel_count,
    'area': float(area),
    'qc_pass': qc_pass
}
    import json
    with open(f"outputs/manifests/{sample_id}.json", 'w') as f:
        json.dump(manifest, f)
    
    # Save metrics
    metrics_path = "outputs/metrics/pipeline_metrics.csv"
    metrics_df = pd.DataFrame([{
    'sample_id': sample_id,
    'panel_count': panel_count,
    'area': area,
    'qc_pass': qc_pass
}])
    if idx == 0:
        metrics_df.to_csv(metrics_path, index=False)
    else:
        metrics_df.to_csv(metrics_path, mode='a', header=False, index=False)


