from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model("https://ultralytics.com/images/bus.jpg")

r = results[0]
print("Image shape:", r.orig_shape)
print("Detections:", len(r.boxes))


import pydicom
from pathlib import Path
from PIL import Image
import numpy as np
import shutil
 
# ----------------------------
# Folders
# ----------------------------
base_folder = Path.home() / "laura-srp" / "AI project - Anonymised"
output_folder = Path.home() / "laura-srp" / "dataset_tif"
output_folder.mkdir(exist_ok=True)
 
# ----------------------------
# Helpers
# ----------------------------
def normalize_uint8(img):
    img = img.astype(float)
    img -= img.min()
    if img.max() > 0:
        img /= img.max()
    return (img * 255).astype(np.uint8)
 
# ----------------------------
# Loop through patients (1,2,3,...)
# ----------------------------
for patient_folder in base_folder.iterdir():
    if not patient_folder.is_dir():
        continue
 
    patient_id = patient_folder.name
    print(f"\nProcessing patient: {patient_id}")
 
    # Loop through visits
    for visit_folder in patient_folder.iterdir():
        if not visit_folder.is_dir():
            continue
 
        visit_name = visit_folder.name
        print(f"  Visit: {visit_name}")
 
        out_dir = output_folder / patient_id / visit_name
        out_dir.mkdir(parents=True, exist_ok=True)
 
        counter = 0
 
        # ----------------------------
        # Loop through ALL files once
        # ----------------------------
        for file_path in visit_folder.rglob("*"):
            if not file_path.is_file():
                continue
 
            # ---- Handle TIFFs first ----
            if file_path.suffix.lower() in [".tif", ".tiff"]:
                counter += 1
                tiff_name = f"{patient_id}-{visit_name}-{counter:02d}.tif"
                tiff_path = out_dir / tiff_name
                shutil.copy2(file_path, tiff_path)
                print(f"    Copied existing TIFF: {tiff_path}")
                continue
 
            # ---- Try DICOM (extensionless) ----
            try:
                ds = pydicom.dcmread(str(file_path), force=True)
                if "PixelData" not in ds:
                    continue
 
                img_array = normalize_uint8(ds.pixel_array)
                img = Image.fromarray(img_array)
 
                counter += 1
                tiff_name = f"{patient_id}-{visit_name}-{counter:02d}.tif"
                tiff_path = out_dir / tiff_name
                img.save(tiff_path)
 
                print(f"    Saved from DICOM: {tiff_path}")
 
            except Exception:
                # Not a DICOM → ignore silently
                continue
 
print("\n✅ All patients, visits, DICOMs (with or without extensions), and TIFFs processed!")