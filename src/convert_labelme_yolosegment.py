import json
from pathlib import Path

import numpy as np

# Map your labels (exact text from labelme) to class IDs
CLASS_MAP = {
    "pelvic ring": 0,
    "left_obturator_foramen": 1,
    "right_obturator_foramen": 2,
}


def convert_single_labelme_json(json_path: Path, labels_root: Path):
    with open(json_path, "r") as f:
        data = json.load(f)

    img_w = data["imageWidth"]
    img_h = data["imageHeight"]

    image_filename = data["imagePath"]
    stem = Path(image_filename).stem  # name without extension

    # YOLO label file path
    label_path = labels_root / f"{stem}.txt"
    label_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []

    for shape in data.get("shapes", []):
        label = shape["label"]
        if label not in CLASS_MAP:
            # skip shapes with labels we don't care about
            continue

        points = np.array(shape["points"], dtype=np.float32)  # shape: (N, 2)
        xs = points[:, 0]
        ys = points[:, 1]

        # bounding box in pixel coordinates
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        width = x_max - x_min
        height = y_max - y_min

        # normalize bbox to [0,1]
        x_center_norm = x_center / img_w
        y_center_norm = y_center / img_h
        width_norm = width / img_w
        height_norm = height / img_h

        # polygon points normalized
        poly_norm = []
        for x, y in zip(xs, ys):
            poly_x = x / img_w
            poly_y = y / img_h
            poly_norm.extend([poly_x, poly_y])

        class_id = CLASS_MAP[label]

        line = [class_id, x_center_norm, y_center_norm, width_norm, height_norm, *poly_norm]
        line_str = " ".join(str(v) for v in line)
        lines.append(line_str)

    if lines:
        with open(label_path, "w") as f:
            f.write("\n".join(lines))
        print(f"Written {label_path}")
    else:
        print(f"No relevant shapes in {json_path}, skipping.")


def convert_folder(images_dir: Path, labels_root: Path):
    json_files = sorted(images_dir.glob("*.json"))
    if not json_files:
        print(f"No labelme JSON files found in {images_dir}")
        return

    print(f"Found {len(json_files)} labelme JSON files in {images_dir}")
    for json_path in json_files:
        convert_single_labelme_json(json_path, labels_root)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    # TRAIN
    train_images_dir = project_root / "data" / "images" / "train"
    train_labels_dir = project_root / "data" / "labels" / "train"
    convert_folder(train_images_dir, train_labels_dir)

    # VAL
    val_images_dir = project_root / "data" / "images" / "val"
    val_labels_dir = project_root / "data" / "labels" / "val"
    convert_folder(val_images_dir, val_labels_dir)
