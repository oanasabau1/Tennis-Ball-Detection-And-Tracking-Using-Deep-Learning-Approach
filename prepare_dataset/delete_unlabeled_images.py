import os

base_dir = "/tennis_ball_dataset_1"
splits = ["train", "valid", "test"]

deleted_count = 0

for split in splits:
    img_dir = os.path.join(base_dir, split, "images")
    lbl_dir = os.path.join(base_dir, split, "labels")

    for fname in os.listdir(img_dir):
        if not fname.endswith((".jpg", ".png")):
            continue

        label_fname = fname.replace(".jpg", ".txt").replace(".png", ".txt")
        label_path = os.path.join(lbl_dir, label_fname)

        if not os.path.exists(label_path):
            img_path = os.path.join(img_dir, fname)
            try:
                os.remove(img_path)
                print(f"Deleted: {img_path}")
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {img_path}: {e}")

print(f"\n Total unlabeled images deleted: {deleted_count}")
