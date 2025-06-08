import os
import shutil
import yaml


datasets = {
    "dataset1": "../tennis_ball_dataset_1",
    "dataset2": "../tennis_ball_dataset_2",
}
combined_path = "../combined_tennis_ball_dataset"
splits = ["train", "valid", "test"]
class_names = ["tennis ball"]


def make_dirs():
    for split in splits:
        os.makedirs(os.path.join(combined_path, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(combined_path, split, "labels"), exist_ok=True)


def copy_dataset_content(dataset_path):
    for split in splits:
        img_src = os.path.join(dataset_path, split, "images")
        lbl_src = os.path.join(dataset_path, split, "labels")
        img_dst = os.path.join(combined_path, split, "images")
        lbl_dst = os.path.join(combined_path, split, "labels")

        if not os.path.exists(img_src) or not os.path.exists(lbl_src):
            print(f"Skipping {split} from {dataset_path} (not found)")
            continue

        for fname in os.listdir(img_src):
            src_file = os.path.join(img_src, fname)
            dst_file = os.path.join(img_dst, fname)

            if os.path.exists(dst_file):
                print(f"Skipping duplicate image: {fname}")
                continue

            shutil.copy2(src_file, dst_file)

        for fname in os.listdir(lbl_src):
            src_file = os.path.join(lbl_src, fname)
            dst_file = os.path.join(lbl_dst, fname)

            if os.path.exists(dst_file):
                print(f"Skipping duplicate label: {fname}")
                continue

            shutil.copy2(src_file, dst_file)


def write_yaml():
    data = {
        "train": os.path.abspath(os.path.join(combined_path, "train", "images")),
        "val": os.path.abspath(os.path.join(combined_path, "valid", "images")),
        "test": os.path.abspath(os.path.join(combined_path, "test", "images")),
        "nc": len(class_names),
        "names": class_names
    }
    with open(os.path.join(combined_path, "data.yaml"), "w") as f:
        yaml.dump(data, f)


if __name__ == "__main__":
    make_dirs()
    for tag, dataset_path in datasets.items():
        copy_dataset_content(dataset_path)
    write_yaml()
    print(f"Combined dataset created at: {os.path.abspath(combined_path)}")