import os
import random
import shutil
import argparse


def move_images(source_dir, dest_dir, count):
    img_src_dir = os.path.join(source_dir, "images")
    lbl_src_dir = os.path.join(source_dir, "labels")
    img_dst_dir = os.path.join(dest_dir, "images")
    lbl_dst_dir = os.path.join(dest_dir, "labels")

    os.makedirs(img_dst_dir, exist_ok=True)
    os.makedirs(lbl_dst_dir, exist_ok=True)

    img_files = [f for f in os.listdir(img_src_dir) if f.endswith((".jpg", ".png"))]
    random.shuffle(img_files)

    moved = 0
    for fname in img_files[:count]:
        img_src = os.path.join(img_src_dir, fname)
        lbl_src = os.path.join(lbl_src_dir, fname.replace(".jpg", ".txt").replace(".png", ".txt"))

        img_dst = os.path.join(img_dst_dir, fname)
        lbl_dst = os.path.join(lbl_dst_dir, os.path.basename(lbl_src))

        try:
            shutil.move(img_src, img_dst)
            if os.path.exists(lbl_src):
                shutil.move(lbl_src, lbl_dst)
            moved += 1
        except Exception as e:
            print(f"Failed to move {fname}: {e}")

    print(f"Moved {moved} image(s) from '{source_dir}' to '{dest_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Move labeled images between dataset splits.")
    parser.add_argument("--source", required=True, help="Source split folder (e.g., train)")
    parser.add_argument("--dest", required=True, help="Destination split folder (e.g., valid)")
    parser.add_argument("--count", type=int, required=True, help="Number of images to move")

    args = parser.parse_args()
    move_images(args.source, args.dest, args.count)
