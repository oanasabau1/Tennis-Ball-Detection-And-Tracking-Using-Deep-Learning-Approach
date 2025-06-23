import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from matplotlib.colors import LinearSegmentedColormap
from mini_court import MiniCourt
from court_line_detector import CourtLineDetector
import cv2

matplotlib.use('TkAgg')

def create_heatmap(video_name, output_dir=None):
    stub_path = f'tracker_stub/tennis_ball_detections_for_{video_name}.pkl'
    if not os.path.exists(stub_path):
        print(f"Error: Stub file not found: {stub_path}")
        return

    with open(stub_path, 'rb') as file:
        tennis_ball_positions = pickle.load(file)

    frame_path = f'input_videos/{video_name}.mp4'
    cap = cv2.VideoCapture(frame_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Error: Could not read video frame.")
        return

    court_detector = CourtLineDetector(model_path='models/tennis_court_keypoints_model.pth')
    court_keypoints = court_detector.predict(frame)

    mini_court = MiniCourt(frame)

    mini_positions = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
        tennis_ball_positions, court_keypoints
    )

    xs = []
    ys = []
    for pos in mini_positions:
        coords = pos.get(1)
        if coords and not (np.isnan(coords[0]) or np.isnan(coords[1])):
            xs.append(coords[0])
            ys.append(coords[1])

    if not xs or not ys:
        print("Error: No valid data points after conversion.")
        return

    cmap = LinearSegmentedColormap.from_list(
        'tennis_cmap', [(0, 0, 0.5), (0, 0.5, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)], N=256
    )

    plt.figure(figsize=(6, 12))
    plt.hist2d(xs, ys, bins=40, cmap=cmap)
    plt.colorbar(label='Frequency')
    plt.title('Tennis Ball Heatmap in Mini-Court Coordinates')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')

    if output_dir is None:
        output_dir = f'output_videos/{video_name}'
    os.makedirs(output_dir, exist_ok=True)
    heatmap_path = os.path.join(output_dir, f'heatmap_for_{video_name}.png')
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"Heatmap saved to {heatmap_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python heatmap_visualization.py <video_name>")
    else:
        video_name = sys.argv[1]
        create_heatmap(video_name)
