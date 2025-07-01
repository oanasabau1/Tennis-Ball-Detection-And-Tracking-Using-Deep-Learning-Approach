
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


def load_tennis_ball_positions(stub_path):
    if not os.path.exists(stub_path):
        raise FileNotFoundError(f"Stub file not found: {stub_path}")
    with open(stub_path, 'rb') as file:
        return pickle.load(file)


def load_video_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError("Could not read video frame.")
    return frame


def draw_court_lines(ax, mini_court):
    k = mini_court.keypoints
    lines = mini_court.lines
    for pt1_idx, pt2_idx in lines:
        x1, y1 = k[pt1_idx * 2], k[pt1_idx * 2 + 1]
        x2, y2 = k[pt2_idx * 2], k[pt2_idx * 2 + 1]
        ax.plot([x1, x2], [y1, y2], color='black', linewidth=1)

    # Draw net line
    net_y = (k[1] + k[5]) / 2
    ax.plot([k[0], k[2]], [net_y, net_y], color='blue', linewidth=1)


def generate_heatmap(xs, ys, mini_court, output_dir, video_name):
    if not xs or not ys:
        raise ValueError("No valid data points after conversion.")

    fig, ax = plt.subplots(figsize=(6, 12))
    draw_court_lines(ax, mini_court)
    ax.scatter(xs, ys, color='red', s=10, label='Ball Positions')
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()
    plt.legend()
    plt.title('Raw Ball Positions on Mini-Court')
    plt.show()

    cmap = LinearSegmentedColormap.from_list(
        'tennis_cmap', [(0, 0, 0.5), (0, 0.5, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)], N=256
    )

    fig, ax = plt.subplots(figsize=(6, 12))  # Tennis court is vertical

    h = ax.hist2d(
        xs, ys, bins=60,
        range=[
            [mini_court.court_start_x, mini_court.court_end_x],
            [mini_court.court_start_y, mini_court.court_end_y]
        ],
        cmap=cmap
    )
    plt.colorbar(h[3], ax=ax, label='Frequency')

    draw_court_lines(ax, mini_court)

    ax.set_title('Tennis Ball Heatmap on Mini-Court')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()

    os.makedirs(output_dir, exist_ok=True)
    heatmap_path = os.path.join(output_dir, f'heatmap_for_{video_name}.png')
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"Heatmap saved to {heatmap_path}")


def create_heatmap(video_name, output_dir=None):
    try:
        stub_path = f'tracker_stub/tennis_ball_detections_for_{video_name}.pkl'
        tennis_ball_positions = load_tennis_ball_positions(stub_path)

        frame_path = f'input_videos/{video_name}.mp4'
        frame = load_video_frame(frame_path)

        court_detector = CourtLineDetector(model_path='models/tennis_court_keypoints_model.pth')
        court_keypoints = court_detector.predict(frame)

        if court_keypoints is None or len(court_keypoints) < 28:
            raise ValueError("Court keypoints detection failed or incomplete.")

        mini_court = MiniCourt(frame)
        mini_positions = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
            tennis_ball_positions, court_keypoints
        )

        xs, ys = [], []
        for pos in mini_positions:
            coords = pos.get(1)
            if coords and not (np.isnan(coords[0]) or np.isnan(coords[1])):
                xs.append(coords[0])
                ys.append(coords[1])

        if output_dir is None:
            output_dir = f'outputs/{video_name}'

        generate_heatmap(xs, ys, mini_court, output_dir, video_name)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python heatmap_visualization.py <video_name>")
    else:
        create_heatmap(sys.argv[1])
