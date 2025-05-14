import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from matplotlib.colors import LinearSegmentedColormap

matplotlib.use('TkAgg')


def create_heatmap_visualization(video_name, output_dir=None):

    stub_path = f'tracker_stub/tennis_ball_detections_for_{video_name}.pkl'
    if not os.path.exists(stub_path):
        print(f"Error: Stub file not found: {stub_path}")
        return

    with open(stub_path, 'rb') as file:
        tennis_ball_positions = pickle.load(file)

    positions = [x.get(1, []) for x in tennis_ball_positions]
    df = pd.DataFrame(positions, columns=['x1', 'y1', 'x2', 'y2']).interpolate()

    center_x = (df['x1'] + df['x2']) / 2
    center_y = (df['y1'] + df['y2']) / 2

    valid = ~(np.isnan(center_x) | np.isnan(center_y))
    center_x = center_x[valid].to_numpy()
    center_y = center_y[valid].to_numpy()

    if len(center_x) == 0 or len(center_y) == 0:
        print("Error: No valid data points after filtering.")
        return

    x_mid = (np.max(center_x) + np.min(center_x)) / 2
    y_mid = (np.max(center_y) + np.min(center_y)) / 2

    cmap = LinearSegmentedColormap.from_list(
        'tennis_cmap', [(0, 0, 0.5), (0, 0.5, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)], N=256
    )

    plt.figure(figsize=(12, 8))
    h = plt.hist2d(center_x, center_y, bins=40, range=[[center_x.min(), center_x.max()], [center_y.min(), center_y.max()]], cmap=cmap)

    plt.colorbar(h[3], label='Frequency')
    plt.axvline(x=x_mid, color='white', linestyle='--', alpha=0.7)
    plt.axhline(y=y_mid, color='white', linestyle='--', alpha=0.7)
    plt.title('Tennis Ball Position Heatmap')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')

    if output_dir is None:
        output_dir = f'outputs/{video_name}'
    os.makedirs(output_dir, exist_ok=True)
    heatmap_path = os.path.join(output_dir, f'ball_heatmap_for_{video_name}.png')

    plt.savefig(heatmap_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python heatmap_visualization.py <video_name>")
    else:
        video_name = sys.argv[1]
        create_heatmap_visualization(video_name)
