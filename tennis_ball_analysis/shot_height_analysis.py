# shot_height_analysis.py
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib

matplotlib.use('TkAgg')


def analyze_shot_heights():
    # Load data
    with open('../tracker_stub/tennis_ball_detections.pkl', 'rb') as file:
        tennis_ball_positions = pickle.load(file)

    # Process data
    tennis_ball_positions = [x.get(1, []) for x in tennis_ball_positions]
    df = pd.DataFrame(tennis_ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
    df = df.interpolate().bfill()

    df['center_x'] = (df['x1'] + df['x2']) / 2
    df['center_y'] = (df['y1'] + df['y2']) / 2
    df['height'] = df['y2'] - df['y1']  # Ball size can approximate relative height from camera

    # Calculate moving average of height
    df['height_rolling'] = df['height'].rolling(window=5, min_periods=1).mean()

    # Calculate net line position (assuming middle of frame for simplicity)
    # This is an approximation - ideally use court detection for precise net position
    net_y_position = df['center_y'].median()

    # Plot height distribution as a function of Y position (approximating court position)
    plt.figure(figsize=(10, 8))

    # Add court regions (approximated)
    court_height = df['center_y'].max() - df['center_y'].min()
    baseline_top = df['center_y'].min() + court_height * 0.1
    baseline_bottom = df['center_y'].max() - court_height * 0.1

    # Add approximate court regions
    plt.axhspan(df['center_y'].min(), baseline_top, alpha=0.2, color='green', label='Behind baseline (top)')
    plt.axhspan(baseline_bottom, df['center_y'].max(), alpha=0.2, color='blue', label='Behind baseline (bottom)')
    plt.axhspan(baseline_top, net_y_position, alpha=0.2, color='lightgreen', label='Court (top)')
    plt.axhspan(net_y_position, baseline_bottom, alpha=0.2, color='lightblue', label='Court (bottom)')

    # Plot height vs position
    plt.scatter(df['center_x'], df['center_y'], c=df['height_rolling'], cmap='viridis',
                alpha=0.7, s=10)

    plt.colorbar(label='Ball Height (pixels)')
    plt.axhline(y=net_y_position, color='red', linestyle='--', label='Approximate Net Position')

    plt.title('Ball Height Analysis')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.gca().invert_yaxis()  # Invert Y-axis to match video coordinates

    plt.tight_layout()
    plt.savefig('ball_height_analysis.png', dpi=300)
    plt.show()

    # Height distribution histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df['height_rolling'], bins=30, alpha=0.7, color='blue')
    plt.axvline(df['height_rolling'].mean(), color='red', linestyle='--',
                label=f'Mean Height: {df["height_rolling"].mean():.2f} pixels')

    plt.title('Ball Height Distribution')
    plt.xlabel('Ball Height (pixels)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig('ball_height_histogram.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    analyze_shot_heights()