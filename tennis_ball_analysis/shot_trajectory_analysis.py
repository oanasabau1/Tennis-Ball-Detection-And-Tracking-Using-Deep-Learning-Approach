# shot_trajectory_analysis.py
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import make_interp_spline

matplotlib.use('TkAgg')


def analyze_shot_trajectories():
    # Load data
    with open('../tracker_stub/tennis_ball_detections.pkl', 'rb') as file:
        tennis_ball_positions = pickle.load(file)

    # Process and prepare data similar to main script
    tennis_ball_positions = [x.get(1, []) for x in tennis_ball_positions]
    df = pd.DataFrame(tennis_ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
    df = df.interpolate().bfill()

    df['center_x'] = (df['x1'] + df['x2']) / 2
    df['center_y'] = (df['y1'] + df['y2']) / 2

    # Get hit frames from original analysis
    df['center_y_rolling_mean'] = df['center_y'].rolling(window=5, min_periods=1, center=False).mean()
    df['delta_y'] = df['center_y_rolling_mean'].diff()

    # Identify hits (simplified from main script)
    df['tennis_ball_hit'] = 0
    minimum_change_frames_for_hit = 25

    for i in range(1, len(df) - int(minimum_change_frames_for_hit * 1.2)):
        neg_change = df['delta_y'].iloc[i] > 0 > df['delta_y'].iloc[i + 1]
        pos_change = df['delta_y'].iloc[i] < 0 < df['delta_y'].iloc[i + 1]

        if neg_change or pos_change:
            change_count = 0
            for cf in range(i + 1, i + int(minimum_change_frames_for_hit * 1.2) + 1):
                if cf >= len(df):
                    break
                neg_change_follow = df['delta_y'].iloc[i] > 0 > df['delta_y'].iloc[cf]
                pos_change_follow = df['delta_y'].iloc[i] < 0 < df['delta_y'].iloc[cf]

                if neg_change and neg_change_follow:
                    change_count += 1
                elif pos_change and pos_change_follow:
                    change_count += 1

            if change_count > minimum_change_frames_for_hit - 1:
                df.at[i, 'tennis_ball_hit'] = 1

    hit_frames = df[df['tennis_ball_hit'] == 1].index.tolist()

    # Analyze trajectories between consecutive hits
    plt.figure(figsize=(12, 8))

    for i in range(len(hit_frames) - 1):
        start_frame = hit_frames[i]
        end_frame = hit_frames[i + 1]

        # Get trajectory segment
        trajectory = df.loc[start_frame:end_frame, ['center_x', 'center_y']]

        # Smooth trajectory for visualization
        if len(trajectory) > 3:
            X = trajectory.index.values
            Y_x = trajectory['center_x'].values
            Y_y = trajectory['center_y'].values

            X_smooth = np.linspace(X.min(), X.max(), 100)
            try:
                spl_x = make_interp_spline(X, Y_x, k=3)
                spl_y = make_interp_spline(X, Y_y, k=3)

                x_smooth = spl_x(X_smooth)
                y_smooth = spl_y(X_smooth)

                plt.plot(x_smooth, y_smooth, label=f'Shot {i + 1}')
            except:
                # Fall back to original data if smoothing fails
                plt.plot(trajectory['center_x'], trajectory['center_y'], label=f'Shot {i + 1}')
        else:
            plt.plot(trajectory['center_x'], trajectory['center_y'], label=f'Shot {i + 1}')

        # Mark start and end points
        plt.scatter(df.loc[start_frame, 'center_x'], df.loc[start_frame, 'center_y'],
                    color='green', s=100, marker='o', label=f'Hit {i + 1}' if i == 0 else "")
        plt.scatter(df.loc[end_frame, 'center_x'], df.loc[end_frame, 'center_y'],
                    color='red', s=100, marker='x', label=f'Hit {i + 2}' if i == 0 else "")

    plt.title('Tennis Ball Shot Trajectories')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True)
    plt.legend()
    plt.gca().invert_yaxis()  # Invert Y-axis to match video coordinates
    plt.savefig('shot_trajectories.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    analyze_shot_trajectories()