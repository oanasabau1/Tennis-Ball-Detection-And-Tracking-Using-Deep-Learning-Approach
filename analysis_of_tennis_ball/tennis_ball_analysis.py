import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.use('TkAgg')


def detect_ball_hits(video_name):
    stub_path = f'tracker_stub/tennis_ball_detections_for_{video_name}.pkl'

    if not os.path.exists(stub_path):
        print(f"Stub file not found: {stub_path}")
        return

    with open(stub_path, 'rb') as file:
        tennis_ball_positions = pickle.load(file)

    tennis_ball_positions = [x.get(1, []) for x in tennis_ball_positions]
    df = pd.DataFrame(tennis_ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
    df = df.interpolate().bfill()

    df['center_x'] = (df['x1'] + df['x2']) / 2
    df['center_y'] = (df['y1'] + df['y2']) / 2
    df['center_y_rolling_mean'] = df['center_y'].rolling(window=5, min_periods=1).mean()

    plt.plot(df['center_y_rolling_mean'])
    plt.title('Smoothed Vertical Ball Movement')
    plt.show()

    df['delta_y'] = df['center_y_rolling_mean'].diff()

    plt.plot(df['delta_y'])
    plt.title('Delta Y between Frames')
    plt.show()

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
                neg_follow = df['delta_y'].iloc[i] > 0 > df['delta_y'].iloc[cf]
                pos_follow = df['delta_y'].iloc[i] < 0 < df['delta_y'].iloc[cf]

                if neg_change and neg_follow:
                    change_count += 1
                elif pos_change and pos_follow:
                    change_count += 1

            if change_count > minimum_change_frames_for_hit - 1:
                df.at[i, 'tennis_ball_hit'] = 1

    hits = df[df['tennis_ball_hit'] == 1]

    print("Detected ball hits at frames:")
    print(hits[['center_x', 'center_y', 'tennis_ball_hit']])
    return hits.index.tolist()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python detect_ball_hits.py <video_name>")
    else:
        video_name = sys.argv[1]
        detect_ball_hits(video_name)
