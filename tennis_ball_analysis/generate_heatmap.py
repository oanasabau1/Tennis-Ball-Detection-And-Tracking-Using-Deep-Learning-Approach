import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from matplotlib.colors import LinearSegmentedColormap

matplotlib.use('TkAgg')


def generate_ball_heatmaps():
    # Load data
    with open('../tracker_stub/tennis_ball_detections.pkl', 'rb') as file:
        tennis_ball_positions = pickle.load(file)

    # Process data
    tennis_ball_positions = [x.get(1, []) for x in tennis_ball_positions]
    df = pd.DataFrame(tennis_ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
    df = df.interpolate().bfill()

    # Calculate center positions
    df['center_x'] = (df['x1'] + df['x2']) / 2
    df['center_y'] = (df['y1'] + df['y2']) / 2

    # Identify ball hits using the same method from tennis_ball_analysis.py
    df['center_y_rolling_mean'] = df['center_y'].rolling(window=5, min_periods=1, center=False).mean()
    df['delta_y'] = df['center_y_rolling_mean'].diff()
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

    # Create custom colormap for tennis
    colors = [(0, 0, 0.5), (0, 0.5, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)]
    tennis_cmap = LinearSegmentedColormap.from_list('tennis_cmap', colors, N=256)

    # 1. Basic Density Heatmap
    plt.figure(figsize=(12, 8))
    plt.hist2d(df['center_x'], df['center_y'], bins=40, cmap=tennis_cmap)
    plt.colorbar(label='Frequency')
    plt.title('Tennis Ball Position Heatmap')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.savefig('ball_position_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Kernel Density Estimation (KDE) Heatmap for smoother result
    plt.figure(figsize=(12, 8))
    sns.kdeplot(x=df['center_x'], y=df['center_y'], cmap=tennis_cmap, fill=True, levels=20)
    plt.title('Tennis Ball Position KDE Heatmap')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.savefig('ball_position_kde_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Heatmap of hit positions only
    hit_df = df.loc[hit_frames]
    plt.figure(figsize=(12, 8))
    sns.kdeplot(x=hit_df['center_x'], y=hit_df['center_y'], cmap=tennis_cmap, fill=True, levels=15)
    plt.scatter(hit_df['center_x'], hit_df['center_y'], c='black', s=20, alpha=0.6)
    plt.title('Tennis Ball Hit Positions Heatmap')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.savefig('ball_hit_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Split court heatmap (top half vs bottom half)
    court_midpoint = df['center_y'].mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Top half
    top_df = df[df['center_y'] < court_midpoint]
    sns.kdeplot(x=top_df['center_x'], y=top_df['center_y'], cmap=tennis_cmap, fill=True, ax=ax1, levels=15)
    ax1.set_title('Top Half Court Heatmap')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')

    # Bottom half
    bottom_df = df[df['center_y'] >= court_midpoint]
    sns.kdeplot(x=bottom_df['center_x'], y=bottom_df['center_y'], cmap=tennis_cmap, fill=True, ax=ax2, levels=15)
    ax2.set_title('Bottom Half Court Heatmap')
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')

    plt.tight_layout()
    plt.savefig('split_court_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Heatmaps generated successfully!")
    return df, hit_frames


if __name__ == "__main__":
    df, hit_frames = generate_ball_heatmaps()

    # Print basic statistics
    print(f"Total frames analyzed: {len(df)}")
    print(f"Total detected hits: {len(hit_frames)}")