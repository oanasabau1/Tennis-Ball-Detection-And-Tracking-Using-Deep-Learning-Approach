import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

with open('../tracker_stub/tennis_ball_detections.pkl', 'rb') as file:
    tennis_ball_positions = pickle.load(file)

tennis_ball_positions = [x.get(1, []) for x in tennis_ball_positions]
df_tennis_ball_positions = pd.DataFrame(tennis_ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
df_tennis_ball_positions = df_tennis_ball_positions.interpolate()
df_tennis_ball_positions = df_tennis_ball_positions.bfill()

df_tennis_ball_positions['center_x'] = (df_tennis_ball_positions['x1'] + df_tennis_ball_positions['x2']) / 2
df_tennis_ball_positions['center_y'] = (df_tennis_ball_positions['y1'] + df_tennis_ball_positions['y2']) / 2
print(df_tennis_ball_positions[['center_x', 'center_y']])

df_tennis_ball_positions['center_y_rolling_mean'] = df_tennis_ball_positions['center_y'].rolling(window=5, min_periods=1, center=False).mean()
plt.plot(df_tennis_ball_positions['center_y_rolling_mean'])
plt.title('Smoothed Vertical Ball Movement')
plt.show()

df_tennis_ball_positions['delta_y'] = df_tennis_ball_positions['center_y_rolling_mean'].diff()
plt.plot(df_tennis_ball_positions['delta_y'])
plt.title('Delta Y between Frames')
plt.show()

df_tennis_ball_positions['tennis_ball_hit'] = 0
minimum_change_frames_for_hit = 25

for i in range(1, len(df_tennis_ball_positions) - int(minimum_change_frames_for_hit * 1.2)):
    neg_change = df_tennis_ball_positions['delta_y'].iloc[i] > 0 > df_tennis_ball_positions['delta_y'].iloc[i + 1]
    pos_change = df_tennis_ball_positions['delta_y'].iloc[i] < 0 < df_tennis_ball_positions['delta_y'].iloc[i + 1]

    if neg_change or pos_change:
        change_count = 0
        for cf in range(i + 1, i + int(minimum_change_frames_for_hit * 1.2) + 1):
            neg_change_follow = df_tennis_ball_positions['delta_y'].iloc[i] > 0 > df_tennis_ball_positions['delta_y'].iloc[cf]
            pos_change_follow = df_tennis_ball_positions['delta_y'].iloc[i] < 0 < df_tennis_ball_positions['delta_y'].iloc[cf]

            if neg_change and neg_change_follow:
                change_count += 1
            elif pos_change and pos_change_follow:
                change_count += 1

        if change_count > minimum_change_frames_for_hit - 1:
            df_tennis_ball_positions.at[i, 'tennis_ball_hit'] = 1


frame_nums_with_ball_hits = df_tennis_ball_positions[df_tennis_ball_positions['tennis_ball_hit'] == 1].index.tolist()

print("Detected ball hits at frames:")
print(df_tennis_ball_positions[df_tennis_ball_positions['tennis_ball_hit'] == 1])
