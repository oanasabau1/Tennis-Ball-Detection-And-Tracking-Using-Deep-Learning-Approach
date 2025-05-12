from ultralytics import YOLO
import cv2
import pickle
import pandas as pd


class TennisBallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def interpolate_tennis_ball_positions(self, tennis_ball_positions):
        tennis_ball_positions = [x.get(1, []) for x in tennis_ball_positions]
        df_tennis_ball_positions = pd.DataFrame(tennis_ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        df_tennis_ball_positions = df_tennis_ball_positions.interpolate()
        df_tennis_ball_positions = df_tennis_ball_positions.bfill()
        tennis_ball_positions = [{1: x} for x in df_tennis_ball_positions.to_numpy().tolist()]
        return tennis_ball_positions

    def get_tennis_ball_shot_frames(self, tennis_ball_positions):
        tennis_ball_positions = [x.get(1, []) for x in tennis_ball_positions]
        df_tennis_ball_positions = pd.DataFrame(tennis_ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        df_tennis_ball_positions['tennis_ball_hit'] = 0
        df_tennis_ball_positions['center_x'] = (df_tennis_ball_positions['x1'] + df_tennis_ball_positions['x2']) / 2
        df_tennis_ball_positions['center_y'] = (df_tennis_ball_positions['y1'] + df_tennis_ball_positions['y2']) / 2
        df_tennis_ball_positions['center_y_rolling_mean'] = df_tennis_ball_positions['center_y'].rolling(window=5,
                                                                                                         min_periods=1, center=False).mean()
        df_tennis_ball_positions['delta_y'] = df_tennis_ball_positions['center_y_rolling_mean'].diff()
        minimum_change_frames_for_hit = 25
        for i in range(1, len(df_tennis_ball_positions) - int(minimum_change_frames_for_hit * 1.2)):
            neg_change = df_tennis_ball_positions['delta_y'].iloc[i] > 0 > df_tennis_ball_positions['delta_y'].iloc[
                i + 1]
            pos_change = df_tennis_ball_positions['delta_y'].iloc[i] < 0 < df_tennis_ball_positions['delta_y'].iloc[
                i + 1]
            if neg_change or pos_change:
                change_count = 0
                for cf in range(i + 1, i + int(minimum_change_frames_for_hit * 1.2) + 1):
                    neg_change_follow = df_tennis_ball_positions['delta_y'].iloc[i] > 0 > \
                                        df_tennis_ball_positions['delta_y'].iloc[cf]
                    pos_change_follow = df_tennis_ball_positions['delta_y'].iloc[i] < 0 < \
                                        df_tennis_ball_positions['delta_y'].iloc[cf]

                    if neg_change and neg_change_follow:
                        change_count += 1
                    elif pos_change and pos_change_follow:
                        change_count += 1

                if change_count > minimum_change_frames_for_hit - 1:
                    df_tennis_ball_positions.at[i, 'tennis_ball_hit'] = 1

        frame_nums_with_ball_hits = df_tennis_ball_positions[df_tennis_ball_positions['tennis_ball_hit'] == 1].index.tolist()
        return frame_nums_with_ball_hits

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        tennis_ball_detections = []

        if read_from_stub is True and stub_path is not None:
            with open(stub_path, "rb") as f:
                tennis_ball_detections = pickle.load(f)
            return tennis_ball_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            tennis_ball_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(tennis_ball_detections, f)

        return tennis_ball_detections

    def detect_frame(self, frame):
        results = self.model.predict(frame, conf=0.15)[0]
        tennis_ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            tennis_ball_dict[1] = result
        return tennis_ball_dict

    def draw_bounding_boxes(self, video_frames, player_detections):
        output_video_frames = []
        for frame, tennis_ball_dict in zip(video_frames, player_detections):
            for track_id, bbox in tennis_ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Ball", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            output_video_frames.append(frame)
        return output_video_frames
