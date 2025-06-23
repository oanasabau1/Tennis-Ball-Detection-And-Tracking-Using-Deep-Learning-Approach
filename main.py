import os
import cv2
import numpy as np
import constants
from process_video import read_video, save_video
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
from trackers import TennisBallTracker
from bounding_boxes import measure_distance
from utils import convert_pixel_distance_to_meters
from analysis_of_tennis_ball import create_heatmap, detect_ball_hits


def main(video_path):
    input_video_path = f"input_videos/{video_path}.mp4"
    video_frames = read_video(input_video_path)

    tennis_ball_tracker = TennisBallTracker(
        model_path="D:/tennis_thesis/runs/train/tennis_ball_yolov5m/weights/best.pt"
    )
    stub_path = f"tracker_stub/tennis_ball_detections_for_{video_path}.pkl"
    read_from_stub = os.path.exists(stub_path)
    tennis_ball_detections = tennis_ball_tracker.detect_frames(
        video_frames, read_from_stub=read_from_stub, stub_path=stub_path
    )
    tennis_ball_detections = tennis_ball_tracker.interpolate_tennis_ball_positions(tennis_ball_detections)

    court_line_detector = CourtLineDetector(model_path="models/tennis_court_keypoints_model.pth")
    court_keypoints = court_line_detector.predict(video_frames[0])

    mini_court = MiniCourt(video_frames[0])
    tennis_ball_shot_frames = tennis_ball_tracker.get_tennis_ball_shot_frames(tennis_ball_detections)
    tennis_ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
        tennis_ball_detections, court_keypoints
    )

    shot_speeds = []
    for i in range(len(tennis_ball_shot_frames) - 1):
        start_frame = tennis_ball_shot_frames[i]
        end_frame = tennis_ball_shot_frames[i + 1]
        shot_time = (end_frame - start_frame) / 24

        start_pos = tennis_ball_mini_court_detections[start_frame].get(1)
        end_pos = tennis_ball_mini_court_detections[end_frame].get(1)

        if start_pos is None or end_pos is None:
            continue

        pixel_distance = measure_distance(start_pos, end_pos)
        meter_distance = convert_pixel_distance_to_meters(
            pixel_distance,
            constants.DOUBLE_LINE_WIDTH,
            mini_court.get_width_of_mini_court()
        )
        speed_kmh = meter_distance / shot_time * 3.6
        shot_speeds.append(speed_kmh)

        print(f"Shot {i + 1} | Speed: {speed_kmh:.2f} km/h")

    if shot_speeds:
        avg_speed = sum(shot_speeds) / len(shot_speeds)
        print(f"Average speed: {avg_speed:.2f} km/h")

    output_video_frames = tennis_ball_tracker.draw_bounding_boxes(video_frames, tennis_ball_detections)
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    blank_frames = [np.ones((600, 600, 3), dtype=np.uint8) * 255 for _ in video_frames]
    mini_court_frames = mini_court.draw_mini_court(blank_frames)
    mini_court_frames = mini_court.draw_points_on_mini_court(mini_court_frames, tennis_ball_mini_court_detections)

    for i, frame in enumerate(mini_court_frames):
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    output_dir = f"output_videos/{video_path}"
    os.makedirs(output_dir, exist_ok=True)
    save_video(output_video_frames, f"{output_dir}/{video_path}.avi")
    save_video(mini_court_frames, f"{output_dir}/mini_court_for_{video_path}.avi")

    create_heatmap(video_name=video_path, output_dir=output_dir)
    hits = detect_ball_hits(video_name=video_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python main.py <video_name>")
    else:
        main(sys.argv[1])
