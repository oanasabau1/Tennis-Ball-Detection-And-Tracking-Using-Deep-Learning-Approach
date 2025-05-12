import constants
from process_video import read_video, save_video
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
from trackers import TennisBallTracker
import cv2
import numpy as np
from bounding_boxes import measure_distance
from utils import convert_pixel_distance_to_meters


def main(video_path):
    input_video_path = f"input_videos/{video_path}.mp4"
    video_frames = read_video(input_video_path)

    # Track tennis ball
    tennis_ball_tracker = TennisBallTracker(model_path="D:\\tennis_thesis\\runs\\detect\\tennis_ball_detection_yolov5m.pt\\weights\\best.pt")
    tennis_ball_detections = tennis_ball_tracker.detect_frames(video_frames, read_from_stub=False,
                                                                stub_path="tracker_stub/tennis_ball_detections.pkl")
    tennis_ball_detections = tennis_ball_tracker.interpolate_tennis_ball_positions(tennis_ball_detections)

    # Detect court keypoints
    court_line_detector = CourtLineDetector(model_path="models/tennis_court_keypoints_model.pth")
    court_keypoints = court_line_detector.predict(video_frames[0])

    # Set up mini court
    mini_court = MiniCourt(video_frames[0])

    # Detect shot events
    tennis_ball_shot_frames = tennis_ball_tracker.get_tennis_ball_shot_frames(tennis_ball_detections)

    # Convert ball positions to mini court coordinates
    tennis_ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
        tennis_ball_detections,
        court_keypoints)

    # Track only shot speeds
    shot_speeds = []

    for tennis_ball_shot_index in range(len(tennis_ball_shot_frames) - 1):
        start_frame = tennis_ball_shot_frames[tennis_ball_shot_index]
        end_frame = tennis_ball_shot_frames[tennis_ball_shot_index + 1]
        tennis_ball_shot_time_in_seconds = (end_frame - start_frame) / 24

        start_pos = tennis_ball_mini_court_detections[start_frame].get(1)
        end_pos = tennis_ball_mini_court_detections[end_frame].get(1)

        if start_pos is None or end_pos is None:
            continue

        distance_covered_by_ball_in_pixels = measure_distance(start_pos, end_pos)
        distance_covered_by_ball_in_meters = convert_pixel_distance_to_meters(
            distance_covered_by_ball_in_pixels,
            constants.DOUBLE_LINE_WIDTH,
            mini_court.get_width_of_mini_court()
        )
        speed_of_the_ball_shot = distance_covered_by_ball_in_meters / tennis_ball_shot_time_in_seconds * 3.6
        shot_speeds.append(speed_of_the_ball_shot)

        print(f"Shot {tennis_ball_shot_index + 1} | Speed: {speed_of_the_ball_shot:.2f} km/h")

    # Display average shot speed
    if shot_speeds:
        avg_speed = sum(shot_speeds) / len(shot_speeds)
        print(f"\n=== Shot Stats ===")
        print(f"Number of shots: {len(shot_speeds)}")
        print(f"Average speed: {avg_speed:.2f} km/h")
        print(f"Max speed: {max(shot_speeds):.2f} km/h")

    # Draw output videos
    output_video_frames = tennis_ball_tracker.draw_bounding_boxes(video_frames, tennis_ball_detections)
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    blank_frames = [np.ones((600, 600, 3), dtype=np.uint8) * 255 for _ in video_frames]
    mini_court_frames = mini_court.draw_mini_court(blank_frames)
    mini_court_frames = mini_court.draw_points_on_mini_court(mini_court_frames, tennis_ball_mini_court_detections)

    for i, frame in enumerate(mini_court_frames):
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    save_video(output_video_frames, f"output_videos/{video_path}.avi")
    save_video(mini_court_frames, f"output_videos/mini_court_for_{video_path}.avi")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python main.py <video_name>")
    else:
        video_name = sys.argv[1]
        main(video_name)