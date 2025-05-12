import constants
from process_video import read_video, save_video
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
from trackers import TennisBallTracker
import cv2
import numpy as np
from bounding_boxes import measure_distance
from utils import convert_pixel_distance_to_meters


def main():
    input_video_path = "input_videos/input_video.mp4"
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

    # Initialize player stats (using ball position to guess who hit)
    player_stats = {
        1: {'shots': 0, 'total_speed': 0},
        2: {'shots': 0, 'total_speed': 0}
    }

    court_center_y = (mini_court.court_start_y + mini_court.court_end_y) / 2

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

        # Estimate which player hit the shot based on y-position
        if start_pos[1] > court_center_y:
            player_who_hit = 1  # bottom
        else:
            player_who_hit = 2  # top

        player_stats[player_who_hit]['shots'] += 1
        player_stats[player_who_hit]['total_speed'] += speed_of_the_ball_shot

        print(f"Shot {tennis_ball_shot_index + 1} | Player {player_who_hit} | Speed: {speed_of_the_ball_shot:.2f} km/h")

    # Display final stats
    print("\n=== Player Shot Stats (Estimated) ===")
    for pid in [1, 2]:
        shots = player_stats[pid]['shots']
        avg_speed = player_stats[pid]['total_speed'] / shots if shots > 0 else 0
        print(f"Player {pid}: {shots} shots, average speed: {avg_speed:.2f} km/h")

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

    save_video(output_video_frames, "output_videos/output_video.avi")
    save_video(mini_court_frames, "output_videos/mini_court_video.avi")


if __name__ == "__main__":
    main()
