import cv2
import sys
import numpy as np
import constants
from bounding_boxes.bounding_boxes_utils import get_center_of_bounding_box, get_closest_keypoint_index
from utils import convert_meters_to_pixel_distance

sys.path.append("../")


class MiniCourt:
    def __init__(self, frame):
        self.drawing_rectangle_width = 250
        self.drawing_rectangle_height = 500
        self.buffer = 50
        self.padding_court = 20
        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_keypoints()
        self.set_court_lines()

    def convert_meters_to_pixels(self, meters):
        return convert_meters_to_pixel_distance(
            meters,
            constants.DOUBLE_LINE_WIDTH,
            self.court_drawing_width
        )

    def set_canvas_background_box_position(self, frame):
        frame_height, frame_width = 600, 600
        self.start_x = (frame_width - self.drawing_rectangle_width) // 2
        self.end_x = self.start_x + self.drawing_rectangle_width
        self.start_y = (frame_height - self.drawing_rectangle_height) // 2
        self.end_y = self.start_y + self.drawing_rectangle_height

    def set_mini_court_position(self):
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x

    def set_court_drawing_keypoints(self):
        k = [0] * 28
        k[0], k[1] = int(self.court_start_x), int(self.court_start_y)
        k[2], k[3] = int(self.court_end_x), int(self.court_start_y)
        k[4] = int(self.court_start_x)
        k[5] = self.court_start_y + self.convert_meters_to_pixels(constants.HALF_COURT_LINE_HEIGHT * 2)
        k[6] = k[0] + self.court_drawing_width
        k[7] = k[5]
        k[8] = k[0] + self.convert_meters_to_pixels(constants.DOUBLE_ALLEY_WIDTH)
        k[9] = k[1]
        k[10] = k[4] + self.convert_meters_to_pixels(constants.DOUBLE_ALLEY_WIDTH)
        k[11] = k[5]
        k[12] = k[2] - self.convert_meters_to_pixels(constants.DOUBLE_ALLEY_WIDTH)
        k[13] = k[3]
        k[14] = k[6] - self.convert_meters_to_pixels(constants.DOUBLE_ALLEY_WIDTH)
        k[15] = k[7]
        k[16] = k[8]
        k[17] = k[9] + self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        k[18] = k[16] + self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        k[19] = k[17]
        k[20] = k[10]
        k[21] = k[11] - self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        k[22] = k[20] + self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        k[23] = k[21]
        k[24] = int((k[16] + k[18]) / 2)
        k[25] = k[17]
        k[26] = int((k[20] + k[22]) / 2)
        k[27] = k[21]
        self.keypoints = k

    def set_court_lines(self):
        self.lines = [
            (0, 2), (4, 5), (6, 7), (1, 3),
            (0, 1), (8, 9), (10, 11), (2, 3)
        ]

    def draw_background_rectangle(self, frame):
        shapes = np.zeros_like(frame, np.uint8)
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), -1)
        output = frame.copy()
        alpha = 0.5
        mask = shapes.astype(bool)
        output[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]
        return output

    def draw_court(self, frame):
        for i in range(0, len(self.keypoints), 2):
            x, y = int(self.keypoints[i]), int(self.keypoints[i + 1])
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        for line in self.lines:
            start = (int(self.keypoints[line[0] * 2]), int(self.keypoints[line[0] * 2 + 1]))
            end = (int(self.keypoints[line[1] * 2]), int(self.keypoints[line[1] * 2 + 1]))
            cv2.line(frame, start, end, (0, 0, 0), 2)

        net_y = int((self.keypoints[1] + self.keypoints[5]) / 2)
        cv2.line(frame, (self.keypoints[0], net_y), (self.keypoints[2], net_y), (255, 0, 0), 2)
        return frame

    def draw_mini_court(self, frames):
        return [self.draw_court(self.draw_background_rectangle(f)) for f in frames]

    def get_court_drawing_keypoints(self):
        return self.keypoints

    def get_start_point_of_mini_court(self):
        return self.start_x, self.start_y

    def get_width_of_mini_court(self):
        return self.drawing_rectangle_width


    def convert_bounding_boxes_to_mini_court_coordinates(self, ball_boxes, original_court_keypoints):
        output_ball_boxes = []

        real_court_x1 = original_court_keypoints[0]
        real_court_x2 = original_court_keypoints[2]
        real_court_y1 = original_court_keypoints[1]
        real_court_y2 = original_court_keypoints[5]  # bottom of half court

        real_court_width = abs(real_court_x2 - real_court_x1)
        real_court_height = abs(real_court_y2 - real_court_y1)

        for frame_num, ball_box_dict in enumerate(ball_boxes):
            ball_box = ball_box_dict.get(1)
            if ball_box is None:
                output_ball_boxes.append({})
                continue

            ball_x, ball_y = get_center_of_bounding_box(ball_box)

            closest_kp_idx = get_closest_keypoint_index((ball_x, ball_y), original_court_keypoints, [0, 2, 12, 13])
            closest_kp = (
                original_court_keypoints[closest_kp_idx * 2],
                original_court_keypoints[closest_kp_idx * 2 + 1]
            )
            mini_kp = (
                self.keypoints[closest_kp_idx * 2],
                self.keypoints[closest_kp_idx * 2 + 1]
            )

            dx = (ball_x - closest_kp[0]) / real_court_width * self.court_drawing_width
            dy = (ball_y - closest_kp[1]) / real_court_height * (self.court_end_y - self.court_start_y)

            mini_court_pos = (mini_kp[0] + dx, mini_kp[1] + dy)
            output_ball_boxes.append({1: mini_court_pos})

        return output_ball_boxes

    def draw_points_on_mini_court(self, frames, positions, color=(0, 255, 0)):
        for frame_num, frame in enumerate(frames):
            for _, (x, y) in positions[frame_num].items():
                cv2.circle(frame, (int(x), int(y)), 5, color, -1)
        return frames
