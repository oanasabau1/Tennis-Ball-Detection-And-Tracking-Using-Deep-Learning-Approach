import cv2
import numpy as np
import constants
from bounding_boxes.bounding_boxes_utils import get_center_of_bounding_box
from utils import convert_meters_to_pixel_distance


class MiniCourt:
    def __init__(self, frame):
        self.drawing_rectangle_width = 250
        self.drawing_rectangle_height = 500
        self.padding_court = 20
        self.setup_canvas(600, 600)  # Standard frame size
        self.setup_court_dimensions()
        self.setup_court_keypoints()
        self.lines = [
            (0, 2), (4, 5), (6, 7), (1, 3),
            (0, 1), (8, 9), (10, 11), (2, 3)
        ]

    def setup_canvas(self, frame_width, frame_height):
        self.start_x = (frame_width - self.drawing_rectangle_width) // 2
        self.end_x = self.start_x + self.drawing_rectangle_width
        self.start_y = (frame_height - self.drawing_rectangle_height) // 2
        self.end_y = self.start_y + self.drawing_rectangle_height

    def setup_court_dimensions(self):
        self.court_start_x = self.start_x + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x
        self.court_drawing_height = self.court_end_y - self.court_start_y

    def convert_meters_to_pixels(self, meters):
        return convert_meters_to_pixel_distance(
            meters,
            constants.DOUBLE_LINE_WIDTH,
            self.court_drawing_width
        )

    def setup_court_keypoints(self):
        k = np.zeros(28, dtype=np.float32)
        k[0], k[1] = self.court_start_x, self.court_start_y
        k[2], k[3] = self.court_end_x, self.court_start_y
        k[4] = k[0]
        k[5] = self.court_start_y + self.convert_meters_to_pixels(constants.HALF_COURT_LINE_HEIGHT * 2)
        k[6], k[7] = k[2], k[5]
        k[8] = k[0] + self.convert_meters_to_pixels(constants.DOUBLE_ALLEY_WIDTH)
        k[9] = k[1]
        k[10], k[11] = k[8], k[5]
        k[12] = k[2] - self.convert_meters_to_pixels(constants.DOUBLE_ALLEY_WIDTH)
        k[13] = k[3]
        k[14], k[15] = k[12], k[7]
        service_line_height = self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        k[16], k[17] = k[8], k[9] + service_line_height
        k[18] = k[16] + self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        k[19] = k[17]
        k[20], k[21] = k[10], k[11] - service_line_height
        k[22], k[23] = k[20] + self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH), k[
            21]
        k[24], k[25] = (k[16] + k[18]) // 2, k[17]
        k[26], k[27] = (k[20] + k[22]) // 2, k[21]

        self.keypoints = k

    def draw_mini_court(self, frames):
        return [self._draw_court_on_frame(frame) for frame in frames]

    def _draw_court_on_frame(self, frame):
        frame = self._draw_background(frame)

        for i in range(0, len(self.keypoints), 2):
            x, y = int(self.keypoints[i]), int(self.keypoints[i + 1])
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

        for line in self.lines:
            pt1 = (int(self.keypoints[line[0] * 2]), int(self.keypoints[line[0] * 2 + 1]))
            pt2 = (int(self.keypoints[line[1] * 2]), int(self.keypoints[line[1] * 2 + 1]))
            cv2.line(frame, pt1, pt2, (0, 0, 0), 2)

        left_x = int((self.keypoints[8] + self.keypoints[10]) / 2)
        right_x = int((self.keypoints[12] + self.keypoints[14]) / 2)
        y_top, y_bottom = int(self.keypoints[1]), int(self.keypoints[5])

        cv2.line(frame, (left_x, y_top), (left_x, y_bottom), (0, 0, 0), 2)
        cv2.line(frame, (right_x, y_top), (right_x, y_bottom), (0, 0, 0), 2)

        service_top_y = int(self.keypoints[17])
        service_bottom_y = int(self.keypoints[21])
        center_x = int((self.keypoints[8] + self.keypoints[12]) / 2)
        cv2.line(frame, (center_x, service_top_y), (center_x, service_bottom_y), (0, 0, 0), 2)

        net_y = int((self.keypoints[1] + self.keypoints[5]) / 2)
        cv2.line(frame, (int(self.keypoints[0]), net_y), (int(self.keypoints[2]), net_y), (255, 0, 0), 2)

        return frame

    def _draw_background(self, frame):
        shapes = np.zeros_like(frame, np.uint8)
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), -1)
        return cv2.addWeighted(frame, 0.5, shapes, 0.5, 0)

    def draw_points_on_mini_court(self, frames, positions, color=(0, 255, 0)):
        result = frames.copy()
        for i, frame in enumerate(result):
            for _, (x, y) in positions[i].items():
                if not (np.isnan(x) or np.isnan(y)):
                    cv2.circle(frame, (int(x), int(y)), 5, color, -1)
        return result

    def convert_bounding_boxes_to_mini_court_coordinates(self, ball_boxes, original_court_keypoints):
        real_x1 = original_court_keypoints[0]
        real_x2 = original_court_keypoints[2]
        real_y1 = original_court_keypoints[1]
        real_y2 = original_court_keypoints[5]

        real_width = abs(real_x2 - real_x1)
        real_height = abs(real_y2 - real_y1)

        output = []
        for ball_box_dict in ball_boxes:
            ball_box = ball_box_dict.get(1)
            if ball_box is None or len(ball_box) != 4:
                output.append({})
                continue

            ball_x, ball_y = get_center_of_bounding_box(ball_box)

            norm_x = (ball_x - real_x1) / real_width if real_width else 0
            norm_y = (ball_y - real_y1) / real_height if real_height else 0

            mini_x = self.court_start_x + norm_x * self.court_drawing_width
            mini_y = self.court_start_y + norm_y * self.court_drawing_height

            output.append({1: (mini_x, mini_y)})

        return output

    def get_width_of_mini_court(self):
        return self.court_drawing_width