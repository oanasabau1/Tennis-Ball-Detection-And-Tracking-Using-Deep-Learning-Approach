import cv2
import numpy as np
import constants
from bounding_boxes.bounding_boxes_utils import get_center_of_bounding_box
from utils import convert_meters_to_pixel_distance


class MiniCourt:
    def __init__(self, frame):
        # Initialize court dimensions
        self.drawing_rectangle_width = 250
        self.drawing_rectangle_height = 500
        self.padding_court = 20

        # Set up canvas and court dimensions
        self.setup_canvas(600, 600)  # Standard frame size
        self.setup_court_dimensions()
        self.setup_court_keypoints()

        # Define court line connections
        self.lines = [
            (0, 2), (4, 5), (6, 7), (1, 3),
            (0, 1), (8, 9), (10, 11), (2, 3)
        ]

    def setup_canvas(self, frame_width, frame_height):
        """Set up the canvas dimensions for the mini court"""
        self.start_x = (frame_width - self.drawing_rectangle_width) // 2
        self.end_x = self.start_x + self.drawing_rectangle_width
        self.start_y = (frame_height - self.drawing_rectangle_height) // 2
        self.end_y = self.start_y + self.drawing_rectangle_height

    def setup_court_dimensions(self):
        """Calculate the actual court area within the canvas"""
        self.court_start_x = self.start_x + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x
        self.court_drawing_height = self.court_end_y - self.court_start_y

    def convert_meters_to_pixels(self, meters):
        """Convert real-world meters to pixels on the mini court"""
        return convert_meters_to_pixel_distance(
            meters,
            constants.DOUBLE_LINE_WIDTH,
            self.court_drawing_width
        )

    def setup_court_keypoints(self):
        """Calculate all court keypoints based on tennis court dimensions"""
        # Initialize keypoints array
        k = np.zeros(28, dtype=np.float32)

        # Base court corners
        k[0], k[1] = self.court_start_x, self.court_start_y  # Top left
        k[2], k[3] = self.court_end_x, self.court_start_y  # Top right

        # Bottom corners
        k[4] = k[0]  # Bottom left x
        k[5] = self.court_start_y + self.convert_meters_to_pixels(constants.HALF_COURT_LINE_HEIGHT * 2)  # Bottom left y
        k[6], k[7] = k[2], k[5]  # Bottom right

        # Double sidelines
        k[8] = k[0] + self.convert_meters_to_pixels(constants.DOUBLE_ALLEY_WIDTH)  # Top left inner x
        k[9] = k[1]  # Top left inner y
        k[10], k[11] = k[8], k[5]  # Bottom left inner
        k[12] = k[2] - self.convert_meters_to_pixels(constants.DOUBLE_ALLEY_WIDTH)  # Top right inner x
        k[13] = k[3]  # Top right inner y
        k[14], k[15] = k[12], k[7]  # Bottom right inner

        # Service line points
        service_line_height = self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        k[16], k[17] = k[8], k[9] + service_line_height  # Service line left outer
        k[18] = k[16] + self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)  # Service line left inner x
        k[19] = k[17]  # Service line left inner y
        k[20], k[21] = k[10], k[11] - service_line_height  # Service line bottom left
        k[22], k[23] = k[20] + self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH), k[
            21]  # Service line bottom right

        # Center service line points
        k[24], k[25] = (k[16] + k[18]) // 2, k[17]  # Center mark top
        k[26], k[27] = (k[20] + k[22]) // 2, k[21]  # Center mark bottom

        self.keypoints = k

    def draw_mini_court(self, frames):
        """Draw the entire mini court on all frames"""
        return [self._draw_court_on_frame(frame) for frame in frames]

    def _draw_court_on_frame(self, frame):
        """Draw court on a single frame"""
        # Create background
        frame = self._draw_background(frame)

        # Draw keypoints
        for i in range(0, len(self.keypoints), 2):
            x, y = int(self.keypoints[i]), int(self.keypoints[i + 1])
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

        # Draw main court lines
        for line in self.lines:
            pt1 = (int(self.keypoints[line[0] * 2]), int(self.keypoints[line[0] * 2 + 1]))
            pt2 = (int(self.keypoints[line[1] * 2]), int(self.keypoints[line[1] * 2 + 1]))
            cv2.line(frame, pt1, pt2, (0, 0, 0), 2)

        # Draw center lines
        left_x = int((self.keypoints[8] + self.keypoints[10]) / 2)
        right_x = int((self.keypoints[12] + self.keypoints[14]) / 2)
        y_top, y_bottom = int(self.keypoints[1]), int(self.keypoints[5])

        cv2.line(frame, (left_x, y_top), (left_x, y_bottom), (0, 0, 0), 2)
        cv2.line(frame, (right_x, y_top), (right_x, y_bottom), (0, 0, 0), 2)

        # Draw service center line
        service_top_y = int(self.keypoints[17])
        service_bottom_y = int(self.keypoints[21])
        center_x = int((self.keypoints[8] + self.keypoints[12]) / 2)
        cv2.line(frame, (center_x, service_top_y), (center_x, service_bottom_y), (0, 0, 0), 2)

        # Draw net
        net_y = int((self.keypoints[1] + self.keypoints[5]) / 2)
        cv2.line(frame, (int(self.keypoints[0]), net_y), (int(self.keypoints[2]), net_y), (255, 0, 0), 2)

        return frame

    def _draw_background(self, frame):
        """Draw the white background rectangle for the court"""
        shapes = np.zeros_like(frame, np.uint8)
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), -1)
        return cv2.addWeighted(frame, 0.5, shapes, 0.5, 0)

    def draw_points_on_mini_court(self, frames, positions, color=(0, 255, 0)):
        """Draw ball positions on all frames"""
        result = frames.copy()
        for i, frame in enumerate(result):
            for _, (x, y) in positions[i].items():
                if not (np.isnan(x) or np.isnan(y)):
                    cv2.circle(frame, (int(x), int(y)), 5, color, -1)
        return result

    def convert_bounding_boxes_to_mini_court_coordinates(self, ball_boxes, original_court_keypoints):
        """Convert ball positions from the real court to mini court coordinates"""
        # Extract real court bounds
        real_x1 = original_court_keypoints[0]
        real_x2 = original_court_keypoints[2]
        real_y1 = original_court_keypoints[1]
        real_y2 = original_court_keypoints[5]

        real_width = abs(real_x2 - real_x1)
        real_height = abs(real_y2 - real_y1)

        # Process each ball position
        output = []
        for ball_box_dict in ball_boxes:
            ball_box = ball_box_dict.get(1)
            if ball_box is None or len(ball_box) != 4:
                output.append({})
                continue

            # Calculate center of bounding box
            ball_x, ball_y = get_center_of_bounding_box(ball_box)

            # Normalize coordinates
            norm_x = (ball_x - real_x1) / real_width if real_width else 0
            norm_y = (ball_y - real_y1) / real_height if real_height else 0

            # Convert to mini court coordinates
            mini_x = self.court_start_x + norm_x * self.court_drawing_width
            mini_y = self.court_start_y + norm_y * self.court_drawing_height

            output.append({1: (mini_x, mini_y)})

        return output

    def get_width_of_mini_court(self):
        """Get the width of the miniature court drawing area"""
        return self.court_drawing_width