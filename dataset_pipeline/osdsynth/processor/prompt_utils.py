import math
import random
import string

import numpy as np


def generate_random_string(length):
    letters = string.ascii_letters + string.digits
    return "".join(random.choice(letters) for _ in range(length))


def calculate_angle_clockwise(A_pos, B_pos, x_right=False):
    # Vector from A to B
    if x_right:
        vector_A_to_B = (A_pos[0] - B_pos[0], B_pos[1] - A_pos[1])
    else:
        vector_A_to_B = (B_pos[0] - A_pos[0], B_pos[1] - A_pos[1])

    # Angle of this vector w.r.t. positive z-axis
    angle_rad = math.atan2(vector_A_to_B[0], vector_A_to_B[1])  # atan2 handles all quadrants
    angle_deg = math.degrees(angle_rad)

    # Convert angle to clock position, 360 degrees => 12 hours, so 1 hour = 30 degrees
    # We adjust the angle to be positive and then calculate the clock position
    angle_deg = (angle_deg + 360) % 360
    clock_position = 12 - angle_deg // 30
    clock_position = clock_position if clock_position > 0 else 12 + clock_position

    return clock_position


def is_aligned_vertically(A, B):
    # Convert Open3D point cloud to NumPy array
    A_points = np.asarray(A["pcd"].points)
    B_points = np.asarray(B["pcd"].points)

    # Calculate vertical (y) extents for A
    A_min, A_max = np.min(A_points[:, 1]), np.max(A_points[:, 1])
    # Calculate vertical (y) extents for B
    B_min, B_max = np.min(B_points[:, 1]), np.max(B_points[:, 1])

    # Determine vertical overlap
    overlap = max(0, min(A_max, B_max) - max(A_min, B_min))
    A_overlap_percentage = overlap / (A_max - A_min) if A_max != A_min else 0
    B_overlap_percentage = overlap / (B_max - B_min) if B_max != B_min else 0

    # Return True if both overlaps are greater than 50%
    return A_overlap_percentage > 0.5 and B_overlap_percentage > 0.5


def is_aligned_horizontally(A, B):
    # The high level logic is to check if the x-axis of one object is fully contained by the x-axis of the other object

    # Extract the bounding boxes for both A and B
    A_box = A["pcd"].get_axis_aligned_bounding_box()
    B_box = B["pcd"].get_axis_aligned_bounding_box()

    # Get the min and max x-axis values for both A and B
    A_min_x, A_max_x = A_box.get_min_bound()[0], A_box.get_max_bound()[0]
    B_min_x, B_max_x = B_box.get_min_bound()[0], B_box.get_max_bound()[0]

    # Check if A and B are almost the same size on the x-axis
    A_width, B_width = A_max_x - A_min_x, B_max_x - B_min_x
    is_almost_same_size = max(A_width, B_width) / min(A_width, B_width) <= 1.5
    if not is_almost_same_size:
        return False

    overlap_min, overlap_max = max(A_min_x, B_min_x), min(A_max_x, B_max_x)
    overlap_width = max(0, overlap_max - overlap_min)
    overlap_percent = max(overlap_width / A_width, overlap_width / B_width)

    return overlap_percent > 0.95


def is_y_axis_overlapped(A, B):
    # Extract the y-axis values (height) of the bounding boxes
    A_box = A["pcd"].get_axis_aligned_bounding_box()
    B_box = B["pcd"].get_axis_aligned_bounding_box()

    # Get the min and max y-axis values for both A and B
    A_min_y, A_max_y = A_box.get_min_bound()[1], A_box.get_max_bound()[1]
    B_min_y, B_max_y = B_box.get_min_bound()[1], B_box.get_max_bound()[1]

    # Check if there's any overlap in the y-axis values
    # There are four possible scenarios for overlap, but we can check them with a simpler logic:
    # If one box's minimum is between the other's min and max, or one box's max is.
    overlap = (A_min_y <= B_max_y and A_max_y >= B_min_y) or (B_min_y <= A_max_y and B_max_y >= A_min_y)

    return overlap


def is_supporting(A, B):
    # Extract bounding boxes
    A_box = A["pcd"].get_axis_aligned_bounding_box()
    B_box = B["pcd"].get_axis_aligned_bounding_box()

    # Get the corners of the bounding boxes
    A_min, A_max = A_box.get_min_bound(), A_box.get_max_bound()
    B_min, B_max = B_box.get_min_bound(), B_box.get_max_bound()

    # Check vertical contact:
    # The bottom of the upper object is at or above the top of the lower object
    vertical_contact = (A_min[2] <= B_max[2] and A_min[2] >= B_min[2]) or (
        B_min[2] <= A_max[2] and B_min[2] >= A_min[2]
    )

    if not vertical_contact:
        # If there's no vertical contact, they are not supporting each other
        return False

    # Determine which object is on top and which is on bottom
    if A_min[2] < B_min[2]:
        top, bottom = B, A
        top_min, top_max = B_min, B_max
        bottom_min, bottom_max = A_min, A_max
    else:
        top, bottom = A, B
        top_min, top_max = A_min, A_max
        bottom_min, bottom_max = B_min, B_max

    # Check horizontal coverage:
    # The larger (top) object's bounding box completely covers the smaller (bottom) object's bounding box
    horizontal_coverage = (
        top_min[0] <= bottom_min[0]
        and top_max[0] >= bottom_max[0]
        and top_min[1] <= bottom_min[1]
        and top_max[1] >= bottom_max[1]
    )

    return horizontal_coverage
