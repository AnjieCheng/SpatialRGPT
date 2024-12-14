import random
import numpy as np
from itertools import combinations
import json

from osdsynth.processor.instruction_template import *
from osdsynth.processor.prompt_utils import *
from osdsynth.processor.pointcloud import human_like_distance, calculate_distances_between_point_clouds


def left_predicate(A, B):
    true_responses = left_true_responses
    false_responses = left_false_responses

    A_desc, A_cloud = A["caption"], A["pcd"]
    B_desc, B_cloud = B["caption"], B["pcd"]
    A_desc, B_desc = A_desc.lower(), B_desc.lower()

    A_pos = A_cloud.get_center()
    B_pos = B_cloud.get_center()

    is_left = A_pos[0] > B_pos[0]  # Compare X coordinates

    response_template = random.choice(true_responses if is_left else false_responses)
    answer = response_template.replace("[A]", A_desc).replace("[B]", B_desc)

    return answer


def below_predicate(A, B):
    true_responses = below_true_responses
    false_responses = below_false_responses

    A_desc, A_cloud = A["caption"], A["pcd"]
    B_desc, B_cloud = B["caption"], B["pcd"]
    A_desc, B_desc = A_desc.lower(), B_desc.lower()

    A_pos = A_cloud.get_center()
    B_pos = B_cloud.get_center()

    is_below = A_pos[1] < B_pos[1]

    response_template = random.choice(true_responses if is_below else false_responses)

    answer = response_template.replace("[A]", A_desc).replace("[B]", B_desc)

    return answer


def short_predicate(A, B):
    true_responses = short_true_responses
    false_responses = short_false_responses

    A_desc, A_cloud = A["caption"], A["pcd"]
    B_desc, B_cloud = B["caption"], B["pcd"]
    A_desc, B_desc = A_desc.lower(), B_desc.lower()

    height_A = A_cloud.get_axis_aligned_bounding_box().get_extent()[1]
    height_B = B_cloud.get_axis_aligned_bounding_box().get_extent()[1]

    is_shorter = height_A < height_B

    response_template = random.choice(true_responses if is_shorter else false_responses)

    answer = response_template.replace("[A]", A_desc).replace("[B]", B_desc)

    return answer


def thin_predicate(A, B):
    true_responses = thin_true_responses
    false_responses = thin_false_responses

    A_desc, A_cloud = A["caption"], A["pcd"]
    B_desc, B_cloud = B["caption"], B["pcd"]
    A_desc, B_desc = A_desc.lower(), B_desc.lower()

    width_A = A_cloud.get_axis_aligned_bounding_box().get_extent()[0]
    width_B = B_cloud.get_axis_aligned_bounding_box().get_extent()[0]

    is_thinner = width_A < width_B

    response_template = random.choice(true_responses if is_thinner else false_responses)

    answer = response_template.replace("[A]", A_desc).replace("[B]", B_desc)

    return answer


def small_predicate(A, B):
    true_responses = small_true_responses
    false_responses = small_false_responses

    A_desc, A_cloud = A["caption"], A["pcd"]
    B_desc, B_cloud = B["caption"], B["pcd"]
    A_desc, B_desc = A_desc.lower(), B_desc.lower()

    extent_A = A_cloud.get_axis_aligned_bounding_box().get_extent()
    volume_A = extent_A[0] * extent_A[1] * extent_A[2]

    extent_B = B_cloud.get_axis_aligned_bounding_box().get_extent()
    volume_B = extent_B[0] * extent_B[1] * extent_B[2]

    is_smaller = volume_A < volume_B

    response_template = random.choice(true_responses if is_smaller else false_responses)

    answer = response_template.replace("[A]", A_desc).replace("[B]", B_desc)

    return answer


def front_predicate(A, B):
    true_responses = front_true
    false_responses = front_false

    A_desc, A_cloud = A["caption"], A["pcd"]
    B_desc, B_cloud = B["caption"], B["pcd"]
    A_desc, B_desc = A_desc.lower(), B_desc.lower()

    # Calculate the minimum z-value for both A and B
    A_min_z = A_cloud.get_min_bound()[2]
    B_min_z = B_cloud.get_min_bound()[2]
    # Determine if A is behind B based on the minimum z-value
    is_in_front = A_min_z < B_min_z

    response_template = random.choice(true_responses if is_in_front else false_responses)

    answer = response_template.replace("[A]", A_desc).replace("[B]", B_desc)

    return answer


# Distance prompts


def generate_spatial_reasoning_data(A, B, human_readable_dist, template_answers):
    A_desc, B_desc = A["caption"].lower(), B["caption"].lower()

    answer_template = random.choice(template_answers)

    # Replace placeholders with actual values
    answer = answer_template.replace("[A]", A_desc).replace("[B]", B_desc).replace("[X]", human_readable_dist)

    # Add to the dataset
    return answer


def vertical_distance_data(A, B, use_center=True):
    template_answers = vertical_distance_answers

    # Get the bounding boxes for both A and B
    A_box = A["pcd"].get_axis_aligned_bounding_box()
    B_box = B["pcd"].get_axis_aligned_bounding_box()

    if use_center:
        A_center = A_box.get_axis_aligned_bounding_box().get_center()
        B_center = B_box.get_axis_aligned_bounding_box().get_center()
        vertical_distance = abs(A_center[1] - B_center[1])
    else:
        # Determine the highest and lowest points (in terms of y-value) of each object
        A_min_y, A_max_y = A_box.get_min_bound()[1], A_box.get_max_bound()[1]
        B_min_y, B_max_y = B_box.get_min_bound()[1], B_box.get_max_bound()[1]

        # Assuming A is above B, adjust if it's the other way around
        if A_min_y < B_min_y:
            # This means B is above A, swap the values
            A_min_y, A_max_y, B_min_y, B_max_y = B_min_y, B_max_y, A_min_y, A_max_y

        # The vertical distance is now the difference between the lowest point of the higher object (B_max_y)
        # and the highest point of the lower object (A_min_y), considering A is below B after the possible swap.
        vertical_distance = A_min_y - B_max_y if A_min_y > B_max_y else 0

    human_readable_dist = human_like_distance(vertical_distance)

    return generate_spatial_reasoning_data(A, B, human_readable_dist, template_answers)


def distance(A, B):
    template_answers = distance_template_answers
    distance = calculate_distances_between_point_clouds(A["pcd"], B["pcd"])
    return generate_spatial_reasoning_data(
        A,
        B,
        distance,
        template_answers,
    )


def horizontal_distance_data(A, B, use_center=True):
    template_answers = horizontal_distance_answers

    # Extract bounding boxes for A and B
    A_box = A["pcd"].get_axis_aligned_bounding_box()
    B_box = B["pcd"].get_axis_aligned_bounding_box()

    if use_center:
        A_center = A_box.get_center()
        B_center = B_box.get_center()
        horizontal_distance = np.sqrt((A_center[0] - B_center[0]) ** 2)
    else:
        # Extract min and max bounds for A and B on x and z axes
        A_min, A_max = A_box.get_min_bound(), A_box.get_max_bound()
        B_min, B_max = B_box.get_min_bound(), B_box.get_max_bound()

        # Calculate the shortest horizontal (x, z plane) distance between the two boxes
        horizontal_distance = max(A_min[0] - B_max[0], B_min[0] - A_max[0], 0)

    human_readable_dist = human_like_distance(horizontal_distance)

    return generate_spatial_reasoning_data(A, B, human_readable_dist, template_answers)


def width_data(A, B=None):
    A_desc = A["caption"].lower()

    template_answers = width_answers

    width = A["pcd"].get_axis_aligned_bounding_box().get_extent()[0]

    human_readable_width = human_like_distance(width)
    answer_template = random.choice(template_answers)

    answer = answer_template.replace("[A]", A_desc).replace("[X]", human_readable_width)

    return answer


def height_data(A, B=None):
    A_desc = A["caption"].lower()

    template_answers = height_answers

    height = A["pcd"].get_axis_aligned_bounding_box().get_extent()[1]

    human_readable_height = human_like_distance(height)
    answer_template = random.choice(template_answers)

    answer = answer_template.replace("[A]", A_desc).replace("[X]", human_readable_height)

    return answer


def direction(A, B):
    template_responses = direction_responses

    A_desc, A_cloud = A["caption"], A["pcd"]
    B_desc, B_cloud = B["caption"], B["pcd"]
    A_desc, B_desc = A_desc.lower(), B_desc.lower()

    A_pos = (A_cloud.get_center()[0], A_cloud.get_center()[2])  # Only x, z
    B_pos = (B_cloud.get_center()[0], B_cloud.get_center()[2])  # Only x, z

    clock_position = calculate_angle_clockwise(A_pos, B_pos)

    answer_template = random.choice(template_responses)

    answer = answer_template.replace("[X]", str(int(clock_position))).replace("[A]", A_desc).replace("[B]", B_desc)

    return answer


class PromptGenerator:
    def __init__(self, cfg, logger, device):
        """Initialize the class."""
        self.cfg = cfg
        self.logger = logger
        self.device = device
        self.vis = True

    def evaluate_predicates_on_pairs(self, detections):

        all_combinations = list(combinations(range(len(detections)), 2))
        random.shuffle(all_combinations)
        selected_combinations = all_combinations[:3]
        object_pairs = [(detections[i], detections[j]) for i, j in selected_combinations]

        all_prompt_variants = [
            # direction,
            left_predicate,
            thin_predicate,
            small_predicate,
            front_predicate,
            below_predicate,
            short_predicate,
            vertical_distance_data,
            horizontal_distance_data,
            width_data,
            height_data,
            distance,
        ]

        results = []

        for A, B in object_pairs:

            to_remove = set()  # A set to hold items to remove

            # Remove all items in `to_remove` from `all_prompt_variants`, if present
            all_prompt_variants = [item for item in all_prompt_variants if item not in to_remove]

            # selected_predicates_choices = all_prompt_variants
            selected_predicates_choices = random.sample(all_prompt_variants, 3)

            for prompt_func in selected_predicates_choices:
                results.append(prompt_func(A, B))

        return results
