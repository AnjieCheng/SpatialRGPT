import random
from collections import Counter

import cv2
import matplotlib
import numpy as np
import open3d as o3d
import torch
from osdsynth.processor.wrappers.metric3d_v2 import get_depth_model, inference_depth
from osdsynth.processor.wrappers.perspective_fields import (
    create_rotation_matrix,
    get_perspective_fields_model,
    run_perspective_fields_model,
)
from PIL import Image
from scipy.spatial.transform import Rotation
from wis3d import Wis3D


class PointCloudReconstruction:
    """Class to reconstruct point cloud from depth maps."""

    def __init__(self, cfg, logger, device, init_models=True):
        """Initialize the class."""
        self.cfg = cfg
        self.logger = logger
        self.device = device
        self.vis = self.cfg.vis

        if init_models:
            # Initialize the perspective_fields_model
            if self.cfg.perspective_model_variant == "perspective_fields":
                print(f"Using Perspective Fields")
                self.perspective_fields_model = get_perspective_fields_model(cfg, device)
            elif self.cfg.perspective_model_variant == "geo_calib":
                from geocalib import GeoCalib

                print(f"Using Geo Calib")
                self.perspective_fields_model = GeoCalib(weights="distorted").to(device)
            else:
                raise ValueError(f"perspective_model_variant: {self.cfg.perspective_model_variant} not implemented")

            # Initialize the Camera Intrinsics Model
            self.wilde_camera_model = torch.hub.load("ShngJZ/WildCamera", "WildCamera", pretrained=True).to(device)

            # Initialize the Metric3D_v2
            self.depth_model = get_depth_model(device)
        else:
            self.perspective_fields_model = self.wilde_camera_model = self.depth_model = None

    def process(self, filename, image_bgr, detections_list):
        """Reconstruct point cloud from depth map."""

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb_pil = Image.fromarray(image_rgb)

        if self.cfg.perspective_model_variant == "perspective_fields":
            # Run Perspective Fields, this returns the pitch, roll
            (
                vis_perspective_fields,
                perspective_fields,
            ) = run_perspective_fields_model(self.perspective_fields_model, image_bgr)
            roll, pitch = perspective_fields["roll"], perspective_fields["pitch"]

        elif self.cfg.perspective_model_variant == "geo_calib":
            from geocalib.utils import rad2deg

            # load image as tensor in range [0, 1] with shape [C, H, W]
            image_geo = torch.tensor((image_rgb.transpose((2, 0, 1))) / 255.0, dtype=torch.float).to(self.device)
            geo_results = self.perspective_fields_model.calibrate(image_geo, camera_model="simple_radial")
            roll, pitch = rad2deg(geo_results["gravity"].rp).unbind(-1)
            roll, pitch = roll.item(), pitch.item()

        # Perspective to Rotation Matrix
        perspective_R = create_rotation_matrix(
            roll=roll,
            pitch=pitch,
            yaw=0,
            degrees=True,
        )

        # Infer camera intrinsics
        intrinsic, _ = self.wilde_camera_model.inference(image_rgb_pil, wtassumption=False)

        # Infer depth
        metric_depth = inference_depth(image_rgb, intrinsic, self.depth_model)

        # Depth to points
        pts3d = depth_to_points(metric_depth[None], intrinsic=intrinsic)
        cano_pts3d = depth_to_points(metric_depth[None], R=perspective_R.T, intrinsic=intrinsic)

        # Translate the points to ground
        cano_pts3d_flattened = cano_pts3d.reshape(-1, 3)
        sorted_flattened_points = cano_pts3d_flattened[cano_pts3d_flattened[:, 2].argsort()]
        fifty_percent_index = int(sorted_flattened_points.shape[0] * 0.5)
        selected_nearest_points = sorted_flattened_points[:fifty_percent_index]
        min_y = np.min(selected_nearest_points[:, 1])
        cano_pts3d[:, :, 1] -= min_y

        if self.vis:
            wis3d = Wis3D(self.cfg.wis3d_folder, filename)
            wis3d.add_point_cloud(vertices=pts3d.reshape((-1, 3)), colors=image_rgb.reshape(-1, 3), name="pts3d")
            wis3d.add_point_cloud(
                vertices=cano_pts3d.reshape((-1, 3)), colors=image_rgb.reshape(-1, 3), name="cano_pts3d"
            )

        ### Create object points ###
        n_objects = len(detections_list)

        for obj_idx in range(n_objects):

            mask = detections_list[obj_idx]["subtracted_mask"]
            class_name = detections_list[obj_idx]["class_name"]

            object_pcd = create_object_pcd(cano_pts3d, image_rgb, mask)

            # The object should at least contains 5 points
            if len(object_pcd.points) < max(self.cfg.min_points_threshold, 5):
                print("camera_object_pcd points less than threshold, skip this detection")
                continue

            object_pcd = process_pcd(self.cfg, object_pcd)

            if len(object_pcd.points) < self.cfg.min_points_threshold_after_denoise:
                print(f"{class_name} pcd_bbox too less points ({len(object_pcd.points)}), skip this detection")
                continue

            axis_aligned_bbox, oriented_bbox = get_bounding_box(self.cfg, object_pcd)

            if axis_aligned_bbox.volume() < 1e-6:
                print(f"{class_name} pcd_bbox got small volume, skip this detection")
                continue

            detections_list[obj_idx]["pcd"] = object_pcd
            detections_list[obj_idx]["axis_aligned_bbox"] = axis_aligned_bbox
            detections_list[obj_idx]["oriented_bbox"] = oriented_bbox

        # Filter detections to include only those with a 'pcd' key
        filtered_detections = [det for det in detections_list if "pcd" in det]

        instance_colored_pcds = color_by_instance([det["pcd"] for det in filtered_detections])
        axis_aligned_bbox = [det["axis_aligned_bbox"] for det in filtered_detections]
        oriented_bboxes = [det["oriented_bbox"] for det in filtered_detections]

        if self.vis:
            obj_id = 0
            for obj_pcd, obj_aa_box, obj_or_box in zip(instance_colored_pcds, axis_aligned_bbox, oriented_bboxes):
                class_name = filtered_detections[obj_id]["class_name"]
                pcd_points = np.asarray(obj_pcd.points)
                pcd_colors = np.asarray(obj_pcd.colors)

                # Convert bbox to center, euler, extent
                aa_center, aa_eulers, aa_extent = axis_aligned_bbox_to_center_euler_extent(
                    obj_aa_box.get_min_bound(), obj_aa_box.get_max_bound()
                )
                or_center, or_eulers, or_extent = oriented_bbox_to_center_euler_extent(
                    obj_or_box.center, obj_or_box.R, obj_or_box.extent
                )

                wis3d.add_point_cloud(vertices=pcd_points, colors=pcd_colors, name=f"{obj_id:02d}_{class_name}")
                wis3d.add_boxes(
                    positions=aa_center, eulers=aa_eulers, extents=aa_extent, name=f"{obj_id:02d}_{class_name}_aa_bbox"
                )
                # wis3d.add_boxes(positions=or_center, eulers=or_eulers, extents=or_extent, name=f"{obj_id:02d}_{class_name}_or_bbox")
                obj_id += 1

        return filtered_detections


def depth_to_points(depth, R=None, t=None, fov=None, intrinsic=None):
    K = intrinsic
    Kinv = np.linalg.inv(K)
    if R is None:
        R = np.eye(3)
    if t is None:
        t = np.zeros(3)

    # M converts from your coordinate to PyTorch3D's coordinate system
    M = np.eye(3)

    height, width = depth.shape[1:3]

    x = np.arange(width)
    y = np.arange(height)
    coord = np.stack(np.meshgrid(x, y), -1)
    coord = np.concatenate((coord, np.ones_like(coord)[:, :, [0]]), -1)  # z=1
    coord = coord.astype(np.float32)
    coord = coord[None]  # bs, h, w, 3

    D = depth[:, :, :, None, None]
    pts3D_1 = D * Kinv[None, None, None, ...] @ coord[:, :, :, :, None]
    # pts3D_1 live in your coordinate system. Convert them to Py3D's
    pts3D_1 = M[None, None, None, ...] @ pts3D_1
    # from reference to targe tviewpoint
    pts3D_2 = R[None, None, None, ...] @ pts3D_1 + t[None, None, None, :, None]

    # G converts from your coordinate to PyTorch3D's coordinate system
    G = np.eye(3)
    G[0, 0] = -1.0
    G[1, 1] = -1.0

    return pts3D_2[:, :, :, :3, 0][0] @ G.T


def create_object_pcd(image_points, image_rgb, mask):
    points = image_points[mask]
    colors = image_rgb[mask] / 255.0

    # Perturb the points a bit to avoid colinearity
    points += np.random.normal(0, 4e-3, points.shape)

    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def get_bounding_box(cfg, pcd):
    axis_aligned_bbox = pcd.get_axis_aligned_bounding_box()

    try:
        oriented_bbox = pcd.get_oriented_bounding_box(robust=True)
    except RuntimeError as e:
        print(f"Met {e}.")
        oriented_bbox = None

    return axis_aligned_bbox, oriented_bbox


def points_to_pcd(points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.reshape(-1, 3))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors.reshape(-1, 3))
    return pcd


def process_pcd(cfg, pcd, run_dbscan=True):
    scale = np.linalg.norm(np.asarray(pcd.points).std(axis=0)) * 3.0 + 1e-6
    [pcd, _] = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=1.2)
    pcd = pcd.voxel_down_sample(voxel_size=max(0.01, scale / 40))

    if cfg.dbscan_remove_noise and run_dbscan:
        # print("Before dbscan:", len(pcd.points))
        pcd = pcd_denoise_dbscan(pcd, eps=cfg.dbscan_eps, min_points=cfg.dbscan_min_points)  #
        # print("After dbscan:", len(pcd.points))

    return pcd


def pcd_denoise_dbscan(pcd: o3d.geometry.PointCloud, eps=0.02, min_points=10) -> o3d.geometry.PointCloud:
    ### Remove noise via clustering
    pcd_clusters = pcd.cluster_dbscan(
        eps=eps,
        min_points=min_points,
    )

    # Convert to numpy arrays
    obj_points = np.asarray(pcd.points)
    obj_colors = np.asarray(pcd.colors)
    pcd_clusters = np.array(pcd_clusters)

    # Count all labels in the cluster
    counter = Counter(pcd_clusters)

    # Remove the noise label
    if counter and (-1 in counter):
        del counter[-1]

    if counter:
        # Find the label of the largest cluster
        most_common_label, _ = counter.most_common(1)[0]

        # Create mask for points in the largest cluster
        largest_mask = pcd_clusters == most_common_label

        # Apply mask
        largest_cluster_points = obj_points[largest_mask]
        largest_cluster_colors = obj_colors[largest_mask]

        # If the largest cluster is too small, return the original point cloud
        if len(largest_cluster_points) < 5:
            return pcd

        # Create a new PointCloud object
        largest_cluster_pcd = o3d.geometry.PointCloud()
        largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)
        largest_cluster_pcd.colors = o3d.utility.Vector3dVector(largest_cluster_colors)

        pcd = largest_cluster_pcd

    return pcd


def create_object_pcd(image_points, image_rgb, mask):
    points = image_points[mask]
    colors = image_rgb[mask] / 255.0

    # Perturb the points a bit to avoid colinearity
    points += np.random.normal(0, 4e-3, points.shape)

    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def color_by_instance(pcds):
    cmap = matplotlib.colormaps.get_cmap("turbo")
    instance_colors = cmap(np.linspace(0, 1, len(pcds)))
    for i in range(len(pcds)):
        pcd = pcds[i]
        pcd.colors = o3d.utility.Vector3dVector(np.tile(instance_colors[i, :3], (len(pcd.points), 1)))
    return pcds


def oriented_bbox_to_center_euler_extent(bbox_center, box_R, bbox_extent):
    center = np.asarray(bbox_center)
    extent = np.asarray(bbox_extent)
    eulers = Rotation.from_matrix(box_R.copy()).as_euler("XYZ")
    return center, eulers, extent


def axis_aligned_bbox_to_center_euler_extent(min_coords, max_coords):
    # Calculate the center
    center = tuple((min_val + max_val) / 2.0 for min_val, max_val in zip(min_coords, max_coords))

    # Euler angles for an axis-aligned bounding box are always 0
    eulers = (0, 0, 0)

    # Calculate the extents
    extent = tuple(max_val - min_val for min_val, max_val in zip(min_coords, max_coords))

    return center, eulers, extent


# Distance calculations
def human_like_distance(distance_meters):
    # Define the choices with units included, focusing on the 0.1 to 10 meters range
    if distance_meters < 1:  # For distances less than 1 meter
        choices = [
            (
                round(distance_meters * 100, 2),
                "centimeters",
                0.2,
            ),  # Centimeters for very small distances
            (
                round(distance_meters * 39.3701, 2),
                "inches",
                0.8,
            ),  # Inches for the majority of cases under 1 meter
        ]
    elif distance_meters < 3:  # For distances less than 3 meters
        choices = [
            (round(distance_meters, 2), "meters", 0.5),
            (
                round(distance_meters * 3.28084, 2),
                "feet",
                0.5,
            ),  # Feet as a common unit within indoor spaces
        ]
    else:  # For distances from 3 up to 10 meters
        choices = [
            (
                round(distance_meters, 2),
                "meters",
                0.7,
            ),  # Meters for clarity and international understanding
            (
                round(distance_meters * 3.28084, 2),
                "feet",
                0.3,
            ),  # Feet for additional context
        ]

    # Normalize probabilities and make a selection
    total_probability = sum(prob for _, _, prob in choices)
    cumulative_distribution = []
    cumulative_sum = 0
    for value, unit, probability in choices:
        cumulative_sum += probability / total_probability  # Normalize probabilities
        cumulative_distribution.append((cumulative_sum, value, unit))

    # Randomly choose based on the cumulative distribution
    r = random.random()
    for cumulative_prob, value, unit in cumulative_distribution:
        if r < cumulative_prob:
            return f"{value} {unit}"

    # Fallback to the last choice if something goes wrong
    return f"{choices[-1][0]} {choices[-1][1]}"


def calculate_distances_between_point_clouds(A, B):
    dist_pcd1_to_pcd2 = np.asarray(A.compute_point_cloud_distance(B))
    dist_pcd2_to_pcd1 = np.asarray(B.compute_point_cloud_distance(A))
    combined_distances = np.concatenate((dist_pcd1_to_pcd2, dist_pcd2_to_pcd1))
    avg_dist = np.mean(combined_distances)
    return human_like_distance(avg_dist)


def calculate_centroid(pcd):
    """Calculate the centroid of a point cloud."""
    points = np.asarray(pcd.points)
    centroid = np.mean(points, axis=0)
    return centroid


def calculate_relative_positions(centroids):
    """Calculate the relative positions between centroids of point clouds."""
    num_centroids = len(centroids)
    relative_positions_info = []

    for i in range(num_centroids):
        for j in range(i + 1, num_centroids):
            relative_vector = centroids[j] - centroids[i]

            distance = np.linalg.norm(relative_vector)
            relative_positions_info.append(
                {"pcd_pair": (i, j), "relative_vector": relative_vector, "distance": distance}
            )

    return relative_positions_info


def get_bounding_box_height(pcd):
    """
    Compute the height of the bounding box for a given point cloud.

    Parameters:
    pcd (open3d.geometry.PointCloud): The input point cloud.

    Returns:
    float: The height of the bounding box.
    """
    aabb = pcd.get_axis_aligned_bounding_box()
    return aabb.get_extent()[1]  # Assuming the Y-axis is the up-direction


def compare_bounding_box_height(pcd_i, pcd_j):
    """
    Compare the bounding box heights of two point clouds.

    Parameters:
    pcd_i (open3d.geometry.PointCloud): The first point cloud.
    pcd_j (open3d.geometry.PointCloud): The second point cloud.

    Returns:
    bool: True if the bounding box of pcd_i is taller than that of pcd_j, False otherwise.
    """
    height_i = get_bounding_box_height(pcd_i)
    height_j = get_bounding_box_height(pcd_j)

    return height_i > height_j
