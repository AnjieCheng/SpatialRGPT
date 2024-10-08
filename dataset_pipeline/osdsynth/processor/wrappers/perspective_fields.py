import os
import sys

PPF_PATH = "osdsynth/external/PerspectiveFields"
sys.path.append(PPF_PATH)  # This is needed for the following imports in this file

PPF_PATH_ABS = os.path.abspath(PPF_PATH)

import copy
import os

import cv2
import numpy as np
import torch
from perspective2d import PerspectiveFields
from perspective2d.utils import draw_from_r_p_f_cx_cy, draw_perspective_fields


def create_rotation_matrix(
    roll: float,
    pitch: float,
    yaw: float,
    degrees: bool = False,
) -> np.ndarray:
    r"""Create rotation matrix from extrinsic parameters
    Args:
        roll (float): camera rotation about camera frame z-axis
        pitch (float): camera rotation about camera frame x-axis
        yaw (float): camera rotation about camera frame y-axis

    Returns:
        np.ndarray: rotation R_z @ R_x @ R_y
    """
    if degrees:
        roll = np.radians(roll)
        pitch = np.radians(pitch)
        yaw = np.radians(yaw)
    # calculate rotation about the x-axis
    R_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(pitch), np.sin(pitch)],
            [0.0, -np.sin(pitch), np.cos(pitch)],
        ]
    )
    # calculate rotation about the y-axis
    R_y = np.array(
        [
            [np.cos(yaw), 0.0, -np.sin(yaw)],
            [0.0, 1.0, 0.0],
            [np.sin(yaw), 0.0, np.cos(yaw)],
        ]
    )
    # calculate rotation about the z-axis
    R_z = np.array(
        [
            [np.cos(roll), np.sin(roll), 0.0],
            [-np.sin(roll), np.cos(roll), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    return R_z @ R_x @ R_y


def resize_fix_aspect_ratio(img, field, target_width=None, target_height=None):
    height = img.shape[0]
    width = img.shape[1]
    if target_height is None:
        factor = target_width / width
    elif target_width is None:
        factor = target_height / height
    else:
        factor = max(target_width / width, target_height / height)
    if factor == target_width / width:
        target_height = int(height * factor)
    else:
        target_width = int(width * factor)

    img = cv2.resize(img, (target_width, target_height))
    for key in field:
        if key not in ["up", "lati"]:
            continue
        tmp = field[key].numpy()
        transpose = len(tmp.shape) == 3
        if transpose:
            tmp = tmp.transpose(1, 2, 0)
        tmp = cv2.resize(tmp, (target_width, target_height))
        if transpose:
            tmp = tmp.transpose(2, 0, 1)
        field[key] = torch.tensor(tmp)
    return img, field


def run_perspective_fields_model(model, image_bgr):

    pred = model.inference(img_bgr=image_bgr)
    field = {
        "up": pred["pred_gravity_original"].cpu().detach(),
        "lati": pred["pred_latitude_original"].cpu().detach(),
    }
    img, field = resize_fix_aspect_ratio(image_bgr[..., ::-1], field, 640)

    # Draw perspective field from ParamNet predictions
    param_vis = draw_from_r_p_f_cx_cy(
        img,
        pred["pred_roll"].item(),
        pred["pred_pitch"].item(),
        pred["pred_general_vfov"].item(),
        pred["pred_rel_cx"].item(),
        pred["pred_rel_cy"].item(),
        "deg",
        up_color=(0, 1, 0),
    ).astype(np.uint8)
    param_vis = cv2.cvtColor(param_vis, cv2.COLOR_RGB2BGR)

    param = {
        "roll": pred["pred_roll"].cpu().item(),
        "pitch": pred["pred_pitch"].cpu().item(),
    }

    return param_vis, param


def get_perspective_fields_model(cfg, device):
    MODEL_ID = "Paramnet-360Cities-edina-centered"
    # MODEL_ID = 'Paramnet-360Cities-edina-uncentered'
    # MODEL_ID = 'PersNet_Paramnet-GSV-centered'
    # MODEL_ID = 'PersNet_Paramnet-GSV-uncentered'
    # MODEL_ID = 'PersNet-360Cities'
    # feel free to test with uncentered or centered depending on your data

    PerspectiveFields.versions()
    pf_model = PerspectiveFields(MODEL_ID).eval().cuda()
    return pf_model
