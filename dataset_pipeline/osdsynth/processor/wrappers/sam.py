import os
import sys
from typing import Any, Dict, Generator, ItemsView, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

GSA_PATH = "osdsynth/external/Grounded-Segment-Anything"
sys.path.append(GSA_PATH)

from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_hq_model_registry, sam_model_registry

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./sam_vit_h_4b8939.pth")

# Segment-Anything checkpoint
SAM_HQ_ENCODER_VERSION = "vit_h"
SAM_HQ_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./sam_hq_vit_h.pth")

# Prompting SAM with detected boxes
def get_sam_segmentation_from_xyxy(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(box=box, multimask_output=True)
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


def get_sam_predictor(variant: str, device: str | int) -> SamPredictor:
    if variant == "sam":
        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(device)
        sam_predictor = SamPredictor(sam)
        return sam_predictor

    if variant == "sam-hq":
        print("Using SAM-HQ")
        sam = sam_hq_model_registry[SAM_HQ_ENCODER_VERSION](checkpoint=SAM_HQ_CHECKPOINT_PATH)
        sam.to(device)
        sam_predictor = SamPredictor(sam)
        return sam_predictor

    else:
        raise NotImplementedError


def get_sam_mask_generator(variant: str, device: str | int) -> SamAutomaticMaskGenerator:
    if variant == "sam":
        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(device)
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=12,
            points_per_batch=144,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            crop_n_layers=0,
            min_mask_region_area=100,
        )
        return mask_generator
    elif variant == "fastsam":
        raise NotImplementedError
    else:
        raise NotImplementedError


def convert_detections_to_list(detections_dict, classes):
    detection_list = []
    for i in range(len(detections_dict["xyxy"])):
        detection = {
            "class_name": classes[detections_dict["class_id"][i]],  # Lookup class name using class_id
            "xyxy": detections_dict["xyxy"][i],  # Assuming detections.xyxy is a numpy array
            "confidence": detections_dict["confidence"][i].item(),  # Convert numpy scalar to Python scalar
            "class_id": detections_dict["class_id"][i].item(),
            "box_area": detections_dict["box_area"][i].item(),
            "mask": detections_dict["mask"][i],
            "subtracted_mask": detections_dict["subtracted_mask"][i],
            "rle": detections_dict["rle"][i],
            "area": detections_dict["area"][i],
        }
        detection_list.append(detection)
    return detection_list


def convert_detections_to_dict(detections, classes, image_crops=None, image_feats=None, text_feats=None):
    # Convert the detections to a dict. The elements are in np.array
    results = {
        "xyxy": detections.xyxy,
        "confidence": detections.confidence,
        "class_id": detections.class_id,
        "box_area": detections.box_area,
        "mask": detections.mask,
        "area": detections.area,
        "classes": classes,
    }
    return results


def mask_subtract_contained(xyxy: np.ndarray, mask: np.ndarray, th1=0.8, th2=0.7):
    """Compute the containing relationship between all pair of bounding boxes. For each mask, subtract the mask of
    bounding boxes that are contained by it.

    Args:
        xyxy: (N, 4), in (x1, y1, x2, y2) format
        mask: (N, H, W), binary mask
        th1: float, threshold for computing intersection over box1
        th2: float, threshold for computing intersection over box2

    Returns:
        mask_sub: (N, H, W), binary mask
    """
    N = xyxy.shape[0]  # number of boxes

    # Get areas of each xyxy
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])  # (N,)

    # Compute intersection boxes
    lt = np.maximum(xyxy[:, None, :2], xyxy[None, :, :2])  # left-top points (N, N, 2)
    rb = np.minimum(xyxy[:, None, 2:], xyxy[None, :, 2:])  # right-bottom points (N, N, 2)

    inter = (rb - lt).clip(min=0)  # intersection sizes (dx, dy), if no overlap, clamp to zero (N, N, 2)

    # Compute areas of intersection boxes
    inter_areas = inter[:, :, 0] * inter[:, :, 1]  # (N, N)

    inter_over_box1 = inter_areas / areas[:, None]  # (N, N)
    # inter_over_box2 = inter_areas / areas[None, :] # (N, N)
    inter_over_box2 = inter_over_box1.T  # (N, N)

    # if the intersection area is smaller than th2 of the area of box1,
    # and the intersection area is larger than th1 of the area of box2,
    # then box2 is considered contained by box1
    contained = (inter_over_box1 < th2) & (inter_over_box2 > th1)  # (N, N)
    contained_idx = contained.nonzero()  # (num_contained, 2)

    mask_sub = mask.copy()  # (N, H, W)
    # mask_sub[contained_idx[0]] = mask_sub[contained_idx[0]] & (~mask_sub[contained_idx[1]])
    for i in range(len(contained_idx[0])):
        mask_sub[contained_idx[0][i]] = mask_sub[contained_idx[0][i]] & (~mask_sub[contained_idx[1][i]])

    return mask_sub, contained


def filter_detections(cfg, detections_dict: dict, image: np.ndarray):
    # If no detection at all
    if len(detections_dict["xyxy"]) == 0:
        return detections_dict

    # Filter out the objects based on various criteria
    idx_to_keep = []
    for obj_idx in range(len(detections_dict["xyxy"])):
        class_name = detections_dict["classes"][detections_dict["class_id"][obj_idx]]

        # Skip masks that are too small
        if detections_dict["mask"][obj_idx].sum() < max(cfg.mask_area_threshold, 10):
            print(f"Skipping {class_name} mask with too few points")
            continue

        # Skip the BG classes
        if cfg.skip_bg and class_name in cfg.bg_classes:
            print(f"Skipping {class_name} as it is a background class")
            continue

        # Skip the non-background boxes that are too large
        if class_name not in cfg.bg_classes:
            x1, y1, x2, y2 = detections_dict["xyxy"][obj_idx]
            bbox_area = (x2 - x1) * (y2 - y1)
            image_area = image.shape[0] * image.shape[1]
            if bbox_area > cfg.max_bbox_area_ratio * image_area:
                print(f"Skipping {class_name} with area {bbox_area} > {cfg.max_bbox_area_ratio} * {image_area}")
                continue

        # Skip masks with low confidence
        if detections_dict["confidence"][obj_idx] < cfg.mask_conf_threshold:
            print(
                f"Skipping {class_name} with confidence {detections_dict['confidence'][obj_idx]} < {cfg.mask_conf_threshold}"
            )
            continue

        idx_to_keep.append(obj_idx)

    for k in detections_dict.keys():
        if isinstance(detections_dict[k], str) or k == "classes":  # Captions
            continue
        elif isinstance(detections_dict[k], list):
            detections_dict[k] = [detections_dict[k][i] for i in idx_to_keep]
        elif isinstance(detections_dict[k], np.ndarray):
            detections_dict[k] = detections_dict[k][idx_to_keep]
        else:
            raise NotImplementedError(f"Unhandled type {type(detections_dict[k])}")

    return detections_dict


def sort_detections_by_area(detections_dict):
    # Sort the detections by area, use negative to sort from large to small
    sorted_indices = np.argsort(-detections_dict["area"])
    for key in detections_dict.keys():
        if isinstance(detections_dict[key], np.ndarray):  # Check to ensure it's an array
            detections_dict[key] = detections_dict[key][sorted_indices]
    return detections_dict


def post_process_mask(detections_dict):
    sam_masks = torch.tensor(detections_dict["subtracted_mask"])
    uncompressed_mask_rles = mask_to_rle_pytorch(sam_masks)
    rle_masks_list = [coco_encode_rle(uncompressed_mask_rles[i]) for i in range(len(uncompressed_mask_rles))]
    detections_dict["rle"] = rle_masks_list
    return detections_dict


def crop_image_and_mask(image: Image, mask: np.ndarray, x1: int, y1: int, x2: int, y2: int, padding: int = 0):
    """Crop the image and mask with some padding.

    I made a single function that crops both the image and the mask at the same time because I was getting shape
    mismatches when I cropped them separately.This way I can check that they are the same shape.
    """

    image = np.array(image)
    # Verify initial dimensions
    if image.shape[:2] != mask.shape:
        print(f"Initial shape mismatch: Image shape {image.shape} != Mask shape {mask.shape}")
        return None, None

    # Define the cropping coordinates
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.shape[1], x2 + padding)
    y2 = min(image.shape[0], y2 + padding)
    # round the coordinates to integers
    x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)

    # Crop the image and the mask
    image_crop = image[y1:y2, x1:x2]
    mask_crop = mask[y1:y2, x1:x2]

    # Verify cropped dimensions
    if image_crop.shape[:2] != mask_crop.shape:
        print(
            "Cropped shape mismatch: Image crop shape {} != Mask crop shape {}".format(
                image_crop.shape, mask_crop.shape
            )
        )
        return None, None

    # convert the image back to a pil image
    image_crop = Image.fromarray(image_crop)

    return image_crop, mask_crop


def crop_detections_with_xyxy(cfg, image, detections_list):
    for idx, detection in enumerate(detections_list):
        x1, y1, x2, y2 = detection["xyxy"]
        image_crop, mask_crop = crop_image_and_mask(image, detection["mask"], x1, y1, x2, y2, padding=10)
        if cfg.masking_option == "blackout":
            image_crop_modified = blackout_nonmasked_area(image_crop, mask_crop)
        elif cfg.masking_option == "red_outline":
            image_crop_modified = draw_red_outline(image_crop, mask_crop)
        else:
            image_crop_modified = image_crop  # No modification
        detections_list[idx]["image_crop"] = image_crop
        detections_list[idx]["mask_crop"] = mask_crop
        detections_list[idx]["image_crop_modified"] = image_crop_modified
    return detections_list


def mask_to_rle_pytorch(tensor: torch.Tensor) -> List[Dict[str, Any]]:
    """
    Encodes masks to an uncompressed RLE, in the format expected by
    pycoco tools.
    """
    # Put in fortran order and flatten h,w
    b, h, w = tensor.shape
    tensor = tensor.permute(0, 2, 1).flatten(1)

    # Compute change indices
    diff = tensor[:, 1:] ^ tensor[:, :-1]
    change_indices = diff.nonzero()

    # Encode run length
    out = []
    for i in range(b):
        cur_idxs = change_indices[change_indices[:, 0] == i, 1]
        cur_idxs = torch.cat(
            [
                torch.tensor([0], dtype=cur_idxs.dtype, device=cur_idxs.device),
                cur_idxs + 1,
                torch.tensor([h * w], dtype=cur_idxs.dtype, device=cur_idxs.device),
            ]
        )
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        counts = [] if tensor[i, 0] == 0 else [0]
        counts.extend(btw_idxs.detach().cpu().tolist())
        out.append({"size": [h, w], "counts": counts})
    return out


def rle_to_mask(rle: Dict[str, Any]) -> np.ndarray:
    """Compute a binary mask from an uncompressed RLE."""
    h, w = rle["size"]
    mask = np.empty(h * w, dtype=bool)
    idx = 0
    parity = False
    for count in rle["counts"]:
        mask[idx : idx + count] = parity
        idx += count
        parity ^= True
    mask = mask.reshape(w, h)
    return mask.transpose()  # Put in C order


def coco_encode_rle(uncompressed_rle: Dict[str, Any]) -> Dict[str, Any]:
    from pycocotools import mask as mask_utils  # type: ignore

    h, w = uncompressed_rle["size"]
    rle = mask_utils.frPyObjects(uncompressed_rle, h, w)
    rle["counts"] = rle["counts"].decode("utf-8")  # Necessary to serialize with json
    return rle
