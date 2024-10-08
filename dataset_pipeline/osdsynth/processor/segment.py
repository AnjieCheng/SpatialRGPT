import cv2
import torch
import torchvision
from osdsynth.processor.wrappers.grounding_dino import get_grounding_dino_model
from osdsynth.processor.wrappers.ram import get_tagging_model, run_tagging_model
from osdsynth.processor.wrappers.sam import (
    convert_detections_to_dict,
    convert_detections_to_list,
    crop_detections_with_xyxy,
    filter_detections,
    get_sam_predictor,
    get_sam_segmentation_from_xyxy,
    mask_subtract_contained,
    post_process_mask,
    sort_detections_by_area,
)
from osdsynth.utils.logger import SkipImageException
from osdsynth.visualizer.som import draw_som_on_image
from PIL import Image


class SegmentImage:
    """Class to segment the image."""

    def __init__(self, cfg, logger, device, init_gdino=True, init_tagging=True, init_sam=True):
        self.cfg = cfg
        self.logger = logger
        self.device = device

        if init_gdino:
            # Initialize the Grounding Dino Model
            self.grounding_dino_model = get_grounding_dino_model(cfg, device)
        else:
            self.grounding_dino_model = None

        if init_tagging:
            # Initialize the tagging Model
            self.tagging_transform, self.tagging_model = get_tagging_model(cfg, device)
        else:
            self.tagging_transform = self.tagging_model = None

        if init_sam:
            # Initialize the SAM Model
            self.sam_predictor = get_sam_predictor(cfg.sam_variant, device)
        else:
            self.sam_predictor = None

        pass

    def process(self, image_bgr, plot_som=True):
        """Segment the image."""

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb_pil = Image.fromarray(image_rgb)

        img_tagging = image_rgb_pil.resize((384, 384))
        img_tagging = self.tagging_transform(img_tagging).unsqueeze(0).to(self.device)

        # Tag2Text
        classes = run_tagging_model(self.cfg, img_tagging, self.tagging_model)

        # Using GroundingDINO to detect and SAM to segment
        detections = self.grounding_dino_model.predict_with_classes(
            image=image_bgr,  # This function expects a BGR image...
            classes=classes,
            box_threshold=self.cfg.box_threshold,
            text_threshold=self.cfg.text_threshold,
        )

        if len(detections.class_id) < 1:
            raise SkipImageException("No object detected.")

        # Non-maximum suppression
        nms_idx = (
            torchvision.ops.nms(
                torch.from_numpy(detections.xyxy),
                torch.from_numpy(detections.confidence),
                self.cfg.nms_threshold,
            )
            .numpy()
            .tolist()
        )

        print(f"Before NMS: {len(detections.xyxy)} detections")
        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]
        print(f"After NMS: {len(detections.xyxy)} detections")

        # Somehow some detections will have class_id=-1, remove them
        valid_idx = detections.class_id != -1
        detections.xyxy = detections.xyxy[valid_idx]
        detections.confidence = detections.confidence[valid_idx]
        detections.class_id = detections.class_id[valid_idx]

        # Segment Anything
        detections.mask = get_sam_segmentation_from_xyxy(
            sam_predictor=self.sam_predictor, image=image_rgb, xyxy=detections.xyxy
        )

        # Convert the detection to a dict. Elements are np.ndarray
        detections_dict = convert_detections_to_dict(detections, classes)

        # Filter out the objects based on various criteria
        detections_dict = filter_detections(self.cfg, detections_dict, image_rgb)

        if len(detections_dict["xyxy"]) < 1:
            raise SkipImageException("No object detected after filtering.")

        # Subtract the mask of bounding boxes that are contained by it
        detections_dict["subtracted_mask"], mask_contained = mask_subtract_contained(
            detections_dict["xyxy"], detections_dict["mask"], th1=0.05, th2=0.05
        )

        # Sort the dets by area
        detections_dict = sort_detections_by_area(detections_dict)

        # Add RLE to dict
        detections_dict = post_process_mask(detections_dict)

        # Convert the detection to a list. Each element is a dict
        detections_list = convert_detections_to_list(detections_dict, classes)

        detections_list = crop_detections_with_xyxy(self.cfg, image_rgb_pil, detections_list)

        if plot_som:
            # Visualize with SoM
            vis_som = draw_som_on_image(
                detections_dict,
                image_rgb,
                label_mode="1",
                alpha=0.4,
                anno_mode=["Mask", "Mark", "Box"],
            )
        else:
            vis_som = None

        return vis_som, detections_list
