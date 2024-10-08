import os
import sys
from typing import List

import torchvision.transforms as TS
from ram import inference_ram
from ram.models import ram

# sys.path.append("osdsynth/external/recognize-anything")


def run_tagging_model(cfg, raw_image, tagging_model):
    res = inference_ram(raw_image, tagging_model)
    caption = "NA"
    tags = res[0].strip(" ").replace("  ", " ").replace(" |", ",")
    print("Tags: ", tags)

    # Currently ", " is better for detecting single tags
    # while ". " is a little worse in some case
    text_prompt = res[0].replace(" |", ",")

    if cfg.rm_bg_classes:
        cfg.remove_classes += cfg.bg_classes

    classes = process_tag_classes(
        text_prompt,
        add_classes=cfg.add_classes,
        remove_classes=cfg.remove_classes,
    )
    print("Tags (Final): ", classes)
    return classes


def process_tag_classes(text_prompt: str, add_classes: List[str] = [], remove_classes: List[str] = []) -> list[str]:
    """Convert a text prompt from Tag2Text to a list of classes."""
    classes = text_prompt.split(",")
    classes = [obj_class.strip() for obj_class in classes]
    classes = [obj_class for obj_class in classes if obj_class != ""]

    for c in add_classes:
        if c not in classes:
            classes.append(c)

    for c in remove_classes:
        classes = [obj_class for obj_class in classes if c not in obj_class.lower()]

    return classes


def get_tagging_model(cfg, device):
    RAM_CHECKPOINT_PATH = os.path.abspath(
        "osdsynth/external/Grounded-Segment-Anything/recognize-anything/ram_swin_large_14m.pth"
    )
    tagging_model = ram(pretrained=RAM_CHECKPOINT_PATH, image_size=384, vit="swin_l")

    tagging_model = tagging_model.eval().to(device)
    tagging_transform = TS.Compose(
        [
            TS.Resize((384, 384)),
            TS.ToTensor(),
            TS.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return tagging_transform, tagging_model
