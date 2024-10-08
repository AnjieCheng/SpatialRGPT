import os
import sys

GSA_PATH = "osdsynth/external/Grounded-Segment-Anything"
sys.path.append(GSA_PATH)

from groundingdino.util.inference import Model

# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = os.path.abspath(
    os.path.join(GSA_PATH, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
)
GROUNDING_DINO_CHECKPOINT_PATH = os.path.abspath(os.path.join(GSA_PATH, "./groundingdino_swint_ogc.pth"))


def get_grounding_dino_model(cfg, device):
    grounding_dino_model = Model(
        model_config_path=GROUNDING_DINO_CONFIG_PATH,
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
        device=device,
    )

    return grounding_dino_model
