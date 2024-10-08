# This file is modified from https://github.com/haotian-liu/LLaVA/

import os

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel

from .base_extractor import RegionExtractor, RegionExtractorConfig


def build_region_extractor(model_type_or_path: str, config: PretrainedConfig) -> PreTrainedModel:
    if model_type_or_path is None:
        return None

    ## load from pretrained model
    if config.resume_path and os.path.exists(model_type_or_path):
        print("Resuming region extractor from: ", model_type_or_path)
        return RegionExtractor.from_pretrained(model_type_or_path, config, torch_dtype=eval(config.model_dtype))
    else:
        print("Build region extractor from scratch.")
        region_extractor_cfg = RegionExtractorConfig(model_type_or_path)
        region_extractor = RegionExtractor(region_extractor_cfg, config).to(eval(config.model_dtype))
        return region_extractor
