import os
import os.path as osp
import re
import sys

import einops
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MaskPooling(nn.Module):
    def __init__(self, mask_threshold=0.5):
        super().__init__()
        self.mask_threshold = mask_threshold

    def forward(self, x, mask_list, return_list=False, return_mask=False):
        """
        Args:
            x: [B, (HW), C]
            mask_list: List( tensor[M, IH, IW] )
        """
        batch_size = x.size(0)
        if mask_list is None:
            mask_list = [None for i in range(batch_size)]

        output = []
        attn_mask_list = []
        for i in range(batch_size):
            x_len = x.size(1)
            mask = mask_list[i]
            if mask is None:
                output.append(None)
                attn_mask_list.append(None)
            else:
                # resize mask from image shape to feature map shape
                mask_hw = mask.size(-1) * mask.size(-2)
                scale_factor = (x_len / mask_hw) ** 0.5

                mask = mask.detach()
                mask = mask.float()[None, ...]
                mask = nn.functional.interpolate(mask, scale_factor=scale_factor, mode="bilinear")
                mask = mask.to(x.dtype)
                mask = mask[0]
                feature = x[i]

                denorm = mask.sum(dim=(-1, -2)) + 1e-8  # M
                denorm = denorm.unsqueeze(-1)  # M, 1

                mask = mask.flatten(start_dim=1)  # M, H, W -> M, HW

                attn_mask_list.append((mask > self.mask_threshold).to(mask.dtype))  # M, HW

                mask_pooled_x = torch.einsum(
                    "lc,ml->mc",
                    feature,
                    mask / denorm,
                )
                # mc output
                output.append(mask_pooled_x)

        if return_list:
            if return_mask:
                return output, attn_mask_list
            return output
        else:
            # FIXME: Not support Nonetype
            output = torch.cat(output)
            return output


def get_feature_refinement_module(vision_hidden_size, feature_refinement_type="deconv2x"):
    deconv_match = re.match(r"^deconv(\d+)x$", feature_refinement_type)
    if deconv_match:
        deconv_depth = int(deconv_match.group(1))
        modules = []
        for i in range(deconv_depth - 1):
            modules.append(nn.ConvTranspose2d(vision_hidden_size, vision_hidden_size, kernel_size=2, stride=2))
            modules.append(LayerNorm2d(vision_hidden_size))
            modules.append(nn.GELU())
        modules.append(nn.ConvTranspose2d(vision_hidden_size, vision_hidden_size, kernel_size=2, stride=2))
        modules.append(nn.GELU())

        return nn.Sequential(*modules)

    raise ValueError(f"Unknown feature refinement type: {feature_refinement_type}")


class RegionExtractorConfig(PretrainedConfig):
    model_type = "region_extractor"

    def __init__(self, region_extractor_type: str = None, **kwargs):
        super().__init__()
        self.region_extractor_type = region_extractor_type


class RegionExtractor(PreTrainedModel):
    config_class = RegionExtractorConfig

    def __init__(self, region_extractor_cfg: RegionExtractorConfig, config: PretrainedConfig):
        super().__init__(region_extractor_cfg)
        region_extractor_type = region_extractor_cfg.region_extractor_type

        if region_extractor_type == "regiongpt":
            self.mask_pooling = MaskPooling()
            self.feature_refinement_module = get_feature_refinement_module(config.mm_hidden_size)
            # TODO: hardcoded pooling size here, should be inside cfg
            self.ada_pooling = nn.AdaptiveAvgPool2d(27)
            self.rgb_projector = nn.Linear(config.mm_hidden_size, config.hidden_size)
            self.depth_projector = nn.Linear(config.mm_hidden_size, config.hidden_size)
        elif region_extractor_type == "duplicate":
            self.mask_pooling = MaskPooling()
            self.rgb_projector = None
            self.depth_projector = None
        elif region_extractor_type == "duplicate_deconv":
            self.feature_refinement_module = get_feature_refinement_module(config.mm_hidden_size)
            self.ada_pooling = nn.AdaptiveAvgPool2d(27)
            self.mask_pooling = MaskPooling()
            self.rgb_projector = None
            self.depth_projector = None

    def feature_refinement(self, tower_features):
        HW = tower_features.shape[1]
        tower_features = einops.rearrange(tower_features, "N (H W) C -> N C H W", H=int(HW**0.5))
        hres_tower_features = self.feature_refinement_module(tower_features)
        # local feature branch
        hres_tower_features_flatten = einops.rearrange(hres_tower_features, "N C H W -> N (H W) C")

        # global feature branch
        ada_image_feature = self.ada_pooling(hres_tower_features)
        lres_tower_features_flatten = einops.rearrange(ada_image_feature, "N C H W -> N (H W) C")
        return hres_tower_features_flatten, lres_tower_features_flatten

    def extract_region_features(self, hres_tower_features, masks, connector):
        # assume is already flattened -> 'N (H W) C'
        if self.config.region_extractor_type == "regiongpt":
            mask_embeds = self.mask_pooling(hres_tower_features, masks, return_list=True)
            _mask_embeds = []
            for mask_embed in mask_embeds:
                if mask_embed is None:
                    _mask_embeds.append(None)
                else:
                    _mask_embeds.append(connector(mask_embed))

        elif self.config.region_extractor_type in ["duplicate", "duplicate_deconv"]:
            raise NotImplementedError(f"{self.config.region_extractor_type} not implemented")

        mask_embeds = _mask_embeds

        return mask_embeds

    def forward(self, image_features, depth_features, masks, *args, **kwargs):
        mask_embeds = self.extract_region_features(image_features, masks, self.rgb_projector)
        if depth_features is not None:
            depth_embeds = self.extract_region_features(depth_features, masks, self.depth_projector)
        else:
            depth_embeds = None
        return mask_embeds, depth_embeds


AutoConfig.register("region_extractor", RegionExtractorConfig)
AutoModel.register(RegionExtractorConfig, RegionExtractor)
