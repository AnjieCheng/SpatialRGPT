#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import logging
import os
import os.path as osp
import sys
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict

import torch
import torch.distributed as dist
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_utils import ContextManagers, no_init_weights

from llava.constants import (
    DEFAULT_DEPTH_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_MASK_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from llava.model.configuration_llava import LlavaConfig
from llava.model.language_model.builder import build_llm_and_tokenizer
from llava.model.multimodal_encoder.builder import build_vision_tower
from llava.model.multimodal_projector.builder import build_mm_projector
from llava.model.region_extractor.builder import build_region_extractor
from llava.model.utils import get_model_config
from llava.train.sequence_parallel import get_pg_manager
from llava.train.sequence_parallel.globals import get_ulysess_sp_pg
from llava.train.utils import (
    get_checkpoint_path,
    prepare_config_for_training,
    unit_test_rope_scaling,
    vision_resolution_elevation,
)


# TODO decide whether should we use metaclass
class LlavaMetaModel(ABC):
    def init_vlm(self, config: PreTrainedModel = None, *args, **kwargs):
        # TODO(ligeng): figure out how from_config and from_pretrained works in HF implementation.
        if (
            hasattr(self, "llm")
            or hasattr(self, "vision_tower")
            or hasattr(self, "mm_projector")
            or hasattr(self, "region_extractor")
        ):
            # already initialized, skipped
            return

        model_dtype = getattr(config, "model_dtype", "torch.float16")
        if not hasattr(config, "model_dtype"):
            warnings.warn("model_dtype not found in config, defaulting to torch.float16.")
            config.model_dtype = model_dtype

        cfgs = get_model_config(config)
        if len(cfgs) == 4:
            llm_cfg, vision_tower_cfg, mm_projector_cfg, region_extractor_cfg = cfgs
        elif len(cfgs) == 3:
            llm_cfg, vision_tower_cfg, mm_projector_cfg = cfgs
            region_extractor_cfg = ""
        else:
            raise ValueError(
                "`llm_cfg` `mm_projector_cfg` `vision_tower_cfg` `region_extractor_cfg` not found in the config."
            )

        # print("Before init in Config")
        # if hasattr(config, "deepspeed") and "mics" in config.deepspeed:
        #     print("Using MiCS_Init")
        #     import deepspeed
        #     with deepspeed.zero.MiCS_Init():
        #         self.llm, self.tokenizer = build_llm_and_tokenizer(llm_cfg, config, *args, **kwargs)
        #         self.vision_tower = build_vision_tower(vision_tower_cfg, config)
        #         self.mm_projector = build_mm_projector(mm_projector_cfg, config)
        # else:
        self.llm, self.tokenizer = build_llm_and_tokenizer(llm_cfg, config, *args, **kwargs)
        self.vision_tower = build_vision_tower(vision_tower_cfg, config)
        self.mm_projector = build_mm_projector(mm_projector_cfg, config)
        self.region_extractor = build_region_extractor(region_extractor_cfg, config)

        self.post_config()
        self.is_loaded = True

        assert (
            self.llm is not None or self.vision_tower is not None or self.mm_projector is not None
        ), "At least one of the components must be instantiated."

    @classmethod
    def load_from_config(cls, model_path_or_config, *args, **kwargs):
        pass

    ## FIXME we will use this function to load model in the future
    @classmethod
    def load_pretrained(cls, model_path_or_config, *args, **kwargs):
        kwargs.pop("config", None)

        if isinstance(model_path_or_config, str):
            config = AutoConfig.from_pretrained(model_path_or_config)
        elif isinstance(model_path_or_config, LlavaConfig):
            config = model_path_or_config
        else:
            raise NotImplementedError(
                f"wrong type, {type(model_path_or_config)} \
                                      {isinstance(model_path_or_config, LlavaConfig)}"
            )

        model_dtype = getattr(config, "model_dtype", "torch.float16")
        if not hasattr(config, "model_dtype"):
            warnings.warn("model_dtype not found in config, defaulting to torch.float16.")
            config.model_dtype = model_dtype

        cfgs = get_model_config(config)
        if len(cfgs) == 4:
            llm_cfg, vision_tower_cfg, mm_projector_cfg, region_extractor_cfg = cfgs
        elif len(cfgs) == 3:
            llm_cfg, vision_tower_cfg, mm_projector_cfg = cfgs
            region_extractor_cfg = ""
        else:
            raise ValueError(
                "`llm_cfg` `mm_projector_cfg` `vision_tower_cfg` `region_extractor_cfg` not found in the config."
            )

        # print(llm_cfg, vision_tower_cfg, mm_projector_cfg); input("DEBUG load_pretrained")
        init_context = [
            no_init_weights(_enable=True),
        ]
        # print("Before Init Context")
        # if hasattr(config, "deepspeed") and "mics" in config.deepspeed:
        #     print("Using MiCS_Init")
        #     import deepspeed
        #     init_context.append(deepspeed.zero.MiCS_Init(config_dict_or_path=config.deepspeed))
        with ContextManagers(init_context):
            vlm = cls(config, *args, **kwargs)
        # print(llm_cfg, vision_tower_cfg, mm_projector_cfg); input("DEBUG load_pretrained finish")

        if hasattr(vlm, "llm") or hasattr(vlm, "vision_tower") or hasattr(vlm, "mm_projector"):
            if vlm.is_loaded:
                return vlm

        vlm.llm, vlm.tokenizer = build_llm_and_tokenizer(llm_cfg, config, *args, **kwargs)
        vlm.vision_tower = build_vision_tower(vision_tower_cfg, config)
        vlm.mm_projector = build_mm_projector(mm_projector_cfg, config)
        vlm.region_extractor = build_region_extractor(region_extractor_cfg, config)

        self.post_config()
        self.is_loaded = True

        # FIXME(ligeng, yunhao): llm should never be none here.
        assert (
            vlm.llm is not None
            or vlm.vision_tower is not None
            or vlm.mm_projector is not None
            or vlm.region_extractor is not None
        ), "At least one of the components must be instantiated."
        return vlm

    ## FIXME we will use this function to save the model in the future
    def save_pretrained(self, output_dir, state_dict=None):
        if state_dict is None:
            # other wise fetch from deepspeed
            # state_dict = accelerator.get_state_dict(is_deepspeed_enabled)
            state_dict = self.state_dict()

        if getattr(self, "tokenizer", None):
            self.tokenizer.save_pretrained(osp.join(output_dir, "llm"))

        if self.get_llm():
            print(f"saving llm to {osp.join(output_dir, 'llm')}")
            self.llm.config._name_or_path = osp.join(output_dir, "llm")
            llm_state_dict = OrderedDict({k.split("llm.")[-1]: v for k, v in state_dict.items() if "llm" in k})
            self.llm.save_pretrained(os.path.join(output_dir, "llm"), state_dict=llm_state_dict)
            self.config.llm_cfg = self.llm.config

        if self.get_vision_tower():
            print(f"saving vision_tower to {osp.join(output_dir, 'vision_tower')}")
            self.vision_tower.config._name_or_path = osp.join(output_dir, "vision_tower")
            vision_tower_state_dict = OrderedDict(
                {k.split("vision_tower.vision_tower.")[-1]: v for k, v in state_dict.items() if "vision_tower" in k}
            )
            self.vision_tower.vision_tower.save_pretrained(
                os.path.join(output_dir, "vision_tower"),
                state_dict=vision_tower_state_dict,
            )
            self.vision_tower.image_processor.save_pretrained(os.path.join(output_dir, "vision_tower"))
            self.config.vision_tower_cfg = self.vision_tower.config
            if hasattr(self.config.vision_tower_cfg, "auto_map"):
                if "radio" not in self.get_vision_tower().__class__.__name__.lower():
                    delattr(self.config.vision_tower_cfg, "auto_map")

        if self.get_mm_projector():
            print(f"saving mm_projector to {osp.join(output_dir, 'mm_projector')}")
            self.mm_projector.config._name_or_path = osp.join(output_dir, "mm_projector")
            mm_projector_state_dict = OrderedDict(
                {k.split("mm_projector.")[-1]: v for k, v in state_dict.items() if "mm_projector" in k}
            )
            self.mm_projector.save_pretrained(
                os.path.join(output_dir, "mm_projector"),
                state_dict=mm_projector_state_dict,
            )
            self.config.mm_projector_cfg = self.mm_projector.config

        if self.get_region_extractor() and self.get_region_extractor().config.region_extractor_type != "duplicate":
            print(f"saving region_extractor to {osp.join(output_dir, 'region_extractor')}")
            self.region_extractor.config._name_or_path = osp.join(output_dir, "region_extractor")
            region_extractor_state_dict = OrderedDict(
                {k.split("region_extractor.")[-1]: v for k, v in state_dict.items() if "region_extractor" in k}
            )
            self.region_extractor.save_pretrained(
                os.path.join(output_dir, "region_extractor"),
                state_dict=region_extractor_state_dict,
            )
            self.config.region_extractor_cfg = self.region_extractor.config

        ## update and save top-level config
        if hasattr(self.config, "enable_region") and self.config.enable_region:
            self.config.enable_region = True
        else:
            self.config.enable_region = False

        if hasattr(self.config, "enable_depth") and self.config.enable_depth:
            self.config.enable_depth = True
        else:
            self.config.enable_depth = False

        self.config._name_or_path = output_dir
        self.config.architectures = [self.__class__.__name__]
        self.config.save_pretrained(output_dir)

    def get_llm(self):
        llm = getattr(self, "llm", None)
        if type(llm) is list:
            llm = llm[0]
        return llm

    def get_lm_head(self):
        lm_head = getattr(self.get_llm(), "lm_head", None)
        return lm_head

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_mm_projector(self):
        mm_projector = getattr(self, "mm_projector", None)
        if type(mm_projector) is list:
            mm_projector = mm_projector[0]
        return mm_projector

    def get_region_extractor(self):
        region_extractor = getattr(self, "region_extractor", None)
        if type(region_extractor) is list:
            region_extractor = region_extractor[0]
        return region_extractor

    def post_config(self):
        self.training = self.get_llm().training
        ## configuration
        if getattr(self.config, "llm_cfg", None) is None:
            self.config.llm_cfg = self.llm.config
        if getattr(self.config, "vision_tower_cfg", None) is None:
            self.config.vision_tower_cfg = self.vision_tower.config
        if getattr(self.config, "mm_projector_cfg", None) is None:
            self.config.mm_projector_cfg = self.mm_projector.config
        if getattr(self.config, "region_extractor_cfg", None) is None:
            self.config.region_extractor_cfg = self.region_extractor.config

    def freezed_module_patch(self):
        """
        Huggingface will call model.train() at each training_step. To ensure the expected behaviors for modules like dropout, batchnorm, etc., we need to call model.eval() for the freezed modules.
        """
        if self.training:
            if self.get_llm() and not getattr(self.config, "tune_language_model", False):
                pass
                # logging.warning("Caution: Your LLM is currently in training mode, ensuring accurate gradient computation. Please be vigilant, particularly regarding BatchNorm and Dropout operations.")
            if self.get_vision_tower() and not getattr(self.config, "tune_vision_tower", False):
                self.get_vision_tower().eval()
            if self.get_mm_projector() and not getattr(self.config, "tune_mm_projector", False):
                self.get_mm_projector().eval()
            if self.get_region_extractor() and not getattr(self.config, "tune_region_extractor", False):
                self.get_region_extractor().eval()

    def encode_images(self, images):
        image_features = self.get_vision_tower()(images)
        image_features = self.get_mm_projector()(image_features)
        return image_features

    ## @yunhao: is there a better way to handle function call and attributes for llm?
    ## support beam search
    def _temporary_reorder_cache(self, past_key_values, sorted_idx):
        return self.get_llm()._temporary_reorder_cache(past_key_values, sorted_idx)

    def get_input_embeddings(self):
        return self.get_llm().get_input_embeddings()

    def get_output_embeddings(self):
        return self.get_llm().get_output_embeddings()

    def resize_token_embeddings(self, embed_size):
        self.get_llm().resize_token_embeddings(embed_size)


class LlavaMetaForCausalLM(ABC):
    """This class is originally implemented by the LLaVA team and
    modified by Haotian Tang and Jason Lu based on Ji Lin's implementation
    to support multiple images and input packing."""

    ## TODO move the forward function here if there is no need to override it
    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
        masks=None,
        depths=None,
    ):

        # Handle sequence parallelism
        PROCESS_GROUP_MANAGER = get_pg_manager()
        if PROCESS_GROUP_MANAGER is None:
            sp_degree = -1
            sp_rank = -1
        else:
            sp_degree = PROCESS_GROUP_MANAGER.sp_degree
            sp_rank = PROCESS_GROUP_MANAGER.sp_rank

        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or (input_ids.shape[1] == 1 and PROCESS_GROUP_MANAGER is None):
            if (
                past_key_values is not None
                and vision_tower is not None
                and images is not None
                and input_ids.shape[1] == 1
            ):
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones(
                            (
                                attention_mask.shape[0],
                                target_shape - attention_mask.shape[1],
                            ),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                    ),
                    dim=1,
                )
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                None,
                labels,
            )
        # handle different image dtypes for packing
        if type(images) is list:
            images = torch.cat(images, dim=0)
        elif images.ndim == 5:  # batch_size x seq_len x image_channels
            images = images.flatten(0, 1)

        if depths is not None:
            if type(depths) is list:
                depths = torch.cat(depths, dim=0)
            elif depths.ndim == 5:  # batch_size x seq_len x image_channels
                depths = depths.flatten(0, 1)

        tower_features = self.get_vision_tower()(images)
        tower_features = tower_features.to(self.device)

        if hasattr(self.config, "enable_region") and self.config.enable_region:
            hres_tower_features, lres_tower_features = self.get_region_extractor().feature_refinement(tower_features)
            if hasattr(self.config, "enable_depth") and self.config.enable_depth and depths is not None:
                depth_features = self.get_vision_tower()(depths).to(self.device)
                mask_embeds, depth_embeds = self.get_region_extractor()(hres_tower_features, depth_features, masks)
            else:
                mask_embeds, depth_embeds = self.get_region_extractor()(hres_tower_features, None, masks)
        else:
            lres_tower_features = tower_features

        image_features = self.get_mm_projector()(lres_tower_features).to(self.device)

        # Note (kentang-mit@): image start / end is not implemented here to support pretraining.
        if getattr(self.config, "turn_mm_projector", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask
        input_ids_copy = input_ids.clone()
        # kentang-mit@: Otherwise tokenizer out of bounds. Embeddings of image tokens will not be used.
        input_ids_copy[input_ids_copy == IMAGE_TOKEN_INDEX] = 0
        input_embeds = self.llm.model.embed_tokens(input_ids_copy)

        input_ids = [
            cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        input_embeds_1 = [
            cur_input_embeds[cur_attention_mask]
            for cur_input_embeds, cur_attention_mask in zip(input_embeds, attention_mask)
        ]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0

        # kentang-mit@: If some part of the model is executed in the loop, the the loop length needs to be a constant.
        for batch_idx, cur_input_ids in enumerate(input_ids):
            cur_input_ids = input_ids[batch_idx]
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[0]
                cur_input_embeds_1 = input_embeds_1[batch_idx]
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                # kenang-mit@: we do not have placeholdr image for text-only data now.
                continue

            cur_input_embeds = input_embeds_1[batch_idx]
            image_token_indices = (
                [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            )

            if hasattr(self.config, "enable_region") and self.config.enable_region:
                pos_mask = cur_input_ids == vision_tower.config.llm_mask_token_id
                num_mask = pos_mask.sum().item()
                # we pad the mask from num_batch to num_images, use cur_image_idx instead of batch_idx
                mask_embed = mask_embeds[cur_image_idx]

                if mask_embed is None and num_mask > 0:
                    print("Error: mask embed is None, but the num of <mask> is not 0!!!")

                if mask_embed is not None:
                    mask_embed = mask_embed[:num_mask]
                    zeros_embeds = torch.zeros_like(cur_input_embeds)
                    zeros_embeds[pos_mask] = mask_embed.to(device=zeros_embeds.device, dtype=zeros_embeds.dtype)
                    cur_input_embeds = (
                        cur_input_embeds * (~pos_mask).to(cur_input_embeds.dtype).unsqueeze(-1) + zeros_embeds
                    )

            if hasattr(self.config, "enable_depth") and self.config.enable_depth and depths is not None:
                pos_depth = cur_input_ids == vision_tower.config.llm_depth_token_id
                num_depth = pos_depth.sum().item()
                depth_embed = depth_embeds[cur_image_idx]

                if depth_embed is None and num_depth > 0:
                    print("Error: depth embed is None, but the num of <depth> is not 0!!!")

                if depth_embed is not None:
                    depth_embed = depth_embed[:num_depth]
                    zeros_embeds = torch.zeros_like(cur_input_embeds)
                    zeros_embeds[pos_depth] = depth_embed.to(device=zeros_embeds.device, dtype=zeros_embeds.dtype)
                    cur_input_embeds = (
                        cur_input_embeds * (~pos_depth).to(cur_input_embeds.dtype).unsqueeze(-1) + zeros_embeds
                    )

            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            cur_input_embeds_no_im = []
            for i in range(len(image_token_indices) - 1):
                if sp_degree > 1 and i == 0 and sp_rank != 0:  # Handle sequence parallelism
                    cur_input_ids_noim.append(cur_input_ids[0:0])
                    cur_labels_noim.append(cur_labels[0:0])
                    cur_input_embeds_no_im.append(cur_input_embeds[0:0])
                    continue
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_input_embeds_no_im.append(cur_input_embeds[image_token_indices[i] + 1 : image_token_indices[i + 1]])

            cur_new_input_embeds = []
            cur_new_labels = []
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(
                        torch.full(
                            (cur_image_features.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.llm.config, "tokenizer_model_max_length", None)
        if tokenizer_model_max_length is not None:
            if any(len(x) > tokenizer_model_max_length for x in new_input_embeds):
                warnings.warn("Inputs truncated!")
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        # max_len = tokenizer_model_max_length
        # print("Warning: using max_len as tokenizer_model_max_length")
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.llm.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )
            else:
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        # if sp_degree > 1:  # Handle sequence parallelism
        #     if sp_rank not in self.global_seq_len:
        #         self.global_seq_len[sp_rank] = position_ids.shape[-1]
        #     else:
        #         assert self.global_seq_len[sp_rank] == position_ids.shape[-1]

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        # We will not use packing here when sequence parallelism is enabled.
        if PROCESS_GROUP_MANAGER is not None:
            return (
                None,
                _position_ids,
                attention_mask,
                past_key_values,
                new_input_embeds,
                new_labels,
            )

        return (
            None,
            position_ids,
            attention_mask,
            past_key_values,
            new_input_embeds,
            new_labels,
        )

    def repack_multimodal_data(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        inputs_embeds,
        labels,
    ):
        # Handle sequence parallelism
        PROCESS_GROUP_MANAGER = get_pg_manager()
        # if PROCESS_GROUP_MANAGER is None:
        #     sp_degree = -1
        #     sp_rank = -1
        # else:
        #     sp_degree = PROCESS_GROUP_MANAGER.sp_degree
        #     sp_rank = PROCESS_GROUP_MANAGER.sp_rank

        # We will not use packing here when sequence parallelism is enabled.
        # However, we do resharding here to ensure the sequence length is the same across all ranks.
        if PROCESS_GROUP_MANAGER is not None:

            sp_degree = PROCESS_GROUP_MANAGER.sp_degree
            sp_rank = PROCESS_GROUP_MANAGER.sp_rank
            sp_group = PROCESS_GROUP_MANAGER.ulysses_pg
            bs, shard_seqlen = position_ids.shape
            ulysess_seq_len = [torch.zeros(1, dtype=torch.int64, device=position_ids.device) for _ in range(sp_degree)]
            dist.all_gather(ulysess_seq_len, torch.tensor(shard_seqlen, device=position_ids.device), group=sp_group)
            # global_seq_len = torch.sum(torch.cat(ulysess_seq_len, dim=0)).item()

            # Gather attention_mask and reshard it evenly
            attention_mask_list = [
                torch.zeros((bs, ulysess_seq_len[i]), dtype=attention_mask.dtype, device=attention_mask.device)
                for i in range(sp_degree)
            ]
            dist.all_gather(attention_mask_list, attention_mask, group=sp_group)
            effective_seqlen_list = [attention_mask_list[i].sum(dim=-1) for i in range(sp_degree)]
            effective_seqlen = torch.stack(effective_seqlen_list, dim=-1)
            effective_seqlen_batch_list = torch.unbind(effective_seqlen, dim=0)

            global_attention_mask_list = []
            for i in range(bs):
                global_attention_mask_batch_list = []
                for j in range(sp_degree):
                    global_attention_mask_batch_list.append(
                        attention_mask_list[j][i, : effective_seqlen_batch_list[i][j]]
                    )
                global_attention_mask_list.append(torch.cat(global_attention_mask_batch_list, dim=0))
            global_attention_mask = torch.nn.utils.rnn.pad_sequence(
                global_attention_mask_list, batch_first=True, padding_value=False
            )

            # Hyperparameters for sequence parallelism resharding
            global_seq_len = global_attention_mask.shape[-1]
            seq_len_sharded = global_seq_len // sp_degree
            start_idx_reshard = seq_len_sharded * sp_rank
            end_idx_reshard = start_idx_reshard + seq_len_sharded if sp_rank < sp_degree - 1 else global_seq_len
            # if sp_rank == 0:
            #     start_idx = 0
            # else:
            #     start_idx = torch.sum(torch.cat(ulysess_seq_len[:sp_rank], dim=0)).item()

            new_attention_mask = torch.narrow(
                global_attention_mask, 1, start_idx_reshard, end_idx_reshard - start_idx_reshard
            )

            # Gather position_ids and reshard it evenly
            position_ids_list = [
                torch.zeros((bs, ulysess_seq_len[i]), dtype=position_ids.dtype, device=position_ids.device)
                for i in range(sp_degree)
            ]
            dist.all_gather(position_ids_list, position_ids, group=sp_group)
            global_position_ids_list = []
            for i in range(bs):
                global_position_ids_batch_list = []
                for j in range(sp_degree):
                    global_position_ids_batch_list.append(position_ids_list[j][i, : effective_seqlen_batch_list[i][j]])
                global_position_ids_list.append(torch.cat(global_position_ids_batch_list, dim=0))
            global_position_ids = torch.nn.utils.rnn.pad_sequence(
                global_position_ids_list, batch_first=True, padding_value=-1
            )
            new_position_ids = torch.narrow(
                global_position_ids, 1, start_idx_reshard, end_idx_reshard - start_idx_reshard
            )

            # Gather labels and reshard it evenly
            labels_list = [
                torch.zeros((bs, ulysess_seq_len[i]), dtype=labels.dtype, device=labels.device)
                for i in range(sp_degree)
            ]
            dist.all_gather(labels_list, labels, group=sp_group)
            global_labels_list = []
            for i in range(bs):
                global_labels_batch_list = []
                for j in range(sp_degree):
                    global_labels_batch_list.append(labels_list[j][i, : effective_seqlen_batch_list[i][j]])
                global_labels_list.append(torch.cat(global_labels_batch_list, dim=0))
            global_labels = torch.nn.utils.rnn.pad_sequence(
                global_labels_list, batch_first=True, padding_value=IGNORE_INDEX
            )
            new_labels = torch.narrow(global_labels, 1, start_idx_reshard, end_idx_reshard - start_idx_reshard)

            # Gather inputs_embeds and reshard it evenly
            # TODO: Fix the non-enough images.
            # inputs_embeds_list = [torch.zeros((bs, ulysess_seq_len[i], inputs_embeds.shape[-1]), dtype=inputs_embeds.dtype, device=inputs_embeds.device, requires_grad=True) for i in range(sp_degree)]
            # dist.all_gather(inputs_embeds_list, inputs_embeds, group=sp_group)
            # global_inputs_embeds_list = []
            # for i in range(bs):
            #     global_inputs_embeds_batch_list = []
            #     for j in range(sp_degree):
            #         global_inputs_embeds_batch_list.append(inputs_embeds_list[j][i, :effective_seqlen_batch_list[i][j]])
            #     global_inputs_embeds_list.append(torch.cat(global_inputs_embeds_batch_list, dim=0))
            # global_inputs_embeds = torch.nn.utils.rnn.pad_sequence(global_inputs_embeds_list, batch_first=True, padding_value=0)
            # new_inputs_embeds = torch.narrow(global_inputs_embeds, 1, start_idx_reshard, end_idx_reshard - start_idx_reshard)

            # Gather all hidden states and flaten them
            ulysess_seq_len_cat = torch.cat(ulysess_seq_len, dim=0)
            global_inputs_embeds_list = []
            if sp_rank == 0:
                original_start_id = 0
                original_end_id = torch.sum(ulysess_seq_len_cat[: sp_rank + 1]).item()
            elif sp_rank == sp_degree - 1:
                original_start_id = torch.sum(ulysess_seq_len_cat[:sp_rank]).item()
                original_end_id = torch.sum(ulysess_seq_len_cat[: sp_rank + 1]).item()
            else:
                original_start_id = torch.sum(ulysess_seq_len_cat[:sp_rank]).item()
                original_end_id = torch.sum(ulysess_seq_len_cat[: sp_rank + 1]).item()
            all_inputs_embeds = torch.zeros(
                bs,
                torch.sum(ulysess_seq_len_cat),
                inputs_embeds.shape[-1],
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            ).contiguous()
            all_inputs_embeds[:, original_start_id:original_end_id, :] += inputs_embeds
            dist.barrier(group=sp_group)
            dist.all_reduce(all_inputs_embeds, group=sp_group)
            dist.barrier(group=sp_group)
            for i in range(bs):
                global_inputs_embeds_batch_list = []
                for j in range(sp_degree):
                    prev_len = torch.sum(ulysess_seq_len_cat[:j]).item() if j > 0 else 0
                    start_id = prev_len
                    end_id = prev_len + effective_seqlen_batch_list[i][j]
                    global_inputs_embeds_batch_list.append(all_inputs_embeds[i, start_id:end_id])
                global_inputs_embeds_list.append(torch.cat(global_inputs_embeds_batch_list, dim=0))
            global_inputs_embeds = torch.nn.utils.rnn.pad_sequence(
                global_inputs_embeds_list, batch_first=True, padding_value=0
            )
            new_inputs_embeds = torch.narrow(
                global_inputs_embeds, 1, start_idx_reshard, end_idx_reshard - start_idx_reshard
            )

            return (
                None,
                new_position_ids,
                new_attention_mask,
                past_key_values,
                new_inputs_embeds,
                new_labels,
                None,  # sorted_seqlens_in_batch set as None for sequence parallelism
            )

        # kentang-mit@: reorder and repack (reduce computation overhead)
        # requires transformers replacement.
        new_inputs_embeds = []
        new_position_ids = []
        new_labels = []
        seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
        sorted_seqlens_in_batch, sorted_idx = torch.sort(seqlens_in_batch, descending=True)
        max_seqlen = inputs_embeds.shape[1]

        cur_inputs_embeds = []
        cur_position_ids = []
        cur_labels = []
        cur_batch_len = 0
        for i in range(len(sorted_seqlens_in_batch)):
            cur_seqlen = sorted_seqlens_in_batch[i].item()
            if cur_seqlen + cur_batch_len <= max_seqlen:
                cur_batch_len += cur_seqlen
                # each item: num_tokens x num_channels
                # remove padding on-the-fly
                cur_inputs_embeds.append(inputs_embeds[sorted_idx[i]][attention_mask[sorted_idx[i]]])
                cur_position_ids.append(
                    torch.arange(
                        cur_inputs_embeds[-1].shape[0],
                        device=cur_inputs_embeds[-1].device,
                    )
                )
                # each item: num_tokens
                # remove padding on-the-fly
                cur_labels.append(labels[sorted_idx[i]][attention_mask[sorted_idx[i]]])
            else:
                new_inputs_embeds.append(torch.cat(cur_inputs_embeds, 0))
                new_position_ids.append(torch.cat(cur_position_ids, 0))
                new_labels.append(torch.cat(cur_labels, 0))
                # The current batch is too long. We will start a new batch.
                cur_batch_len = cur_seqlen
                cur_inputs_embeds = [inputs_embeds[sorted_idx[i]][attention_mask[sorted_idx[i]]]]
                cur_position_ids = [
                    torch.arange(
                        cur_inputs_embeds[-1].shape[0],
                        device=cur_inputs_embeds[-1].device,
                    )
                ]
                cur_labels = [labels[sorted_idx[i]][attention_mask[sorted_idx[i]]]]
            # Mask the first token in the labels for every sample
            # cur_labels[-1][0] = IGNORE_INDEX

        if len(cur_inputs_embeds):
            new_inputs_embeds.append(torch.cat(cur_inputs_embeds, 0))
            new_position_ids.append(torch.cat(cur_position_ids, 0))
            new_labels.append(torch.cat(cur_labels, 0))

        new_inputs_embeds = torch.nn.utils.rnn.pad_sequence(
            new_inputs_embeds, batch_first=True, padding_value=self.llm.pad_token_id
        )

        new_position_ids = torch.nn.utils.rnn.pad_sequence(new_position_ids, batch_first=True, padding_value=-1)

        new_labels = torch.nn.utils.rnn.pad_sequence(new_labels, batch_first=True, padding_value=IGNORE_INDEX)
        ## yunhao: it's currently a workaround to avoid errors for seq_len < 100
        new_attention_mask = new_position_ids.ne(-1)
        # sanity check
        assert new_attention_mask.sum() == attention_mask.sum()

        # Handle sequence parallelism: Calculate the position ids for sequence parallelism
        # NOTE: This implementation only works for [<bos>, <img>, ..., <img>, <caption>] pattern
        # if sp_degree > 1 and sp_rank > 0:
        #     cur_len = new_position_ids.shape[-1]
        #     if sp_rank < sp_degree - 1:  # Intermediate ranks
        #         offset = cur_len * sp_rank + 1
        #         new_position_ids = new_position_ids + offset
        #     elif sp_rank == sp_degree - 1:  # The last rank
        #         assert new_labels[0, -1] != IGNORE_INDEX, "The first sequence should be longest one."
        #         last_img_token_index = torch.where(new_labels[0] == IGNORE_INDEX)[0][-1]
        #         # print(f"last_img_token_index, {last_img_token_index}")
        #         # if sp_degree == 2: # Handle SP=2, because of bos_token
        #         #     offset = last_img_token_index + 3
        #         # else:
        #         #     offset = (last_img_token_index + 2) * sp_rank + 1
        #         offset = (last_img_token_index + 1) * sp_rank + 1
        #         offset_mask = new_position_ids != -1
        #         new_position_ids[offset_mask] += offset
        #     else:
        #         raise ValueError(f"sp_rank {sp_rank} is out of range {sp_degree}")

        return (
            None,
            new_position_ids,
            new_attention_mask,
            past_key_values,
            new_inputs_embeds,
            new_labels,
            sorted_seqlens_in_batch,
        )

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.enable_region:
            tokenizer.add_tokens([DEFAULT_MASK_TOKEN, DEFAULT_DEPTH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            # record mask/depth token id
            vision_tower_cfg = self.get_vision_tower().config
            vision_tower_cfg.llm_mask_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_MASK_TOKEN)
            vision_tower_cfg.llm_depth_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_DEPTH_TOKEN)

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg
            ## TODO yunhao: handle cases for <im_st> <im_end>
            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location="cpu")
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}."
                    )
        elif model_args.mm_use_im_patch_token:
            if model_args.mm_projector:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
