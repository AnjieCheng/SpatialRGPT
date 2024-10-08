import argparse
import copy
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# add spatialrgpt main path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from gradio_box_promptable_image import BoxPromptableImage
from utils.markdown import markdown_default, process_markdown
from utils.sam_utils import get_box_inputs
from utils.som import draw_mask_and_number_on_image

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, process_images, process_regions, tokenizer_image_token
from llava.model.builder import load_pretrained_model

if "DEPTH_ANYTHING_PATH" not in os.environ:
    print("DepthAnything source path not set...please check")
else:
    DEPTH_ANYTHING_PATH = os.environ["DEPTH_ANYTHING_PATH"]

if "SAM_CKPT_PATH" not in os.environ:
    print("SAM checkpoint path not set...please check")
else:
    SAM_CKPT_PATH = os.environ["SAM_CKPT_PATH"]


examples = [
    [{"image": "../demo_images/urban.png", "points": [[]]}],
]

conv = None
conv_history = {"user": [], "model": []}


def get_depth_predictor():
    sys.path.append(DEPTH_ANYTHING_PATH)
    from depth_anything.dpt import DepthAnything
    from depth_anything.util.transform import NormalizeImage, PrepareForNet, Resize
    from torchvision.transforms import Compose

    # Build depth model
    depth_model = DepthAnything(
        {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024], "localhub": False}
    )
    depth_model.load_state_dict(torch.load(f"{DEPTH_ANYTHING_PATH}/checkpoints/depth_anything_vitl14.pth"))

    depth_transform = Compose(
        [
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method="lower_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )
    print("Depth model successfully loaded!")
    return depth_model.cuda(), depth_transform


def get_sam_predictor():
    from segment_anything_hq import SamPredictor, sam_model_registry

    sam = sam_model_registry["vit_h"](checkpoint=SAM_CKPT_PATH).cuda()
    sam_predictor = SamPredictor(sam)
    return sam_predictor


def segment_using_boxes(input_str, prompt_dict, use_segmentation, use_depth, use_bfloat, follow_up):
    raw_image, prompts = prompt_dict["image"], prompt_dict["points"]
    print(prompt_dict["points"])
    orig_h, orig_w = raw_image.shape[:2]
    bboxes = np.array(get_box_inputs(prompts))

    seg_masks = []
    segmented_image = raw_image

    if use_segmentation:
        sam_predictor.set_image(raw_image)
        for bbox in bboxes:
            masks, scores, logits = sam_predictor.predict(box=np.asarray(bbox), multimask_output=True)
            seg_masks.append(masks[np.argmax(scores)].astype(np.uint8))

    else:
        # using bounding box
        for bbox in bboxes:
            zero_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            x1, y1, x2, y2 = map(int, bbox)
            zero_mask[y1:y2, x1:x2] = 1
            seg_masks.append(zero_mask)

    if len(seg_masks) > 0:
        region_labels = [f"Region {i}" for i in range(len(seg_masks))]
        image_rgb = cv2.resize(raw_image, (int(640 / (raw_image.shape[0]) * (raw_image.shape[1])), 640))
        segmented_image = draw_mask_and_number_on_image(
            image_rgb, seg_masks, region_labels, label_mode="1", alpha=0.5, anno_mode=["Mask", "Mark", "Box"]
        )

    return seg_masks, segmented_image


def get_depth_map(input_str, prompt_dict, use_segmentation, use_depth, use_bfloat, follow_up):
    raw_image = prompt_dict["image"]
    orig_h, orig_w = raw_image.shape[:2]

    depth_input_image = depth_transform({"image": raw_image / 255.0})["image"]
    depth_input_image = torch.from_numpy(depth_input_image).unsqueeze(0).cuda()

    # depth/disparity shape: 1xHxW
    raw_depth = depth_model(depth_input_image)
    raw_depth = F.interpolate(raw_depth[None], (orig_h, orig_w), mode="bilinear", align_corners=False)[0, 0]
    raw_depth = raw_depth.detach().cpu().numpy()

    raw_depth = (raw_depth - raw_depth.min()) / (raw_depth.max() - raw_depth.min()) * 255.0
    raw_depth = raw_depth.astype(np.uint8)
    colorized_depth = np.stack([raw_depth, raw_depth, raw_depth], axis=-1)
    return colorized_depth


def inference_vlm(
    input_str, prompt_dict, use_segmentation, use_depth, use_bfloat, follow_up, colorized_depth, seg_masks
):
    global conv, conv_history  # Use the global variables
    raw_image = prompt_dict["image"]

    if use_depth:
        query = re.sub(r"<region\d+>", "<mask> <depth>", input_str)  # replace <regionX>
    else:
        query = re.sub(r"<region\d+>", "<mask>", input_str)  # replace <regionX>

    if not follow_up:
        # only append IMAGE_TOKEN for first round
        query = DEFAULT_IMAGE_TOKEN + "\n" + query

        # reset conv_templates and history
        conv = conv_templates[args.conv_mode].copy()
        conv_history = {"user": [], "model": []}

    print("input: ", query)

    conv_history["user"].append(input_str)
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # find all occurrences of the "<regionX>" where X is a digit
    # loop through all past histories, so compatible no matter followup or not
    region_indices = []
    for user_input in conv_history["user"]:
        _region_indices = re.findall(r"<region(\d+)>", user_input)
        _region_indices = [int(index) for index in _region_indices]
        region_indices.extend(_region_indices)

    images_tensor = process_images([Image.fromarray(raw_image)], image_processor, spatialrgpt_model.config).to(
        spatialrgpt_model.device, dtype=torch.float16
    )
    depths_tensor = process_images([Image.fromarray(colorized_depth)], image_processor, spatialrgpt_model.config).to(
        spatialrgpt_model.device, dtype=torch.float16
    )
    if len(seg_masks) > 0:
        # this masks_tensor contains all non repeat region masks
        masks_tensor = process_regions(seg_masks, image_processor, spatialrgpt_model.config).to(
            spatialrgpt_model.device, dtype=torch.float16
        )
        masks_tensor = masks_tensor[region_indices]
    else:
        masks_tensor = None
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(spatialrgpt_model.device)
    )

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    if use_bfloat:
        selected_dtype = torch.bfloat16
    else:
        selected_dtype = torch.float16

    spatialrgpt_model.to(dtype=selected_dtype)

    with torch.inference_mode():
        output_ids = spatialrgpt_model.generate(
            input_ids,
            images=[images_tensor.to(dtype=selected_dtype).cuda()],
            depths=[depths_tensor.to(dtype=selected_dtype).cuda()],
            masks=[masks_tensor],
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    print(f"Raw output: {outputs}")

    # post processing, remap the output region index to input
    input_region_indices = re.findall(r"<region(\d+)>", input_str)
    mapping_dict = {str(out_index): str(in_index) for out_index, in_index in enumerate(input_region_indices)}
    remapped_outputs = re.sub(r"\[([0-9]+)\]", lambda x: f"[{mapping_dict[x.group(1)]}]", outputs)
    print(f"Post-process output: {remapped_outputs}")

    # After each round,
    # 1. pop out the empty one
    # 2. append the output
    conv.messages.pop()
    conv.append_message(conv.roles[1], outputs)
    conv_history["model"].append(remapped_outputs)

    markdown_out = process_markdown(remapped_outputs, [])
    return markdown_out


def inference(*args):
    seg_masks, segmented_image = segment_using_boxes(*args)
    colorized_depth = get_depth_map(*args)
    generated_texts = inference_vlm(*args, colorized_depth, seg_masks)
    return segmented_image, generated_texts
    # inference_depth(*args)


def build_promptable_segmentation_tab(prompter, label, process):

    output_img = gr.Image()

    with gr.Blocks() as tab_layout:
        with gr.Row(equal_height=False):
            with gr.Column():

                with gr.Column(variant="panel"):
                    input_image = prompter(label=label)

                    with gr.Row():
                        use_segmentation = gr.Checkbox(label="Use SAM", value=True)
                        use_depth = gr.Checkbox(label="Use <depth>", value=True)
                        use_bfloat = gr.Checkbox(label="Use bfloat", value=True)
                        follow_up = gr.Checkbox(label="Followup", value=False)

                    with gr.Row():
                        input_str = gr.Textbox(
                            lines=1,
                            placeholder="Type in the text instruction here.",
                            label="Text Instruction",
                            show_label=False,
                        )

                    with gr.Row():
                        clear_btn = gr.ClearButton(components=[input_image, input_str])
                        submit_btn = gr.Button("Submit", variant="primary")

                with gr.Column():
                    gr.Examples(examples=examples, inputs=[input_image])

            with gr.Column():
                with gr.Column(variant="panel"):
                    output_img.render()
                    output_txt = gr.Markdown(markdown_default)

            submit_btn.click(
                lambda *args: process(*args),
                [input_str, input_image, use_segmentation, use_depth, use_bfloat, follow_up],
                [output_img, output_txt],
            )

    return tab_layout


def build_demo():
    box_label = "Click to add region(s)"
    box_segmentation = build_promptable_segmentation_tab(BoxPromptableImage, box_label, inference)

    with gr.Blocks(theme="bethecloud/storj_theme") as demo:
        title_formatting = "<h1>SpatialRGPT: Grounded Spatial Reasoning in Vision Language Models</h1>"
        description = """
            This demo of SpatialRGPT allows you to prompt using boxes/SAM masks.

            ### Instructions ###

            * To reference a specific region, use `<region0>`, `<region1>`, and so on.
            * Unselect `Use SAM segmentation` checkbox to use box as mask.
            * When using RegionGPT, it is recommended to unselect `Use <depth>`.

            """

        gr.Markdown(title_formatting)
        gr.Markdown(description)

        gr.TabbedInterface(
            interface_list=[box_segmentation],
            tab_names=[
                "Main Demo",
            ],
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=str, default="pytorch", choices=["pytorch"])
    parser.add_argument("--conv-mode", type=str, default="llama_3")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
    )
    parser.add_argument("--model-name", type=str, default="vila-siglip-llama-3b")
    args = parser.parse_args()

    tokenizer, spatialrgpt_model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_name)
    mask_processor = copy.deepcopy(image_processor)
    mask_processor.do_normalize = False
    mask_processor.do_convert_rgb = False
    mask_processor.rescale_factor = 1.0

    spatialrgpt_model = spatialrgpt_model.cuda()

    sam_predictor = get_sam_predictor()
    depth_model, depth_transform = get_depth_predictor()

    demo = build_demo()
    demo.launch(share=True, debug=True)
