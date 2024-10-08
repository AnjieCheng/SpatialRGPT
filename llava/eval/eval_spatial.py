import argparse
import copy
import json
import math
import os
import re
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools import mask as cocomask
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

if "DEPTH_ANYTHING_PATH" not in os.environ:
    print("DepthAnything source path not set...please check")
else:
    DEPTH_ANYTHING_PATH = os.environ["DEPTH_ANYTHING_PATH"]


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


def pad_to_square(array):
    H, W = array.shape
    max_side = max(H, W)

    padded_array = np.zeros((max_side, max_side), dtype=np.uint8)
    pad_h = (max_side - H) // 2
    pad_w = (max_side - W) // 2
    padded_array[pad_h : pad_h + H, pad_w : pad_w + W] = array

    return padded_array


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def clamp_box(bbox, image_info):
    h, w = image_info["height"], image_info["width"]
    bbox[0] = max(min(w, bbox[0]), 0)
    bbox[2] = max(min(w, bbox[2]), 0)

    bbox[1] = max(min(h, bbox[1]), 0)
    bbox[3] = max(min(h, bbox[3]), 0)


def get_depth_map(raw_image, depth_model, depth_transform):
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


def eval_model(args):
    disable_torch_init()

    # Depth Model
    depth_model, depth_transform = get_depth_predictor()
    print("Depth model successfully loaded!")

    # Model
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_name, args.model_base)

    with open(args.annotation_file) as f:
        questions = json.load(f)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    ans_file = open(answers_file, "w")

    mask_processer = copy.deepcopy(image_processor)
    mask_processer.do_normalize = False
    mask_processer.do_convert_rgb = False
    mask_processer.rescale_factor = 1.0

    for line in tqdm(questions, total=len(questions)):
        question_id = line["id"]
        image_file = line["image_info"]["file_path"]
        image_info = line["image_info"]
        text_question = line["text_q"]
        qa_info = line["qa_info"]
        # generate mask

        if args.use_mask:

            masks = []
            try:
                rles = line["rle"]
                for rle in rles:
                    m = cocomask.decode(rle)
                    m = m.astype(np.uint8)
                    image_aspect_ratio = getattr(model.config, "image_aspect_ratio", None)
                    if image_aspect_ratio == "pad":
                        m = pad_to_square(m)
                    masks.append(m)

            except:
                bboxes = line["bbox"]
                for bbox in bboxes:
                    zero_mask = np.zeros((image_info["height"], image_info["width"]), dtype=np.uint8)
                    clamp_box(bbox, image_info)
                    x1, y1, x2, y2 = map(int, bbox)
                    zero_mask[y1:y2, x1:x2] = 1
                    image_aspect_ratio = getattr(model.config, "image_aspect_ratio", None)
                    if image_aspect_ratio == "pad":
                        zero_mask = pad_to_square(zero_mask)
                    masks.append(zero_mask)

        else:
            masks = []
            print("using box!")
            bboxes = line["bbox"]
            for bbox in bboxes:
                zero_mask = np.zeros((image_info["height"], image_info["width"]), dtype=np.uint8)
                clamp_box(bbox, image_info)
                x1, y1, x2, y2 = map(int, bbox)
                zero_mask[y1:y2, x1:x2] = 1
                image_aspect_ratio = getattr(model.config, "image_aspect_ratio", None)
                if image_aspect_ratio == "pad":
                    zero_mask = pad_to_square(zero_mask)
                masks.append(zero_mask)

        if len(masks) > 0:
            masks_pt = []
            for m in masks:
                m = mask_processer.preprocess(m[None, ...], return_tensors="pt")["pixel_values"][0]
                masks_pt.append(m)
            masks = torch.vstack(masks_pt).float()  # (n, h, w)
        else:
            masks = None

        image = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")

        depth_input_image = np.array(Image.open(os.path.join(args.image_folder, image_file)).convert("RGB"))
        colorized_depth = get_depth_map(depth_input_image, depth_model, depth_transform)

        images_tensor = process_images([image], image_processor, model.config).to(model.device, dtype=torch.float16)
        depths_tensor = process_images([Image.fromarray(colorized_depth)], image_processor, model.config).to(
            model.device, dtype=torch.float16
        )

        conv = conv_templates[args.conv_mode].copy()
        conversations = line["conversations"]

        num_question = len(conversations) // 2
        for i in range(num_question):
            question = conversations[i * 2]["value"]
            question, _ = re.subn(r"<mask>", "<mask> <depth>", question)
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            input_ids = input_ids.to(device="cuda", non_blocking=True)
            input_ids = input_ids.unsqueeze(0)

            stop_str = (
                conv_templates[args.conv_mode].sep
                if conv_templates[args.conv_mode].sep_style != SeparatorStyle.TWO
                else conv_templates[args.conv_mode].sep2
            )

            model.to(dtype=torch.bfloat16)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images_tensor.to(dtype=torch.bfloat16, device="cuda", non_blocking=True),
                    depths=depths_tensor.to(dtype=torch.bfloat16, device="cuda", non_blocking=True),
                    masks=[masks.to(dtype=torch.bfloat16, device="cuda", non_blocking=True)]
                    if masks is not None
                    else None,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=128,
                    use_cache=True,
                )

            outputs = outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)]
            outputs = outputs.strip()

            ans_file.write(
                json.dumps(
                    {
                        "question_id": question_id,
                        "image": image_file,
                        "question": text_question,
                        "pred": outputs,
                        "gt": conversations[i * 2 + 1]["value"],
                        "model_id": model_name,
                        "qa_info": qa_info,
                    }
                )
                + "\n"
            )

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--annotation-file", type=str, default="")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--use-mask", type=bool, default=True)
    args = parser.parse_args()

    eval_model(args)
