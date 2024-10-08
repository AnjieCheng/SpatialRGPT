import argparse
import copy
import json
import math
import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as cocomask
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

prompts = [
    "Identify the object or feature present in the region denoted by <mask>.",
    "What category best describes the area represented by <mask>?",
    "Describe the content of the image section highlighted by <mask>.",
    "Can you specify the type of object or landscape within the bounds of <mask>?",
    "Which of the following categories best fits the region marked by <mask>? Provide your answer.",
    "What can you discern from the area indicated by <mask> in the image?",
    "Categorize the visual element within the area designated by <mask>.",
    "Give a brief description of the item or scene captured in the segment marked by <mask>.",
    "Which classification would you assign to the visual content found at <mask>?",
    "Determine and describe the primary subject located within <mask>.",
    "How would you label the section of the image encompassed by <mask>?",
    "Assess and classify the feature present within the confines of <mask>.",
    "If you were to tag the section indicated by <mask>, what tag would you use?",
    "What stands out to you in the region demarcated by <mask>? Please classify it.",
    "Evaluate the content of the image portion pinpointed by <mask> and provide its category.",
]


def keepit(ann):
    try:
        keep = ann["iscrowd"] == 0
        return keep
    except:
        return True


def get_crop_box(bboxes, image_info):
    short_size = min(image_info["height"], image_info["width"])
    bbox = bboxes[0]

    if bbox[3] - bbox[1] > short_size or bbox[2] - bbox[0] > short_size:
        return [0, 0, image_info["width"], image_info["height"]]

    center_x, center_y = int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)

    x_left, x_right = center_x - short_size // 2, center_x + short_size // 2
    y_top, y_bottom = center_y - short_size // 2, center_y + short_size // 2

    if x_left < 0:
        x_left, x_right = 0, short_size
    if x_right > short_size:
        x_left, x_right = image_info["width"] - short_size, image_info["width"]

    if y_top < 0:
        y_top, y_bottom = 0, short_size
    if y_bottom > short_size:
        y_top, y_bottom = image_info["height"] - short_size, image_info["height"]

    crop_bbox = [x_left, y_top, x_right, y_bottom]
    return crop_bbox


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


def generate_data_list(annotations):
    from pycocotools.coco import COCO

    coco = COCO(annotations)
    ids = list(sorted(coco.imgs.keys()))

    cid2name = {}
    for _, v in coco.cats.items():
        cid = v["id"]
        cname = v["name"]
        cid2name[cid] = cname.lower()

    data_list = []
    for _id in ids:
        data = {}
        img_info = coco.loadImgs(_id)[0]
        filename = img_info["coco_url"].split("/")[-1]
        dirname = img_info["coco_url"].split("/")[-2]
        image = os.path.join("coco", dirname, filename)
        data["image"] = image
        data["image_info"] = {"height": img_info["height"], "width": img_info["width"]}
        data["image_id"] = img_info["id"]

        anns = coco.loadAnns(coco.getAnnIds(_id))
        anns = copy.deepcopy(anns)
        anns = list(filter(keepit, anns))

        _data_list = [copy.deepcopy(data) for ann in anns]

        for _i, ann in enumerate(anns):
            box = copy.deepcopy(ann["bbox"])
            box[2] += box[0]
            box[3] += box[1]
            segmentation = copy.deepcopy(ann["segmentation"])
            _data_list[_i]["bbox"] = [box]
            _data_list[_i]["segmentation"] = [segmentation]

            cid = ann["category_id"]
            category_name = cid2name[cid]
            _data_list[_i]["category_name"] = category_name
            _data_list[_i]["score"] = 1.0

        if len(_data_list) == 0:
            continue

        data_list += _data_list

    return data_list


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(
        self, data_list, image_folder, tokenizer, image_processor, model_config, dataset_name, prompt_type, args
    ):
        self.data_list = data_list
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.dataset_name = dataset_name
        self.prompt_type = prompt_type
        self.args = args

        mask_processer = copy.deepcopy(self.image_processor)
        mask_processer.do_normalize = False
        mask_processer.do_convert_rgb = False
        mask_processer.rescale_factor = 1.0

        self.mask_processer = mask_processer

    def __getitem__(self, index):
        line = self.data_list[index]
        # image, bbox, image_info,
        image_file = line["image"]
        bboxes = line["bbox"]
        segmentations = line["segmentation"]
        image_info = line["image_info"]

        assert len(bboxes) == 1, "obo only support 1 bbox"

        crop_bbox = get_crop_box(bboxes, image_info)

        masks = []
        if self.prompt_type == "seg":
            for segmentation in segmentations:
                rle = cocomask.frPyObjects(segmentation, image_info["height"], image_info["width"])
                m = cocomask.decode(rle)
                m = np.sum(m, axis=2)  # sometimes there are multiple binary map (corresponding to multiple segs)
                m = m.astype(np.uint8)  # convert to np.uint8
                if args.erosion:
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (args.kernel, args.kernel))
                    m = cv2.erode(m, kernel, iterations=args.iterations)
                if args.dilation:
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (args.kernel, args.kernel))
                    m = cv2.dilate(m, kernel, iterations=args.iterations)

                image_aspect_ratio = getattr(self.model_config, "image_aspect_ratio", None)
                m = m[crop_bbox[1] : crop_bbox[3], crop_bbox[0] : crop_bbox[2]]
                if image_aspect_ratio == "pad":
                    m = pad_to_square(m)
                masks.append(m)
        else:
            for bbox in bboxes:
                zero_mask = np.zeros((image_info["height"], image_info["width"]), dtype=np.uint8)
                x1, y1, x2, y2 = map(int, bbox)
                zero_mask[y1:y2, x1:x2] = 1
                image_aspect_ratio = getattr(self.model_config, "image_aspect_ratio", None)
                zero_mask = zero_mask[crop_bbox[1] : crop_bbox[3], crop_bbox[0] : crop_bbox[2]]
                if image_aspect_ratio == "pad":
                    zero_mask = pad_to_square(zero_mask)
                masks.append(zero_mask)

        masks_pt = []
        for m in masks:
            m = self.mask_processer.preprocess(m[None, ...], return_tensors="pt")["pixel_values"][0]
            masks_pt.append(m)

        masks = torch.vstack(masks_pt).float()  # (n, h, w)

        num_reg = len(bboxes)
        question = random.choice(prompts)
        if num_reg == 1:
            pass
        else:
            question_chunk = question.split("<mask>")
            mask_tokens = ",".join([" <mask>"] * (num_reg - 1)) + " and <mask>"
            question = question_chunk[0] + mask_tokens + question_chunk[1]

        if self.dataset_name == "coco":
            qs = question + " Answer the question using a single word or phrase from COCO-80 categories."
        else:
            qs = question + " Answer the question using a single word or phrase from LVIS categories."

        if self.model_config.mm_use_im_start_end:
            raise ValueError
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert("RGB")
        image = image.crop(tuple(crop_bbox))

        images_tensor = process_images([image], self.image_processor, self.model_config)

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")

        return input_ids, images_tensor, masks, str(bboxes)

    def __len__(self):
        return len(self.data_list)


# DataLoader
def create_data_loader(
    data_list,
    image_folder,
    tokenizer,
    image_processor,
    model_config,
    batch_size=1,
    num_workers=4,
    dataset_name="lvis",
    prompt_type="seg",
    args=None,
):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(
        data_list, image_folder, tokenizer, image_processor, model_config, dataset_name, prompt_type, args
    )
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_name, args.model_base)

    data_list = generate_data_list(args.annotation_file)
    data_list = get_chunk(data_list, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    ans_file = open(answers_file, "w")

    data_loader = create_data_loader(
        data_list,
        args.image_folder,
        tokenizer,
        image_processor,
        model.config,
        dataset_name=args.dataset,
        prompt_type=args.prompt_type,
        args=args,
    )

    for (input_ids, image_tensor, masks, bboxes), line in tqdm(zip(data_loader, data_list), total=len(data_list)):
        idx = line["image"]

        stop_str = (
            conv_templates[args.conv_mode].sep
            if conv_templates[args.conv_mode].sep_style != SeparatorStyle.TWO
            else conv_templates[args.conv_mode].sep2
        )
        input_ids = input_ids.to(device="cuda", non_blocking=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True),
                masks=[masks[0].to(dtype=torch.float16, device="cuda", non_blocking=True)],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=64,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        outputs = outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()

        image_id = line["image_id"]
        ans_file.write(
            json.dumps(
                {
                    "question_id": idx,
                    "text": outputs,
                    "gt_name": line["category_name"],
                    "score": line["score"],
                    "bbox": line["bbox"],
                    "image_id": image_id,
                    "model_id": model_name,
                    "metadata": {},
                }
            )
            + "\n"
        )
        ans_file.flush()

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
    parser.add_argument("--dataset", type=str, default="lvis")
    parser.add_argument("--prompt_type", type=str, default="seg")
    parser.add_argument("--erosion", type=bool, default=False)
    parser.add_argument("--dilation", type=bool, default=False)
    parser.add_argument("--kernel", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
