import random
from colorsys import hls_to_rgb, rgb_to_hls

import cv2
import gradio as gr

markdown_default = """
<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap" rel="stylesheet">
<style>
        .highlighted-text {
            font-family: 'Montserrat', sans-serif;
            font-weight: 600;
            font-size: 14px;
            color: rgb(255, 255, 239);
            background-color: rgb(225, 231, 254);
            border-radius: 7px;
            padding: 5px 7px;
            display: inline-block;
        }
        .regular-text {
            font-family: 'Montserrat', sans-serif;
            font-weight: 400;
            font-size: 14px;
        }
        .highlighted-response {
            font-family: 'Montserrat', sans-serif;
            font-weight: 600;
            font-size: 14px;
            border-radius: 6px;
            padding: 3px 4px;
            display: inline-block;
        }
</style>
<span class="highlighted-text" style='color:rgb(107, 100, 239)'>SpatialRGPT</span>

"""

title = "SpatialRGPT: Grounded Spatial Reasoning in Vision Language Models"


colors = [
    [255, 0, 0],  # Red
    [0, 255, 0],  # Green
    [0, 0, 255],  # Blue
    [255, 255, 0],  # Yellow
    [0, 255, 255],  # Cyan
    [255, 0, 255],  # Magenta
    [255, 192, 203],  # Pink
    [165, 42, 42],  # Brown
    [255, 165, 0],  # Orange
    [128, 0, 128],  # Purple
    [0, 0, 128],  # Navy
    [128, 0, 0],  # Maroon
    [128, 128, 0],  # Olive
    [70, 130, 180],  # Steel Blue
    [173, 216, 230],  # Light Blue
    [255, 192, 0],  # Gold
    [255, 165, 165],  # Light Salmon
    [255, 20, 147],  # Deep Pink
]


def process_markdown(output_str, color_history):
    markdown_out = output_str.replace("[SEG]", "")
    markdown_out = markdown_out.replace(
        "<p>", "<span class='highlighted-response' style='background-color:rgb[COLOR]'>"
    )
    markdown_out = markdown_out.replace("</p>", "</span>")

    for color in color_history:
        markdown_out = markdown_out.replace("[COLOR]", str(desaturate(tuple(color))), 1)

    markdown_out = f"""
    <br>
    {markdown_out}

    """
    markdown_out = markdown_default + "<p><span class='regular-text'>" + markdown_out  # + '</span></p>'
    return markdown_out


def desaturate(rgb, factor=0.65):
    """
    Desaturate an RGB color by a given factor.

    :param rgb: A tuple of (r, g, b) where each value is in [0, 255].
    :param factor: The factor by which to reduce the saturation.
                   0 means completely desaturated, 1 means original color.
    :return: A tuple of desaturated (r, g, b) values in [0, 255].
    """
    r, g, b = (x / 255.0 for x in rgb)
    h, l, s = rgb_to_hls(r, g, b)
    l = factor
    new_r, new_g, new_b = hls_to_rgb(h, l, s)
    return (int(new_r * 255), int(new_g * 255), int(new_b * 255))


def draw_bbox(image, boxes, color_history=[]):

    colors = [
        [255, 0, 0],  # Red
        [0, 255, 0],  # Green
        [0, 0, 255],  # Blue
        [255, 255, 0],  # Yellow
        [0, 255, 255],  # Cyan
        [255, 0, 255],  # Magenta
        [255, 192, 203],  # Pink
        [165, 42, 42],  # Brown
        [255, 165, 0],  # Orange
        [128, 0, 128],  # Purple
        [0, 0, 128],  # Navy
        [128, 0, 0],  # Maroon
        [128, 128, 0],  # Olive
        [70, 130, 180],  # Steel Blue
        [173, 216, 230],  # Light Blue
        [255, 192, 0],  # Gold
        [255, 165, 165],  # Light Salmon
        [255, 20, 147],  # Deep Pink
    ]
    new_image = image
    text = "<region_0>"
    font = cv2.FONT_HERSHEY_TRIPLEX
    font_scale = 1.0
    thickness = 4
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    for bbox_id, box in enumerate(boxes):
        if len(color_history) == 0:
            color = tuple(random.choice(colors))
        else:
            color = color_history[bbox_id]

        start_point = int(box[0]), int(box[1])
        end_point = int(box[2]), int(box[3])
        new_image = cv2.rectangle(new_image, start_point, end_point, color, thickness)
        if len(color_history) == 0:
            new_image = cv2.putText(
                new_image,
                f"<region {bbox_id}>",
                (int(box[0]), int(box[1]) + text_size[1]),
                font,
                font_scale,
                color,
                thickness,
            )

    return new_image
