import random

from osdsynth.utils.logger import SkipImageException

global_qs_list = [
    "Can you provide a detailed description of this image in one paragraph of 30 words or less?",
    "Could you give a concise, detailed account of what's depicted in this image, all in one paragraph, aiming for no more than 30 words?",
    "Please offer a detailed portrayal of this image, condensed into one paragraph, keeping it under 30 words.",
    "Could you sketch out a detailed narrative of this image within a 30-word limit, all in a single paragraph?",
    "Would you be able to distill the essence of this image into one detailed paragraph of exactly 30 words?",
    "Can you unpack this image's details in a succinct paragraph, ensuring it's contained to up to 30 words?",
    "Could you elaborate on what this image shows, using no more than 30 words, all within one paragraph?",
    "In 30 words or fewer, can you dissect the details of this image, presented in a single paragraph?",
    "Could you render a vivid description of this image within the confines of 30 words, all in one paragraph?",
    "Please distill the details of this image into a brief yet rich description, not exceeding 30 words, all in one paragraph.",
    "Can you encapsulate this image's details in a comprehensive paragraph, without exceeding 30 words?",
    "Would you mind providing a detailed explanation of this image, adhering to a 30-word limit, all in one paragraph?",
    "Could you convey the intricate details of this image in a brief composition of no more than 30 words, contained in one paragraph?",
    "Please craft a detailed depiction of this image, ensuring it's concise with a maximum of 30 words, all within a single paragraph.",
    "Can you delineate the specifics of this image in a succinct narrative, capped at 30 words, all in one paragraph?",
]

landmark_prompt = [
    "In the context of: {global_caption} Try to categorize the following image into one of these categories ['indoor', 'outdoor'], use 'others' if it is not a natural image."
]


class CaptionImage:
    def __init__(self, cfg, logger, device, init_lava=False):
        self.cfg = cfg
        self.logger = logger
        self.device = device

        # Initialize LLava, deprecated, only use placeholders
        if init_lava:
            from osdsynth.processor.wrappers.llava import LLavaWrapper

            self.llava_processor = LLavaWrapper(cfg, logger, device)
        else:
            self.llava_processor = None

    def process_landmark(self, image_bgr):
        image_tensor, image_size = self.llava_processor.process_image(image_bgr)

        global_qs = random.choice(global_qs_list)
        global_caption = self.llava_processor.process_vqa(image_tensor, image_size, global_qs, 1024)

        landmark_qs = random.choice(landmark_prompt)
        landmark_qs = landmark_qs.format(global_caption=global_caption)
        landmark_caption = self.llava_processor.process_vqa(image_tensor, image_size, landmark_qs, 50)

        if "indoor" in landmark_caption.lower():
            landmark = "indoor"
        elif "outdoor" in landmark_caption.lower():
            landmark = "outdoor"
        else:
            raise SkipImageException("LLava failed to predict the landmark.")

        return landmark, global_caption

    def process_local_caption(self, detections, global_caption="", use_placeholder=True):
        n_objects = len(detections)
        if n_objects < 2:
            raise SkipImageException("Ddetected objects less than 2")

        for obj_idx in range(n_objects):
            if use_placeholder:
                detections[obj_idx]["caption"] = f"<region{obj_idx}>"
            else:
                assert self.llava_processor is not None
                detections[obj_idx]["caption"] = f"<region{obj_idx}>"

                # prepare dense caption
                cropped_image = detections[obj_idx]["image_crop_modified"]
                image_tensor, image_size = self.llava_processor.process_image(cropped_image, is_pil_rgb=True)

                local_qs_template = r""""Can you describe the {class_name} in this close-up within five words? Highlight its color, appearance, style. For example: 'Man in red hat walking', 'Wooden pallet with boxes'."""
                local_qs = local_qs_template.format(
                    global_caption=global_caption, class_name=detections[obj_idx]["class_name"]
                )
                local_caption = self.llava_processor.process_vqa(image_tensor, image_size, local_qs, 50)
                detections[obj_idx]["dense_caption"] = local_caption.lower().replace(".", "").strip('"')
        return detections
