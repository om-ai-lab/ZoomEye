import torch
from torch.nn import CrossEntropyLoss
from typing import List
from PIL import Image
import numpy as np

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from ZoomEye.tree import ImageTree, Node
from ZoomEye.utils import *

BOX_COLOR = "red"

class ZoomModelQwenVL:
    def __init__(self, model_path: str, device: str = "cuda:0", torch_dtype=torch.bfloat16, **kwargs) -> None:
        self.device = device
        self.dtype = torch_dtype
        load_kwargs = {}
        load_in_8bit = kwargs.get("load_in_8bit", False)
        if load_in_8bit:
            device_map = "auto"
        else:
            device_map = self.device

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            attn_implementation="flash_attention_2",
            device_map=device_map, 
            **load_kwargs
        )
        max_pixels = kwargs.get("max_pixels", 12845056)
        min_pixels = kwargs.get("min_pixels", 3136)
        print("max_pixels:", max_pixels)
        print("min_pixels:", min_pixels)
        self.processor.image_processor.max_pixels = max_pixels
        self.processor.image_processor.min_pixels = min_pixels

        self.bias_value = kwargs.get("bias_value", 0.6)
        print("bias_value:", self.bias_value)

        # We hard code the input size to 448x448 for QwenVL
        self.input_size = (448, 448)
        print("input size:", self.input_size)

        self.background_color = tuple(int(x*255) for x in self.processor.image_processor.image_mean)
        print("background color:", self.background_color)

        self.patch_scale = kwargs.get("patch_scale", None)
        print("patch scale:", self.patch_scale)

        self.init_prompts()
        self.init_index_yes_no()
    
    def init_index_yes_no(self):
        print("Yes:",self.tokenizer("Yes").input_ids)
        print("No:",self.tokenizer("No").input_ids)
        if len(self.tokenizer("Yes").input_ids) == 1 and len(self.tokenizer("No").input_ids) == 1: 
            self.index_yes = self.tokenizer("Yes").input_ids[0]
            self.index_no = self.tokenizer("No").input_ids[0]
        else:
            assert len(self.tokenizer("Yes").input_ids) == 2 and len(self.tokenizer("No").input_ids) == 2
            self.index_yes = self.tokenizer("Yes").input_ids[1]
            self.index_no = self.tokenizer("No").input_ids[1]
        print("index_yes:", self.index_yes)
        print("index_no:", self.index_no)
    
    def get_confidence_weight(self, node: Node, max_depth: int):
        coeff = (1 - self.bias_value) / (max_depth ** 2)
        return coeff * (node.depth**2) + self.bias_value

    def init_prompts(self):
        self.prompts = {
            "global":{
                "pre_information": "<image>\n",
                "latent_prompt": "According to your common sense knowledge and the content of image, is it possible to find a {} by further zooming in the image? Answer Yes or No and tell the reason.",
                "existence_prompt": "Is there a {} in the image? Answer Yes or No.",
                "answering_prompt": "Question: {}\nCould you answer the question based on the the available visual information? Answer Yes or No.",
            },
            "zoom":{
                "pre_information": f"<image>\nThis is the main image, and the section enclosed by the {BOX_COLOR} rectangle is the focus region.\n<image>\nThis is the zoomed-in view of the focus region.\n",
                "latent_prompt": "According to your common sense knowledge and the content of the zoomed-in view, along with its location in the image, is it possible to find a {} by further zooming in the current view? Answer Yes or No and tell the reason.",
                "existence_prompt": "Is there a {} in the zoomed-in view? Answer Yes or No.",
                "answering_prompt": "Question: {}\nCould you answer the question based on the the available visual information? Answer Yes or No.",
            },
        }
    
    @torch.no_grad()
    def generate_visual_cues_using_ic(self, ic_examples, question: str, split_tag = r' and |, '):
        ic_question_template = ic_examples["question_template"]
        ic_question_list = ic_examples["question_list"]
        ic_response_list = ic_examples["response_list"]
        
        message = []
        for q, a in zip(ic_question_list, ic_response_list):
            message.extend([
                {"role": "user", "content": [{"type": "text", "text": ic_question_template.format(q)}]},
                {"role": "assistant", "content": [{"type": "text", "text": a}]}
            ])
        message.extend([
            {"role": "user", "content": [{"type": "text", "text": ic_question_template.format(question)}]},
        ])
        texts = [self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)]
        model_inputs = self.processor(
            text=texts,
            images=None,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False)
        model_inputs = model_inputs.to(self.device)
        generated_ids = self.model.generate(**model_inputs, use_cache=True, max_new_tokens=256, do_sample=False)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        targets_sentence = extract_targets(response)
        if targets_sentence is not None:
            targets = split_targets_sentence(targets_sentence, split_tag)
        else:
            targets = []
        
        return targets
    
    def filter_visual_cues(self, image_pil:Image.Image, targets, root_node, decomposed_question_template, answering_confidence_threshold_upper, show_value=False):
        assert root_node.is_root 
        ret = []
        if len(targets) > 1:
            for target in targets:
                if target.startswith("all "):
                    ret.append(target)
                    continue
                inputs = {
                    "node": root_node,
                    "image_pil": image_pil,
                    "confidence_type": "answering",
                    "input_ele": decomposed_question_template.format(target)
                }
                inputs['root_anyres'] = False
                conf_root = self.get_confidence_value(**inputs)
                if show_value:
                    print(target, conf_root)
                if conf_root < answering_confidence_threshold_upper:
                    ret.append(target)
        else:
            ret = targets[:]
        return ret
    
    def get_prompt_from_qs(self, qs, response=None, show_prompt=False):
        message = []
        message.append({"role": "user", "content": []})
        while "<image>\n" in qs:
            index = qs.find("<image>\n")
            if index == 0:
                message[0]["content"].append({"type": "image"})
                qs = qs[len("<image>\n"):]
            else:
                message[0]["content"].append({"type": "text", "text": qs[:index]})
                qs = qs[index:]
        if len(qs) > 0:
            message[0]["content"].append({"type": "text", "text": qs})
        if response is not None:
            message.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
        texts = [self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True if response is None else False)]
        return texts[0]

    def resize_image(self, image_pil):
        img = deepcopy(image_pil)
        input_size = self.input_size[0]
        if img.width > img.height:
            img = img.resize((input_size, int(img.height * input_size / img.width)))
        else:
            img = img.resize((int(img.width * input_size / img.height), input_size))
        return img
    
    def get_prompt_tag(self, image_list):
        if len(image_list) == 1:
            prompt_tag = "global"
        elif len(image_list) == 2:
            prompt_tag = "zoom"
        else:
            raise ValueError
        return prompt_tag
    
    def save_crop(self, image_pil, node, image_path):
        resized_bboxes = self.get_patch(node.state.bbox, image_pil.width, image_pil.height, patch_size=self.input_size[0], patch_scale=self.patch_scale)
        cropped_image = image_pil.crop(resized_bboxes)
        cropped_image.save(image_path)

    def is_root_only(self, nodes: List[Node]):
        return (len(nodes)==1 and nodes[0].is_root)
    
    def include_root(self, nodes: List[Node]):
        return any(node.is_root for node in nodes)

    def get_bbox_in_square_image(self, bbox, left, top):
        x1, y1, x2, y2 = bbox
        return [x1+left, y1+top, x2+left, y2+top]
    
    def draw_bbox_arrow_in_square_image(self, square_image, resized_bbox, color):
        thickness = square_image.width//120
        new_bbox = visualize_bbox_and_arrow(square_image, resized_bbox, color, thickness, xyxy=True)
        return new_bbox

    def get_patch(self, bbox, image_width, image_height, patch_size, patch_scale=None):
        object_width = int(np.ceil(bbox[2]))
        object_height = int(np.ceil(bbox[3]))

        object_center_x = int(bbox[0] + bbox[2]/2)
        object_center_y = int(bbox[1] + bbox[3]/2)

        patch_width = max(object_width, patch_size)
        patch_height = max(object_height, patch_size)
        if patch_scale is not None:
            patch_width = int(patch_width*patch_scale)
            patch_height = int(patch_height*patch_scale)

        left = max(0, object_center_x-patch_width//2)
        right = min(left+patch_width, image_width)

        top = max(0, object_center_y-patch_height//2)
        bottom = min(top+patch_height, image_height)

        return [left, top, right, bottom]

    def get_original_image_with_color_bbox(self, square_image, left, top, image_pil):
        image_width, image_height = image_pil.size
        crop_coords = [left, top, left + image_width, top + image_height]
        cropped_image = square_image.crop(crop_coords)
        return cropped_image

    
    def process_nodes_to_image_list(self, nodes, image_pil, root_anyres=True):
        square_image, left, top = expand2square(image_pil, self.background_color)
        if self.is_root_only(nodes):
            return [deepcopy(image_pil)] if root_anyres else [self.resize_image(image_pil)]
        if len(nodes) == 0 or self.include_root(nodes):
            return [deepcopy(image_pil)]

        resized_bboxes = [self.get_patch(node.state.bbox, image_pil.width, image_pil.height, patch_size=self.input_size[0], patch_scale=self.patch_scale) for node in nodes]
        resized_bboxes = merge_bbox_list(resized_bboxes, threshold=0)

        full_color_bboxes = []
        for i in range(len(resized_bboxes)):
            resized_bbox = self.get_bbox_in_square_image(resized_bboxes[i], left, top)
            color_bbox = self.draw_bbox_arrow_in_square_image(square_image, resized_bbox, BOX_COLOR)
            full_color_bboxes.append(color_bbox)
        
        union_color_bboxes = union_all_bboxes(full_color_bboxes)
        if union_color_bboxes is None:
            return [square_image]
        zoomed_view = square_image.crop(union_color_bboxes)

        original_image_with_color_bbox = self.get_original_image_with_color_bbox(square_image, left, top, image_pil)

        # x = self.resize_image(original_image_with_color_bbox)
        # x.save("/training/shz/project/ZoomEye/demo/saves/x.jpg")
        # zoomed_view.save("/training/shz/project/ZoomEye/demo/saves/zoomed_view.jpg")

        return [self.resize_image(original_image_with_color_bbox), zoomed_view]
    

    @torch.no_grad()
    def get_confidence_value(self, node: Node, image_pil: Image.Image, confidence_type: str, input_ele, root_anyres=True):
        assert confidence_type in ['existence', 'latent', 'answering']
        image_list = self.process_nodes_to_image_list([node], image_pil, root_anyres=root_anyres)

        prompt_tag = self.get_prompt_tag(image_list)
        qs = self.prompts[prompt_tag]["pre_information"] + self.prompts[prompt_tag][f"{confidence_type}_prompt"].format(input_ele)
        prompt = self.get_prompt_from_qs(qs)

        model_inputs = self.processor(
            text=[prompt],
            images=image_list,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        )
        model_inputs = model_inputs.to(self.device)

        outputs = self.model(**model_inputs)     
        
        return self._cal_confidence(outputs)

    @torch.no_grad()
    def _cal_confidence(self, outputs):
        logits_yesno = outputs.logits[0, -1, [self.index_yes, self.index_no]]
        confidence = torch.softmax(logits_yesno, dim=-1)[0] 
        confidence = 2 * (confidence.item() - 0.5) # [-1, 1]
        return confidence

    @torch.inference_mode()
    def free_form_using_nodes(self, image_pil, question, searched_nodes: List[Node], return_zoomed_view=False):
        image_list = self.process_nodes_to_image_list(searched_nodes, image_pil)
        prompt_tag = self.get_prompt_tag(image_list)
        qs = self.prompts[prompt_tag]["pre_information"] + question
        prompt = self.get_prompt_from_qs(qs)
        model_inputs = self.processor(
            text=[prompt],
            images=image_list,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        )
        model_inputs = model_inputs.to(self.device)
        generated_ids = self.model.generate(**model_inputs, use_cache=True, max_new_tokens=256, do_sample=False)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        outputs = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        if return_zoomed_view:
            response = {'text': outputs, 'output_image': image_list[1] if len(image_list) > 1 else image_list[0]}
            return response
        
        return outputs

    @torch.inference_mode()
    def multiple_choices_inference(self, image_pil, question, options, searched_nodes: List[Node] = None):
        image_list = self.process_nodes_to_image_list(searched_nodes, image_pil)

        # zoomed_view = image_list[1] if len(image_list) > 1 else image_list[0]
        # zoomed_view.save("/training/shz/project/ZoomEye/demo/saves/zoomed_view.jpg")

        prompt_tag = self.get_prompt_tag(image_list)
        qs = self.prompts[prompt_tag]["pre_information"] + question
        prompt = self.get_prompt_from_qs(qs)
        
        model_inputs = self.processor(
            text=[prompt],
            images=image_list,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        )
        model_inputs = model_inputs.to(self.device)

        question_input_ids = model_inputs.input_ids
        len_question_input_ids = question_input_ids.shape[1]

        output_question = self.model(**model_inputs)
        len_question_logits = output_question.logits.shape[1]

        loss_list = []

        for option in options:
            full_prompt = self.get_prompt_from_qs(qs, option)

            full_model_inputs = self.processor(
                text=[full_prompt],
                images=image_list,
                return_tensors="pt",
                padding=True,
                padding_side="left",
            )
            full_model_inputs = full_model_inputs.to(self.device)
            full_input_ids = full_model_inputs.input_ids

            output_option = self.model(**full_model_inputs)
            logits = output_option.logits[:, len_question_logits-1:-1]

            loss_fct = CrossEntropyLoss()
            logits = logits.view(-1, self.model.config.vocab_size)
            labels = full_input_ids[:, len_question_input_ids:].view(-1)
            # print(logits.shape, labels.shape)
            loss = loss_fct(logits, labels)
            loss_list.append(loss)

        option_chosen = torch.stack(loss_list).argmin()

        return option_chosen.cpu().item()

