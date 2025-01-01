import torch
from torch.nn import CrossEntropyLoss
from abc import ABC, abstractmethod
from typing import List
from PIL import Image
import numpy as np

from transformers import AutoTokenizer
from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM, LlavaConfig
from llava.utils import disable_torch_init
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, process_images

from ZoomEye.tree import ImageTree, Node
from ZoomEye.utils import *

class ZoomModel(ABC):
    def __init__(self, model_path: str, conv_type: str = "qwen_1_5", device: str = "cuda:0", torch_dtype=torch.float16, attn_implementation="flash_attention_2", padding_side="left", **kwargs) -> None:
        disable_torch_init()
        self.device = device
        self.dtype = torch_dtype
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_size=padding_side)
        self.tokenizer.padding_side = padding_side
        print("padding side:", self.tokenizer.padding_side)
        
        if "qwen" in model_path:
            self.model = LlavaQwenForCausalLM.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
                torch_dtype=self.dtype,
                attn_implementation=attn_implementation,
            )
        else:
            llava_cfg = LlavaConfig.from_pretrained(model_path)
            if "v1.5" in model_path.lower():
                llava_cfg.delay_load = True
            self.model = LlavaLlamaForCausalLM.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
                torch_dtype=self.dtype,
                attn_implementation=attn_implementation,
                config=llava_cfg
            )
        print("model:", type(self.model))
        self.model.config.tokenizer_padding_side = padding_side
        self.model.to(self.device)

        vision_tower = self.model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=self.device)
            vision_tower.to(device=self.device, dtype=torch.float16)
        self.image_processor = vision_tower.image_processor

        self.conv_type = conv_type
        self.bias_value = kwargs.get("bias_value", 0.2)
        print("bias_value:", self.bias_value)

        self.input_size = (self.image_processor.crop_size['width'], self.image_processor.crop_size['height'])
        self.background_color = tuple(int(x*255) for x in self.image_processor.image_mean)
        print("input size:", self.input_size)
        print("background color:", self.background_color)

        self.patch_scale = kwargs.get("patch_scale", None)
        print("patch scale:", self.patch_scale)

        self.init_prompts()
        self.init_index_yes_no()

    @abstractmethod
    def init_prompts(self):
        pass

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
    
    @abstractmethod
    def process_nodes_to_image_list(self, nodes: List[Node], image_pil: Image.Image) -> List[Image.Image]:
        pass
    
    @abstractmethod
    def get_prompt_tag(self, image_list: List[Image.Image]):
        pass

    @abstractmethod
    def process_image_list_to_tensor(self, image_list: List[Image.Image]):
        pass
    
    def get_prompt_from_qs(self, qs, response=None, show_prompt=False):
        conv = conv_templates[self.conv_type].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], response)
        prompt = conv.get_prompt()
        return prompt

    @torch.no_grad()
    def get_confidence_value(self, node: Node, image_pil: Image.Image, confidence_type: str, input_ele):
        assert confidence_type in ['existence', 'latent', 'answering']
        image_list = self.process_nodes_to_image_list([node], image_pil)
        prompt_tag = self.get_prompt_tag(image_list)
        qs = self.prompts[prompt_tag]["pre_information"] + self.prompts[prompt_tag][f"{confidence_type}_prompt"].format(input_ele)
        prompt = self.get_prompt_from_qs(qs)

        image_tensor, image_sizes = self.process_image_list_to_tensor(image_list)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        outputs = self.model(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            modalities=["image"] * len(input_ids),
            return_dict=True
        )
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

        image_tensor, image_sizes = self.process_image_list_to_tensor(image_list)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        
        output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                max_new_tokens=256,
            )
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        
        if return_zoomed_view:
            response = {'text': outputs, 'output_image': image_list[1] if len(image_list) > 1 else image_list[0]}
            return response
        
        return outputs
    
    @torch.inference_mode()
    def multiple_choices_inference(self, image_pil, question, options, searched_nodes: List[Node] = None):
        image_list = self.process_nodes_to_image_list(searched_nodes, image_pil)
        prompt_tag = self.get_prompt_tag(image_list)
        qs = self.prompts[prompt_tag]["pre_information"] + question
        prompt = self.get_prompt_from_qs(qs)
        image_tensor, image_sizes = self.process_image_list_to_tensor(image_list)

        question_input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)

        output_question = self.model(
            question_input_ids,
            use_cache=True,
            images=image_tensor,
            image_sizes=image_sizes)

        question_logits = output_question.logits
        question_past_key_values = output_question.past_key_values

        loss_list = []

        for option in options:
            full_prompt = self.get_prompt_from_qs(qs, option)
            full_input_ids = tokenizer_image_token(full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
            option_answer_input_ids = full_input_ids[:, question_input_ids.shape[1]:]

            output_option = self.model(input_ids=option_answer_input_ids,
                                use_cache=True,
                                attention_mask=torch.ones(1, question_logits.shape[1]+option_answer_input_ids.shape[1], device=full_input_ids.device),
                                past_key_values=question_past_key_values)
            
            logits = torch.cat([question_logits[:, -1:], output_option.logits[:, :-1]], 1)

            loss_fct = CrossEntropyLoss()
            logits = logits.view(-1, self.model.config.vocab_size)
            labels = option_answer_input_ids.view(-1)
            loss = loss_fct(logits, labels)
            loss_list.append(loss)

        option_chosen = torch.stack(loss_list).argmin()

        return option_chosen.cpu().item()
    
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

    def save_crop(self, image_pil, node, image_path):
        resized_bboxes = self.get_patch(node.state.bbox, image_pil.width, image_pil.height, patch_size=self.input_size[0], patch_scale=self.patch_scale)
        cropped_image = image_pil.crop(resized_bboxes)
        cropped_image.save(image_path)

    @torch.no_grad()
    def generate_visual_cues_using_ic(self, ic_examples, question: str, split_tag = r' and |, '):
        ic_question_template = ic_examples["question_template"]
        ic_question_list = ic_examples["question_list"]
        ic_response_list = ic_examples["response_list"]
        fs_prompt = []
        for q, a in zip(ic_question_list, ic_response_list):
            fs_prompt.append((ic_question_template.format(q), a))
        conv = conv_templates[self.conv_type].copy()
        for qa in fs_prompt:
            conv.append_message(conv.roles[0], qa[0])
            conv.append_message(conv.roles[1], qa[1])
        conv.append_message(conv.roles[0], ic_question_template.format(question))
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids.to(self.device)

        output_ids = self.model.generate(
            input_ids,
            images=None,
            do_sample=False,
            temperature=0,
            max_new_tokens=256,
        )
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        targets_sentence = extract_targets(outputs)
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
                if "anyres" in getattr(self.model.config, "image_aspect_ratio", "square"):
                    inputs['root_anyres'] = False
                conf_root = self.get_confidence_value(**inputs)
                if show_value:
                    print(target, conf_root)
                if conf_root < answering_confidence_threshold_upper:
                    ret.append(target)
        else:
            ret = targets[:]
        return ret
        



class ZoomModelLocal(ZoomModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.side_threshold = kwargs.get("side_threshold", 1000)
    
    def init_prompts(self):
        self.prompts = {
            "local":{
                "pre_information": f"{DEFAULT_IMAGE_TOKEN}\n",
                "latent_prompt": "According to your common sense knowledge and the content of the image, is it possible to find a {} in the image? Answer Yes or No and tell the reason.",
                "existence_prompt": "Is there a {} in the image? Answer Yes or No.",
                "answering_prompt": "Question: {}\nCould you answer the question based on the the available visual information? Answer Yes or No.",
            }
        }
    
    def get_bbox_in_square_image(self, bbox, left, top):
        x1, y1, x2, y2 = bbox
        return [x1+left, y1+top, x2+left, y2+top]
    
    def process_nodes_to_image_list(self, nodes: List[Node], image_pil: Image.Image):
        square_image, left, top = expand2square(image_pil, self.background_color)
        if len(nodes) == 0 or (len(nodes)==1 and nodes[0].is_root):
            return [square_image.resize(self.input_size)]

        resized_bboxes = [self.get_patch(node.state.bbox, image_pil.width, image_pil.height, patch_size=self.input_size[0], patch_scale=self.patch_scale) for node in nodes]
        resized_bboxes = merge_bbox_list(resized_bboxes)
        resized_bboxes = [self.get_bbox_in_square_image(bbox, left, top) for bbox in resized_bboxes]
        union_bbox = union_all_bboxes(resized_bboxes)
        w, h = union_bbox[2] - union_bbox[0], union_bbox[3] - union_bbox[1]
        if max(w,h) > self.side_threshold and len(resized_bboxes)==2:
            result = union_blocks_independent(square_image, resized_bboxes[0], resized_bboxes[1], self.input_size[0], self.background_color)
        else:
            result = square_image.crop(union_bbox)

        return [result]
    
    def get_prompt_tag(self, image_list):
        return "local"
    
    def process_image_list_to_tensor(self, image_list):
        image_tensor = process_images(image_list, self.image_processor, self.model.config)
        image_tensor = [_image.to(dtype=self.dtype, device=self.device) for _image in image_tensor]
        image_sizes = [image.size for image in image_list]
        return image_tensor, image_sizes
    

BOX_COLOR = 'red'

class ZoomModelGlobalLocal(ZoomModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.set_anyres_num(kwargs.get("max_anyres_num", 12))
    
    def set_anyres_num(self, max_anyres_num):
        patch_size = self.input_size[0]
        new_possible_resolutions = []
        for p in self.model.config.image_grid_pinpoints:
            x = p[0] // patch_size
            y = p[1] // patch_size
            if x * y <= max_anyres_num:
                new_possible_resolutions.append(p)
        self.model.config.image_grid_pinpoints = new_possible_resolutions
        print("new_image_grid_pinpoints:", self.model.config.image_grid_pinpoints)
        self.max_anyres_num = max_anyres_num
    
    def init_prompts(self):
        self.prompts = {
            "global":{
                "pre_information": f"{DEFAULT_IMAGE_TOKEN}\n",
                "latent_prompt": "According to your common sense knowledge and the content of image, is it possible to find a {} by further zooming in the image? Answer Yes or No and tell the reason.",
                "existence_prompt": "Is there a {} in the image? Answer Yes or No.",
                "answering_prompt": "Question: {}\nCould you answer the question based on the the available visual information? Answer Yes or No.",
            },
            "zoom":{
                "pre_information": f"{DEFAULT_IMAGE_TOKEN}\nThis is the main image, and the section enclosed by the {BOX_COLOR} rectangle is the focus region.\n{DEFAULT_IMAGE_TOKEN}\nThis is the zoomed-in view of the focus region.\n",
                "latent_prompt": "According to your common sense knowledge and the content of the zoomed-in view, along with its location in the image, is it possible to find a {} by further zooming in the current view? Answer Yes or No and tell the reason.",
                "existence_prompt": "Is there a {} in the zoomed-in view? Answer Yes or No.",
                "answering_prompt": "Question: {}\nCould you answer the question based on the the available visual information? Answer Yes or No.",
            },
        }
    
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
    
    def process_nodes_to_image_list(self, nodes, image_pil, root_anyres=True):
        square_image, left, top = expand2square(image_pil, self.background_color)
        if self.is_root_only(nodes):
            # return [square_image] if root_anyres else [square_image.resize(self.input_size)]
            return [deepcopy(image_pil)] if root_anyres else [square_image.resize(self.input_size)]
        if len(nodes) == 0 or self.include_root(nodes):
            # return [square_image]
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

        return [square_image.resize(self.input_size), zoomed_view]
    
    def get_prompt_tag(self, image_list):
        if len(image_list) == 1:
            prompt_tag = "global"
        elif len(image_list) == 2:
            prompt_tag = "zoom"
        else:
            raise ValueError
        return prompt_tag

    def process_image_list_to_tensor(self, image_list):
        image_tensor = process_images(image_list, self.image_processor, self.model.config)
        image_tensor = [_image.to(dtype=self.dtype, device=self.device) for _image in image_tensor]
        # (2, 3, 384, 384) --> (1, 3, 384, 384)
        image_tensor = [_image[[0], ...] if _image.shape[0]==2 else _image for _image in image_tensor]
        image_sizes = [image.size for image in image_list]
        return image_tensor, image_sizes

    @torch.no_grad()
    def get_confidence_value(self, node: Node, image_pil: Image.Image, confidence_type: str, input_ele, root_anyres=True):
        assert confidence_type in ['existence', 'latent', 'answering']
        image_list = self.process_nodes_to_image_list([node], image_pil, root_anyres=root_anyres)
        prompt_tag = self.get_prompt_tag(image_list)
        qs = self.prompts[prompt_tag]["pre_information"] + self.prompts[prompt_tag][f"{confidence_type}_prompt"].format(input_ele)
        prompt = self.get_prompt_from_qs(qs)

        image_tensor, image_sizes = self.process_image_list_to_tensor(image_list)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        outputs = self.model(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            modalities=["image"] * len(input_ids),
            return_dict=True
        )
        return self._cal_confidence(outputs)

    