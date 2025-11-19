import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)
from PIL import Image
import argparse
import os
import json
import warnings
warnings.filterwarnings("ignore")
import torch
from tqdm import tqdm
from ZoomEye.zoom_model import ZoomModelGlobalLocal, ZoomModelLocal
from ZoomEye.zoom_model_internvl import ZoomModelInternvl
from ZoomEye.zoom_model_qwenvl import ZoomModelQwenVL
from ZoomEye.zoom_eye import get_zoom_eye_response, get_direct_response
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="lmms-lab/llava-onevision-qwen2-7b-ov")
    parser.add_argument("--input_image", type=str, default="demo/demo.jpg")
    parser.add_argument("--question", type=str, default="What is the color of the soda can?")
    args = parser.parse_args()


    model_path = args.model_path
    config_path = os.path.join(model_path, "config.json")
    config = json.load(open(config_path, "r"))
    if "llava" in model_path.lower():
        if "anyres" in config['image_aspect_ratio']:
            zoom_model = ZoomModelGlobalLocal(model_path=model_path, conv_type="qwen_1_5", patch_scale=1.2, bias_value=0.6)
        else:
            zoom_model = ZoomModelLocal(model_path=model_path, conv_type="v1", patch_scale=None, bias_value=0.2)
    elif "internvl" in model_path.lower():
        zoom_model = ZoomModelInternvl(model_path=model_path, device="cuda:0", torch_dtype=torch.bfloat16, patch_scale=1.2)
    elif "qwen" in model_path.lower():
        zoom_model = ZoomModelQwenVL(model_path=model_path, device="cuda:0", torch_dtype=torch.bfloat16, patch_scale=1.2)
    else:
        raise ValueError(f"Model {model_path} not supported")
    
    def pop_limit_func(max_depth):
        return max_depth * 3
    search_kwargs = {
        "pop_limit": pop_limit_func,
        "threshold_descrease": [0.1, 0.1, 0.2],
        "answering_confidence_threshold_lower": 0,
        "answering_confidence_threshold_upper": 0.6,
        "visual_cue_threshold": 0.6,
    }

    decomposed_question_template = "What is the appearance of the {}?" 
    ic_examples_path = "ZoomEye/ic_examples/vstar.json"

    annotation = {
        "input_image": args.input_image,
        "question": args.question
    }
    if not args.direct_answer:
        response = get_zoom_eye_response(
            zoom_model=zoom_model,
            annotation=annotation,
            ic_examples=json.load(open(ic_examples_path, "r")),
            decomposed_question_template=decomposed_question_template,
            mode="debug",
            **search_kwargs,
        )
    else:
        response = get_direct_response(
            zoom_model=zoom_model,
            annotation=annotation,
            image_folder=None,
        )
    if args.direct_answer:
        print(response)
    else:
        print(response['text'])
        response['output_image'].save("/training/shz/project/ZoomEye/demo/saves/zoomed_view.jpg")