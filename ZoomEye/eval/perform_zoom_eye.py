import sys
sys.path.insert(0, "/data9/shz/project/ZoomEye")
from PIL import Image
import argparse
import os
import json
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from ZoomEye.zoom_model import ZoomModelGlobalLocal, ZoomModelLocal
from ZoomEye.zoom_eye import get_zoom_eye_response, get_direct_response

def get_chunk(lst, n, k):
    subarrays = [[] for _ in range(n)]
    for i in range(n):
        subarrays[i] = lst[i::n]
    return subarrays[k]

def get_basename(path):
    return os.path.basename(path)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--answers-file", type=str, default=None)
    parser.add_argument("--model-path", type=str, default="/data9/shz/ckpt/llava-onevision-qwen2-0.5b-ov")
    parser.add_argument("--annotation_path", type=str, default="/data9/shz/dataset/zoom_eye_data")
    parser.add_argument("--benchmark",  type=str, choices=["vstar", "hr-bench_4k", "hr-bench_8k"], default="vstar")
    parser.add_argument("--direct-answer", action="store_true")
    args = parser.parse_args()

    model_path = args.model_path
    annoataion_path = args.annotation_path
    benchmark = args.benchmark

    if args.answers_file is None:
        answers_dir = f"ZoomEye/eval/answers/{benchmark}"
        answers_dir = os.path.join(answers_dir, os.path.basename(args.model_path))
        os.makedirs(answers_dir, exist_ok=True)
        answer_tag = "zoom_eye" if not args.direct_answer else "direct_answer"
        args.answers_file = os.path.join(answers_dir, f"{answer_tag}.jsonl")
        print(args.answers_file)

    config_path = os.path.join(model_path, "config.json")
    config = json.load(open(config_path, "r"))
    if "anyres" in config['image_aspect_ratio']:
        zoom_model = ZoomModelGlobalLocal(model_path=model_path, conv_type="qwen_1_5", patch_scale=1.2, bias_value=0.6)
    else:
        zoom_model = ZoomModelLocal(model_path=model_path, conv_type="v1", patch_scale=None, bias_value=0.2)
    
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
    ic_examples_path = f"ZoomEye/ic_examples/{benchmark}.json"
    m = json.load(open(os.path.join(annoataion_path, f"{benchmark}/annotation_{benchmark}.json"), "r"))
    m = get_chunk(m, args.num_chunks, args.chunk_idx)

    results_file = open(args.answers_file, 'w')
    for annotation in tqdm(m):
        if not args.direct_answer:
            response = get_zoom_eye_response(
                zoom_model=zoom_model,
                annotation=annotation,
                ic_examples=json.load(open(ic_examples_path, "r")),
                decomposed_question_template=decomposed_question_template,
                image_folder=os.path.join(annoataion_path, f"{benchmark}"),
                **search_kwargs,
            )
        else:
            response = get_direct_response(
                zoom_model=zoom_model,
                annotation=annotation,
                image_folder=os.path.join(annoataion_path, f"{benchmark}"),
            )
        annotation['output'] = response
        results_file.write(json.dumps(annotation) + "\n")
    results_file.close()

