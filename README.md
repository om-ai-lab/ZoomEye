# ZoomEye
<p align="center"> <img src='docs/examples.jpg' align="center" height="400px"> </p>

> [**ZoomEye: Enhancing Multimodal LLMs with Human-Like Zooming Capabilities through Tree-Based Image Exploration**](https://arxiv.org/abs/2411.16044)

Zoom Eye enables MLLMs to **(a)** answer the question directly when the visual information is adequate, **(b)** zoom in gradually for a closer examination, and **(c)** zoom out to the previous view and explore other regions if the desired information is not initially found.


## 🛠️ Installation
This project is built based on [LLaVA-Next](https://github.com/LLaVA-VL/LLaVA-NeXT). If you encounter unknown errors during installation, you can refer to the issues and solutions in it.

### 1. Clone this repository
```bash
git clone https://github.com/om-ai-lab/ZoomEye.git
cd ZoomEye
```
### 2. Install dependencies
```bash
conda create -n zoom_eye python=3.10 -y
conda activate zoom_eye
pip install --upgrade pip  # Enable PEP 660 support.
pip install -e ".[train]"
```

## 📚 Preparation
### 1. MLLM checkpoints
In our work, we implement Zoom Eye with LLaVA-v1.5 and LLaVA-OneVision(ov) series, you could download these checkpoints before running or automatically download them when executing the **from_pretrained** method in transformers.
* [LLaVA-v1.5-7B](https://huggingface.co/liuhaotian/llava-v1.5-7b)
* [LLaVA-v1.5-13B](https://huggingface.co/liuhaotian/llava-v1.5-13b)
* [LLaVA-ov-0.5B](https://huggingface.co/lmms-lab/llava-onevision-qwen2-0.5b-ov)
* [LLaVA-ov-7B](https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov)

### 2. Evaluation data
The evaluation data we will use has been packaged together, and the link is provided [here](https://huggingface.co/datasets/omlab/zoom_eye_data). After downloading, please unzip it and its path is referred as to **anno path**.

Its folder tree is that:
```
zoom_eye_data 
  ├── hr-bench_4k                                  
  │   └── annotation_hr-bench_4k.json
  │   └── images
  │     └── some.jpg
  │    ...
  ├── hr-bench_8k
  │   └── annotation_hr-bench_8k.json
  │   └── images
  │     └── some.jpg
  │    ...
  │── vstar
  │   └── annotation_vstar.json
  │   └── direct_attributes
  │     └── some.jpg
  │    ...
  │   └── relative_positions
  │     └── some.jpg
  │    ...
 ...
```

## 🚀 Evaluation
### 1. Run the demo
We provide a demo file of Zoom Eye accepting any input Image-Question pair.
```bash
python ZoomEye/demo.py \
    --model-path lmms-lab/llava-onevision-qwen2-7b-ov \
    --input_image demo/demo.jpg \
    --question "What is the color of the soda can?"
```
and the zoomed views of Zoom Eye will be saved into the demo folder.

### 2. Results of V<sup>*</sup> Bench
```bash
# After excute this script, the result will be saved in the answers dir: ZoomEye/eval/answers/vstar/<mllm model base name>/merge.jsonl
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ZoomEye/eval/perform_zoom_eye.sh \
<mllm model> \
<anno path> \
vstar

# Get the result
python ZoomEye/eval/eval_results_vstar.py --answers-file ZoomEye/eval/answers/vstar/<mllm model base name>/merge.jsonl
```
The \<mllm model\> could be referred as to the above MLLM checkpoints, and the \<anno path\> is the path of the evaluation data.

If you don't have multi-gpu environment, you can set CUDA_VISIBLE_DEVICES=0.

### 3. Results of HR-Bench 4k
```bash
# After excute this script, the result will be saved in the answers dir: ZoomEye/eval/answers/hr-bench_4k/<mllm model base name>/merge.jsonl
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ZoomEye/eval/perform_zoom_eye.sh \
<mllm model> \
<anno path> \
hr-bench_4k

# Get the result
python ZoomEye/eval/eval_results_vstar.py --answers-file ZoomEye/eval/answers/vstar/hr-bench_4k/merge.jsonl
```


### 4. Results of HR-Bench 8k
```bash
# After excute this script, the result will be saved in the answers dir: ZoomEye/eval/answers/hr-bench_4k/<mllm model base name>/merge.jsonl
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ZoomEye/eval/perform_zoom_eye.sh \
<mllm model> \
<anno path> \
hr-bench_8k

# Get the result
python ZoomEye/eval/eval_results_vstar.py --answers-file ZoomEye/eval/answers/vstar/hr-bench_8k/merge.jsonl
```

### 5. Results for MLLMs with direct answering
```bash
# After excute this script, the result will be saved in the answers dir: ZoomEye/eval/answers/<bench name>/<mllm model base name>/direct_answer.jsonl
python ZoomEye/eval/perform_zoom_eye.py \
    --model-path <mllm model> \
    --annotation_path <anno path> \
    --benchmark <bench name> \
    --direct-answer

# Get the result
python ZoomEye/eval/eval_results_{vstar/hr-bench}.py --answers-file ZoomEye/eval/answers/<bench name>/<mllm model base name>/direct_answer.jsonl
```

## 🔗 Related works
If you are intrigued by multimodal large language models, and agent technologies, we invite you to delve deeper into our research endeavors:  
🔆 [OmAgent: A Multi-modal Agent Framework for Complex Video Understanding with Task Divide-and-Conquer](https://arxiv.org/abs/2406.16620) (EMNLP24)   
🏠 [GitHub Repository](https://github.com/om-ai-lab/OmAgent)

🔆 [How to Evaluate the Generalization of Detection? A Benchmark for Comprehensive Open-Vocabulary Detection](https://arxiv.org/abs/2308.13177) (AAAI24)   
🏠 [GitHub Repository](https://github.com/om-ai-lab/OVDEval/tree/main)

🔆 [OmDet: Large-scale vision-language multi-dataset pre-training with multimodal detection network](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cvi2.12268) (IET Computer Vision)  
🏠 [Github Repository](https://github.com/om-ai-lab/OmDet)

## ⭐️ Citation

If you find our repository beneficial, please cite our paper:  
```angular2
@article{shen2024zoomeye,
  title={ZoomEye: Enhancing Multimodal LLMs with Human-Like Zooming Capabilities through Tree-Based Image Exploration},
  author={Shen, Haozhan and Zhao, Kangjia and Zhao, Tiancheng and Xu, Ruochen and Zhang, Zilun and Zhu, Mingwei and Yin, Jianwei},
  journal={arXiv preprint arXiv:2411.16044},
  year={2024}
}
```