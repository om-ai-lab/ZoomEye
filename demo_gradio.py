import sys
sys.path.insert(0, "/data9/shz/project/ZoomEye")
from PIL import Image
import os
import json
import warnings
import gradio as gr
from ZoomEye.zoom_model import ZoomModelGlobalLocal, ZoomModelLocal
from ZoomEye.zoom_eye import get_zoom_eye_response

warnings.filterwarnings("ignore")

# ------------------ Global Model Initialization ------------------
# Load the model globally so it's initialized only once
MODEL_PATH = "/data9/shz/ckpt/llava-onevision-qwen2-7b-ov"

def initialize_model(model_path):
    """
    Initialize the ZoomEye model based on the model configuration.
    """
    config_path = os.path.join(model_path, "config.json")
    config = json.load(open(config_path, "r"))
    if "anyres" in config['image_aspect_ratio']:
        zoom_model = ZoomModelGlobalLocal(
            model_path=model_path, conv_type="qwen_1_5", patch_scale=1.2, bias_value=0.6
        )
    else:
        zoom_model = ZoomModelLocal(
            model_path=model_path, conv_type="v1", patch_scale=None, bias_value=0.2
        )
    return zoom_model, config

# Initialize the model only once
ZOOM_MODEL, CONFIG = initialize_model(MODEL_PATH)

# ------------------ Main Function for Processing ------------------
def process_image_and_question(image, question):
    """
    Process the input image and question to generate a response.
    Returns a textual answer and an output image.
    """
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
    ic_examples_path = "ZoomEye/ic_examples/hr-bench_4k.json"

    # Save the input image as a file (for ZoomEye processing)
    input_image_path = "temp_image.jpg"
    image.save(input_image_path)

    annotation = {
        "input_image": input_image_path,
        "question": question,
    }

    # Get the response, which includes a textual answer and an output image
    response = get_zoom_eye_response(
        zoom_model=ZOOM_MODEL,  # Use the globally initialized model
        annotation=annotation,
        ic_examples=json.load(open(ic_examples_path, "r")),
        decomposed_question_template=decomposed_question_template,
        **search_kwargs,
    )

    text_response = response["text"]  # Model's textual answer
    output_image = response["output_image"]  # Output image (PIL.Image object)

    return text_response, output_image  # Return text and image


# ------------------ Gradio Interface ------------------
def gradio_interface():
    """
    Define the Gradio interface with input fields, outputs, and examples.
    """
    examples = [
        ["demo/demo.jpg", "What is the color of the soda can?"],
        ["demo/demo1.jpg", "What is the message written on the sign?"],
        ["demo/demo2.jpg", "Is the red car on the left or right side of the police car?"],
    ]

    inputs = [
        gr.Image(type="pil", label="Image"),  # Image upload input
        gr.Textbox(label="Question", placeholder="Please input your question here.")  # Text input
    ]
    outputs = [
        gr.Textbox(label="Model Response"),  # Textual response output
        gr.Image(label="Zoomed View")    # Zoomed view of ZoomEye
    ]

    interface = gr.Interface(
        fn=process_image_and_question,
        inputs=inputs,
        outputs=outputs,
        title="ZoomEye Demo",
        description="Upload an image and ask a question to get the response and zoomed view from ZoomEye.",
        theme="compact",
        examples=examples,
    )
    return interface


if __name__ == "__main__":
    # Launch the Gradio interface
    print("lauch gradio interface")
    app = gradio_interface()
    app.launch(share=True)