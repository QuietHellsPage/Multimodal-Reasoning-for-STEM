import os
import streamlit as st
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

BASE_MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"
CHECKPOINT_PATH = "sft-2-datasets"
INSTRUCTION = "Write the LaTeX formula shown in this image."
MAX_NEW_TOKENS = 2048


@st.cache_resource(show_spinner=False)
def load_model():
    """
    Method that loads model
    """
    with st.spinner("Model is loading"):
        processor = AutoProcessor.from_pretrained(BASE_MODEL)
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.padding_side = "right"

        model = AutoModelForImageTextToText.from_pretrained(BASE_MODEL, dtype=torch.float32)

        if os.path.exists(CHECKPOINT_PATH):
            try:
                model = PeftModel.from_pretrained(model, CHECKPOINT_PATH)
                model = model.merge_and_unload()
            except Exception:
                pass

        model.eval()
    return model, processor


def predict_latex(model, processor, image: Image.Image):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": INSTRUCTION},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=text,
        images=[image],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_NEW_TOKENS,
    )
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()


st.set_page_config(page_title="Formula to LaTeX", layout="centered")
st.title("Formula to LaTeX")

model, processor = load_model()

uploaded_file = st.file_uploader("Upload an image with a formula", type=["png", "jpg", "jpeg", "bmp", "webp"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_column_width=True)

    if st.button("Convert to LaTeX", type="primary", use_container_width=True):
        with st.spinner("Generating..."):
            try:
                latex = predict_latex(model, processor, image)
                if latex:
                    st.code(latex, language="latex")
                    st.latex(latex)
                else:
                    st.warning("Model returned an empty result.")
            except Exception as e:
                st.error(f"Error: {e}")
