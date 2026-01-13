# vlmeval/vlm/ayavision.py

import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from .base import BaseModel


class AyaVision(BaseModel):
    def __init__(self, model_path="CohereLabs/aya-vision-8b", **kwargs):
        super().__init__()
        self.model_path = model_path

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            **kwargs
        )
        self.model.eval()

    @torch.no_grad()
    def generate_inner(self, message, dataset=None):
        content = []
        image_path = None

        # 🔧 FIX: this loop MUST be inside the method
        for item in message:
            if item["type"] == "image":
                image_path = item["value"]
                assert os.path.exists(image_path), f"Image not found: {image_path}"
                content.append({"type": "image"})
            elif item["type"] == "text":
                content.append({"type": "text", "text": item["value"]})

        if image_path is None:
            raise ValueError("No image provided")

        # Load image (processor will use it internally)
        image = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        # Safety check
        if "pixel_values" not in inputs:
            raise RuntimeError("Image features not processed! Check processor.")

        gen_tokens = self.model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.3,
        )

        input_len = inputs["input_ids"].shape[1]
        output = self.processor.decode(
            gen_tokens[0][input_len:],
            skip_special_tokens=True
        ).strip()

        return output
