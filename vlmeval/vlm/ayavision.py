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
    def generate_inner(self, message, dataset=None):
        # --------------------------------------------------
        # 1. Normalize VLMEvalKit shorthand
        # ['img.jpg', 'question']
        # --------------------------------------------------
        if (
            isinstance(message, list)
            and len(message) == 2
            and isinstance(message[0], str)
            and isinstance(message[1], str)
        ):
            message = [
                {"type": "image", "value": message[0]},
                {"type": "text", "value": message[1]},
            ]

        image_path = None
        text = ""

        for item in message:
            if item["type"] == "image":
                image_path = item["value"]
            elif item["type"] == "text":
                text += item["value"]

        if image_path is None:
            raise ValueError("No image provided")

        # --------------------------------------------------
        # 2. Load image (VLMEvalKit gives path, HF expects PIL)
        # --------------------------------------------------
        image = Image.open(image_path).convert("RGB")

        # --------------------------------------------------
        # 3. Build conversation EXACTLY like your example
        # --------------------------------------------------
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text},
                ],
            }
        ]

        # --------------------------------------------------
        # 4. Apply chat template (CRITICAL: tokenize=True)
        # --------------------------------------------------
        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            padding=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        # --------------------------------------------------
        # 5. Generate (T4-safe)
        # --------------------------------------------------
        gen_tokens = self.model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.3,
            do_sample=True,
        )

        textout = self.processor.tokenizer.decode(gen_tokens[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        return textout
