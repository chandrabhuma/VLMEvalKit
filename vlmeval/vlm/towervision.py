# vlmeval/vlm/towervision.py

import os
import torch
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from .base import BaseModel


class TowerVision(BaseModel):
    def __init__(self, model_path="utter-project/TowerVision-2B", **kwargs):
        super().__init__()
        self.model_path = model_path

        self.processor = LlavaNextProcessor.from_pretrained(model_path)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            **kwargs
        )
        self.model.eval()

    def generate_inner(self, message, dataset=None):
        # Normalize input
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
        text_prompt = ""

        for item in message:
            if item["type"] == "image":
                image_path = item["value"]
            elif item["type"] == "text":
                text_prompt = item["value"]

        if not image_path or not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")

        # ✅ CRITICAL: Build message with {"type": "image"} (no data)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},          # ← placeholder only
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]

        # ✅ Let processor handle prompt + image binding
        inputs = self.processor(
            conversation,
            images=image,
            return_tensors="pt"
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            gen_tokens = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.0,
            )

        # Decode
        response = self.processor.batch_decode(
            gen_tokens, skip_special_tokens=True
        )[0].strip()

        return response
