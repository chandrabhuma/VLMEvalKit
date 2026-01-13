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

        # 🔑 Normalize VLMEvalKit shorthand
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

        content = []
        image_path = None

        for item in message:
            if item["type"] == "image":
                image_path = item["value"]
                content.append({"type": "image"})
            elif item["type"] == "text":
                content.append({"type": "text", "text": item["value"]})

        if image_path is None:
            raise ValueError("No image provided")

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

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            use_cache=False,
        )

        input_len = inputs["input_ids"].shape[1]
        output = self.processor.batch_decode(
            outputs[:, input_len:],
            skip_special_tokens=True,
        )[0]

        return output.strip()
