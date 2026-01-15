# vlmeval/vlm/sa2va.py

import os
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModel
from .base import BaseModel


class SA2VA(BaseModel):
    def __init__(self, model_path="ByteDance/Sa2VA-1B", **kwargs):
        super().__init__()
        self.model_path = model_path

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False
        )

        # Load model
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            **kwargs
        ).eval().cuda()

    def generate_inner(self, message, dataset=None):
        """
        VLMEvalKit passes message as:
          [{'type': 'image', 'value': '/path/to/image.jpg'},
           {'type': 'text', 'value': 'Describe the image.'}]
        or shorthand: ['/path.jpg', 'prompt']
        """
        # Normalize shorthand input
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

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Format prompt with <image> token (required by Sa2VA)
        formatted_prompt = f"<image>{text_prompt}"

        # Build input dict as expected by Sa2VA's predict_forward
        input_dict = {
            "image": image,
            "text": formatted_prompt,
            "past_text": "",          # for multi-turn; leave empty for single-turn
            "mask_prompts": None,
            "tokenizer": self.tokenizer,
        }

        # Run inference
        with torch.no_grad():
            output = self.model.predict_forward(**input_dict)

        # Extract answer
        answer = output.get("prediction", "").strip()
        return answer
