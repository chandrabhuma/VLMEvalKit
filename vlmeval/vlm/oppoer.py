# vlmeval/vlm/oppoer.py

import os
import torch
from PIL import Image
import base64
from io import BytesIO
from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor
from .base import BaseModel


class AndesVL(BaseModel):
    def __init__(self, model_path, **kwargs):
        super().__init__()
        self.model_path = model_path

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.image_processor = CLIPImageProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            **kwargs
        ).cuda()
        self.model.eval()

    def _image_to_data_url(self, image_path):
        """Convert local image to base64 data URL."""
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{img_b64}"

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

        # ✅ Convert local image to base64 data URL
        data_url = self._image_to_data_url(image_path)

        # Build messages with data URL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url}
                    }
                ],
            }
        ]

        # Run inference
        with torch.no_grad():
            response = self.model.chat(
                messages=messages,
                tokenizer=self.tokenizer,
                image_processor=self.image_processor,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.6,
            )

        return response.strip()
