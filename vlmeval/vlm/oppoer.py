# vlmeval/vlm/oppoer.py

import os
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor
from .base import BaseModel


class AndesVL(BaseModel):
    def __init__(self, model_path="OPPOer/AndesVL-0_6B-Instruct", **kwargs):
        super().__init__()
        self.model_path = model_path

        # Load components
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

    def generate_inner(self, message, dataset=None):
        """
        VLMEvalKit input format:
          [{'type': 'image', 'value': '/path/to/image.jpg'},
           {'type': 'text', 'value': 'Describe this image.'}]
        or shorthand: ['/path.jpg', 'prompt']
        """
        # Normalize shorthand
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

        # Load image as PIL
        image = Image.open(image_path).convert("RGB")

        # Build messages in the format expected by .chat()
        # Note: AndesVL's .chat() ignores the URL if you pass image separately,
        # but we must include a dummy "image_url" to satisfy schema.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": "dummy"}  # placeholder; actual image passed via image_processor
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
                image=image,  # ← Pass PIL image directly (critical!)
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.6,
            )

        return response.strip()
