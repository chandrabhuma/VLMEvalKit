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

        # Load processor and model
        self.processor = LlavaNextProcessor.from_pretrained(model_path)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            **kwargs
        )
        self.model.eval()

    def generate_inner(self, message, dataset=None):
        """
        VLMEvalKit passes message as:
          [{'type': 'image', 'value': '/path/to/image.jpg'},
           {'type': 'text', 'value': 'Describe this image.'}]
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

        # Format prompt: <image>\n{query}
        prompt = f"<image>\n{text_prompt}"

        # Tokenize + embed image
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            gen_tokens = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,      # deterministic for eval
                temperature=0.0,
            )

        # Decode only the new tokens
        input_len = inputs.input_ids.shape[1]
        response = self.processor.tokenizer.decode(
            gen_tokens[0][input_len:],
            skip_special_tokens=True
        ).strip()

        return response
