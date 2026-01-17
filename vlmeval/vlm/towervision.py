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

        # ✅ Use chat template for correct prompt formatting
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]

        # Apply chat template to get full prompt with roles and <image>
        prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )

        # Process
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
                do_sample=False,
                temperature=0.0,
            )

        # ✅ Decode full output and remove input part safely
        full_output = self.processor.batch_decode(gen_tokens, skip_special_tokens=True)[0]

        # Optional: strip input prompt if needed (usually not necessary with skip_special_tokens)
        # But if you want to be sure:
        input_text = self.processor.batch_decode(inputs.input_ids, skip_special_tokens=True)[0]
        if full_output.startswith(input_text):
            response = full_output[len(input_text):].strip()
        else:
            response = full_output.strip()

        return response
