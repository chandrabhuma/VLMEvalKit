import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from .base import BaseModel


class AyaVision(BaseModel):
    def __init__(self, model_path="CohereLabs/aya-vision-8b", temperature=0.3, max_new_tokens=300, **kwargs):
        super().__init__()
        self.model_path = model_path
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            **kwargs
        )
        self.model.eval()

    def generate_inner(self, message, dataset=None):
        # Normalize VLMEvalKit shorthand
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

        # ✅ Step 1: Build message for apply_chat_template (NO image data)
        chat_msg = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},  # placeholder only
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]

        # ✅ Step 2: Get formatted prompt WITH <image> token
        prompt_text = self.processor.apply_chat_template(
            chat_msg,
            add_generation_prompt=True,
            tokenize=False  # ← returns string
        )

        # ✅ Step 3: Process text + image together
        inputs = self.processor(
            text=prompt_text,
            images=image,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            gen_tokens = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=0.95,
            )

        # Decode only the new part
        input_len = inputs.input_ids.shape[1]
        output_text = self.processor.tokenizer.decode(
            gen_tokens[0][input_len:], skip_special_tokens=True
        ).strip()

        return output_text
