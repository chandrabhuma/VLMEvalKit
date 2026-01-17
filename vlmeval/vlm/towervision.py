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
        query = ""
    
        for item in message:
            if item["type"] == "image":
                image_path = item["value"]
            elif item["type"] == "text":
                query = item["value"]
    
        if not image_path or not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
    
        image = Image.open(image_path).convert("RGB")
    
        # --------------------------------------------------
        # 2. Build TowerVision conversation (OFFICIAL FORMAT)
        # --------------------------------------------------
        conversation = [
            {
                "role": "user",
                "content": f"<image>\n{query}"
            }
        ]
    
        prompt = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
    
        # --------------------------------------------------
        # 3. Process inputs
        # --------------------------------------------------
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.model.device)
    
        # --------------------------------------------------
        # 4. Generate
        # --------------------------------------------------
        with torch.no_grad():
            gen_tokens = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.0,
            )
    
        # --------------------------------------------------
        # 5. Decode EXACTLY like repo
        # --------------------------------------------------
        response = self.processor.tokenizer.decode(
            gen_tokens[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()
    
        return response
