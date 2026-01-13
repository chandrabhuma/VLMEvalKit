# vlmeval/vlm/ayavision.py

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from .base import BaseModel
import os

class AyaVision(BaseModel):
    def __init__(self, model_path="CohereLabs/aya-vision-8b", **kwargs):
        super().__init__()
        self.model_path = model_path
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
        """
        VLMEvalKit expects this method.
        `message` is a list of mixed text/image elements in VLMEvalKit format.
        Example:
          [{'type': 'image', 'value': '/path/to/image.jpg'},
           {'type': 'text', 'value': 'What is shown?'}]
        """
        # Reconstruct messages in Aya Vision chat format
        content = []
        image_path = None

        for item in message:
            if item['type'] == 'image':
                image_path = item['value']
                content.append({"type": "image"})
            elif item['type'] == 'text':
                content.append({"type": "text", "text": item['value']})

        if image_path is None:
            raise ValueError("No image provided in the message.")

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Build chat message
        messages = [{"role": "user", "content": content}]

        # Apply chat template and tokenize
        inputs = self.processor.apply_chat_template(
            messages,
            padding=True,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            gen_tokens = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,  # VLMEvalKit usually uses greedy unless specified
                temperature=0.0,
            )

        # Decode only the generated part
        input_len = inputs.input_ids.shape[1]
        output_text = self.processor.tokenizer.decode(
            gen_tokens[0][input_len:], skip_special_tokens=True
        ).strip()

        return output_text
