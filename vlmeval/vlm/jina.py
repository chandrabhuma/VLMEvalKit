# vlmeval/vlm/jina.py

import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig
from .base import BaseModel


class JinaVLM(BaseModel):
    def __init__(self, model_path="jinaai/jina-vlm", **kwargs):
        super().__init__()
        self.model_path = model_path

        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(
            model_path, use_fast=False, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
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

        # Load image as PIL
        image = Image.open(image_path).convert("RGB")

        # Build conversation (same structure as official example)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},  # PIL.Image object
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]

        # Apply chat template to get prompt string
        prompt_text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

        # Tokenize + embed image
        inputs = self.processor(
            text=[prompt_text],
            images=[image],
            padding="longest",
            return_tensors="pt"
        )
        inputs = {
            k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        # Generate
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                generation_config=GenerationConfig(
                    max_new_tokens=512,
                    do_sample=False,
                    temperature=0.0,
                ),
                return_dict_in_generate=True,
                use_model_defaults=True,
            )

        # Decode only the new tokens
        input_len = inputs["input_ids"].shape[-1]
        response = self.processor.tokenizer.decode(
            output.sequences[0][input_len:],
            skip_special_tokens=True
        ).strip()

        return response
