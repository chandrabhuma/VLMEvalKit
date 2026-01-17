# vlmeval/vlm/kanana.py

import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from .base import BaseModel


class Kanana(BaseModel):
    def __init__(self, model_path="kakaocorp/kanana-1.5-v-3b-instruct", **kwargs):
        super().__init__()
        self.model_path = model_path

        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            **kwargs
        )
        self.model.eval()

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

        # Build conversation format expected by Kanana
        # Note: Kanana uses a two-turn format in examples, but single user turn works
        conv = [
            {"role": "user", "content": "<image>"},
            {"role": "user", "content": text_prompt},
        ]

        # Prepare batch (even for single sample)
        batch = [{
            "image": [image],
            "conv": conv
        }]

        # Tokenize and collate
        inputs = self.processor.batch_encode_collate(
            batch,
            padding_side="left",
            add_generation_prompt=True,
            max_length=8192
        )
        inputs = {
            k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        # Generate
        with torch.no_grad():
            gen_tokens = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.0,
                top_p=1.0,
                num_beams=1,
                do_sample=False,
            )

        # Decode output
        outputs = self.processor.tokenizer.batch_decode(
            gen_tokens, skip_special_tokens=True
        )
        return outputs[0].strip()
