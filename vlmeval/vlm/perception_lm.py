import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

from .base import BaseModel


class PerceptionLM(BaseModel):
    def __init__(
        self,
        model_path="facebook/Perception-LM-1B",
        device="cuda",
        dtype=torch.float16,
        **kwargs
    ):
        super().__init__()

        self.device = device

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            use_fast=True
        )

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=dtype
        ).to(device).eval()

    @torch.no_grad()
    def generate(self, message, **kwargs):
        """
        VLMEvalKit message format:
        [
            {"type": "image", "value": image_path},
            {"type": "text", "value": question}
        ]
        """

        image_path = None
        question = ""

        for m in message:
            if m["type"] == "image":
                image_path = m["value"]
            elif m["type"] == "text":
                question += m["value"]

        # ---- Build Perception-LM conversation ----
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": image_path,   # local path works
                    },
                    {
                        "type": "text",
                        "text": question,
                    },
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            [conversation],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = inputs.to(self.device)

        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=64
        )

        # Remove input tokens (VERY IMPORTANT)
        input_len = inputs["input_ids"].shape[1]
        gen_ids = generate_ids[:, input_len:]

        output = self.processor.batch_decode(
            gen_ids,
            skip_special_tokens=True
        )[0]

        return output.strip()
