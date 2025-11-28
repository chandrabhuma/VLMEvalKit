import torch
from PIL import Image
import sys
import os.path as osp
from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE, DATASET_MODALITY


from transformers import AutoProcessor, AutoModelForImageTextToText


class PerceptionLM(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True  # supports images anywhere

    def __init__(self, model_path="facebook/Perception-LM-1B", max_new_tokens=512, **kwargs):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            trust_remote_code=True,
        ).to("cuda")
        self.max_new_tokens = max_new_tokens

    def generate_inner(self, message, dataset=None):
        images = []
        contents = []

        # Convert VLMEvalKit messages → Perception-LM format
        for msg in message:
            if msg["type"] == "text":
                contents.append({"type": "text", "text": msg["value"]})
            elif msg["type"] == "image":
                contents.append({"type": "image", "url": msg["value"]})

        conversation = [{
            "role": "user",
            "content": contents,
        }]

        # Tokenize
        inputs = self.processor.apply_chat_template(
            [conversation],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        # Generate
        generate_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        input_length = inputs["input_ids"].shape[1]
        outputs = generate_ids[:, input_length:]

        # Decode
        answer = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        if "</think>" in answer:
            answer = answer.split("</think>")[-1].strip()
        return answer
