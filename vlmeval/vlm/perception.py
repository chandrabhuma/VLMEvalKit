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
    INTERLEAVE = False  # IMPORTANT: disable interleaved images
    USE_CUSTOM_PROMPT = False  # IMPORTANT: disable long prompts

    def __init__(self, model_path="PIA-SPACE-LAB/Perception-LM-1B", max_new_tokens=256, **kwargs):
        super().__init__()

        self.processor = AutoProcessor.from_pretrained(model_path)

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            attn_implementation="eager"   # or eager
        )

        self.max_new_tokens = max_new_tokens

    # override VLMEvalKit's heavy prompt version
    def generate_inner(self, image, text, dataset=None):
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        return self.processor.decode(output[0], skip_special_tokens=True)
