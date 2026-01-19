# vlmeval/vlm/hulu.py

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from vlmeval.smp import *
from vlmeval.vlm.base import BaseModel

class Hulu(BaseModel):
    def __init__(self, model_path, **kwargs):
        assert model_path is not None
        self.model_path = model_path
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self.tokenizer = self.processor.tokenizer

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            attn_implementation='eager',
        )
        self.model.eval()
        self.kwargs = kwargs
        self.max_new_tokens = kwargs.get('max_new_tokens', 1024)

    def use_custom_prompt(self, dataset):
        # Return True if you want to handle prompt formatting yourself
        # For medical VQA datasets like VQA-Med, PathVQA, etc., you may want custom prompts
        return False  # Let VLMEvalKit handle standard prompting

    def generate_inner(self, message, dataset=None):
        """
        message: list of dicts with 'type' ('image' or 'text') and corresponding content
        Example:
          [{'type': 'image', 'value': 'path/to/image.jpg'},
           {'type': 'text', 'value': 'Describe this image.'}]
        """
        # Parse input
        conversation = [{"role": "user", "content": []}]

        for msg in message:
            if msg['type'] == 'image':
                # Hulu expects image as dict with 'image_path'
                conversation[0]['content'].append({
                    "type": "image",
                    "image": {"image_path": msg['value']}
                })
            elif msg['type'] == 'text':
                conversation[0]['content'].append({
                    "type": "text",
                    "text": msg['value']
                })

        # Process inputs
        inputs = self.processor(
            conversation=conversation,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        # Move tensors to GPU and cast pixel_values to bfloat16
        inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # Set to True if needed, but VLMEvalKit usually uses greedy
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        outputs = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            use_think=False  # Important: disable internal reasoning tokens
        )[0].strip()

        return outputs
