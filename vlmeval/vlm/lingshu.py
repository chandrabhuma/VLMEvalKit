import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from vlmeval.smp import *
from vlmeval.vlm.base import BaseModel


class Lingshu(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self,
                 model_path='lingshu-medical-mllm/Lingshu-7B',
                 torch_dtype=torch.bfloat16,
                 device='cuda',
                 **kwargs):
        assert model_path is not None
        self.model_path = model_path
        self.torch_dtype = torch_dtype

        # Load model with flash attention if available
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=self.torch_dtype,
            attn_implementation="eager",
            device_map="auto",
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model.eval()

    def use_custom_prompt(self, dataset):
        return False

    def generate_inner(self, message, dataset=None):
        # Validate and convert VLMEvalKit format to Qwen format
        if not isinstance(message, list):
            raise ValueError("Input message must be a list of dicts with 'type' and 'value'.")

        qwen_content = []
        for item in message:
            if not (isinstance(item, dict) and 'type' in item and 'value' in item):
                raise ValueError(f"Invalid message item: {item}. Expected {{'type': ..., 'value': ...}}")
            
            t, v = item['type'], item['value']
            if t == 'image':
                qwen_content.append({'type': 'image', 'image': v})
            elif t == 'text':
                qwen_content.append({'type': 'text', 'text': v})
            else:
                raise ValueError(f"Unsupported type: {t}. Only 'image' and 'text' are supported.")

        messages = [{'role': 'user', 'content': qwen_content}]

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process vision inputs
        image_inputs, video_inputs = process_vision_info(messages)

        # Tokenize + preprocess
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # Get tokenizer for special token IDs
        tokenizer = self.processor.tokenizer

        # Generate
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Decode output
        input_len = inputs.input_ids.shape[1]
        output_text = tokenizer.batch_decode(
            generated_ids[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return output_text.strip()
