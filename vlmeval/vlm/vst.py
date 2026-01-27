# vlmeval/vlm/vst.py

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from vlmeval.smp import *
from vlmeval.vlm import BaseModel

class VST(BaseModel):
    def __init__(self, model_path, **kwargs):
        super().__init__()
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and processor
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",  # or "flash_attention_2" if supported
            device_map="auto"
        ).eval()
        
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28
        )

    def use_custom_prompt(self, dataset):
        # Return True if you want to control the prompt format per dataset
        # For generality, we'll let the model handle chat templates
        return False

    def generate_inner(self, message, dataset=None):
        """
        message: list of dicts with keys 'type' ('image' or 'text') and corresponding values.
        Example:
          [{'type': 'image', 'value': 'path/to/image.jpg'},
           {'type': 'text', 'value': 'Describe this image.'}]
        """
        # Convert VLMEvalKit message format to Qwen-VL format
        qwen_messages = [{"role": "user", "content": []}]
        
        for msg in message:
            if msg['type'] == 'image':
                # VLMEvalKit passes local paths or URLs; Qwen accepts both
                qwen_messages[0]["content"].append({
                    "type": "image",
                    "image": msg['value']
                })
            elif msg['type'] == 'text':
                qwen_messages[0]["content"].append({
                    "type": "text",
                    "text": msg['value']
                })

        # Apply chat template
        text = self.processor.apply_chat_template(
            qwen_messages, tokenize=False, add_generation_prompt=True
        )
        
        # Process vision inputs
        image_inputs, video_inputs = process_vision_info(qwen_messages)
        
        # Tokenize and prepare inputs
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.0
            )

        # Trim input tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return output_text.strip()
