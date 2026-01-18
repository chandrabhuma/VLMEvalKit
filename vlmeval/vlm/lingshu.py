import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from vlmeval.smp import *
from vlmeval.vlm.base import BaseModel


class Lingshu(BaseModel):
    INSTALL_REQ = False  # No extra installation required beyond standard dependencies
    INTERLEAVE = True    # Supports interleaved image-text input

    def __init__(self,
                 model_path='lingshu-medical-mllm/Lingshu-7B',
                 torch_dtype=torch.bfloat16,
                 device='cuda',
                 **kwargs):
        """
        Initialize the Lingshu model.
        Args:
            model_path (str): HuggingFace model path or local path.
            torch_dtype: Data type for model weights (default: bfloat16).
            device (str): Device to load the model on.
        """
        assert model_path is not None
        self.model_path = model_path
        self.device = device
        self.torch_dtype = torch_dtype

        # Load model with FlashAttention-2 if available
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=self.torch_dtype,
            attn_implementation="flash_attention_2",
            device_map="auto",
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.processor.eos_token_id = self.model.config.eos_token_id
        self.model.eval()

        self.kwargs = kwargs

    def use_custom_prompt(self, dataset):
        # Use default chat template; no custom prompt needed
        return False

    def generate_inner(self, message, dataset=None):
        """
        Generate a response given a message (list of content dicts with 'type': 'image'/'text').
        This aligns with VLMEvalKit's expected input format.
        """
        # Convert message to the format expected by Lingshu
        messages = [{'role': 'user', 'content': message}]

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Extract vision inputs
        image_inputs, video_inputs = process_vision_info(messages)

        # Preprocess inputs
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # Generate
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.0,
        )

        # Trim input tokens
        input_len = inputs.input_ids.shape[1]
        generated_text = self.processor.batch_decode(
            generated_ids[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return generated_text.strip()
