import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from vlmeval.smp import *
from vlmeval.utils import CustomPrompt
from vlmeval.vlm.base import BaseModel

class MiMo(BaseModel):
    """
    Wrapper for XiaomiMiMo/MiMo-VL-7B-RL model
    """
    def __init__(self, model_path='XiaomiMiMo/MiMo-VL-7B-RL', **kwargs):
        self.model_path = model_path
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map='auto'
        )
        self.kwargs = kwargs
        
    def generate_inner(self, message, dataset=None):
        # Process the message to extract image and text
        content_list = []
        image_path = None
        
        for msg in message:
            if msg['type'] == 'image':
                image_path = msg['value']
            elif msg['type'] == 'text':
                content_list.append({"type": "text", "text": msg['value']})
        
        # Create messages format required by the model
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "url": image_path}] + content_list
            }
        ]
        
        # Apply chat template
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                **self.kwargs
            )
            
        # Decode only the generated part (excluding input tokens)
        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        response = self.processor.decode(generated_ids, skip_special_tokens=True)
        
        return response.strip()
    
    def use_custom_prompt(self, dataset):
        # Check if the dataset has a custom prompt defined
        if dataset is not None and listinstr(['MCQ', 'Y/N'], dataset):
            return True
        return False
    
    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert isinstance(line, dict)
        
        tgt_path = self.dump_image(line, dataset)
        
        question = line['question']
        
        if dataset is not None and 'MMMU' in dataset:
            # For MMMU dataset, format accordingly
            prompt = question
        elif dataset is not None and listinstr(['MCQ'], dataset):
            # For multiple choice questions
            options = {
                cand: line[cand]
                for cand in string.ascii_uppercase
                if cand in line and not pd.isna(line[cand])
            }
            options_prompt = '\n'.join([f'{key}. {value}' for key, value in options.items()])
            prompt = f'{question}\n{options_prompt}'
        elif dataset is not None and listinstr(['Y/N'], dataset):
            # For yes/no questions
            prompt = f'{question}'
        else:
            # Default case
            prompt = question
            
        message = [dict(type='image', value=tgt_path)]
        message.extend([dict(type='text', value=prompt)])
        return message
