import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from vlmeval.smp import *
from vlmeval.utils import CustomPrompt
from vlmeval.vlm.base import BaseModel

class YoutuVL(BaseModel):
    """
    Wrapper for Tencent Youtu-VL-4B-Instruct model
    """
    def __init__(self, model_path='tencent/Youtu-VL-4B-Instruct', **kwargs):
        self.model_path = model_path
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            attn_implementation="eager",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        ).eval()
        
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            use_fast=True,
            trust_remote_code=True
        )
        self.kwargs = kwargs
        
    def generate_inner(self, message, dataset=None):
        # Extract image and text from the message
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
                "content": [{"type": "image", "image": image_path}] + content_list
            }
        ]
        
        # Apply chat template
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        # Generate response with the special img_input parameter
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                temperature=0.1,
                top_p=0.001,
                repetition_penalty=1.05,
                do_sample=True,
                max_new_tokens=512,  # Reduced from 32768 for evaluation efficiency
                img_input=image_path,
                **self.kwargs
            )
        
        # Decode only the generated part (excluding input tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        outputs = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        generated_text = outputs[0].strip()
        return generated_text
    
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
