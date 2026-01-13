import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

from .base import BaseModel




#class PerceptionLM(BaseModel):



class PerceptionLM(BaseModel):
    def __init__(self, model_path, **kwargs):
        self.default_instruction_prompt = (
            "\nPlease answer directly with only the final answer, "
            "do not give any explanation."
        )
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        self.model = (
            AutoModelForImageTextToText.from_pretrained(model_path).to("cuda") )

        kwargs_default = {"max_new_tokens": 256, "use_cache": True}
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default

    def custom_instruction_prompt_by_dataset(self, dataset):
        if dataset == "MathVista_MINI" or dataset == "MM-IFEval" or dataset == "MMVet":
            return ""
        else:
            return self.default_instruction_prompt

    def message_to_chat_messages(self, message, instruction_prompt, dataset):
        single_turn_messages = []
        image_paths = []

        for item in message:
            if item["type"] == "image":
                image_paths.append(item["value"])
                single_turn_messages.append({"type": "image", "url": item["value"]})
            elif item["type"] == "text":
                single_turn_messages.append({"type": "text", "text": item["value"]})
        if instruction_prompt:
            single_turn_messages.append({"type": "text", "text": instruction_prompt})

        if dataset == "MM-IFEval":
            # move images to the beginning of the conversation list, in the same order as they appear in the message
            index = 0
            image_index = 0
            while index < len(single_turn_messages):
                if single_turn_messages[index]["type"] == "image":
                    single_turn_messages.insert(
                        image_index, single_turn_messages.pop(index)
                    )
                    image_index += 1
                index += 1

        chat_messages = [{"role": "user", "content": single_turn_messages}]
        images_pil = [Image.open(p).convert("RGB") for p in image_paths]

        return chat_messages, images_pil

   
    def generate_inner(self, message, dataset=None):
    
        instruction_prompt = self.custom_instruction_prompt_by_dataset(dataset)
        chat_messages, _ = self.message_to_chat_messages(
            message, instruction_prompt, dataset
        )
    
        # 🔑 MUST tokenize here (Perception-LM style)
        inputs = self.processor.apply_chat_template(
            chat_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device=self.model.device, dtype=torch.float16)
    
        # 🔑 Disable KV cache to avoid OOM on T4
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=128,
            use_cache=False,
        )
    
        # 🔑 Trim prompt tokens (VERY IMPORTANT)
        input_len = inputs["input_ids"].shape[1]
        gen_ids = output_ids[:, input_len:]
    
        output = self.processor.batch_decode(
            gen_ids,
            skip_special_tokens=True
        )[0]
    
        return output.strip()
