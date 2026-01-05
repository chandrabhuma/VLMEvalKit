import ast
from .image_base import ImageBaseDataset
from ..smp import *


import re

def normalize_text(s):
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)  # remove punctuation
    s = re.sub(r"\s+", " ", s)
    return s

def relaxed_match(pred, gt):
    return normalize_text(pred) == normalize_text(gt)

def vqa_score(pred, answers):
    """
    answers: list of ground truth answers (strings)
    Standard VQA accuracy: min(#matches / 3, 1)
    """
    pred = normalize_text(pred)
    matches = sum(pred == normalize_text(a) for a in answers)
    return min(matches / 3.0, 1.0)

class traffic_vqa_test(ImageBaseDataset):

    TYPE = "VQA"

    DATASET_URL = {
        "traffic_vqa_test": "https://huggingface.co/datasets/chandrabhuma/vlmevalkit_files/resolve/main/tafficvqa_vlm_test.tsv",
		"traffic_vqa_train": "https://huggingface.co/datasets/chandrabhuma/vlmevalkit_files/resolve/main/tafficvqa_vlm_train.tsv",
		"screenspot_vqa_train": "https://huggingface.co/datasets/chandrabhuma/vlmevalkit_files/resolve/main/screenspot_vlm_train.tsv",
		"screenspot_vqa_test": "https://huggingface.co/datasets/chandrabhuma/vlmevalkit_files/resolve/main/screenspot_vlm_test.tsv",
    }

def build_prompt(self, line):
	        msgs = []
	        if 'image' in line:
	            image_url = line['image']
	            if isinstance(image_url, str) and image_url.startswith("http"):
	                msgs.append(dict(type='image', value=image_url))
	        if 'question' in line:
	            question = line['question']
	            if isinstance(question, str) and len(question) > 0:
	                msgs.append(dict(type='text', value=question))
	        return msgs
# def evaluate(self, eval_file, **judge_kwargs):
# 	        data = pd.read_excel(eval_file)
# 	        correct = (data['prediction'] == data['answer']).sum()
# 	        total = len(data)
# 	        accuracy = correct / total
# 	        return {'accuracy': accuracy}
def evaluate(self, eval_file, **judge_kwargs):
    data = load(eval_file)

    correct = (data['prediction'] == data['answer']).sum()
    total = len(data)

    acc = correct / total * 100

    return {
        "Strict Accuracy (%)": round(acc, 2),
        "Correct": int(correct),
        "Total": int(total),
    }


