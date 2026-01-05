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
    data = pd.read_excel(eval_file)

    exact_correct = 0
    relaxed_correct = 0
    vqa_total_score = 0.0

    for _, row in data.iterrows():
        pred = row["prediction"]
        gt = row["answer"]

        # Exact accuracy
        if pred == gt:
            exact_correct += 1

        # Relaxed accuracy
        if relaxed_match(pred, gt):
            relaxed_correct += 1

        # VQA accuracy
        if isinstance(gt, list):
            answers = gt
        else:
            try:
                answers = ast.literal_eval(gt) if isinstance(gt, str) and gt.startswith("[") else [gt]
            except Exception:
                answers = [gt]

        vqa_total_score += vqa_score(pred, answers)

    total = len(data)

    exact_acc = exact_correct / total
    relaxed_acc = relaxed_correct / total
    vqa_acc = vqa_total_score / total

    # 🔹 PRINT METRICS
    print("=" * 50)
    print(f"[{self.DATASET_NAME}] Evaluation Results")
    print(f"Exact Accuracy   : {exact_acc:.4f}")
    print(f"Relaxed Accuracy : {relaxed_acc:.4f}")
    print(f"VQA Accuracy     : {vqa_acc:.4f}")
    print("=" * 50)

    # 🔹 RETURN for VLMEvalKit logging
    return {
        "accuracy": exact_acc,
        "relaxed_accuracy": relaxed_acc,
        "vqa_accuracy": vqa_acc,
    }



   
