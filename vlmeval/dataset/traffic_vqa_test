import ast
from .image_base import ImageBaseDataset
from ..smp import *
from .utils.sarena_mini import evaluate_sarena_mini


class traffic_vqa_test(ImageBaseDataset):

    TYPE = "VQA"

    DATASET_URL = {
        "traffic_vqa_test": "https://huggingface.co/datasets/chandrabhuma/vlmevalkit_files/resolve/main/tafficvqa_vlm_test.tsv"
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
def evaluate(self, eval_file, **judge_kwargs):
	        data = pd.read_excel(eval_file)
	        correct = (data['prediction'] == data['answer']).sum()
	        total = len(data)
	        accuracy = correct / total
	        return {'accuracy': accuracy}
	



   
