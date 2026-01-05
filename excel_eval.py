import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# -------------------------
# Text normalization
# -------------------------
def normalize(text):
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# -------------------------
# Strict Accuracy
# -------------------------
def strict_accuracy(gt, pred):
    return int(normalize(gt) == normalize(pred))

# -------------------------
# Relaxed Accuracy
# (substring or token overlap)
# -------------------------
def relaxed_accuracy(gt, pred):
    gt_n = normalize(gt)
    pred_n = normalize(pred)
    return int(gt_n in pred_n or pred_n in gt_n)

# -------------------------
# VQA Accuracy (soft match)
# -------------------------
def vqa_accuracy(gt, pred):
    gt_tokens = normalize(gt).split()
    pred_tokens = normalize(pred).split()

    if len(gt_tokens) == 0:
        return 0.0

    common = set(gt_tokens) & set(pred_tokens)
    score = min(len(common) / len(gt_tokens), 1.0)
    return score

# -------------------------
# Cosine Similarity
# -------------------------
def mean_cosine_similarity(gt_list, pred_list):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    gt_emb = model.encode(gt_list, normalize_embeddings=True)
    pred_emb = model.encode(pred_list, normalize_embeddings=True)
    sims = np.diag(cosine_similarity(gt_emb, pred_emb))
    return sims.mean()

# -------------------------
# Main evaluation
# -------------------------
def evaluate_xlsx(xlsx_path):
    df = pd.read_excel(xlsx_path)

    assert {'answer', 'prediction'}.issubset(df.columns)

    strict_scores = []
    relaxed_scores = []
    vqa_scores = []

    for _, row in df.iterrows():
        gt = row['answer']
        pred = row['prediction']

        strict_scores.append(strict_accuracy(gt, pred))
        relaxed_scores.append(relaxed_accuracy(gt, pred))
        vqa_scores.append(vqa_accuracy(gt, pred))

    results = {
        "Strict Accuracy (%)": np.mean(strict_scores) * 100,
        "Relaxed Accuracy (%)": np.mean(relaxed_scores) * 100,
        "VQA Accuracy (%)": np.mean(vqa_scores) * 100,
        "Mean Cosine Similarity": mean_cosine_similarity(
            df['answer'].astype(str).tolist(),
            df['prediction'].astype(str).tolist()
        )
    }

    return results

# -------------------------
# Run from CLI
# -------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", type=str, required=True, help="Path to prediction xlsx")
    args = parser.parse_args()

    metrics = evaluate_xlsx(args.xlsx)

    print("\n===== VQA Evaluation Results =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")
