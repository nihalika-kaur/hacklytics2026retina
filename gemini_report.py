import json
import os
from typing import Optional

import google.generativeai as genai

TEMPLATE = """You are a medical ML reviewer. Given validation AUCs for 8 retinal conditions from RETFound fine-tuning, produce:
- 3 key takeaways
- 2 risks or biases
- 3 next-step experiments to raise mean AUC by >=2 points
Metrics:
- mean_auc: {mean_auc:.4f}
- per_class_auc: {per_class}
Classes order: [N, D, G, C, A, H, M, O]
"""

def summarize_with_gemini(metrics_path: str, api_key: Optional[str] = None) -> str:
    if api_key is None:
        api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY env var or pass api_key.")
    with open(metrics_path) as f:
        metrics = json.load(f)
    mean_auc = float(metrics["best_mean_auc"])
    per_class = metrics["per_class_auc"]
    prompt = TEMPLATE.format(mean_auc=mean_auc, per_class=per_class)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    resp = model.generate_content(prompt)
    return resp.text

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True, help="Path to metrics.json from training")
    parser.add_argument("--api_key", default=None)
    args = parser.parse_args()
    print(summarize_with_gemini(args.metrics, args.api_key))