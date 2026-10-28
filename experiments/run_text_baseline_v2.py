"""
受控版 Text Baseline 重跑脚本（v2）

修复的混淆变量：
  1. 提示词哲学：与 Agent 3 一致（不再使用 "if uncertain → Reject"）
  2. 项目要求数据格式：CSV 以 JSON 格式加载（与 Vision Baseline 一致）
  3. 文本截断：取消 3000 字符限制
  4. 后处理：不再基于 matching_score 强制覆盖决策

运行模型：GPT-4o, GPT-5.2, Gemini 2.5 Flash, Gemini 3.0 Flash
评测设置：seed=42，N=33，k=5（与原始实验完全一致）
"""

import json
import os
import random
import math
from typing import Dict, Any

from experiments.few_shot_evaluation import FewShotEvaluator
from experiments.text_baseline_evaluation import TextBaselineEvaluator

OPENAI_KEY = os.getenv(
    "OPENAI_API_KEY",
    os.environ.get("OPENAI_API_KEY", "")
)

MODELS = [
    {"provider": "openai",  "model": "gpt-4o",                 "label": "gpt-4o"},
    {"provider": "openai",  "model": "gpt-5.2-2025-12-11",     "label": "gpt-5.2-2025-12-11"},
    {"provider": "gemini",  "model": "gemini-2.5-flash",       "label": "gemini-2.5-flash"},
    {"provider": "gemini",  "model": "gemini-3-flash-preview",  "label": "gemini-3-flash-preview"},
]

OUTPUT_FILE = "results/text_baseline_v2_all4models.json"


def cm_to_metrics(tp, fp, tn, fn) -> Dict[str, Any]:
    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    denom = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc  = (tp*tn - fp*fn) / denom if denom > 0 else 0.0
    return {
        "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "mcc": round(mcc, 4),
    }


def main():
    # ── 加载数据（与原始实验完全相同的 seed=42 划分）──────────────────────────
    random.seed(42)
    base_ev = FewShotEvaluator("acceptance.json")
    base_ev.load_and_convert_data()
    few_shot_examples, test_set = base_ev.select_few_shot_examples(k=5)
    print(f"Test set: {len(test_set)} (Admit={sum(1 for s in test_set if s.label=='Admit')}, "
          f"Reject={sum(1 for s in test_set if s.label=='Reject')})")
    print(f"Few-shot: {len(few_shot_examples)}")

    all_results = []

    for cfg in MODELS:
        print(f"\n{'='*70}")
        print(f"Running Text Baseline v2: {cfg['label']} ({cfg['provider']})")
        print(f"{'='*70}")

        ev = TextBaselineEvaluator(
            provider=cfg["provider"],
            model_name=cfg["model"],
            openai_api_key=OPENAI_KEY if cfg["provider"] == "openai" else None,
        )

        zs_metrics, fs_metrics = ev.evaluate(
            test_set=test_set,
            few_shot_examples=few_shot_examples,
        )

        # ── 提取混淆矩阵（EvaluationMetrics 对象可能有不同字段名）──────────
        def extract(m):
            cm = getattr(m, 'confusion_matrix', {})
            tp = cm.get('TP', 0)
            fp = cm.get('FP', 0)
            tn = cm.get('TN', 0)
            fn = cm.get('FN', 0)
            return cm_to_metrics(int(tp), int(fp), int(tn), int(fn))

        result = {
            "model": cfg["label"],
            "provider": cfg["provider"],
            "version": "v2_controlled",
            "zero_shot": extract(zs_metrics),
            "few_shot":  extract(fs_metrics),
        }
        all_results.append(result)

        # 实时保存（防止中途中断丢失数据）
        os.makedirs("results", exist_ok=True)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        print(f"\n  Zero-shot: {result['zero_shot']['confusion_matrix']}  "
              f"Acc={result['zero_shot']['accuracy']:.1%}  "
              f"F1={result['zero_shot']['f1']:.3f}  MCC={result['zero_shot']['mcc']:.3f}")
        print(f"  Few-shot:  {result['few_shot']['confusion_matrix']}  "
              f"Acc={result['few_shot']['accuracy']:.1%}  "
              f"F1={result['few_shot']['f1']:.3f}  MCC={result['few_shot']['mcc']:.3f}")

    print(f"\n{'='*70}")
    print(f"All done. Results saved to {OUTPUT_FILE}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
