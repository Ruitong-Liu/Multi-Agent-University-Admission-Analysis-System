"""
System-Ablation 消融实验
将 matching_score < 50 → Reject 规则改为条件激活：
  - 仅当 Agent 2 返回非空要求数据时触发规则
  - 对空占位样本直接使用模型原始 admission_decision
只重跑 System 层（4个模型 × Zero-shot + Few-shot），不重跑 Vision/Text。
"""

import json
import os
import random
from typing import Dict, List, Any

from experiments.few_shot_evaluation import FewShotEvaluator, load_transcript_json, load_requirements_json
from experiments.compare_system_vs_baseline import run_system_evaluation
import google.generativeai as genai  # type: ignore

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY",
    os.environ.get("OPENAI_API_KEY", "")
)
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

MODEL_CONFIGS = [
    {"label": "GPT-4o",           "provider": "openai",  "model": "gpt-4o"},
    {"label": "GPT-5.2",          "provider": "openai",  "model": "gpt-5.2-2025-12-11"},
    {"label": "Gemini 2.5 Flash", "provider": "gemini",  "model": "gemini-2.5-flash"},
    {"label": "Gemini 3.0 Flash", "provider": "gemini",  "model": "gemini-3-flash-preview"},
]

ACCEPTANCE_PATH = "acceptance.json"
SEED = 42
OUTPUT_PATH = "results/system_ablation_4models_seed42.json"


def _cm(metrics) -> Dict:
    return metrics.confusion_matrix


def main():
    genai.configure(api_key=GEMINI_API_KEY)

    # 固定 seed，生成与原始实验完全相同的共享 few-shot / test 集
    random.seed(SEED)
    shared_evaluator = FewShotEvaluator(ACCEPTANCE_PATH)
    shared_evaluator.load_and_convert_data()
    shared_few_shot, shared_test_set = shared_evaluator.select_few_shot_examples(k=5)

    print("=" * 80)
    print("System-Ablation：条件激活 matching_score 规则")
    print(f"  seed={SEED}  |  few-shot={len(shared_few_shot)}  |  test={len(shared_test_set)}")
    print("  规则：仅当 Agent2 返回非空要求数据时触发 score<50→Reject")
    print("=" * 80)
    print(f"\nFew-shot 样本（与原始实验相同）:")
    for s in shared_few_shot:
        print(f"  [{s.label}] {s.student_id} {s.school} / {s.program}")

    all_results = []

    for cfg in MODEL_CONFIGS:
        label   = cfg["label"]
        provider = cfg["provider"]
        model   = cfg["model"]

        print(f"\n{'#'*70}")
        print(f"# {label}  ({model})")
        print(f"{'#'*70}")

        try:
            random.seed(SEED)
            sys_z, sys_f, _, _ = run_system_evaluation(
                acceptance_path=ACCEPTANCE_PATH,
                provider=provider,
                model_name=model,
                openai_api_key=OPENAI_API_KEY if provider == "openai" else None,
                few_shot_examples=shared_few_shot,
                test_set=shared_test_set,
            )

            cm_z = _cm(sys_z)
            cm_f = _cm(sys_f)

            # 计算 Precision / Recall / F1 / MCC
            def metrics_from_cm(cm):
                tp = cm.get("TP", 0); fp = cm.get("FP", 0)
                tn = cm.get("TN", 0); fn = cm.get("FN", 0)
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                denom = ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) ** 0.5
                mcc  = (tp*tn - fp*fn) / denom if denom > 0 else 0.0
                return {"precision": round(prec,3), "recall": round(rec,3),
                        "f1": round(f1,3), "mcc": round(mcc,3)}

            res = {
                "label": label,
                "model": model,
                "ablation": "conditional_rule_activation",
                "zero_shot": {
                    "accuracy": round(sys_z.accuracy, 4),
                    "confusion_matrix": cm_z,
                    **metrics_from_cm(cm_z),
                    "errors": sys_z.errors,
                },
                "few_shot": {
                    "accuracy": round(sys_f.accuracy, 4),
                    "confusion_matrix": cm_f,
                    **metrics_from_cm(cm_f),
                    "errors": sys_f.errors,
                },
            }
            all_results.append(res)

            print(f"\n  Zero-shot: Acc={sys_z.accuracy:.1%}  "
                  f"TP={cm_z.get('TP',0)} FP={cm_z.get('FP',0)} "
                  f"TN={cm_z.get('TN',0)} FN={cm_z.get('FN',0)}")
            print(f"  Few-shot:  Acc={sys_f.accuracy:.1%}  "
                  f"TP={cm_f.get('TP',0)} FP={cm_f.get('FP',0)} "
                  f"TN={cm_f.get('TN',0)} FN={cm_f.get('FN',0)}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            all_results.append({"label": label, "model": model, "error": str(e)})

    os.makedirs("results", exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n\n{'='*80}")
    print("【System-Ablation 汇总】")
    print(f"{'模型':<22} {'Sys-Z(Abl)':>11} {'Sys-F(Abl)':>11}")
    print("-" * 46)
    for r in all_results:
        if "error" in r:
            print(f"{r['label']:<22}  ERROR: {r['error'][:40]}")
        else:
            print(f"{r['label']:<22} {r['zero_shot']['accuracy']:>10.1%} {r['few_shot']['accuracy']:>10.1%}")
    print(f"\n✓ 结果保存至: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
