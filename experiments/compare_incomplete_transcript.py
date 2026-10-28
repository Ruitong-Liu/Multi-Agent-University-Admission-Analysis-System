"""
Compare an incomplete transcript against a complete transcript.
Runs both system (Agent3 with transcript JSON) and Gemini baseline (PDF vision)
on the same set of programs, then summarizes differences.
"""

import argparse
import json
import os
from typing import Dict, Any, List, Optional, Tuple

from experiments.few_shot_evaluation import ApplicationSample, load_requirements_json
from agent3 import MatchingModel, GeminiProvider
from experiments.gemini_baseline_evaluation import GeminiBaselineEvaluator


DEFAULT_GEMINI_KEY = os.environ.get("GOOGLE_API_KEY", "")


def load_programs_from_acceptance(acceptance_path: str) -> List[Dict[str, str]]:
    with open(acceptance_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    seen = set()
    programs = []
    for student_data in data.values():
        for bucket in ("accepted", "rejected"):
            for item in student_data.get(bucket, []):
                school = item.get("school", "").strip()
                program = item.get("major", "").strip()
                key = (school.lower(), program.lower())
                if school and program and key not in seen:
                    seen.add(key)
                    programs.append({"school": school, "program": program})
    return programs


def load_transcript_json(student_id: str, transcripts_dir: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(transcripts_dir, f"transcript{student_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_transcript_pdf(student_id: str, transcripts_dir: str) -> Optional[str]:
    path = os.path.join(transcripts_dir, f"transcript{student_id}.pdf")
    return path if os.path.exists(path) else None


def normalize_prediction(raw: Dict[str, Any]) -> Dict[str, Any]:
    decision_raw = str(raw.get("admission_decision", "")).strip()
    if "admit" in decision_raw.lower() or "录取" in decision_raw:
        decision = "Admit"
    elif "reject" in decision_raw.lower() or "拒绝" in decision_raw:
        decision = "Reject"
    else:
        decision = "Unknown"

    score = raw.get("matching_score", 0)
    try:
        score = int(float(score))
    except Exception:
        score = 0
    score = max(0, min(100, score))
    if score < 50:
        decision = "Reject"

    return {
        "admission_decision": decision,
        "matching_score": score,
        "reasoning": raw.get("reasoning", ""),
    }


def summarize_predictions(preds: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not preds:
        return {
            "count": 0,
            "admit_rate": 0.0,
            "avg_score": 0.0,
        }
    admit_count = sum(1 for p in preds if p["admission_decision"] == "Admit")
    avg_score = sum(p["matching_score"] for p in preds) / len(preds)
    return {
        "count": len(preds),
        "admit_rate": admit_count / len(preds),
        "avg_score": avg_score,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare incomplete transcript with a complete transcript"
    )
    parser.add_argument("--incomplete-id", default="02", help="不完整成绩单ID")
    parser.add_argument("--complete-id", default="01", help="完整成绩单ID")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini模型")
    parser.add_argument("--acceptance", default="acceptance.json", help="acceptance.json路径")
    parser.add_argument("--transcripts-dir", default="Student_Transcripts", help="成绩单目录")
    parser.add_argument("--limit", type=int, default=None, help="只比较前N个项目")
    parser.add_argument("--output", default=None, help="输出JSON路径")
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY", DEFAULT_GEMINI_KEY)

    programs = load_programs_from_acceptance(args.acceptance)
    if args.limit:
        programs = programs[: args.limit]

    incomplete_json = load_transcript_json(args.incomplete_id, args.transcripts_dir)
    complete_json = load_transcript_json(args.complete_id, args.transcripts_dir)

    if incomplete_json is None:
        raise FileNotFoundError(f"缺少 transcript{args.incomplete_id}.json")
    if complete_json is None:
        raise FileNotFoundError(f"缺少 transcript{args.complete_id}.json")

    incomplete_pdf = load_transcript_pdf(args.incomplete_id, args.transcripts_dir)
    complete_pdf = load_transcript_pdf(args.complete_id, args.transcripts_dir)

    # System evaluator (Agent3)
    provider = GeminiProvider(api_key)
    system_model = MatchingModel(provider=provider, model_name=args.model)

    # Baseline evaluator (Gemini Vision)
    baseline = GeminiBaselineEvaluator(
        acceptance_json_path=args.acceptance,
        gemini_api_key=api_key,
        model_name=args.model,
        transcripts_dir=args.transcripts_dir,
    )

    system_results_incomplete = []
    system_results_complete = []
    baseline_results_incomplete = []
    baseline_results_complete = []

    detailed_rows = []

    for idx, program in enumerate(programs, 1):
        school = program["school"]
        prog = program["program"]

        requirements = load_requirements_json(school, prog, use_daad=False)
        if not requirements:
            requirements = {"University": school, "Program": prog, "requirements": {}}

        # System: incomplete
        try:
            res_inc = system_model.evaluate(incomplete_json, requirements)
            res_inc = normalize_prediction(res_inc)
        except Exception as e:
            res_inc = {"admission_decision": "Error", "matching_score": 0, "reasoning": str(e)}

        # System: complete
        try:
            res_com = system_model.evaluate(complete_json, requirements)
            res_com = normalize_prediction(res_com)
        except Exception as e:
            res_com = {"admission_decision": "Error", "matching_score": 0, "reasoning": str(e)}

        system_results_incomplete.append(res_inc)
        system_results_complete.append(res_com)

        # Baseline (PDF)
        baseline_inc = {"admission_decision": "Skipped", "matching_score": 0, "reasoning": "PDF missing"}
        baseline_com = {"admission_decision": "Skipped", "matching_score": 0, "reasoning": "PDF missing"}

        if incomplete_pdf and complete_pdf:
            sample_inc = ApplicationSample(args.incomplete_id, school, prog, "Unknown")
            sample_com = ApplicationSample(args.complete_id, school, prog, "Unknown")

            try:
                prompt_inc = baseline.build_zero_shot_prompt(sample_inc, requirements)
                images_inc = baseline._pdf_to_images(incomplete_pdf)
                try:
                    raw_inc = baseline._generate_with_vision(prompt_inc, images_inc)
                    baseline_inc = normalize_prediction(baseline.parse_response(raw_inc))
                finally:
                    for p in images_inc:
                        try:
                            os.unlink(p)
                        except Exception:
                            pass
            except Exception as e:
                baseline_inc = {"admission_decision": "Error", "matching_score": 0, "reasoning": str(e)}

            try:
                prompt_com = baseline.build_zero_shot_prompt(sample_com, requirements)
                images_com = baseline._pdf_to_images(complete_pdf)
                try:
                    raw_com = baseline._generate_with_vision(prompt_com, images_com)
                    baseline_com = normalize_prediction(baseline.parse_response(raw_com))
                finally:
                    for p in images_com:
                        try:
                            os.unlink(p)
                        except Exception:
                            pass
            except Exception as e:
                baseline_com = {"admission_decision": "Error", "matching_score": 0, "reasoning": str(e)}

        baseline_results_incomplete.append(baseline_inc)
        baseline_results_complete.append(baseline_com)

        detailed_rows.append(
            {
                "school": school,
                "program": prog,
                "system_incomplete": res_inc,
                "system_complete": res_com,
                "baseline_incomplete": baseline_inc,
                "baseline_complete": baseline_com,
            }
        )

        print(f"[{idx}/{len(programs)}] {school} - {prog}")

    summary = {
        "system_incomplete": summarize_predictions(system_results_incomplete),
        "system_complete": summarize_predictions(system_results_complete),
        "baseline_incomplete": summarize_predictions(baseline_results_incomplete),
        "baseline_complete": summarize_predictions(baseline_results_complete),
    }

    output = {
        "incomplete_id": args.incomplete_id,
        "complete_id": args.complete_id,
        "model": args.model,
        "program_count": len(programs),
        "summary": summary,
        "details": detailed_rows,
    }

    output_path = (
        args.output
        if args.output
        else f"incomplete_vs_complete_{args.incomplete_id}_vs_{args.complete_id}.json"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n对比完成：")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
