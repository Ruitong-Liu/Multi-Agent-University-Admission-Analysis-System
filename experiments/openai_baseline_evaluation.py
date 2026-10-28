"""
OpenAI Baseline Evaluation System
使用 GPT-4o / GPT-5.2 Vision API 直接评估（不使用 Agent 系统）作为 baseline 对比
直接读取 PDF 成绩单图像，不使用 Agent1 预处理后的 JSON
支持 zero-shot 和 few-shot 两种模式

架构与 gemini_baseline_evaluation.py 完全对称：
  - Baseline：PDF → GPT Vision → 决策
  - System  ：PDF → Agent1(JSON) → Agent3(JSON+structured prompt) → 决策
"""

import base64
import csv
import json
import os
import random
import re
import tempfile
from typing import Dict, List, Any, Optional, Tuple

try:
    from pdf2image import convert_from_path  # type: ignore
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

from experiments.few_shot_evaluation import (
    ApplicationSample,
    EvaluationMetrics,
    FewShotEvaluator,
)


class OpenAIBaselineEvaluator:
    """
    OpenAI Baseline 评估器
    直接把 PDF 成绩单图像送入 GPT Vision API，不经过 Agent1 预处理
    """

    def __init__(
        self,
        acceptance_json_path: str = "acceptance.json",
        openai_api_key: str = None,
        model_name: str = "gpt-4o",
        transcripts_dir: str = "Student_Transcripts",
        csv_path: str = "university_programs.csv",
    ):
        self.acceptance_json_path = acceptance_json_path
        self.model_name = model_name
        self.transcripts_dir = transcripts_dir
        self.csv_path = csv_path

        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("需要提供 OpenAI API 密钥")

        from openai import OpenAI  # type: ignore
        self.client = OpenAI(api_key=self.openai_api_key)

        # Fallback 模型列表（quota 不足时自动切换）
        self.fallback_models = ["gpt-4o", "gpt-4o-mini"]

        # 复用 FewShotEvaluator 的数据加载 / few-shot 选择逻辑
        self.base_evaluator = FewShotEvaluator(acceptance_json_path)

    # ── PDF 工具 ──────────────────────────────────────────────────────────────

    def _pdf_to_images(self, pdf_path: str) -> List[str]:
        """将 PDF 转换为 PNG 图像，返回临时文件路径列表"""
        if not PDF2IMAGE_AVAILABLE:
            raise ImportError("需要安装 pdf2image：pip install pdf2image")
        images = convert_from_path(pdf_path)
        paths = []
        for img in images:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            img.save(tmp.name, "PNG")
            paths.append(tmp.name)
        return paths

    def _image_to_base64(self, image_path: str) -> str:
        """将图像文件编码为 base64 字符串（OpenAI Vision 格式要求）"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _load_transcript_pdf(self, student_id: str) -> Optional[str]:
        """获取成绩单 PDF 路径"""
        path = os.path.join(self.transcripts_dir, f"transcript{student_id}.pdf")
        return path if os.path.exists(path) else None

    # ── CSV 要求加载（与 Gemini Baseline 完全一致）────────────────────────────

    def _load_requirements_from_csv(self, school: str, program: str) -> Optional[Dict]:
        """从 CSV 直接加载项目要求（简化版，不使用 Agent2 的复杂逻辑）"""
        try:
            if not os.path.exists(self.csv_path):
                return None
            with open(self.csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row_school = row.get("University", "").strip()
                    row_program = row.get("Program", "").strip()
                    if school.lower() in row_school.lower() or row_school.lower() in school.lower():
                        if program.lower() in row_program.lower() or row_program.lower() in program.lower():
                            return {
                                "University": row_school,
                                "Program": row_program,
                                "requirements": {
                                    k: v for k, v in row.items()
                                    if k not in ["University", "Program"] and v
                                },
                            }
        except Exception as e:
            print(f"    ⚠ CSV 加载错误: {e}")
        return None

    # ── Vision API 调用 ───────────────────────────────────────────────────────

    def _generate_with_vision(self, prompt: str, image_paths: List[str]) -> str:
        """使用 OpenAI Vision API 生成响应（PDF 图像 + 文本）"""
        import time

        # 构建包含图像的消息
        image_contents = []
        for path in image_paths:
            b64 = self._image_to_base64(path)
            image_contents.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
            })

        messages = [
            {
                "role": "system",
                "content": "You are an expert admission officer. Always respond in valid JSON format.",
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}] + image_contents,
            },
        ]

        models_to_try = [self.model_name] + [m for m in self.fallback_models if m != self.model_name]
        last_error = None
        for model in models_to_try:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={"type": "json_object"},
                )
                return response.choices[0].message.content
            except Exception as e:
                last_error = e
                error_str = str(e)
                if "429" in error_str or "quota" in error_str.lower() or "insufficient_quota" in error_str:
                    print(f"    ⚠ {model} 配额不足，尝试下一个模型...")
                    time.sleep(2)
                elif "connection" in error_str.lower() or "timeout" in error_str.lower():
                    print(f"    ⚠ {model} 连接错误，等待后重试...")
                    time.sleep(5)
                else:
                    print(f"    ⚠ {model} 错误: {error_str[:120]}，尝试下一个模型...")
                    time.sleep(1)
                continue

        print(f"    ✗ 所有模型都失败，最后错误: {str(last_error)[:200]}")
        return '{"admission_decision": "Unknown", "matching_score": 50, "reasoning": "API error"}'

    def _generate_text_only(self, prompt: str) -> str:
        """纯文本调用（用于 few-shot 摘要生成）"""
        import time
        models_to_try = [self.model_name] + [m for m in self.fallback_models if m != self.model_name]
        last_error = None
        for model in models_to_try:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful academic assistant."},
                        {"role": "user", "content": prompt},
                    ],
                )
                return response.choices[0].message.content
            except Exception as e:
                last_error = e
                time.sleep(1)
                continue
        return "Transcript summary unavailable."

    # ── Prompt 构建（与 Gemini Baseline 完全对称）────────────────────────────

    def build_zero_shot_prompt(self, sample: ApplicationSample, requirements_json: Dict) -> str:
        requirements_text = (
            json.dumps(requirements_json, indent=2, ensure_ascii=False)
            if requirements_json else "Not available"
        )
        return f"""You are a strict admission officer for German Master's programmes.

You will receive:
- A student's transcript in PDF format (as images).
- Program admission requirements in JSON format (when available).

IMPORTANT - CRITICAL DISTINCTION:
- The requirement JSON may be incomplete, generic, outdated, or noisy.
- You must treat it only as a hint, NOT as a perfectly reliable source.
- **CRITICAL**: There is a fundamental difference between:

  a) "Cannot fetch/obtain requirements" (信息缺失) - Requirements data is missing or unavailable.
     → Base your decision on the transcript quality and typical admission criteria.
     → DO NOT automatically reject due to missing requirements.
     → If the student's academic profile is strong and matches typical expectations, consider "Admit".

  b) "Does not meet requirements" (条件不达标) - Student clearly lacks prerequisites.
     → Output "Reject" with a clear explanation.

- Your primary signals are:
  1) The student's transcript (GPA, major, core courses) – read directly from the PDF images,
  2) Your knowledge about typical admission criteria for similar programmes at German universities,
  3) The requirement JSON, but only when it contains concrete and reasonable conditions.

Decision rules:
- If the student clearly lacks core prerequisites or has a clearly too low GPA, output "Reject".
- If the student clearly matches the usual background for the programme, output "Admit" even if JSON is incomplete.
- If requirements JSON is unavailable: base decision on transcript + typical standards. DO NOT auto-reject.
- If still unsure after checking everything, prefer "Reject".
- matching_score < 50 always corresponds to "Reject".

---

### NEW SAMPLE

Student ID: {sample.student_id}
School: {sample.school}
Program: {sample.program}

Program Requirements (JSON):
{requirements_text}

Please carefully read the transcript PDF images provided and evaluate the student's application.

Respond STRICTLY in the following JSON format:

{{
  "admission_decision": "Admit" | "Reject",
  "matching_score": <integer 0-100>,
  "reasoning": "<detailed reasoning referencing GPA, prerequisites, language>"
}}"""

    def build_few_shot_prompt(
        self,
        sample: ApplicationSample,
        requirements_json: Dict,
        few_shot_data: List[Dict],
    ) -> str:
        requirements_text = (
            json.dumps(requirements_json, indent=2, ensure_ascii=False)
            if requirements_json else "Not available"
        )

        prompt = """You are a strict admission officer for German Master's programmes.

You will receive:
- A student's transcript in PDF format (as images).
- Program admission requirements in JSON format (when available).

IMPORTANT - CRITICAL DISTINCTION:
- The requirement JSON may be incomplete, generic, outdated, or noisy.
- You must treat it only as a hint, NOT as a perfectly reliable source.
- **CRITICAL**: There is a fundamental difference between:

  a) "Cannot fetch/obtain requirements" (信息缺失) - Requirements data is missing or unavailable.
     → Base your decision on the transcript quality and typical admission criteria.
     → DO NOT automatically reject due to missing requirements.
     → If the student's academic profile is strong and matches typical expectations, consider "Admit".

  b) "Does not meet requirements" (条件不达标) - Student clearly lacks prerequisites.
     → Output "Reject" with a clear explanation.

Decision rules:
- If the student clearly lacks core prerequisites or has a clearly too low GPA, output "Reject".
- If the student clearly matches the usual background for the programme, output "Admit" even if JSON is incomplete.
- If requirements JSON is unavailable: base decision on transcript + typical standards. DO NOT auto-reject.
- If still unsure after checking everything, prefer "Reject".
- matching_score < 50 always corresponds to "Reject".

---

"""
        for i, ex in enumerate(few_shot_data, 1):
            ex_sample = ex["sample"]
            prompt += f"""### FEW-SHOT EXAMPLE {i}

Student ID: {ex_sample.student_id}
School: {ex_sample.school}
Program: {ex_sample.program}
Transcript Summary: {ex.get("transcript_summary", "N/A")}
Program Requirements (JSON):
{json.dumps(ex.get("requirements_json", {}), indent=2, ensure_ascii=False)}

Expected output:
{{
  "admission_decision": "{ex_sample.label}",
  "matching_score": {ex.get("matching_score", 75 if ex_sample.label == "Admit" else 40)},
  "reasoning": "{ex.get("reasoning", "Based on transcript and requirements.")}"
}}

"""

        prompt += f"""---

### NEW SAMPLE

Student ID: {sample.student_id}
School: {sample.school}
Program: {sample.program}

Program Requirements (JSON):
{requirements_text}

Please carefully read the transcript PDF images provided and evaluate the student's application.

Respond STRICTLY in the following JSON format:

{{
  "admission_decision": "Admit" | "Reject",
  "matching_score": <integer 0-100>,
  "reasoning": "<detailed reasoning referencing GPA, prerequisites, language>"
}}"""
        return prompt

    # ── 响应解析 ──────────────────────────────────────────────────────────────

    def parse_response(self, response: str) -> Dict[str, Any]:
        """解析 GPT 响应，提取 JSON"""
        for pattern in [
            r'```(?:json)?\s*(\{.*?\})\s*```',
            r'\{[^{}]*"admission_decision"[^{}]*\}',
        ]:
            m = re.search(pattern, response, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(1) if '```' in pattern else m.group(0))
                except json.JSONDecodeError:
                    pass

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        result = {}
        for key, pat in [
            ("admission_decision", r'"admission_decision"\s*:\s*"([^"]+)"'),
            ("matching_score",     r'"matching_score"\s*:\s*(\d+)'),
            ("reasoning",         r'"reasoning"\s*:\s*"([^"]+)"'),
        ]:
            m = re.search(pat, response, re.IGNORECASE)
            if m:
                val = m.group(1)
                if key == "admission_decision":
                    result[key] = "Admit" if "admit" in val.lower() else "Reject"
                elif key == "matching_score":
                    result[key] = int(val)
                else:
                    result[key] = val
        return result

    # ── 主评估流程 ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        test_set: List[ApplicationSample],
        few_shot_examples: Optional[List[ApplicationSample]] = None,
    ) -> Tuple[EvaluationMetrics, EvaluationMetrics]:
        """
        在 zero-shot 和 few-shot 两种条件下评估（直接读取 PDF）

        Returns:
            (zero_shot_metrics, few_shot_metrics)
        """
        print(f"\n开始 OpenAI Baseline 评估 (测试集大小: {len(test_set)}, 模型: {self.model_name})")
        print("⚠ 注意: 直接读取 PDF，不使用 Agent1/Agent2 预处理")

        # ── 准备 few-shot 示例（使用详细文本摘要替代单句摘要，提升信息密度）────
        few_shot_data = None
        if few_shot_examples:
            print("\n[准备 Few-shot 示例数据]")
            few_shot_data = []
            for example in few_shot_examples:
                requirements = self._load_requirements_from_csv(example.school, example.program)
                transcript_summary = "Transcript data available (see PDF image)"

                transcript_pdf = self._load_transcript_pdf(example.student_id)
                if transcript_pdf:
                    try:
                        image_paths = self._pdf_to_images(transcript_pdf)
                        try:
                            # 生成更详细的摘要（包含 GPA、专业、核心课程），
                            # 避免原有单句摘要信息严重不足的问题
                            summary_prompt = (
                                f"Please read this transcript carefully and provide a structured summary "
                                f"with the following fields:\n"
                                f"- Major/Field of study\n"
                                f"- GPA (exact value if visible)\n"
                                f"- Total ECTS or credit hours (if available)\n"
                                f"- Top 5 most relevant courses for '{example.program}'\n"
                                f"- Language qualifications (if visible)\n"
                                f"Keep each field to one line. Respond in English."
                            )
                            summary = self._generate_with_vision(summary_prompt, image_paths[:1])
                            transcript_summary = summary[:800]
                        finally:
                            for p in image_paths:
                                try:
                                    os.unlink(p)
                                except Exception:
                                    pass
                    except Exception as e:
                        print(f"    ⚠ 无法生成 few-shot 摘要: {e}")

                matching_score = 80 if example.label == "Admit" else 35
                reasoning = (
                    "Major matches programme background, GPA meets threshold, key prerequisite courses satisfied."
                    if example.label == "Admit"
                    else "Major mismatch or GPA below threshold; key prerequisite courses are missing."
                )

                few_shot_data.append({
                    "sample": example,
                    "requirements_json": requirements or {},
                    "transcript_summary": transcript_summary,
                    "matching_score": matching_score,
                    "reasoning": reasoning,
                })

            print(f"  ✓ 成功准备 {len(few_shot_data)}/{len(few_shot_examples)} 个 few-shot 示例")

        # ── 内部评估辅助函数 ──────────────────────────────────────────────────
        def _evaluate_one(sample: ApplicationSample, use_few_shot: bool) -> Optional[Dict]:
            transcript_pdf = self._load_transcript_pdf(sample.student_id)
            if not transcript_pdf:
                print(f"    ⚠ 跳过样本: PDF 不存在")
                return None

            requirements = self._load_requirements_from_csv(sample.school, sample.program) or {
                "University": sample.school,
                "Program": sample.program,
                "requirements": {},
            }

            prompt = (
                self.build_few_shot_prompt(sample, requirements, few_shot_data)
                if use_few_shot and few_shot_data
                else self.build_zero_shot_prompt(sample, requirements)
            )

            image_paths = []
            try:
                image_paths = self._pdf_to_images(transcript_pdf)
                response_text = self._generate_with_vision(prompt, image_paths)
                prediction = self.parse_response(response_text)

                predicted_label = prediction.get("admission_decision", "").strip()
                if "admit" in predicted_label.lower():
                    predicted_label = "Admit"
                elif "reject" in predicted_label.lower():
                    predicted_label = "Reject"
                else:
                    predicted_label = "Unknown"

                score_val = prediction.get("matching_score", 0)
                try:
                    score_val = int(float(score_val))
                except (TypeError, ValueError):
                    score_val = 0
                score_val = max(0, min(100, score_val))

                if score_val < 50:
                    predicted_label = "Reject"

                return {
                    "sample": sample,
                    "prediction": predicted_label,
                    "actual": sample.label,
                    "matching_score": score_val,
                    "reasoning": prediction.get("reasoning", ""),
                    "raw_response": response_text,
                }
            except Exception as e:
                print(f"    ✗ 错误: {str(e)[:150]}")
                return None
            finally:
                for p in image_paths:
                    try:
                        os.unlink(p)
                    except Exception:
                        pass

        def _compute_metrics(preds: List[Dict], errors: List[Dict]) -> EvaluationMetrics:
            tp = fp = tn = fn = 0
            for pred in preds:
                p, a = pred["prediction"], pred["actual"]
                if p == "Admit" and a == "Admit":
                    tp += 1
                elif p == "Admit" and a == "Reject":
                    fp += 1
                elif p == "Reject" and a == "Reject":
                    tn += 1
                elif p == "Reject" and a == "Admit":
                    fn += 1
            total = tp + fp + tn + fn
            accuracy = (tp + tn) / total if total > 0 else 0.0
            return EvaluationMetrics(
                accuracy=accuracy,
                confusion_matrix={"TP": tp, "FP": fp, "TN": tn, "FN": fn},
                errors=errors,
            )

        # ── Zero-shot ─────────────────────────────────────────────────────────
        print("\n[Zero-shot 评估]")
        zs_preds, zs_errors = [], []
        for i, sample in enumerate(test_set):
            print(f"  处理样本 {i+1}/{len(test_set)}: {sample.student_id} - {sample.school} - {sample.program}")
            result = _evaluate_one(sample, use_few_shot=False)
            if result is None:
                continue
            zs_preds.append(result)
            if result["prediction"] != result["actual"]:
                zs_errors.append({
                    "student_id": sample.student_id,
                    "school": sample.school,
                    "program": sample.program,
                    "predicted": result["prediction"],
                    "actual": result["actual"],
                    "matching_score": result["matching_score"],
                })

        zero_shot_metrics = _compute_metrics(zs_preds, zs_errors)
        print(f"\n[Zero-shot 结果] 准确率: {zero_shot_metrics.accuracy:.2%}  "
              f"TP={zero_shot_metrics.confusion_matrix['TP']} "
              f"FP={zero_shot_metrics.confusion_matrix['FP']} "
              f"TN={zero_shot_metrics.confusion_matrix['TN']} "
              f"FN={zero_shot_metrics.confusion_matrix['FN']}")

        # ── Few-shot ──────────────────────────────────────────────────────────
        print("\n[Few-shot 评估]")
        fs_preds, fs_errors = [], []
        for i, sample in enumerate(test_set):
            print(f"  处理样本 {i+1}/{len(test_set)}: {sample.student_id} - {sample.school} - {sample.program}")
            result = _evaluate_one(sample, use_few_shot=True)
            if result is None:
                continue
            fs_preds.append(result)
            if result["prediction"] != result["actual"]:
                fs_errors.append({
                    "student_id": sample.student_id,
                    "school": sample.school,
                    "program": sample.program,
                    "predicted": result["prediction"],
                    "actual": result["actual"],
                    "matching_score": result["matching_score"],
                })

        few_shot_metrics = _compute_metrics(fs_preds, fs_errors)
        print(f"\n[Few-shot 结果] 准确率: {few_shot_metrics.accuracy:.2%}  "
              f"TP={few_shot_metrics.confusion_matrix['TP']} "
              f"FP={few_shot_metrics.confusion_matrix['FP']} "
              f"TN={few_shot_metrics.confusion_matrix['TN']} "
              f"FN={few_shot_metrics.confusion_matrix['FN']}")

        return zero_shot_metrics, few_shot_metrics
