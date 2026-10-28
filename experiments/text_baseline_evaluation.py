"""
Text-only Baseline 评估器（受控版 v2）

变量拆解（三层，严格控制变量）：
  Vision Baseline : PDF 图像 → Vision LLM → 录取决策
  Text Baseline   : PDF 纯文本（pdfplumber/Vision提取）→ Text LLM → 录取决策
  System          : PDF → Agent1 JSON + Agent2 JSON → Agent3 Text LLM → 录取决策

受控变量设计（v2 改动说明）：
  - 提示词哲学：与 Agent 3 完全一致（不再使用 "if uncertain → Reject" 的硬性门控）
  - 项目要求数据来源：与 Vision Baseline 一致（从 CSV 加载为 JSON 格式）
  - 文本截断：取消 3000 字符硬截断（与其他条件保持一致）
  - 后处理规则：移除基于 matching_score 强制覆盖决策的逻辑（与 Agent 3 一致）
  - Few-shot 示例格式：与 Agent 3 格式对齐（原始文本替代 JSON 字段）
  唯一差异：成绩单表示形式（raw text vs Agent 1 JSON）

文本提取策略：
  - pdfplumber 提取（适用于含原生文本的 PDF）
  - 若提取失败（扫描图片 PDF），用 Vision LLM 转录为纯文本（一次性预处理）
"""

from __future__ import annotations

import csv
import json
import os
import re
import time
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import pdfplumber  # type: ignore
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    from pdf2image import convert_from_path  # type: ignore
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

from experiments.few_shot_evaluation import FewShotEvaluator, EvaluationMetrics


TRANSCRIPTS_DIR = "data/Student_Transcripts"
CSV_PATH = "data/university_programs.csv"
ACCEPTANCE_JSON = "acceptance.json"
TEXT_CACHE_FILE = "data/transcript_text_cache.json"

GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY", "")


def _load_disk_cache() -> Dict[str, str]:
    """从磁盘加载已提取的成绩单文本缓存。"""
    if os.path.exists(TEXT_CACHE_FILE):
        try:
            with open(TEXT_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_disk_cache(cache: Dict[str, str]) -> None:
    """将成绩单文本缓存写入磁盘。"""
    os.makedirs(os.path.dirname(TEXT_CACHE_FILE) if os.path.dirname(TEXT_CACHE_FILE) else ".", exist_ok=True)
    with open(TEXT_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# ── 文本提取 ────────────────────────────────────────────────────────────────

def _extract_text_pdfplumber(pdf_path: str) -> str:
    """尝试用 pdfplumber 提取原生文本。"""
    if not PDFPLUMBER_AVAILABLE:
        return ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages_text = [p.extract_text() or "" for p in pdf.pages]
        return "\n".join(pages_text).strip()
    except Exception:
        return ""


def _extract_text_via_vision(pdf_path: str, provider: str, model_name: str,
                              genai=None, openai_client=None) -> str:
    """
    当 pdfplumber 无法提取时（扫描 PDF），
    用 Vision LLM 将 PDF 图像转录为纯文本。
    仅用于文本提取，不做录取决策。
    """
    if not PDF2IMAGE_AVAILABLE:
        return ""
    try:
        pages = convert_from_path(pdf_path)
    except Exception as e:
        print(f"    ⚠ pdf2image 转换失败: {e}")
        return ""

    prompt = (
        "Please transcribe all text you can see in this student transcript image. "
        "Include course names, grades, GPA, major, university name, and any other academic information. "
        "Output plain text only, no JSON. Preserve the original content as accurately as possible."
    )

    extracted_pages = []
    for i, page_img in enumerate(pages):
        try:
            if provider == "gemini" and genai is not None:
                model_instance = genai.GenerativeModel(model_name)
                response = model_instance.generate_content([prompt, page_img])
                extracted_pages.append(response.text)
            elif provider == "openai" and openai_client is not None:
                import base64
                import io
                buf = io.BytesIO()
                page_img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode()
                resp = openai_client.chat.completions.create(
                    model=model_name,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/png;base64,{b64}"
                            }},
                        ],
                    }],
                )
                extracted_pages.append(resp.choices[0].message.content)
        except Exception as e:
            print(f"    ⚠ 第 {i+1} 页文本提取失败: {str(e)[:100]}")
            time.sleep(1)

    return "\n\n--- Page break ---\n\n".join(extracted_pages).strip()


def get_transcript_text(
    student_id: str,
    provider: str,
    model_name: str,
    genai=None,
    openai_client=None,
    text_cache: Optional[Dict[str, str]] = None,
) -> str:
    """
    获取成绩单纯文本。
    优先顺序：内存缓存 → 磁盘缓存 → pdfplumber → Vision LLM 转录
    提取后同时写入内存缓存和磁盘缓存，避免重复调用 Vision API。
    """
    # 1. 内存缓存
    if text_cache is not None and student_id in text_cache:
        return text_cache[student_id]

    # 2. 磁盘缓存
    disk_cache = _load_disk_cache()
    if student_id in disk_cache and len(disk_cache[student_id].strip()) > 50:
        if text_cache is not None:
            text_cache[student_id] = disk_cache[student_id]
        return disk_cache[student_id]

    # 3. pdfplumber 提取
    pdf_path = os.path.join(TRANSCRIPTS_DIR, f"transcript{student_id}.pdf")
    if not os.path.exists(pdf_path):
        return ""

    text = _extract_text_pdfplumber(pdf_path)

    # 4. Vision LLM 转录（扫描 PDF）
    if len(text.strip()) < 50:
        print(f"    → transcript{student_id} 无原生文本，使用 Vision 转录（结果将缓存到磁盘）...")
        text = _extract_text_via_vision(pdf_path, provider, model_name, genai, openai_client)

    # 写入缓存
    if text:
        disk_cache[student_id] = text
        _save_disk_cache(disk_cache)
        if text_cache is not None:
            text_cache[student_id] = text

    return text


# ── 项目要求加载（与 Vision Baseline 保持一致：JSON 格式）────────────────────

def load_requirements_as_json(school: str, program: str) -> Dict:
    """从 CSV 加载项目要求，以 dict/JSON 格式返回（与 Vision Baseline 一致）。"""
    if not os.path.exists(CSV_PATH):
        return {"University": school, "Program": program, "note": "Requirements not available"}
    try:
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_school = row.get("University", "").strip()
                row_program = row.get("Program", "").strip()
                if (school.lower() in row_school.lower() or row_school.lower() in school.lower()) and \
                   (program.lower() in row_program.lower() or row_program.lower() in program.lower()):
                    return {k: v.strip() for k, v in row.items() if v and v.strip()}
    except Exception:
        pass
    return {"University": school, "Program": program, "note": "Requirements not found in database"}


def load_requirements_as_text(school: str, program: str) -> str:
    """兼容旧版调用：以文本格式返回（内部转发至 JSON 加载再序列化）。"""
    return json.dumps(load_requirements_as_json(school, program), ensure_ascii=False, indent=2)


# ── Text Baseline 评估器 ─────────────────────────────────────────────────────

class TextBaselineEvaluator:
    """
    Text-only Baseline：
      成绩单原始文本 + 项目要求文本 → 纯文本 LLM → 录取决策
    """

    def __init__(
        self,
        acceptance_json_path: str = ACCEPTANCE_JSON,
        provider: str = "gemini",
        model_name: str = "gemini-2.5-flash",
        openai_api_key: Optional[str] = None,
    ):
        self.acceptance_json_path = acceptance_json_path
        self.provider = provider
        self.model_name = model_name
        self.openai_api_key = openai_api_key

        self.genai = None
        self.openai_client = None

        if provider == "gemini":
            import google.generativeai as genai_module  # type: ignore
            genai_module.configure(api_key=GEMINI_API_KEY)
            self.genai = genai_module
        elif provider == "openai":
            from openai import OpenAI  # type: ignore
            key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
            self.openai_client = OpenAI(api_key=key)

        self.base_evaluator = FewShotEvaluator(acceptance_json_path)
        self._text_cache: Dict[str, str] = {}

    # ── LLM 纯文本调用 ───────────────────────────────────────────────────────

    def _call_llm_text(self, prompt: str) -> str:
        """调用纯文本 LLM（非 Vision），返回原始响应。"""
        fallback_gemini = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"]
        fallback_openai = ["gpt-4o", "gpt-4o-mini"]

        if self.provider == "gemini":
            models_to_try = [self.model_name] + [m for m in fallback_gemini if m != self.model_name]
            last_err = None
            for m in models_to_try:
                try:
                    instance = self.genai.GenerativeModel(m)
                    resp = instance.generate_content(prompt)
                    return resp.text
                except Exception as e:
                    last_err = e
                    err_str = str(e)
                    if "429" in err_str or "quota" in err_str.lower():
                        time.sleep(2)
                    else:
                        time.sleep(1)
            return '{"admission_decision": "Unknown", "matching_score": 50, "reasoning": "API error"}'

        elif self.provider == "openai":
            models_to_try = [self.model_name] + [m for m in fallback_openai if m != self.model_name]
            last_err = None
            for m in models_to_try:
                try:
                    resp = self.openai_client.chat.completions.create(
                        model=m,
                        messages=[
                            {"role": "system", "content": "You are an expert admission officer. Respond only in valid JSON format."},
                            {"role": "user", "content": prompt},
                        ],
                        response_format={"type": "json_object"},
                    )
                    return resp.choices[0].message.content
                except Exception as e:
                    last_err = e
                    err_str = str(e)
                    if "429" in err_str or "quota" in err_str.lower():
                        time.sleep(2)
                    elif "connection" in err_str.lower() or "timeout" in err_str.lower():
                        time.sleep(5)
                    else:
                        time.sleep(1)
            return '{"admission_decision": "Unknown", "matching_score": 50, "reasoning": "API error"}'

        return '{"admission_decision": "Unknown", "matching_score": 50, "reasoning": "unsupported provider"}'

    # ── Prompt 构建 ──────────────────────────────────────────────────────────

    def _build_zero_shot_prompt(
        self, transcript_text: str, requirements_json: Dict,
        school: str, program: str
    ) -> str:
        """与 Agent 3 提示词哲学完全一致；唯一差异：成绩单为原始文本而非结构化 JSON。"""
        req_str = json.dumps(requirements_json, ensure_ascii=False, indent=2)
        return f"""You are a strict admission officer for German Master's programmes.

You will receive:
- A student's transcript as raw extracted text (from PDF).
- Program admission requirements in JSON format (when available).

IMPORTANT - CRITICAL DISTINCTION:
- The requirement JSON may be incomplete, generic, outdated, or noisy.
- You must treat it only as a hint, NOT as a perfectly reliable source.
- **CRITICAL**: There is a fundamental difference between:

  a) "Cannot fetch/obtain requirements" (信息缺失) - Requirements data is missing or unavailable.
     → Base your decision on transcript quality and typical admission criteria for similar programmes.
     → DO NOT automatically reject due to missing requirements.
     → If the student's academic profile is strong and matches typical expectations, consider "Admit".

  b) "Does not meet requirements" (条件不达标) - Student clearly lacks prerequisites or has insufficient qualifications.
     → In this case, output "Reject" with a clear explanation.

- Your primary signals are:
  1) The student's transcript text (GPA, major, core courses and their relevance),
  2) Your knowledge about typical admission criteria for similar programmes at German universities,
  3) The requirement JSON, but only when it contains concrete and reasonable conditions.

Decision rules:
- If, based on the transcript AND typical expectations, the student clearly lacks core prerequisites or has a clearly too low GPA, output "Reject".
- If the student clearly matches the usual background and level for the programme, output "Admit" even if the requirements JSON is incomplete.
- If you cannot get any useful requirements from the JSON: base your decision on the transcript and typical standards. DO NOT automatically reject.
- If after checking transcript and typical criteria you are still unsure, prefer "Reject".
- matching_score < 50 always corresponds to "Reject".

### NEW APPLICATION TO EVALUATE

Student Transcript (raw text):
{transcript_text}

Program Requirements (JSON):
{req_str}

Respond STRICTLY in the following JSON format:
{{
  "admission_decision": "Admit" or "Reject",
  "matching_score": integer between 0 and 100,
  "reasoning": "detailed reasoning referencing actual GPA, prerequisite courses, language level, and any missing information"
}}"""

    def _build_few_shot_prompt(
        self, transcript_text: str, requirements_json: Dict,
        school: str, program: str,
        few_shot_examples: List[Dict],
    ) -> str:
        """与 Agent 3 few-shot 格式对齐；示例中的成绩单也以原始文本呈现。"""
        req_str = json.dumps(requirements_json, ensure_ascii=False, indent=2)

        few_shot_text = ""
        for i, ex in enumerate(few_shot_examples, 1):
            ex_req = json.dumps(ex.get('requirements_json', {}), ensure_ascii=False, indent=2)
            few_shot_text += f"""
### FEW-SHOT EXAMPLE {i}

Student ID: {ex['student_id']}
School: {ex['school']}
Program: {ex['program']}

Student Transcript (raw text):
{ex.get('transcript_text', '(not available)')}

Program Requirements (JSON):
{ex_req}

Expected output:
{json.dumps({
    "admission_decision": ex['label'],
    "matching_score": ex.get('score', 75 if ex['label'] == 'Admit' else 30),
    "reasoning": ex.get('reasoning', f"Student {'meets' if ex['label'] == 'Admit' else 'does not meet'} programme requirements.")
}, ensure_ascii=False)}
"""

        return f"""You are a strict admission officer for German Master's programmes.

You will receive:
- A student's transcript as raw extracted text (from PDF).
- Program admission requirements in JSON format (when available).

IMPORTANT - CRITICAL DISTINCTION:
- The requirement JSON may be incomplete, generic, outdated, or noisy.
- You must treat it only as a hint, NOT as a perfectly reliable source.
- **CRITICAL**: There is a fundamental difference between:

  a) "Cannot fetch/obtain requirements" (信息缺失) - Requirements data is missing or unavailable.
     → Base your decision on transcript quality and typical admission criteria.
     → DO NOT automatically reject due to missing requirements.
     → If the student's profile is strong and matches typical expectations, consider "Admit".

  b) "Does not meet requirements" (条件不达标) - Student clearly lacks prerequisites.
     → Output "Reject" with a clear explanation.

Decision rules:
- If the student clearly lacks core prerequisites or has a clearly too low GPA, output "Reject".
- If the student clearly matches the usual background for the programme, output "Admit" even if JSON is incomplete.
- If requirements JSON is unavailable: base decision on transcript + typical standards. DO NOT auto-reject.
- If still unsure after checking everything, prefer "Reject".
- matching_score < 50 always corresponds to "Reject".

{few_shot_text}
### NEW APPLICATION TO EVALUATE

Student Transcript (raw text):
{transcript_text}

Program Requirements (JSON):
{req_str}

Respond STRICTLY in the following JSON format:
{{
  "admission_decision": "Admit" or "Reject",
  "matching_score": integer between 0 and 100,
  "reasoning": "detailed reasoning referencing actual GPA, prerequisite courses, language level, and any missing information"
}}"""

    # ── 响应解析 ─────────────────────────────────────────────────────────────

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """从 LLM 响应中提取结构化结果。
        
        与 Agent 3 保持一致：不再基于 matching_score 强制覆盖决策，
        仅对无法解析的输出做保守兜底处理。
        """
        try:
            cleaned = re.sub(r"```(?:json)?\s*", "", response_text)
            cleaned = cleaned.replace("```", "").strip()
            data = json.loads(cleaned)
            decision = data.get("admission_decision", "Unknown")
            if decision not in ("Admit", "Reject"):
                decision = "Reject"
            score = int(data.get("matching_score", 50))
            # 不再强制 "Admit + score<50 → Reject"，保留模型原始决策
            return {
                "admission_decision": decision,
                "matching_score": score,
                "reasoning": data.get("reasoning", ""),
            }
        except Exception:
            pass

        # 最后手段：正则提取（解析失败时保守兜底）
        if re.search(r'"Admit"', response_text):
            return {"admission_decision": "Admit", "matching_score": 70, "reasoning": response_text[:200]}
        return {"admission_decision": "Reject", "matching_score": 30, "reasoning": response_text[:200]}

    # ── 主评估入口 ───────────────────────────────────────────────────────────

    def evaluate(
        self,
        test_set: Optional[List] = None,
        few_shot_examples: Optional[List] = None,
    ) -> Tuple[EvaluationMetrics, EvaluationMetrics]:
        """
        运行 zero-shot 和 few-shot 评估。
        返回 (zero_shot_metrics, few_shot_metrics)
        """
        print(f"\n{'='*80}")
        print(f"Text Baseline 评估（提取纯文本，不使用 Agent 系统）— {self.model_name}")
        print(f"{'='*80}")
        print("⚠ 注意: 使用原始文本输入，不经过 Agent1/Agent2 结构化预处理")

        self.base_evaluator.load_and_convert_data()
        samples = self.base_evaluator.all_samples
        print(f"✓ 成功加载 {len(samples)} 个申请样本")
        admit_count = sum(1 for s in samples if s.label == "Admit")
        reject_count = sum(1 for s in samples if s.label == "Reject")
        print(f"  - Admit: {admit_count}")
        print(f"  - Reject: {reject_count}")

        if few_shot_examples is None or test_set is None:
            few_shot_examples, test_set = self.base_evaluator.select_few_shot_examples(k=5)

        print(f"\n开始 Text Baseline 评估 (测试集大小: {len(test_set)}, 模型: {self.model_name})")

        # ── 预处理 few-shot 示例的文本 ─────────────────────────────────────
        print("\n[准备 Few-shot 示例文本]")
        prepared_few_shot: List[Dict] = []
        for s in few_shot_examples:
            transcript_text = get_transcript_text(
                s.student_id, self.provider, self.model_name,
                self.genai, self.openai_client, self._text_cache
            )
            req_json = load_requirements_as_json(s.school, s.program)
            if not transcript_text:
                print(f"  ⚠ transcript{s.student_id} 文本提取失败，跳过该 few-shot 示例")
                continue
            prepared_few_shot.append({
                "student_id": s.student_id,
                "school": s.school,
                "program": s.program,
                "label": s.label,
                "transcript_text": transcript_text,  # 不截断
                "requirements_json": req_json,
            })
        print(f"  ✓ 成功准备 {len(prepared_few_shot)}/{len(few_shot_examples)} 个 few-shot 示例")

        # ── Zero-shot 评估 ─────────────────────────────────────────────────
        print("\n[Zero-shot 评估]")
        zs_preds: List[Dict] = []
        zs_errors: List[Dict] = []
        for i, sample in enumerate(test_set):
            print(f"  处理样本 {i+1}/{len(test_set)}: {sample.student_id} - {sample.school} - {sample.program}")
            try:
                transcript_text = get_transcript_text(
                    sample.student_id, self.provider, self.model_name,
                    self.genai, self.openai_client, self._text_cache
                )
                req_json = load_requirements_as_json(sample.school, sample.program)

                if not transcript_text:
                    print(f"    ✗ 无法获取成绩单文本，跳过")
                    continue

                prompt = self._build_zero_shot_prompt(
                    transcript_text, req_json, sample.school, sample.program
                )
                raw = self._call_llm_text(prompt)
                result = self._parse_response(raw)
                pred = result["admission_decision"]
                actual = sample.label

                zs_preds.append({"prediction": pred, "actual": actual})
                if pred != actual:
                    zs_errors.append({
                        "student_id": sample.student_id,
                        "school": sample.school,
                        "program": sample.program,
                        "actual": actual,
                        "predicted": pred,
                        "matching_score": result["matching_score"],
                        "reasoning": result.get("reasoning", "")[:200],
                    })
            except Exception as e:
                print(f"    ✗ 错误: {str(e)[:100]}")

        zs_metrics = _compute_metrics(zs_preds, zs_errors)
        print(f"\n[Zero-shot 结果] 准确率: {zs_metrics.accuracy:.2%}  "
              f"TP={zs_metrics.confusion_matrix['TP']} "
              f"FP={zs_metrics.confusion_matrix['FP']} "
              f"TN={zs_metrics.confusion_matrix['TN']} "
              f"FN={zs_metrics.confusion_matrix['FN']}")

        # ── Few-shot 评估 ──────────────────────────────────────────────────
        print("\n[Few-shot 评估]")
        fs_preds: List[Dict] = []
        fs_errors: List[Dict] = []
        for i, sample in enumerate(test_set):
            print(f"  处理样本 {i+1}/{len(test_set)}: {sample.student_id} - {sample.school} - {sample.program}")
            try:
                transcript_text = get_transcript_text(
                    sample.student_id, self.provider, self.model_name,
                    self.genai, self.openai_client, self._text_cache
                )
                req_json = load_requirements_as_json(sample.school, sample.program)

                if not transcript_text:
                    print(f"    ✗ 无法获取成绩单文本，跳过")
                    continue

                prompt = self._build_few_shot_prompt(
                    transcript_text, req_json, sample.school, sample.program,
                    prepared_few_shot
                )
                raw = self._call_llm_text(prompt)
                result = self._parse_response(raw)
                pred = result["admission_decision"]
                actual = sample.label

                fs_preds.append({"prediction": pred, "actual": actual})
                if pred != actual:
                    fs_errors.append({
                        "student_id": sample.student_id,
                        "school": sample.school,
                        "program": sample.program,
                        "actual": actual,
                        "predicted": pred,
                        "matching_score": result["matching_score"],
                        "reasoning": result.get("reasoning", "")[:200],
                    })
            except Exception as e:
                print(f"    ✗ 错误: {str(e)[:100]}")

        fs_metrics = _compute_metrics(fs_preds, fs_errors)
        print(f"\n[Few-shot 结果] 准确率: {fs_metrics.accuracy:.2%}  "
              f"TP={fs_metrics.confusion_matrix['TP']} "
              f"FP={fs_metrics.confusion_matrix['FP']} "
              f"TN={fs_metrics.confusion_matrix['TN']} "
              f"FN={fs_metrics.confusion_matrix['FN']}")

        return zs_metrics, fs_metrics


# ── 辅助函数 ─────────────────────────────────────────────────────────────────

def _compute_metrics(preds: List[Dict], errors: List[Dict]) -> EvaluationMetrics:
    tp = fp = tn = fn = 0
    for p in preds:
        pred, actual = p["prediction"], p["actual"]
        if pred == "Admit" and actual == "Admit":
            tp += 1
        elif pred == "Admit" and actual == "Reject":
            fp += 1
        elif pred == "Reject" and actual == "Reject":
            tn += 1
        elif pred == "Reject" and actual == "Admit":
            fn += 1
    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total > 0 else 0.0
    return EvaluationMetrics(
        accuracy=acc,
        confusion_matrix={"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        errors=errors,
    )
