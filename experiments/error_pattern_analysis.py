"""
错误模式深度分析脚本
分析评估结果中的错误模式，找出共同特征和误判原因
"""

import json
from collections import defaultdict
from typing import Dict, List, Any

def load_comparison_results(filepath: str = "comparison_system_vs_baseline_33samples.json"):
    """加载对比结果"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_error_patterns(results: Dict):
    """分析错误模式"""
    
    analysis = {
        "system_zero_shot": analyze_errors(results["system_evaluation"]["zero_shot"]["errors"], "系统Zero-shot"),
        "system_few_shot": analyze_errors(results["system_evaluation"]["few_shot"]["errors"], "系统Few-shot"),
        "baseline_zero_shot": analyze_errors(results["baseline_evaluation"]["zero_shot"]["errors"], "Baseline Zero-shot"),
        "baseline_few_shot": analyze_errors(results["baseline_evaluation"]["few_shot"]["errors"], "Baseline Few-shot"),
    }
    
    return analysis

def analyze_errors(errors: List[Dict], method_name: str) -> Dict:
    """分析单个方法的错误"""
    
    # 按错误类型分类
    fp_errors = [e for e in errors if e["actual"] == "Reject" and e["predicted"] == "Admit"]
    fn_errors = [e for e in errors if e["actual"] == "Admit" and e["predicted"] == "Reject"]
    
    # 按学生ID分类
    by_student = defaultdict(list)
    for error in errors:
        by_student[error["student_id"]].append(error)
    
    # 按专业领域分类
    by_program_type = defaultdict(list)
    for error in errors:
        program = error["program"].lower()
        if "psychology" in program or "psychologie" in program:
            by_program_type["心理学相关"].append(error)
        elif "computer" in program or "machine learning" in program or "data science" in program or "computing" in program:
            by_program_type["计算机科学相关"].append(error)
        elif "embedded" in program or "systems" in program:
            by_program_type["嵌入式/系统相关"].append(error)
        elif "neuroscience" in program or "neuro" in program or "brain" in program or "cognitive" in program:
            by_program_type["神经科学/认知科学相关"].append(error)
        else:
            by_program_type["其他"].append(error)
    
    # 按学校分类
    by_school = defaultdict(list)
    for error in errors:
        by_school[error["school"]].append(error)
    
    # 分析matching_score分布
    scores = [e.get("matching_score", 0) for e in errors]
    avg_score = sum(scores) / len(scores) if scores else 0
    
    # 找出共同误判的样本
    error_samples = {(e["student_id"], e["school"], e["program"]) for e in errors}
    
    return {
        "method": method_name,
        "total_errors": len(errors),
        "fp_count": len(fp_errors),
        "fn_count": len(fn_errors),
        "fp_errors": fp_errors,
        "fn_errors": fn_errors,
        "by_student": dict(by_student),
        "by_program_type": dict(by_program_type),
        "by_school": dict(by_school),
        "avg_matching_score": avg_score,
        "error_samples": error_samples
    }

def find_common_errors(analysis: Dict) -> Dict:
    """找出所有方法共同误判的样本"""
    
    all_error_samples = []
    for method_analysis in analysis.values():
        all_error_samples.append(method_analysis["error_samples"])
    
    # 找出在所有方法中都出现的错误
    common_errors = set.intersection(*all_error_samples) if all_error_samples else set()
    
    # 找出在至少2个方法中出现的错误
    from collections import Counter
    all_samples_flat = []
    for samples in all_error_samples:
        all_samples_flat.extend(samples)
    
    sample_counts = Counter(all_samples_flat)
    frequent_errors = {sample for sample, count in sample_counts.items() if count >= 2}
    
    return {
        "common_errors": common_errors,
        "frequent_errors": frequent_errors,
        "error_frequency": dict(sample_counts)
    }

def analyze_student_patterns(analysis: Dict) -> Dict:
    """分析特定学生的错误模式"""
    
    student_patterns = defaultdict(lambda: {
        "total_errors": 0,
        "fp_count": 0,
        "fn_count": 0,
        "errors_by_method": defaultdict(list),
        "programs": set()
    })
    
    for method_name, method_analysis in analysis.items():
        for student_id, errors in method_analysis["by_student"].items():
            student_patterns[student_id]["total_errors"] += len(errors)
            student_patterns[student_id]["errors_by_method"][method_name] = errors
            for error in errors:
                student_patterns[student_id]["programs"].add((error["school"], error["program"]))
                if error["actual"] == "Reject" and error["predicted"] == "Admit":
                    student_patterns[student_id]["fp_count"] += 1
                elif error["actual"] == "Admit" and error["predicted"] == "Reject":
                    student_patterns[student_id]["fn_count"] += 1
    
    return dict(student_patterns)

def generate_report(analysis: Dict, common_errors: Dict, student_patterns: Dict):
    """生成分析报告"""
    
    report = []
    report.append("=" * 80)
    report.append("错误模式深度分析报告")
    report.append("=" * 80)
    report.append("")
    
    # 1. 总体错误统计
    report.append("## 1. 总体错误统计")
    report.append("")
    for method_name, method_analysis in analysis.items():
        report.append(f"### {method_analysis['method']}")
        report.append(f"- 总错误数: {method_analysis['total_errors']}")
        report.append(f"- FP (过度预测Admit): {method_analysis['fp_count']}")
        report.append(f"- FN (漏掉应该Admit): {method_analysis['fn_count']}")
        report.append(f"- 平均matching_score: {method_analysis['avg_matching_score']:.1f}")
        report.append("")
    
    # 2. 共同误判样本
    report.append("## 2. 共同误判样本分析")
    report.append("")
    report.append(f"### 在所有方法中都误判的样本: {len(common_errors['common_errors'])}")
    for sample in sorted(common_errors['common_errors']):
        report.append(f"- 学生{sample[0]}: {sample[1]} - {sample[2]}")
    report.append("")
    
    report.append(f"### 在至少2个方法中误判的样本: {len(common_errors['frequent_errors'])}")
    for sample in sorted(common_errors['frequent_errors']):
        count = common_errors['error_frequency'].get(sample, 0)
        report.append(f"- 学生{sample[0]}: {sample[1]} - {sample[2]} (在{count}个方法中误判)")
    report.append("")
    
    # 3. 按学生分析
    report.append("## 3. 按学生分析错误模式")
    report.append("")
    for student_id, pattern in sorted(student_patterns.items()):
        report.append(f"### 学生{student_id}")
        report.append(f"- 总错误数: {pattern['total_errors']}")
        report.append(f"- FP: {pattern['fp_count']}, FN: {pattern['fn_count']}")
        report.append(f"- 涉及项目数: {len(pattern['programs'])}")
        report.append("")
        for method, errors in pattern['errors_by_method'].items():
            if errors:
                report.append(f"  **{method}**: {len(errors)}个错误")
                for error in errors[:3]:  # 只显示前3个
                    report.append(f"    - {error['school']} - {error['program']}: "
                                f"实际{error['actual']}, 预测{error['predicted']}, "
                                f"score={error.get('matching_score', 'N/A')}")
        report.append("")
    
    # 4. 按专业领域分析
    report.append("## 4. 按专业领域分析")
    report.append("")
    for method_name, method_analysis in analysis.items():
        report.append(f"### {method_analysis['method']}")
        for program_type, errors in method_analysis['by_program_type'].items():
            if errors:
                report.append(f"- **{program_type}**: {len(errors)}个错误")
                # 显示前2个示例
                for error in errors[:2]:
                    report.append(f"  - 学生{error['student_id']}: {error['school']} - {error['program']}")
        report.append("")
    
    # 5. FP错误详细分析
    report.append("## 5. FP错误详细分析（过度预测Admit）")
    report.append("")
    for method_name, method_analysis in analysis.items():
        if method_analysis['fp_errors']:
            report.append(f"### {method_analysis['method']} - {len(method_analysis['fp_errors'])}个FP错误")
            # 按学生分组
            fp_by_student = defaultdict(list)
            for error in method_analysis['fp_errors']:
                fp_by_student[error['student_id']].append(error)
            
            for student_id, errors in fp_by_student.items():
                report.append(f"  **学生{student_id}** ({len(errors)}个FP):")
                for error in errors:
                    report.append(f"    - {error['school']} - {error['program']}")
                    report.append(f"      Score: {error.get('matching_score', 'N/A')}")
                    # 提取reasoning的关键部分
                    reasoning = error.get('reasoning', '')[:200]
                    report.append(f"      理由: {reasoning}...")
            report.append("")
    
    # 6. FN错误详细分析
    report.append("## 6. FN错误详细分析（漏掉应该Admit）")
    report.append("")
    for method_name, method_analysis in analysis.items():
        if method_analysis['fn_errors']:
            report.append(f"### {method_analysis['method']} - {len(method_analysis['fn_errors'])}个FN错误")
            for error in method_analysis['fn_errors']:
                report.append(f"  - 学生{error['student_id']}: {error['school']} - {error['program']}")
                report.append(f"    Score: {error.get('matching_score', 'N/A')}")
                reasoning = error.get('reasoning', '')[:200]
                report.append(f"    理由: {reasoning}...")
            report.append("")
    
    # 7. 关键发现和建议
    report.append("## 7. 关键发现和建议")
    report.append("")
    
    # 找出最常见的误判模式
    all_fp_programs = []
    for method_analysis in analysis.values():
        all_fp_programs.extend([e['program'] for e in method_analysis['fp_errors']])
    
    from collections import Counter
    program_counts = Counter(all_fp_programs)
    top_fp_programs = program_counts.most_common(5)
    
    report.append("### 最常见的FP错误（过度预测Admit）")
    for program, count in top_fp_programs:
        report.append(f"- {program}: {count}次误判")
    report.append("")
    
    report.append("### 改进建议")
    report.append("1. **针对特定专业加强检查**: 对于Business Psychology、Work and Organizational Psychology等专业，需要更严格的先修课程检查")
    report.append("2. **提高FN识别能力**: 对于Cognitive-Affective Neuroscience等专业，需要更细致地评估相关背景的适配性")
    report.append("3. **优化matching_score阈值**: 当前score < 50的规则可能不够严格，建议针对不同专业设置不同的阈值")
    report.append("4. **改进requirements数据**: 部分项目的requirements数据不完整，导致模型依赖通用知识而非具体要求")
    report.append("")
    
    return "\n".join(report)

def main():
    """主函数"""
    print("加载对比结果...")
    results = load_comparison_results()
    
    print("分析错误模式...")
    analysis = analyze_error_patterns(results)
    
    print("找出共同错误...")
    common_errors = find_common_errors(analysis)
    
    print("分析学生模式...")
    student_patterns = analyze_student_patterns(analysis)
    
    print("生成报告...")
    report = generate_report(analysis, common_errors, student_patterns)
    
    # 保存报告
    output_file = "ERROR_PATTERN_ANALYSIS.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n✓ 错误模式分析报告已保存到: {output_file}")
    
    # 同时保存JSON格式的详细数据
    json_output = {
        "analysis": {k: {
            "method": v["method"],
            "total_errors": v["total_errors"],
            "fp_count": v["fp_count"],
            "fn_count": v["fn_count"],
            "avg_matching_score": v["avg_matching_score"],
            "by_student": {sid: len(errors) for sid, errors in v["by_student"].items()},
            "by_program_type": {pt: len(errors) for pt, errors in v["by_program_type"].items()}
        } for k, v in analysis.items()},
        "common_errors": {
            "common_count": len(common_errors["common_errors"]),
            "frequent_count": len(common_errors["frequent_errors"]),
            "common_samples": list(common_errors["common_errors"]),
            "frequent_samples": list(common_errors["frequent_errors"])
        },
        "student_patterns": {sid: {
            "total_errors": p["total_errors"],
            "fp_count": p["fp_count"],
            "fn_count": p["fn_count"],
            "program_count": len(p["programs"])
        } for sid, p in student_patterns.items()}
    }
    
    json_file = "error_pattern_analysis_data.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 详细数据已保存到: {json_file}")
    
    # 打印摘要
    print("\n" + "=" * 80)
    print("错误模式分析摘要")
    print("=" * 80)
    print(f"共同误判样本数: {len(common_errors['common_errors'])}")
    print(f"频繁误判样本数: {len(common_errors['frequent_errors'])}")
    print("\n各方法错误统计:")
    for method_name, method_analysis in analysis.items():
        print(f"  {method_analysis['method']}: {method_analysis['total_errors']}个错误 "
              f"(FP: {method_analysis['fp_count']}, FN: {method_analysis['fn_count']})")

if __name__ == "__main__":
    main()

