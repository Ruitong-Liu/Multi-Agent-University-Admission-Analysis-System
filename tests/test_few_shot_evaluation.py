"""
测试Few-shot Learning评估系统的基本功能
"""

from experiments.few_shot_evaluation import FewShotEvaluator, ApplicationSample
import json


def test_data_loading():
    """测试数据加载功能"""
    print("=" * 60)
    print("测试1: 数据加载")
    print("=" * 60)
    
    evaluator = FewShotEvaluator("acceptance.json")
    samples = evaluator.load_and_convert_data()
    
    assert len(samples) > 0, "应该加载到样本"
    assert all(isinstance(s, ApplicationSample) for s in samples), "所有样本应该是ApplicationSample类型"
    
    admit_count = sum(1 for s in samples if s.label == "Admit")
    reject_count = sum(1 for s in samples if s.label == "Reject")
    
    print(f"✓ 成功加载 {len(samples)} 个样本")
    print(f"  - Admit: {admit_count}")
    print(f"  - Reject: {reject_count}")
    print(f"  - 学生数量: {len(set(s.student_id for s in samples))}")
    
    return True


def test_few_shot_selection():
    """测试few-shot示例选择"""
    print("\n" + "=" * 60)
    print("测试2: Few-shot示例选择")
    print("=" * 60)
    
    evaluator = FewShotEvaluator("acceptance.json")
    evaluator.load_and_convert_data()
    
    few_shot_examples, test_set = evaluator.select_few_shot_examples(k=5)
    
    # 验证few-shot示例
    assert len(few_shot_examples) == 5, f"应该有5个few-shot示例，实际有{len(few_shot_examples)}"
    assert any(s.label == "Admit" for s in few_shot_examples), "应该包含至少1个Admit"
    assert any(s.label == "Reject" for s in few_shot_examples), "应该包含至少1个Reject"
    
    # 验证测试集
    assert len(test_set) > 0, "应该有测试集"
    assert len(test_set) == len(evaluator.all_samples) - len(few_shot_examples), "测试集大小应该正确"
    
    # 验证few-shot示例不在测试集中
    few_shot_ids = {(s.student_id, s.school, s.program) for s in few_shot_examples}
    test_set_ids = {(s.student_id, s.school, s.program) for s in test_set}
    assert few_shot_ids.isdisjoint(test_set_ids), "Few-shot示例不应该在测试集中"
    
    print(f"✓ Few-shot示例选择成功")
    print(f"  - Few-shot数量: {len(few_shot_examples)}")
    print(f"  - 测试集大小: {len(test_set)}")
    print(f"  - Few-shot包含Admit: {any(s.label == 'Admit' for s in few_shot_examples)}")
    print(f"  - Few-shot包含Reject: {any(s.label == 'Reject' for s in few_shot_examples)}")
    
    return True


def test_borderline_students():
    """测试边界案例学生识别"""
    print("\n" + "=" * 60)
    print("测试3: 边界案例学生识别")
    print("=" * 60)
    
    evaluator = FewShotEvaluator("acceptance.json")
    evaluator.load_and_convert_data()
    
    borderline_students = evaluator.find_borderline_students(evaluator.all_samples)
    
    print(f"✓ 找到 {len(borderline_students)} 个边界案例学生")
    for student_id in borderline_students:
        student_samples = [s for s in evaluator.all_samples if s.student_id == student_id]
        admit_count = sum(1 for s in student_samples if s.label == "Admit")
        reject_count = sum(1 for s in student_samples if s.label == "Reject")
        print(f"  - Student {student_id}: {admit_count} Admit, {reject_count} Reject")
    
    return True


def test_prompt_generation():
    """测试提示词生成"""
    print("\n" + "=" * 60)
    print("测试4: 提示词生成")
    print("=" * 60)
    
    evaluator = FewShotEvaluator("acceptance.json")
    evaluator.load_and_convert_data()
    
    few_shot_examples, test_set = evaluator.select_few_shot_examples(k=3)
    test_sample = test_set[0] if test_set else evaluator.all_samples[0]
    
    # 测试zero-shot提示词
    zero_shot_prompt = evaluator.build_prompt(test_sample, few_shot_examples=None)
    assert "NEW SAMPLE" in zero_shot_prompt, "应该包含NEW SAMPLE"
    assert "FEW-SHOT EXAMPLES" not in zero_shot_prompt, "Zero-shot不应该包含few-shot示例"
    
    # 测试few-shot提示词
    few_shot_prompt = evaluator.build_prompt(test_sample, few_shot_examples=few_shot_examples)
    assert "FEW-SHOT EXAMPLES" in few_shot_prompt, "应该包含FEW-SHOT EXAMPLES"
    assert test_sample.student_id in few_shot_prompt, "应该包含学生ID"
    assert test_sample.school in few_shot_prompt, "应该包含学校名称"
    assert test_sample.program in few_shot_prompt, "应该包含项目名称"
    
    print("✓ Zero-shot提示词生成成功")
    print(f"  长度: {len(zero_shot_prompt)} 字符")
    print("✓ Few-shot提示词生成成功")
    print(f"  长度: {len(few_shot_prompt)} 字符")
    print(f"  包含 {len(few_shot_examples)} 个few-shot示例")
    
    return True


def test_response_parsing():
    """测试响应解析"""
    print("\n" + "=" * 60)
    print("测试5: 响应解析")
    print("=" * 60)
    
    evaluator = FewShotEvaluator("acceptance.json")
    
    # 测试标准JSON响应
    response1 = '{"admission_decision": "Admit", "matching_score": 85, "reasoning": "Good match"}'
    result1 = evaluator.parse_llm_response(response1)
    assert result1["admission_decision"] == "Admit", "应该正确解析admission_decision"
    assert result1["matching_score"] == 85, "应该正确解析matching_score"
    
    # 测试带额外文本的响应
    response2 = 'Some text before {"admission_decision": "Reject", "matching_score": 45, "reasoning": "Not qualified"} some text after'
    result2 = evaluator.parse_llm_response(response2)
    assert "Reject" in result2.get("admission_decision", ""), "应该从文本中提取JSON"
    
    print("✓ 响应解析测试通过")
    print(f"  测试1: {result1}")
    print(f"  测试2: {result2}")
    
    return True


def test_metrics_calculation():
    """测试指标计算"""
    print("\n" + "=" * 60)
    print("测试6: 指标计算")
    print("=" * 60)
    
    evaluator = FewShotEvaluator("acceptance.json")
    
    # 创建模拟预测结果
    predictions = [
        {"sample": ApplicationSample("01", "School A", "CS", "Admit"), 
         "prediction": "Admit", "actual": "Admit"},
        {"sample": ApplicationSample("01", "School B", "ML", "Admit"), 
         "prediction": "Admit", "actual": "Reject"},
        {"sample": ApplicationSample("02", "School C", "Bio", "Reject"), 
         "prediction": "Reject", "actual": "Reject"},
        {"sample": ApplicationSample("02", "School D", "Bio", "Reject"), 
         "prediction": "Reject", "actual": "Admit"},
    ]
    
    errors = [
        {"student_id": "01", "school": "School B", "program": "ML", 
         "actual": "Admit", "predicted": "Reject"},
        {"student_id": "02", "school": "School D", "program": "Bio", 
         "actual": "Admit", "predicted": "Reject"},
    ]
    
    metrics = evaluator._calculate_metrics(predictions, errors)
    
    assert metrics.accuracy == 0.5, f"准确率应该是0.5，实际是{metrics.accuracy}"
    assert metrics.confusion_matrix["TP"] == 1, "应该有1个TP"
    assert metrics.confusion_matrix["FP"] == 1, "应该有1个FP"
    assert metrics.confusion_matrix["TN"] == 1, "应该有1个TN"
    assert metrics.confusion_matrix["FN"] == 1, "应该有1个FN"
    
    print("✓ 指标计算测试通过")
    print(f"  准确率: {metrics.accuracy:.2%}")
    print(f"  混淆矩阵: {metrics.confusion_matrix}")
    print(f"  错误数量: {len(metrics.errors)}")
    
    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("Few-shot Learning评估系统 - 功能测试")
    print("=" * 60)
    
    tests = [
        test_data_loading,
        test_few_shot_selection,
        test_borderline_students,
        test_prompt_generation,
        test_response_parsing,
        test_metrics_calculation,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
                print(f"✗ {test.__name__} 失败")
        except Exception as e:
            failed += 1
            print(f"✗ {test.__name__} 失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"测试完成: {passed} 通过, {failed} 失败")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)





