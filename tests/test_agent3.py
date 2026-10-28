"""
快速测试 Agent3 功能
"""

from agent3 import MatchingModel, GeminiProvider
import json
import os

def test_agent3():
    """测试 Agent3 匹配模型"""
    print("=" * 60)
    print("Agent3 功能测试")
    print("=" * 60)
    
    # Gemini API 密钥
    gemini_api_key = os.environ.get("GOOGLE_API_KEY", "")
    
    try:
        # 初始化 Gemini 提供者
        print("\n1. 初始化 Gemini 提供者...")
        provider = GeminiProvider(gemini_api_key)
        print("   ✓ Gemini 提供者初始化成功")
        
        # 初始化匹配模型
        print("\n2. 初始化匹配模型...")
        # 尝试多个稳定模型
        models_to_try = [
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ]
        
        matcher = None
        for model_name in models_to_try:
            try:
                print(f"   尝试使用模型: {model_name}...")
                matcher = MatchingModel(provider, model_name=model_name)
                print(f"   ✓ 成功使用模型: {model_name}")
                break
            except Exception as e:
                print(f"   ✗ {model_name} 不可用: {str(e)[:100]}")
                continue
        
        if matcher is None:
            print("   ✗ 所有模型都不可用，请检查 API 密钥和配额")
            return None
        
        # 加载成绩单
        print("\n3. 加载成绩单...")
        transcript_path = "Student_Transcripts/transcript01.json"
        if not os.path.exists(transcript_path):
            print(f"   ✗ 成绩单文件不存在: {transcript_path}")
            return
        
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript_json = json.load(f)
        
        student_name = transcript_json.get('student_information', {}).get('name', 'Unknown')
        student_major = transcript_json.get('student_information', {}).get('major', 'Unknown')
        gpa = transcript_json.get('transcript_totals', {}).get('gpa', 'N/A')
        credits = transcript_json.get('transcript_totals', {}).get('earned_credits', 'N/A')
        
        print(f"   ✓ 成绩单加载成功")
        print(f"     学生: {student_name if student_name else '未填写'}")
        print(f"     专业: {student_major}")
        print(f"     GPA: {gpa}")
        print(f"     学分: {credits} ECTS")
        
        # 创建示例入学要求
        print("\n4. 准备入学要求...")
        requirement_json = {
            "University": "Technical University of Munich",
            "Program": "Computer Science",
            "Degree": "Master of Science",
            "Teaching language": "English",
            "Application deadline": "Winter semester: 31 May; Summer semester: 30 November",
            "requirements": {
                "Description/content": "Bachelor's degree in Computer Science or related field. Strong background in algorithms, data structures, and software engineering. Minimum GPA of 3.5/4.0."
            }
        }
        print(f"   ✓ 入学要求准备完成")
        print(f"     大学: {requirement_json['University']}")
        print(f"     项目: {requirement_json['Program']}")
        
        # 进行评估
        print("\n5. 开始评估匹配度...")
        print("   正在调用 LLM API，请稍候...")
        
        # 如果当前模型失败，尝试其他模型
        result = None
        models_to_try = [matcher.model_name, "gemini-1.5-flash", "gemini-1.5-pro"]
        
        for model_name in models_to_try:
            try:
                if model_name != matcher.model_name:
                    print(f"   尝试使用模型: {model_name}...")
                    matcher.model_name = model_name
                
                result = matcher.evaluate(transcript_json, requirement_json)
                print(f"   ✓ 使用模型 {model_name} 评估成功")
                break
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "quota" in error_msg.lower():
                    print(f"   ✗ {model_name} 配额不足，尝试下一个模型...")
                    continue
                else:
                    print(f"   ✗ {model_name} 评估失败: {error_msg[:100]}")
                    continue
        
        if result is None:
            print("\n✗ 所有模型都不可用或配额不足")
            print("   提示: gemini-3-pro-preview 可能需要付费计划")
            print("   建议: 使用 gemini-1.5-flash 或 gemini-1.5-pro")
            return None
        
        try:
            
            # 打印结果
            print("\n" + "=" * 60)
            print("评估结果:")
            print("=" * 60)
            print(f"录取决定: {result['admission_decision']}")
            print(f"匹配分数: {result['matching_score']}/100")
            print(f"\n评估理由:\n{result['reasoning']}")
            
            # 保存结果
            output_path = "test_matching_result.json"
            matcher.save_result(result, output_path)
            print(f"\n✓ 结果已保存到: {output_path}")
            
            print("\n" + "=" * 60)
            print("测试完成！")
            print("=" * 60)
            
            return result
            
        except Exception as e:
            print(f"\n✗ 处理结果时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
            
    except Exception as e:
        print(f"\n✗ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    test_agent3()

