"""
测试三个Agent的整合
快速验证整合是否正常工作
"""

import os
import json


def test_agent1_output():
    """测试Agent1的输出文件是否存在"""
    print("=" * 60)
    print("测试 Agent1 输出")
    print("=" * 60)
    
    transcript_files = [
        "Student_Transcripts/transcript01.json",
        "Student_Transcripts/transcript02.json",
        "Student_Transcripts/transcript03.json"
    ]
    
    for transcript_file in transcript_files:
        if os.path.exists(transcript_file):
            try:
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                student_name = data.get('student_information', {}).get('name', 'Unknown')
                print(f"✓ {transcript_file}: {student_name}")
            except Exception as e:
                print(f"✗ {transcript_file}: 解析失败 - {str(e)}")
        else:
            print(f"✗ {transcript_file}: 文件不存在")


def test_agent2_output():
    """测试Agent2的CSV文件是否存在"""
    print("\n" + "=" * 60)
    print("测试 Agent2 输入/输出")
    print("=" * 60)
    
    csv_file = "university_programs.csv"
    if os.path.exists(csv_file):
        print(f"✓ CSV文件存在: {csv_file}")
        try:
            import csv
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                print(f"  CSV包含 {len(rows)} 条记录")
                if rows:
                    print(f"  示例: {rows[0].get('University', 'N/A')} - {rows[0].get('Program', 'N/A')}")
        except Exception as e:
            print(f"✗ CSV文件读取失败: {str(e)}")
    else:
        print(f"✗ CSV文件不存在: {csv_file}")
        print("  提示: Agent2需要CSV文件来加载入学要求")


def test_agent3_integration():
    """测试Agent3是否能正确导入和使用"""
    print("\n" + "=" * 60)
    print("测试 Agent3 整合")
    print("=" * 60)
    
    # 测试导入
    try:
        from agent3 import MatchingModel, GeminiProvider, OpenAIProvider
        print("✓ Agent3模块导入成功")
    except ImportError as e:
        print(f"✗ Agent3模块导入失败: {str(e)}")
        return False
    
    # 测试Agent2导入（用于整合）
    try:
        from agent2 import AdmissionRequirementsLoader
        print("✓ Agent2模块导入成功（用于整合）")
    except ImportError as e:
        print(f"⚠ Agent2模块导入失败: {str(e)}")
        print("  提示: Agent3仍可使用，但无法自动加载Agent2的数据")
    
    # 测试Agent1导入（用于整合）
    try:
        from agent1 import TranscriptExtractor
        print("✓ Agent1模块导入成功（用于整合）")
    except ImportError as e:
        print(f"⚠ Agent1模块导入失败: {str(e)}")
        print("  提示: Agent3仍可使用，但需要手动提供成绩单JSON")
    
    return True


def test_data_flow():
    """测试数据流是否完整"""
    print("\n" + "=" * 60)
    print("测试数据流完整性")
    print("=" * 60)
    
    # 检查Agent1输出
    transcript_file = "Student_Transcripts/transcript01.json"
    if not os.path.exists(transcript_file):
        print(f"✗ Agent1输出不存在: {transcript_file}")
        print("  建议: 运行 python agent1.py 生成成绩单JSON")
        return False
    
    # 检查Agent2输入
    csv_file = "university_programs.csv"
    if not os.path.exists(csv_file):
        print(f"⚠ Agent2输入不存在: {csv_file}")
        print("  建议: 创建CSV文件或使用 --fallback 选项")
    
    # 检查数据格式
    try:
        with open(transcript_file, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
        
        required_fields = ['student_information', 'transcript_totals', 'courses']
        missing_fields = [field for field in required_fields if field not in transcript_data]
        
        if missing_fields:
            print(f"✗ Agent1输出格式不完整，缺少字段: {missing_fields}")
            return False
        else:
            print("✓ Agent1输出格式正确")
            student_name = transcript_data.get('student_information', {}).get('name', 'Unknown')
            print(f"  学生: {student_name}")
    
    except Exception as e:
        print(f"✗ 数据格式检查失败: {str(e)}")
        return False
    
    print("\n✓ 数据流检查通过！可以运行整合工作流")
    return True


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("三个Agent整合测试")
    print("=" * 60)
    
    test_agent1_output()
    test_agent2_output()
    test_agent3_integration()
    test_data_flow()
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
    print("\n下一步:")
    print("1. 如果所有测试通过，运行: python example_integrated_agents.py")
    print("2. 或使用命令行: python agent3.py --transcript Student_Transcripts/transcript01.json")
    print("3. 查看完整文档: README.md")


if __name__ == "__main__":
    main()





