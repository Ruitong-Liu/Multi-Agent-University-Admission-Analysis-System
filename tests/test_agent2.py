"""
Agent 2 测试脚本
测试数据: University of Munich - Computer Science
"""

from agent2 import AdmissionRequirementsLoader
import json
import os

def test_university_of_munich():
    """测试 University of Munich - Computer Science"""
    print("=" * 60)
    print("测试: University of Munich - Computer Science")
    print("=" * 60)
    
    # CSV数据库路径
    csv_path = "university_programs.csv"
    
    # 确保CSV文件存在
    if not os.path.exists(csv_path):
        print(f"错误: CSV文件不存在: {csv_path}")
        return
    
    # 初始化加载器
    # 注意: 如果DAAD爬虫遇到问题，可以设置 use_daad=False 来仅使用CSV数据
    loader = AdmissionRequirementsLoader(
        csv_path=csv_path,
        use_daad=True,  # 设置为False可以跳过DAAD爬虫，仅使用CSV数据
        headless=False  # 设置为True可以隐藏浏览器窗口
    )
    
    try:
        # 测试数据
        university = "University of Munich"
        program = "Computer Science"
        
        print(f"\n正在加载入学要求...")
        print(f"大学: {university}")
        print(f"项目: {program}\n")
        
        # 加载入学要求
        result = loader.load_requirements(university, program)
        
        # 打印结果
        print("\n" + "=" * 60)
        print("提取的结果:")
        print("=" * 60)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
        # 保存结果
        output_file = "requirements_University_of_Munich_Computer_Science.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*60}")
        print(f"✓ 结果已保存到: {output_file}")
        print(f"{'='*60}")
        
        # 显示关键信息
        print("\n关键信息摘要:")
        print(f"  大学: {result.get('University', 'N/A')}")
        print(f"  项目: {result.get('Program', 'N/A')}")
        print(f"  学位: {result.get('Degree', 'N/A')}")
        print(f"  教学语言: {result.get('Teaching language', 'N/A')}")
        print(f"  申请截止日期: {result.get('Application deadline', 'N/A')}")
        desc = result.get('requirements', {}).get('Description/content', 'N/A')
        if len(desc) > 100:
            desc = desc[:100] + "..."
        print(f"  要求描述: {desc}")
        
        return result
        
    except Exception as e:
        print(f"\n✗ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        loader.close()


if __name__ == "__main__":
    test_university_of_munich()





