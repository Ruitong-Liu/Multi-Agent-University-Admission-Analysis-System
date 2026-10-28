"""
测试CSV保存功能
"""

from agent2 import CSVLoader
import os

def test_csv_save():
    """测试CSV保存功能"""
    print("=" * 60)
    print("测试CSV保存功能")
    print("=" * 60)
    
    csv_path = "university_programs.csv"
    
    # 确保CSV文件存在
    if not os.path.exists(csv_path):
        print(f"错误: CSV文件不存在: {csv_path}")
        return
    
    # 创建CSV加载器
    loader = CSVLoader(csv_path)
    loader.load()
    
    # 测试数据
    test_data = {
        "University": "Test University",
        "Program": "Test Program",
        "Degree": "Master of Science",
        "Teaching language": "English",
        "Application deadline": "Winter semester: 31 May",
        "requirements": "Bachelor's degree in related field"
    }
    
    print(f"\n测试数据:")
    print(f"  大学: {test_data['University']}")
    print(f"  项目: {test_data['Program']}")
    print(f"  学位: {test_data['Degree']}")
    print(f"  教学语言: {test_data['Teaching language']}")
    print(f"  申请截止日期: {test_data['Application deadline']}")
    print(f"  入学要求: {test_data['requirements']}")
    
    # 测试添加新记录
    print(f"\n正在添加新记录到CSV...")
    success = loader.update_or_add_program(test_data)
    
    if success:
        print("✓ 成功添加/更新记录到CSV")
        
        # 验证数据是否已保存
        loader.load()  # 重新加载
        found = loader.find_program("Test University", "Test Program")
        if found:
            print("✓ 验证成功: 数据已保存到CSV")
            print(f"\n保存的数据:")
            for key, value in found.items():
                print(f"  {key}: {value}")
        else:
            print("✗ 验证失败: 数据未找到")
    else:
        print("✗ 保存失败")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_csv_save()





