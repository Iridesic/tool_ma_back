import json
import os
import shutil

# JSON文件路径
JSON_FILE_PATH = 'modeListSelf.json'

def init_json_file():
    """初始化JSON文件，如果不存在则创建并写入空列表"""
    if not os.path.exists(JSON_FILE_PATH):
        with open(JSON_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=2)

def read_json_data():
    """读取JSON文件数据"""
    init_json_file()
    with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            # 如果文件损坏，返回空列表
            return []

def write_json_data(data):
    """写入数据到JSON文件"""
    with open(JSON_FILE_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def move_images_to_new_folder(new_folder_name):
    # 配置路径 - 请根据实际需求修改这两个路径
    base_path = r"D:\self\code\tool_ma_back\public"  # 新建文件夹的父目录
    source_path = r"D:\self\code\tool_ma_back\public\savepng"   # 源图片所在的文件夹
    
    # 确保基础路径存在
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        print(f"已创建基础目录: {base_path}")
    
    # 新建文件夹的完整路径
    new_folder_path = os.path.join(base_path, new_folder_name)
    
    # 检查新文件夹是否已存在
    if os.path.exists(new_folder_path):
        print(f"警告: 文件夹 '{new_folder_name}' 已存在")
    else:
        # 创建新文件夹
        os.makedirs(new_folder_path)
        print(f"已创建新文件夹: {new_folder_path}")
    
    # 定义图片文件的扩展名
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
    
    # 统计转移的图片数量
    moved_count = 0
    
    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_path):
        # 检查文件是否为图片
        if filename.lower().endswith(image_extensions):
            source_file = os.path.join(source_path, filename)
            target_file = os.path.join(new_folder_path, filename)
            
            # 检查目标文件是否已存在
            if os.path.exists(target_file):
                print(f"跳过已存在文件: {filename}")
                continue
            
            # 移动文件
            shutil.move(source_file, target_file)
            print(f"已移动: {filename}")
            moved_count += 1
    
    print(f"\n操作完成，共移动了 {moved_count} 张图片")
