import json
import os

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
        