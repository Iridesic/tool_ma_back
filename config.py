########## 配置文件 ##########

class Config:
    TUSHARE_TOKEN = '025198f4736f5b8f90f99903c08840a57f974daf1b1f78fe135f652e'
    DEBUG = True
    CORS_ORIGINS = ["http://localhost:8080"]
    MODEL_PATH = "D:/self/code/vuecode/tool_ma_front/src/models/weights/best.pt"  # 训练好的YOLO模型路径
    OUTPUT_DIR = "D:/self/code/vuecode/tool_ma_front/src/models/yolo_results"  # 检测结果保存目录
