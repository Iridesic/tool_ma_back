import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
from ultralytics import YOLO
import os
from pathlib import Path
import numpy as np
from config import Config

plt.switch_backend('Agg')  # 添加此行，使用非GUI后端
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False

# 初始化YOLO模型（全局加载一次）
model = None
if os.path.exists(Config.MODEL_PATH):
    model = YOLO(Config.MODEL_PATH)
    if torch.cuda.is_available():
        model.to("cuda")
        print("YOLO模型已加载到GPU")
    else:
        print("YOLO模型将在CPU上运行")
else:
    print(f"错误：未找到模型文件 {Config.MODEL_PATH}")

def generate_ma_chart(stock_code, ma_data, save_path):
    """生成均线折线图"""
    plt.figure(figsize=(12, 8))  # 增大图表尺寸，便于识别
    for ma in ma_data.columns[1:]:  # 跳过日期列
        plt.plot(ma_data['trade_date'], ma_data[ma], label=f'{ma}日均线')
    
    plt.title(f'{stock_code} 均线走势', fontsize=16)
    plt.xlabel('日期', fontsize=14)
    plt.ylabel('价格', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path

def detect_with_yolo(image_path, nms_threshold=0.4):
    """使用YOLO模型检测图像"""
    if not model:
        print("错误：未加载YOLO模型")
        return None
    
    results = model(image_path)
    results = apply_nms(results, nms_threshold)  # 应用非极大值抑制
    
    # 保存检测结果图像
    save_dir = "D:/self/code/vuecode/tool_ma_front/public/detect_result"
    os.makedirs(save_dir, exist_ok=True)
    img_name = os.path.basename(image_path)
    save_path = os.path.join(save_dir, img_name)
    
    # 仅在检测到目标时保存结果图像
    has_detections = len(results[0].boxes) > 0 if hasattr(results[0], 'boxes') else False
    
    if has_detections:
        results[0].save(save_path)
    
    return has_detections, save_path, results

def apply_nms(results, iou_threshold=0.4):
    """非极大值抑制 - 提高阈值以保留更多检测框"""
    for result in results:
        if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            keep_indices = non_max_suppression(boxes, scores, iou_threshold)
            result.boxes = result.boxes[keep_indices]
    return results

def non_max_suppression(boxes, scores, iou_threshold=0.4):
    """NMS算法实现 - 提高阈值以保留更多检测框"""
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes)
    scores = np.array(scores)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

def check_ma_order(ma_data, ma_list):
    """检查均线是否满足多头排列条件"""
    ma_columns = [f'MA{ma}' for ma in ma_list]
    # 检查均线是否按从短期到长期的顺序排列
    ordered = True
    for i in range(len(ma_columns)-1):
        if not (ma_data[ma_columns[i]] > ma_data[ma_columns[i+1]]).all():
            ordered = False
            break
    
    # 检查均线是否呈上升趋势
    rising = True
    for ma in ma_columns:
        if not (ma_data[ma].iloc[-1] > ma_data[ma].iloc[0]):
            rising = False
            break
    
    return ordered and rising

def is_bullish_arrangement(data_segment, ma_list, extend_days=3):
    """结合YOLO检测和均线数据判断多头排列"""
    # 确保数据段足够长以形成明显的多头排列
    if len(data_segment) < 15:
        return False, None
    
    # 生成临时均线图
    chart_dir = os.path.join(Config.OUTPUT_DIR, "charts")
    os.makedirs(chart_dir, exist_ok=True)
    # stock_code = data_segment['code'] ###################################
    stock_code = str(data_segment['code'].iloc[0])
    print("code--------", stock_code)

    # 修复日期格式：移除时间中的冒号
    start_date = str(data_segment['trade_date'].iloc[0]).replace(':', '-')
    end_date = str(data_segment['trade_date'].iloc[-1]).replace(':', '-')
    date_range = f"{start_date}_to_{end_date}"
    # date_range = f"{data_segment['trade_date'].iloc[0]}_to_{data_segment['trade_date'].iloc[-1]}"
    chart_path = os.path.join(chart_dir, f"{stock_code}_{date_range}_ma.png")
    print("1111")
    # 提取均线数据并生成图表
    ma_data = data_segment[['trade_date'] + [f'MA{ma}' for ma in ma_list]].copy()  # 添加.copy()
    ma_data['trade_date'] = pd.to_datetime(ma_data['trade_date'])
    generate_ma_chart(stock_code, ma_data, chart_path)
    
    print("2222")
    # 执行YOLO检测
    has_detections, detection_path, yolo_results = detect_with_yolo(chart_path)
    print("3333")
    # 如果没有检测到目标，进一步检查均线数据
    if not has_detections:
        # 检查均线排列是否满足多头条件
        is_bullish_by_data = check_ma_order(data_segment, ma_list)
        
        if is_bullish_by_data:
            # 虽然YOLO未检测到，但均线数据满足条件，保存图像并返回
            return True, chart_path
        
        # 既没有检测到也不满足条件，删除原图
        if os.path.exists(chart_path):
            os.remove(chart_path)
        return False, None
    
    # YOLO检测到目标，检查检测置信度
    if hasattr(yolo_results[0], 'boxes') and yolo_results[0].boxes is not None:
        confidences = yolo_results[0].boxes.conf.cpu().numpy()
        if len(confidences) > 0 and np.max(confidences) > 0.2:  # 设置最小置信度阈值
            return True, detection_path
    
    # 检测结果不可靠，删除原图
    if os.path.exists(chart_path):
        os.remove(chart_path)
    return False, None