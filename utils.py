########## 数据处理工具 ##########
import tushare as ts
import pandas as pd
from datetime import timedelta, datetime
import numpy as np
from config import Config
import os
import json

# 设置Tushare的token
ts.set_token(Config.TUSHARE_TOKEN)
pro = ts.pro_api()

# 获取指定股票在指定时间范围内的行情数据
def get_stock_data(ts_code, start_date, end_date):
    # start_date = datetime.strptime(start_date.split('T')[0], '%Y-%m-%d').strftime('%Y%m%d')
    # end_date = datetime.strptime(end_date.split('T')[0], '%Y-%m-%d').strftime('%Y%m%d')
    try:
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df.empty:
            print(f"未获取到 {ts_code} 在 {start_date} 到 {end_date} 的数据。")
            return pd.DataFrame()
        df = df.sort_values('trade_date')
        return df
    except Exception as e:
        print(f"获取 {ts_code} 数据时出错: {e}")
        return pd.DataFrame()
    
def get_stock_data1(ts_code, start_date, end_date):
    start_date = datetime.strptime(start_date.split('T')[0], '%Y-%m-%d').strftime('%Y%m%d')
    end_date = datetime.strptime(end_date.split('T')[0], '%Y-%m-%d').strftime('%Y%m%d')
    try:
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df.empty:
            print(f"未获取到 {ts_code} 在 {start_date} 到 {end_date} 的数据。")
            return pd.DataFrame()
        df = df.sort_values('trade_date')
        return df
    except Exception as e:
        print(f"获取 {ts_code} 数据时出错: {e}")
        return pd.DataFrame()
    

# 计算相应的MA曲线值
def calculate_ma(data, ma_list):
    for ma in ma_list:
        data[f'MA{ma}'] = data['close'].rolling(window=ma).mean()
    return data.dropna()

# 标准化数据，提高相似度计算准确性
def normalize_data(data):
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return np.zeros_like(data)
    return (data - mean) / std

# 计算相似性
def calculate_similarity(seq1, seq2, weights=None):
    if len(seq1) != len(seq2) or len(seq1) == 0:
        return 0
    
    if weights is None:
        weights = [0.4, 0.3, 0.3]
    
    seq1_norm = normalize_data(seq1)
    seq2_norm = normalize_data(seq2)
    
    trend_sim = calculate_trend_similarity(seq1_norm, seq2_norm)
    dist_sim = calculate_distance_similarity(seq1_norm, seq2_norm)
    shape_sim = calculate_cosine_similarity(seq1_norm, seq2_norm)
    
    return sum(w * s for w, s in zip(weights, [trend_sim, dist_sim, shape_sim]))

def calculate_trend_similarity(seq1, seq2):
    trend1 = seq1[-1] - seq1[0]
    trend2 = seq2[-1] - seq2[0]
    
    direction = 0
    if trend1 * trend2 > 0:
        direction = 1
    elif trend1 * trend2 < 0:
        direction = -1
    
    if trend1 == 0 and trend2 == 0:
        magnitude_sim = 1.0
    else:
        max_trend = max(abs(trend1), abs(trend2))
        min_trend = min(abs(trend1), abs(trend2))
        magnitude_sim = min_trend / max_trend
    
    return (1 + direction * magnitude_sim) / 2

def calculate_distance_similarity(seq1, seq2):
    mse = np.mean((seq1 - seq2) ** 2)
    return np.exp(-mse)

def calculate_cosine_similarity(seq1, seq2):
    dot_product = np.dot(seq1, seq2)
    norm_seq1 = np.linalg.norm(seq1)
    norm_seq2 = np.linalg.norm(seq2)
    
    if norm_seq1 == 0 or norm_seq2 == 0:
        return 0
    
    return dot_product / (norm_seq1 * norm_seq2)

# 判断股票代码所属交易所
def get_exchange_index(stock_code):
    if stock_code.startswith('6') or stock_code.endswith('.SH'):
        return '000001'  # 上证
    else:
        return '399001'  # 深证

# 从JSON文件获取原始指数数据（不进行日期筛选）
def get_raw_index_data_from_json(index_code):
    """
    从JSON文件获取完整的指数原始数据
    参数:
    index_code: 指数代码
    返回:
    DataFrame: 包含完整历史数据的DataFrame
    """
    if index_code == '000001':  # 上证指数
        json_file_path = 'D:/self/code/vuecode/tool_ma_front/public/data/000001.json'
    elif index_code == '399001':  # 深证成指
        json_file_path = 'D:/self/code/vuecode/tool_ma_front/public/data/399001.json'
    else:
        print(f"不支持的指数代码: {index_code}")
        return pd.DataFrame()
    
    if not os.path.exists(json_file_path):
        print(f"指数JSON文件不存在: {json_file_path}")
        return pd.DataFrame()
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data, columns=['trade_date', 'low', 'high', 'close', 'open', 'volume'])
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values('trade_date')
        return df
    
    except Exception as e:
        print(f"读取指数JSON文件时出错: {e}")
        return pd.DataFrame()

# 计算指定时间点的指数MA值
def calculate_index_ma_at_points(index_code, target_dates, ma_list):
    """
    从原始数据计算MA并返回指定时间点的MA值
    参数:
    index_code: 指数代码
    target_dates: 目标日期列表 (YYYY-MM-DD格式)
    ma_list: 需要计算的MA周期列表
    返回:
    list: 每个目标日期对应的MA值列表
    """
    if not target_dates or not ma_list:
        return []
    
    # 获取完整历史数据
    raw_data = get_raw_index_data_from_json(index_code)
    if raw_data.empty:
        print(f"无法获取指数 {index_code} 的原始数据")
        return [[0.0]*len(ma_list) for _ in range(len(target_dates))]  # 返回全零数据
    
    # 计算所有可能的MA值
    calculated_data = calculate_ma(raw_data.copy(), ma_list)
    if calculated_data.empty:
        print(f"计算指数 {index_code} 的MA值失败")
        return [[0.0]*len(ma_list) for _ in range(len(target_dates))]  # 返回全零数据
    
    # 查找目标日期的MA值
    result = []
    for date_str in target_dates:
        date_obj = pd.to_datetime(date_str)
        
        # 查找最接近的日期
        mask = calculated_data['trade_date'] <= date_obj
        if not mask.any():
            # 没有早于或等于目标日期的数据
            result.append([0.0]*len(ma_list))
            continue
        
        closest_row = calculated_data[mask].iloc[-1]
        
        # 提取MA值
        ma_values = [closest_row[f'MA{ma}'] if f'MA{ma}' in closest_row else 0.0 for ma in ma_list]
        result.append(ma_values)
    
    return result

# 获取指数在指定区间的MA值（返回指定的5个时间点）
def get_index_ma_values(index_code, start_date, end_date, ma_list):
    """
    计算并返回指定时间区间内5个均匀分布时间点的指数MA值
    参数:
    index_code: 指数代码
    start_date, end_date: 目标日期范围 (YYYY-MM-DD格式)
    ma_list: 需要计算的MA周期列表
    返回:
    list: 5个时间点的MA值二维列表
    """
    # 生成5个均匀分布的日期
    start_obj = pd.to_datetime(start_date)
    end_obj = pd.to_datetime(end_date)
    
    # 确保时间区间足够
    if (end_obj - start_obj).days < 4:
        # 时间区间太短，直接返回开始和结束日期
        target_dates = [
            start_date,
            (start_obj + timedelta(days=1)).strftime('%Y-%m-%d'),
            (start_obj + timedelta(days=2)).strftime('%Y-%m-%d'),
            (start_obj + timedelta(days=3)).strftime('%Y-%m-%d'),
            end_date
        ]
    else:
        # 生成5个均匀分布的日期
        days_between = (end_obj - start_obj).days
        target_dates = [
            start_date,
            (start_obj + timedelta(days=int(days_between * 0.25))).strftime('%Y-%m-%d'),
            (start_obj + timedelta(days=int(days_between * 0.5))).strftime('%Y-%m-%d'),
            (start_obj + timedelta(days=int(days_between * 0.75))).strftime('%Y-%m-%d'),
            end_date
        ]
    
    # 计算这些日期的MA值
    return calculate_index_ma_at_points(index_code, target_dates, ma_list)