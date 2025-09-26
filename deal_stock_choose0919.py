import tempfile
import pandas as pd
import os
import chardet
import numpy as np
from datetime import datetime

from aaa_classify_code.get_stage1_class_stage2copy import predict_from_csv as predict_stage2
from aaa_classify_code.get_stage1_class_stage3copy import predict_from_csv as predict_stage3
from aaa_classify_code.get_stage1_class import predict_from_csv as predict_stage1

# 模型路径设置（区分长短周期）
MODEL_PATHS = {
    'stage1': {
        'long': {
            'model_path': r"D:\self\code\tool_ma_back\models\stage1_long_model.pkl",
            'training_feature_path': r"D:\self\code\tool_ma_back\models\stage1_long_features.csv"
        },
        'short': {
            'model_path': r"D:\self\code\tool_ma_back\models\stage1_short_model.pkl",
            'training_feature_path': r"D:\self\code\tool_ma_back\models\stage1_short_features.csv"
        }
    },
    'stage2': {
        'long': {
            'model_path': r"D:\self\code\tool_ma_back\models\stage2_long_model.pkl",
            'training_feature_path': r"D:\self\code\tool_ma_back\models\stage2_long_features.csv"
        },
        'short': {
            'model_path': r"D:\self\code\tool_ma_back\models\stage2_short_model.pkl",
            'training_feature_path': r"D:\self\code\tool_ma_back\models\stage2_short_features.csv"
        }
    },
    'stage3': {
        'long': {
            'model_path': r"D:\self\code\tool_ma_back\models\stage3_long_model.pkl",
            'training_feature_path': r"D:\self\code\tool_ma_back\models\stage3_long_features.csv"
        },
        'short': {
            'model_path': r"D:\self\code\tool_ma_back\models\stage3_short_model.pkl",
            'training_feature_path': r"D:\self\code\tool_ma_back\models\stage3_short_features.csv"
        }
    }
}

# 长短周期判断阈值
DURATION_THRESHOLDS = {
    'stage1': 10,  # 超过10天为长周期，否则为短周期
    'stage2': 4,   # 超过4天为长周期，否则为短周期
    'stage3': 10   # 超过10天为长周期，否则为短周期
}

def get_stock_data(stock_code, start_date, end_date, data_folder):
    """获取股票在指定时间区间的数据"""
    try:
        file_path = os.path.join(data_folder, f"{stock_code}.csv")
        if not os.path.exists(file_path):
            return pd.DataFrame()
            
        # 检测文件编码
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
        encoding = result['encoding'] or 'utf-8'
        
        df = pd.read_csv(file_path, encoding=encoding)
        df['timestamps'] = pd.to_datetime(df['timestamps'])
        mask = (df['timestamps'] >= start_date) & (df['timestamps'] <= end_date)
        return df[mask]
    except Exception as e:
        print(f"获取股票数据出错: {str(e)}")
        return pd.DataFrame()

def get_full_stock_data(stock_code, data_folder):
    """获取股票的完整数据"""
    try:
        file_path = os.path.join(data_folder, f"{stock_code}.csv")
        if not os.path.exists(file_path):
            return pd.DataFrame()
            
        # 检测文件编码
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
        encoding = result['encoding'] or 'utf-8'
        
        df = pd.read_csv(file_path, encoding=encoding)
        df['timestamps'] = pd.to_datetime(df['timestamps'])
        return df.sort_values('timestamps').reset_index(drop=True)
    except Exception as e:
        print(f"获取完整股票数据出错: {str(e)}")
        return pd.DataFrame()

def calculate_ma(df, ma_periods=[4, 8, 12, 16, 20, 47]):
    """计算指定周期的均线"""
    df_copy = df.copy()
    for period in ma_periods:
        df_copy[f'MA{period}'] = df_copy['close'].rolling(window=period).mean()
    return df_copy

def extract_stage_features(stage_data, ma_periods=[4, 8, 12, 16, 20, 47]):
    """提取阶段特征用于分类计算"""
    features = []
    # 计算各均线的平均斜率
    for period in ma_periods:
        ma_col = f'MA{period}'
        ma_values = stage_data[ma_col].values
        if len(ma_values) > 1:
            slope = (ma_values[-1] - ma_values[0]) / len(ma_values)
            features.append(slope)
        else:
            features.append(0)
    
    # 计算均线之间的平均距离
    for i in range(len(ma_periods) - 1):
        current_ma = f'MA{ma_periods[i]}'
        next_ma = f'MA{ma_periods[i+1]}'
        diff = (stage_data[current_ma] - stage_data[next_ma]).mean()
        features.append(diff)
    
    # 将NumPy数组转换为Python列表
    return features  # 原代码是 return np.array(features)


def classify_stages_from_data(stock_code, full_data, ma_periods=[4, 8, 12, 16, 20, 47],
                             duration_thresholds={'stage1': 10, 'stage2': 4, 'stage3': 10}):
    """对已有数据进行阶段分类（仅分类，不进行阶段判断）"""
    classified_stages = {
        'stage1': [],
        'stage2': [],
        'stage3': []
    }
    
    # 确保数据按时间排序
    full_data = full_data.sort_values('timestamps').reset_index(drop=True)
    full_data['timestamps'] = pd.to_datetime(full_data['timestamps'])
    
    # 计算均线（如果尚未计算）
    if 'MA4' not in full_data.columns:
        full_data = calculate_ma(full_data, ma_periods)
        if full_data.empty:
            print(f"股票 {stock_code} 均线计算失败，无法分类")
            return classified_stages
    
    # 提取所有可能的阶段区间并进行分类
    # 1. 处理stage1分类
    stage1_intervals = extract_stage_intervals(full_data, ma_periods, 'stage1')
    for interval in stage1_intervals:
        stage_data = full_data.loc[interval['start_idx']:interval['end_idx']]
        feature_vector = extract_stage_features(stage_data, ma_periods)
        
        # 使用分类器获取分类结果
        probabilities = predict_stage1(stage_data)
        
        classified_stages['stage1'].append({
            'stock_code': stock_code,
            'start': interval['start_date'].strftime('%Y-%m-%d'),
            'end': interval['end_date'].strftime('%Y-%m-%d'),
            'duration_days': interval['duration'],
            'duration_type': "long" if interval['duration'] > duration_thresholds['stage1'] else "short",
            # 转换为列表
            'feature_vector': feature_vector.tolist() if hasattr(feature_vector, 'tolist') else feature_vector,
            'probabilities': probabilities
        })
    
    # 2. 处理stage2分类
    stage2_intervals = extract_stage_intervals(full_data, ma_periods, 'stage2')
    for interval in stage2_intervals:
        stage_data = full_data.loc[interval['start_idx']:interval['end_idx']]
        feature_vector = extract_stage_features(stage_data, ma_periods)
        
        # 使用分类器获取分类结果
        probabilities = predict_stage2(stage_data)
        
        classified_stages['stage2'].append({
            'stock_code': stock_code,
            'start': interval['start_date'].strftime('%Y-%m-%d'),
            'end': interval['end_date'].strftime('%Y-%m-%d'),
            'duration_days': interval['duration'],
            'duration_type': "long" if interval['duration'] > duration_thresholds['stage2'] else "short",
            'feature_vector': feature_vector,
            'probabilities': probabilities
        })
    
    # 3. 处理stage3分类
    stage3_intervals = extract_stage_intervals(full_data, ma_periods, 'stage3')
    for interval in stage3_intervals:
        stage_data = full_data.loc[interval['start_idx']:interval['end_idx']]
        feature_vector = extract_stage_features(stage_data, ma_periods)
        
        # 使用分类器获取分类结果
        probabilities = predict_stage3(stage_data)
        
        classified_stages['stage3'].append({
            'stock_code': stock_code,
            'start': interval['start_date'].strftime('%Y-%m-%d'),
            'end': interval['end_date'].strftime('%Y-%m-%d'),
            'duration_days': interval['duration'],
            'duration_type': "long" if interval['duration'] > duration_thresholds['stage3'] else "short",
            'feature_vector': feature_vector,
            'probabilities': probabilities
        })
    
    return classified_stages

def extract_stage_intervals(full_data, ma_periods, stage_type):
    """提取指定类型的阶段区间（仅提取区间，不做有效性判断）"""
    intervals = []
    n = len(full_data)
    if n < 2:
        return intervals
    
    # 简单按时间窗口提取可能的阶段区间
    window_size = 5  # 最小窗口大小
    step = 1         # 滑动步长
    
    for i in range(0, n - window_size + 1, step):
        end_idx = i + window_size - 1
        start_date = full_data.loc[i, 'timestamps']
        end_date = full_data.loc[end_idx, 'timestamps']
        duration = end_idx - i + 1
        
        intervals.append({
            'start_idx': i,
            'end_idx': end_idx,
            'start_date': start_date,
            'end_date': end_date,
            'duration': duration
        })
    
    return intervals

def classify_from_csv(csv_file_path, ma_periods=[4, 8, 12, 16, 20, 47]):
    """从CSV文件读取数据并进行阶段分类"""
    try:
        # 检测文件编码
        with open(csv_file_path, 'rb') as f:
            result = chardet.detect(f.read())
        encoding = result['encoding'] or 'utf-8'
        
        # 读取CSV数据
        df = pd.read_csv(csv_file_path, encoding=encoding)
        df['timestamps'] = pd.to_datetime(df['timestamps'])
        
        # 提取股票代码
        stock_code = os.path.basename(csv_file_path).split('.')[0]
        
        # 进行分类
        return classify_stages_from_data(stock_code, df, ma_periods)
    except Exception as e:
        print(f"从CSV分类出错: {str(e)}")
        return None

def get_stage_class_from_csv0919(ma_periods=[4, 8, 12, 16, 20, 47], recent_days=60):
    # 配置测试参数
    print("执行:处理csv过程")
    data_folder = r"D:\self\code\tool_ma_back\bbb_fragments"  # 包含stage标记的CSV文件目录
    result_folder = r"D:\self\code\tool_ma_back\bbb_result"
    ma_data_folder = r"D:\self\data\final_data_0821"  # 新增：存储均线数据的文件夹

    os.makedirs(result_folder, exist_ok=True)
    feature_window_size = 60  # 特征计算窗口大小（60条数据）
    
    print(f"===== 开始基于实际stage标记的分类测试 =====")
    print(f"数据文件夹: {data_folder}")
    print(f"均线数据文件夹: {ma_data_folder}")  # 新增：显示均线数据文件夹
    print(f"均线周期: {ma_periods}")
    print(f"特征计算窗口: {feature_window_size}条数据")
    print(f"长短周期阈值: {DURATION_THRESHOLDS}")
    print(f"返回的近期MA数据天数: {recent_days}")
    print("---------------------------------\n")
    
    # 修改返回结果的结构，包含详细信息
    detailed_results = []
    csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    print(f"发现 {len(csv_files)} 个CSV文件，开始处理...\n")
    
    for csv_file in csv_files:
        stock_code = os.path.splitext(csv_file)[0]
        csv_path = os.path.join(data_folder, csv_file)
        ma_csv_path = os.path.join(ma_data_folder, f"{stock_code[:6]}.csv")  # 对应股票的均线数据文件
        
        print(f"处理文件: {csv_file} (股票代码: {stock_code})")
        
        # 检查均线数据文件是否存在
        if not os.path.exists(ma_csv_path):
            print(f"  警告: 对应的均线数据文件不存在: {ma_csv_path}，跳过\n")
            continue
        
        try:
            # 读取stage标记的CSV文件
            with open(csv_path, 'rb') as f:
                encoding = chardet.detect(f.read())['encoding'] or 'utf-8'
            
            df_stage = pd.read_csv(csv_path, encoding=encoding)
            df_stage['timestamps'] = pd.to_datetime(df_stage['timestamps'])
            df_stage = df_stage.sort_values('timestamps').reset_index(drop=True)  # 确保数据按时间排序
            

            if 'stage' not in df_stage.columns:
                print(f"警告: 文件 {csv_file} 不包含stage标记，跳过\n")
                continue

            df_stage = calculate_ma(df_stage, ma_periods)
            
            # [MA数据提取]读取对应股票的均线数据CSV文件 ###########################
            with open(ma_csv_path, 'rb') as f:
                ma_encoding = chardet.detect(f.read())['encoding'] or 'utf-8'
            
            df_ma = pd.read_csv(ma_csv_path, encoding=ma_encoding)
            df_ma['timestamps'] = pd.to_datetime(df_ma['timestamps'])
            df_ma = df_ma.sort_values('timestamps').reset_index(drop=True)  # 确保均线数据按时间排序
            
            # 验证均线列是否存在
            required_ma_columns = [f'MA{p}' for p in ma_periods]
            missing_columns = [col for col in required_ma_columns if col not in df_ma.columns]
            if missing_columns:
                print(f"  错误: 均线文件中缺失列 {missing_columns}")
                continue  # 跳过当前文件处理
            
            # 提取近N天的MA数据（最后N条记录）
            recent_ma_data = df_ma.tail(recent_days)[['timestamps'] + required_ma_columns].copy()
            # 转换为字典格式，便于前端处理
            recent_ma_list = recent_ma_data.to_dict('records')
            for item in recent_ma_list:
                item['timestamps'] = item['timestamps'].isoformat()  # 转换为ISO格式日期字符串
            
            ##########################################################
            
            # 按不同stage值分组，获取每个stage的首个出现位置
            stage_first_occurrences = {}
            for stage in [1, 2, 3]:
                mask = (df_stage['stage'] == stage)
                if mask.any():
                    first_idx = df_stage[mask].index.min()
                    stage_first_occurrences[stage] = first_idx
            
            # 处理每个stage的首个出现
            for stage, first_idx in stage_first_occurrences.items():
                stage_name = f'stage{stage}'
                print(f"  处理{stage_name}首个出现位置: 索引 {first_idx}")
                
                # 获取该位置的时间戳，用于定位均线数据
                target_timestamp = df_stage.loc[first_idx, 'timestamps']
                
                # 计算特征窗口的时间范围（首个stage数据点之前的60条数据）
                # 在均线数据中找到对应的时间点
                ma_target_idx = df_stage[df_stage['timestamps'] == target_timestamp].index
                if not len(ma_target_idx):
                    print(f"  警告: 在均线数据中未找到对应时间点 {target_timestamp}，跳过\n")
                    continue
                ma_target_idx = ma_target_idx[0]
                
                # 计算特征窗口的起始索引
                feature_start_idx = max(0, ma_target_idx - feature_window_size)
                feature_end_idx = ma_target_idx - 1  # 不包含首个stage数据点本身
                
                # 检查特征窗口是否有足够数据
                if feature_start_idx > feature_end_idx:
                    print(f"  警告: {stage_name} 特征窗口数据不足，跳过\n")
                    continue
                
                # 从均线数据中提取特征计算用数据
                feature_data = df_stage.loc[feature_start_idx:feature_end_idx].copy()
                print(f"  特征计算窗口: 索引 {feature_start_idx} 至 {feature_end_idx} ({len(feature_data)}条数据)")
                
                # 提取该stage的实际数据（从首个出现位置开始）
                stage_mask = (df_stage['stage'] == stage)
                group_stage_data = df_stage[stage_mask].copy()

                # 取group_data的第一条数据和最后一条数据的日期
                start_date = group_stage_data['timestamps'].iloc[0]
                end_date = group_stage_data['timestamps'].iloc[-1]
                
                # 从均线数据中提取对应时间段的数据
                group_ma_data = df_stage[
                    (df_stage['timestamps'] >= start_date) & 
                    (df_stage['timestamps'] <= end_date)
                ].copy()

                duration_days = len(group_ma_data)
                
                # 判断周期类型
                duration_type = 'long' if duration_days > DURATION_THRESHOLDS[stage_name] else 'short'
                print(f"  周期类型: {duration_type} ({duration_days}天)")
                
                # 获取模型路径
                model_path = MODEL_PATHS[stage_name][duration_type]['model_path']
                training_feature_path = MODEL_PATHS[stage_name][duration_type]['training_feature_path']
                
                # 检查模型文件是否存在
                if not os.path.exists(model_path):
                    print(f"  警告: {stage_name} {duration_type}模型文件不存在: {model_path}，跳过该阶段\n")
                    continue
                
                # 创建临时文件保存特征数据（使用均线数据）
                combined_data = pd.concat([feature_data, group_ma_data], ignore_index=True)
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', encoding='utf-8') as temp_file:
                    combined_data.to_csv(temp_file, index=False)
                    temp_file_path = temp_file.name
                
                # 根据阶段选择对应的预测模型
                probabilities = None
                try:
                    if stage == 1:
                        probabilities = predict_stage1(
                            temp_file_path,
                            model_path=model_path, 
                            training_feature_path=training_feature_path
                        )
                    elif stage == 2:
                        probabilities = predict_stage2(
                            temp_file_path,
                            model_path=model_path, 
                            training_feature_path=training_feature_path
                        )
                    else:  # stage == 3
                        probabilities = predict_stage3(
                            temp_file_path,
                            model_path=model_path, 
                            training_feature_path=training_feature_path
                        )
                finally:
                    # 无论预测是否成功，都删除临时文件
                    os.unlink(temp_file_path)
                
                # 处理概率结果，解决数组转换问题
                formatted_probabilities = {}
                if probabilities is not None:
                    try:
                        # 处理numpy数组情况
                        if hasattr(probabilities, 'ndim'):
                            # 如果是二维数组，取第一行
                            if probabilities.ndim == 2 and probabilities.shape[0] > 0:
                                probabilities = probabilities[0]
                            # 展平为一维数组
                            probabilities = probabilities.flatten()
                        
                        # 处理列表嵌套情况
                        while isinstance(probabilities, (list, np.ndarray)) and len(probabilities) == 1:
                            probabilities = probabilities[0]
                        
                        print(probabilities)
                        # 提取概率数组（第二个元素的第一行）
                        probabilities = probabilities[1][0]  # 结果是 numpy 数组：array([0.54, 0.45, 0.01, 0.  ])
                        # 确保是一维序列
                        if isinstance(probabilities, (list, np.ndarray)) and len(probabilities) > 0:
                            formatted_probabilities = {
                                f"class{i+1}": float(prob) for i, prob in enumerate(probabilities)
                            }
                        else:
                            print(f"  警告: 概率数据不是有效的序列: {probabilities}")
                            formatted_probabilities = {"error": "invalid_probability_format"}
                    except Exception as e:
                        print(f"  处理概率时出错: {str(e)}")
                        formatted_probabilities = {"error": str(e)}
                
                # 添加当前stage的详细结果（只要包含类别预测概率即可）
                detailed_results.append({
                    'stock_code': stock_code,
                    'recent_ma_data': recent_ma_list,
                    'detected_stage': stage,
                    'prediction_results': formatted_probabilities  # 确保包含类别预测概率
                })
            
            print(f"  处理完成\n")

            
        except Exception as e:
            print(f"  处理出错: {str(e)}\n")
            continue
    return detailed_results
    
def main():
    # 配置测试参数
    data_folder = r"D:\self\code\tool_ma_back\bbb_fragments"  # 包含stage标记的CSV文件目录
    result_folder = r"D:\self\code\tool_ma_back\bbb_result"
    
    os.makedirs(result_folder, exist_ok=True)
    ma_periods = [4, 8, 12, 16, 20, 47]
    feature_window_size = 60  # 特征计算窗口大小（60条数据）
    
    print(f"===== 开始基于实际stage标记的分类测试 =====")
    print(f"数据文件夹: {data_folder}")
    print(f"均线周期: {ma_periods}")
    print(f"特征计算窗口: {feature_window_size}条数据")
    print(f"长短周期阈值: {DURATION_THRESHOLDS}")
    print("---------------------------------\n")
    
    all_results = []
    csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    print(f"发现 {len(csv_files)} 个CSV文件，开始处理...\n")
    
    for csv_file in csv_files:
        stock_code = os.path.splitext(csv_file)[0]
        csv_path = os.path.join(data_folder, csv_file)
        
        print(f"处理文件: {csv_file} (股票代码: {stock_code})")
        
        try:
            # 读取CSV文件（包含stage标记）
            with open(csv_path, 'rb') as f:
                encoding = chardet.detect(f.read())['encoding'] or 'utf-8'
            
            df = pd.read_csv(csv_path, encoding=encoding)
            df['timestamps'] = pd.to_datetime(df['timestamps'])
            df = df.sort_values('timestamps').reset_index(drop=True)  # 确保数据按时间排序
            
            if 'stage' not in df.columns:
                print(f"警告: 文件 {csv_file} 不包含stage标记，跳过\n")
                continue
            
            # 计算均线（保留所有行，不删除NA值）
            df = calculate_ma(df, ma_periods)
            
            # 按不同stage值分组，获取每个stage的首个出现位置
            stage_first_occurrences = {}
            for stage in [1, 2, 3]:
                mask = (df['stage'] == stage)
                if mask.any():
                    first_idx = df[mask].index.min()
                    stage_first_occurrences[stage] = first_idx
            
            # 处理每个stage的首个出现
            stage_results = {
                'stage1': {'long': [], 'short': []},
                'stage2': {'long': [], 'short': []},
                'stage3': {'long': [], 'short': []}
            }
            
            for stage, first_idx in stage_first_occurrences.items():
                stage_name = f'stage{stage}'
                print(f"  处理{stage_name}首个出现位置: 索引 {first_idx}")
                
                # 计算特征窗口的起始索引（首个stage数据点之前的60条数据）
                feature_start_idx = max(0, first_idx - feature_window_size)
                feature_end_idx = first_idx - 1  # 不包含首个stage数据点本身
                
                # 检查特征窗口是否有足够数据
                if feature_start_idx > feature_end_idx:
                    print(f"  警告: {stage_name} 特征窗口数据不足，跳过\n")
                    continue
                
                # 提取特征计算用数据
                feature_data = df.loc[feature_start_idx:feature_end_idx].copy()
                print(f"  特征计算窗口: 索引 {feature_start_idx} 至 {feature_end_idx} ({len(feature_data)}条数据)")
                
                # 提取该stage的实际数据（从首个出现位置开始）
                stage_mask = (df['stage'] == stage)
                # group_data 仅包含stage=N的数据
                group_data = df[stage_mask].copy()

                # 取group_data的第一条数据和最后一条数据的日期
                start_date = group_data['timestamps'].iloc[0]
                end_date = group_data['timestamps'].iloc[-1]

                duration_days = len(group_data)
                
                # 判断周期类型
                duration_type = 'long' if duration_days > DURATION_THRESHOLDS[stage_name] else 'short'
                print(f"  周期类型: {duration_type} ({duration_days}天)")
                
                # 提取特征（使用首个stage数据点之前的60条数据）
                features = extract_stage_features(feature_data, ma_periods)
                
                # 获取模型路径
                model_path = MODEL_PATHS[stage_name][duration_type]['model_path']
                training_feature_path = MODEL_PATHS[stage_name][duration_type]['training_feature_path']
                
                # 检查模型文件是否存在
                if not os.path.exists(model_path):
                    print(f"  警告: {stage_name} {duration_type}模型文件不存在: {model_path}，跳过该阶段\n")
                    continue
                
                # 创建临时文件保存特征数据
                # 拼接feature_data和group_data用于预测
                combined_data = pd.concat([feature_data, group_data], ignore_index=True)
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', encoding='utf-8') as temp_file:
                    combined_data.to_csv(temp_file, index=False)
                    temp_file_path = temp_file.name
                
                # 根据阶段选择对应的预测模型
                try:
                    if stage == 1:
                        probabilities = predict_stage1(
                            temp_file_path,
                            model_path=model_path, 
                            training_feature_path=training_feature_path
                        )
                    elif stage == 2:
                        probabilities = predict_stage2(
                            temp_file_path,
                            model_path=model_path, 
                            training_feature_path=training_feature_path
                        )
                    else:  # stage == 3
                        probabilities = predict_stage3(
                            temp_file_path,
                            model_path=model_path, 
                            training_feature_path=training_feature_path
                        )
                finally:
                    # 无论预测是否成功，都删除临时文件
                    os.unlink(temp_file_path)
                
                # 存储结果
                stage_results[stage_name][duration_type].append({
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': end_date.strftime('%Y-%m-%d'),
                    'duration_days': duration_days,
                    'features': features.tolist(),
                    'probabilities': probabilities,
                    'model_used': os.path.basename(model_path),
                    'feature_window': f"{feature_start_idx}-{feature_end_idx}"
                })
            
            # 保存结果
            result_file = os.path.join(result_folder, f"{stock_code}_stage_based_results.csv")
            save_stage_based_results(stock_code, stage_results, result_file)
            
            # 收集汇总信息
            all_results.append({
                'stock_code': stock_code,
                'stage1_long': len(stage_results['stage1']['long']),
                'stage1_short': len(stage_results['stage1']['short']),
                'stage2_long': len(stage_results['stage2']['long']),
                'stage2_short': len(stage_results['stage2']['short']),
                'stage3_long': len(stage_results['stage3']['long']),
                'stage3_short': len(stage_results['stage3']['short']),
            })
            
            print(f"  处理完成，结果已保存至: {result_file}\n")
            
        except Exception as e:
            print(f"  处理出错: {str(e)}\n")
            continue
    
    # 打印汇总结果
    print("\n===== 测试汇总结果 =====")
    print(f"总处理文件数: {len(csv_files)}")
    print(f"成功处理文件数: {len(all_results)}\n")
    
    for result in all_results:
        print(f"股票代码: {result['stock_code']}")
        print(f"stage1: 长周期={result['stage1_long']}, 短周期={result['stage1_short']}")
        print(f"stage2: 长周期={result['stage2_long']}, 短周期={result['stage2_short']}")
        print(f"stage3: 长周期={result['stage3_long']}, 短周期={result['stage3_short']}")
        print("---")

def save_stage_based_results(stock_code, stage_results, output_file):
    """保存基于实际stage标记的分类结果"""
    all_data = []
    
    for stage, duration_data in stage_results.items():
        for duration_type, items in duration_data.items():
            for item in items:
                record = {
                    'stock_code': stock_code,
                    'stage': stage,
                    'duration_type': duration_type,
                    'start_date': item['start_date'],
                    'end_date': item['end_date'],
                    'duration_days': item['duration_days'],
                    'features': str(item['features']),
                    'probabilities': str(item['probabilities']),
                    'model_used': item['model_used'],
                    'feature_window': item['feature_window']
                }
                all_data.append(record)
    
    if all_data:
        pd.DataFrame(all_data).to_csv(output_file, index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    main()