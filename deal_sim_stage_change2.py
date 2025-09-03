import pandas as pd
import os
import joblib
import chardet
import shutil
from datetime import timedelta
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datetime import datetime, timedelta
import setter

################################################
# 功能扩展：
# 1. 测试片段阶段判断与特征提取
# 2. 股票池相似区间查找（同阶段）
# 3. 基于长度和模型的分类与结果输出
################################################

# 假设这些是必要的外部函数，根据实际情况调整导入
def get_stock_data(stock_code, start_date, end_date, data_folder):
    """获取股票数据（实际实现需根据数据源调整）"""
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

def calculate_ma(df, ma_periods):
    """计算均线"""
    df_copy = df.copy()
    for period in ma_periods:
        df_copy[f'MA{period}'] = df_copy['close'].rolling(window=period).mean()
    return df_copy.dropna()

def prepare_data_for_gp(folder_path):
    """准备特征数据（实际实现需根据特征工程逻辑调整）"""
    try:
        # 简单示例：读取临时文件并返回
        files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        if not files:
            return None, None
            
        file_path = os.path.join(folder_path, files[0])
        df = pd.read_csv(file_path, parse_dates=['timestamps'])
        # 这里添加实际的特征工程逻辑
        return df, None
    except Exception as e:
        print(f"特征准备出错: {str(e)}")
        return None, None

def find_source_csv_with_history(start_date, end_date, data_folder="D:/self/data/kline-data"):
    """查找包含指定日期范围的源CSV文件"""
    try:
        for filename in os.listdir(data_folder):
            if filename.endswith('.csv'):
                file_path = os.path.join(data_folder, filename)
                # 检测文件编码
                with open(file_path, 'rb') as f:
                    result = chardet.detect(f.read(10000))
                encoding = result['encoding'] or 'utf-8'
                
                # 只读取日期列来检查
                df = pd.read_csv(file_path, usecols=['timestamps'], parse_dates=['timestamps'], encoding=encoding)
                if not df.empty:
                    min_date = df['timestamps'].min()
                    max_date = df['timestamps'].max()
                    if min_date <= start_date and max_date >= end_date:
                        return file_path
        return None
    except Exception as e:
        print(f"查找源文件出错: {str(e)}")
        return None

"""检测文件编码"""
def detect_encoding(file_path):
    """检测文件编码"""
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

"""加载训练时的原始特征列名"""
def load_training_feature_columns(column_path):
    """加载训练时使用的原始特征列名"""
    if not os.path.exists(column_path):
        raise FileNotFoundError(f"训练特征列名文件 {column_path} 不存在，请先运行训练脚本生成")
    return pd.read_csv(column_path)['feature'].tolist()

"""使用对应阶段的模型进行预测"""
def predict_stage_type(stock_code, start_date, end_date, stage, duration_type, model_base_path, kline_data_folder):
    """
    使用对应阶段的模型预测多空类型
    参数:
        stock_code: 股票代码
        start_date: 阶段开始日期
        end_date: 阶段结束日期
        stage: 阶段类型 (stage1, stage2, stage3)
        duration_type: 长度类型 (short: 短周期, long: 长周期)
        model_base_path: 模型存放的基础路径
        kline_data_folder: K线数据文件夹路径
    返回:
        预测结果和概率
    """
    # 确定使用的模型路径（新增长度类型区分）
    model_path = os.path.join(model_base_path, f"{stage}_{duration_type}_model.pkl")
    feature_path = os.path.join(model_base_path, f"{stage}_{duration_type}_features.csv")
    
    if not os.path.exists(model_path):
        print(f"模型文件 {model_path} 不存在，无法进行预测")
        return None, None
        
    if not os.path.exists(feature_path):
        print(f"特征列文件 {feature_path} 不存在，无法进行预测")
        return None, None
    
    try:
        # 计算需要回溯的历史数据范围
        history_days = 50
        history_start_date = start_date - timedelta(days=history_days)
        
        # 查找包含所需历史数据的源文件
        source_csv_path = find_source_csv_with_history(history_start_date, end_date, kline_data_folder)
        if not source_csv_path:
            raise FileNotFoundError(f"找不到包含 {history_start_date} 到 {end_date} 日期范围的源CSV文件")
        
        # 读取源数据
        source_encoding = detect_encoding(source_csv_path)
        source_df = pd.read_csv(source_csv_path, parse_dates=['timestamps'], encoding=source_encoding)
        required_df = source_df[(source_df['timestamps'] >= history_start_date) & 
                               (source_df['timestamps'] <= end_date)]
        
        if required_df.empty:
            raise ValueError(f"源CSV文件中不包含 {history_start_date} 到 {end_date} 的数据")
        
        # 保存临时文件用于特征提取
        temp_folder = 'temp_prediction_data'
        os.makedirs(temp_folder, exist_ok=True)
        
        # 创建子文件夹结构
        stock_folder = os.path.join(temp_folder, str(stock_code))
        os.makedirs(stock_folder, exist_ok=True)
        
        temp_file_path = os.path.join(stock_folder, 'data.csv')
        required_df.to_csv(temp_file_path, index=False)
        
        # 提取原始特征
        features_df, _ = prepare_data_for_gp(temp_folder)
        
        if features_df is None or features_df.empty:
            print("直接传入文件夹路径未能提取到特征，尝试传入文件所在的父文件夹...")
            features_df, _ = prepare_data_for_gp(stock_folder)
            
            if features_df is None or features_df.empty:
                print("仍然无法提取有效特征，检查prepare_data_for_gp函数实现")
                return None, None
        
        # 清理临时文件
        try:
            shutil.rmtree(temp_folder)
        except Exception as e:
            print(f"清理临时文件时出错: {str(e)}")
        
        # 加载训练时的原始特征列名，确保特征一致性
        training_feature_columns = load_training_feature_columns(column_path=feature_path)
        
        # 对齐原始特征（补充缺失列，删除多余列）
        # 补充缺失的特征列（用0填充，实际应用中应使用训练集的中位数）
        missing_cols = [col for col in training_feature_columns if col not in features_df.columns]
        if missing_cols:
            # print(f"补充缺失的原始特征: {missing_cols}")
            for col in missing_cols:
                features_df[col] = 0
        
        # 删除多余的特征列
        extra_cols = [col for col in features_df.columns if col not in training_feature_columns]
        if extra_cols:
            print(f"删除多余的原始特征: {extra_cols}")
            features_df = features_df.drop(columns=extra_cols)
        
        # 筛选与当前阶段对应的特征
        if 'timestamps' in features_df.columns:
            final_features = features_df[features_df['timestamps'].between(start_date, end_date)]
            # 移除timestamps列（训练时没有该列）
            final_features = final_features.drop(columns=['timestamps'], errors='ignore')
        elif isinstance(features_df.index, pd.DatetimeIndex):
            final_features = features_df.loc[start_date:end_date]
        else:
            final_features = features_df
            
        if final_features.empty:
            print("筛选后没有剩余有效特征")
            return None, None
            
        # 确保特征顺序与训练时一致
        final_features = final_features[training_feature_columns]
        
        # 加载模型并进行预测
        pipeline = joblib.load(model_path)
        predictions = pipeline.predict(final_features)
        prediction_probs = pipeline.predict_proba(final_features)
        
        # 返回平均预测结果（整个阶段的平均）
        avg_prediction = int(predictions.mean() > 0.5)  # 0为short，1为long
        avg_probs = prediction_probs.mean(axis=0)
        
        return ("long" if avg_prediction == 1 else "short", 
                {"long_prob": avg_probs[1], "short_prob": avg_probs[0]})
                
    except Exception as e:
        print(f"预测过程中出错: {str(e)}")
        return None, None

"""
    检测股票池中各股票在指定时间区间内均线呈现多头排列的区间，并划分stage1、stage2和stage3
    结果按阶段分类，包含时间区间、股票代码、对应均线数据和多空预测
    """
def detect_bullish_arrangement(stock_pool, start_date, end_date, 
                              ma_periods=[4, 8, 12, 16, 20, 47], 
                              data_folder="D:/self/data/kline-data",
                              model_base_path="./models",
                              duration_thresholds={'stage1': 10, 'stage2': 4, 'stage3': 10}):
    """
    检测股票池中各股票在指定时间区间内均线呈现多头排列的区间，并划分stage1、stage2和stage3
    结果按阶段分类，包含时间区间、股票代码、对应均线数据和多空预测
    """
    # 最终结果按阶段分类
    result = {
        'stage1': [],
        'stage2': [],
        'stage3': []
    }
    
    for stock_code in stock_pool:
        try:
            # 获取股票在指定时间区间的数据
            df = get_stock_data(stock_code, start_date, end_date, data_folder)
            if df.empty:
                print(f"股票 {stock_code} 在指定区间内无有效数据，跳过")
                continue
            
            # 计算均线
            df_with_ma = calculate_ma(df.copy(), ma_periods)
            if df_with_ma.empty:
                print(f"股票 {stock_code} 均线计算失败，跳过")
                continue
            
            # 确保数据按时间排序，使用'trade_date'列名保持一致
            df_with_ma = df_with_ma.sort_values('trade_date').reset_index(drop=True)
            # 确保日期为datetime类型以便比较
            df_with_ma['trade_date'] = pd.to_datetime(df_with_ma['trade_date'])
            # 保存原始数据用于提取均线信息
            full_data = df_with_ma.copy()
            
            # 检测所有阶段区间
            current_stage2_start = None
            # 存储需要后续处理stage3的stage2结束索引
            stage2_ends_for_stage3 = []
            # 存储当前股票的各阶段信息
            stock_stages = {
                'stage1': [],
                'stage2': [],
                'stage3': []
            }
            
            # 遍历每一天检测多头排列状态
            for i in range(1, len(full_data)):
                # 检查均线排列顺序（短周期在上，长周期在下）
                order_valid = True
                for j in range(len(ma_periods) - 1):
                    current_ma = f'MA{ma_periods[j]}'
                    next_ma = f'MA{ma_periods[j+1]}'
                    if full_data.loc[i, current_ma] <= full_data.loc[i, next_ma]:
                        order_valid = False
                        break
                
                # 检查所有均线是否向上（当前值 > 前一天值）
                all_rising = True
                for period in ma_periods:
                    ma_col = f'MA{period}'
                    if full_data.loc[i, ma_col] <= full_data.loc[i-1, ma_col]:
                        all_rising = False
                        break
                
                # 判断当前是否处于多头排列（stage2）
                is_stage2 = order_valid and all_rising
                
                # 记录stage2区间
                if is_stage2:
                    if current_stage2_start is None:
                        current_stage2_start = i  # 记录索引而非日期，方便后续计算
                else:
                    if current_stage2_start is not None:
                        # 计算stage2区间长度（交易日数量）
                        stage2_days = i - current_stage2_start
                        
                        # 分析stage2结束时的均线状态
                        dropping_ma = []
                        for period in ma_periods:
                            ma_col = f'MA{period}'
                            if full_data.loc[i, ma_col] <= full_data.loc[i-1, ma_col]:
                                dropping_ma.append(period)
                        
                        # 判断是否是"结束时只有MA4下降，其余均线不下降"的stage2
                        is_qualified_stage2 = (len(dropping_ma) == 1 and 4 in dropping_ma)
                        
                        # 寻找stage1的起始点
                        stage1_start_idx = None
                        for j in range(current_stage2_start - 1, 0, -1):
                            rising_count = 0
                            for period in ma_periods:
                                ma_col = f'MA{period}'
                                if full_data.loc[j, ma_col] > full_data.loc[j-1, ma_col]:
                                    rising_count += 1
                            
                            if rising_count < 4:
                                stage1_start_idx = j + 1
                                break
                        
                        if stage1_start_idx is None:
                            stage1_start_idx = 1
                        
                        # 计算stage1持续天数
                        stage1_days = (current_stage2_start - 1) - stage1_start_idx + 1
                        
                        # 提取日期（使用trade_date保持一致）
                        stage1_start_date = full_data.loc[stage1_start_idx, 'trade_date']
                        stage1_end_date = full_data.loc[current_stage2_start - 1, 'trade_date']
                        stage2_start_date = full_data.loc[current_stage2_start, 'trade_date']
                        stage2_end_date = full_data.loc[i-1, 'trade_date']
                        
                        # 检查有效性（保持与原方法相同的阈值）
                        valid_stage1 = (stage1_start_idx <= current_stage2_start - 1) and (stage1_days > 3)
                        valid_stage2 = (stage2_days > 3)
                        
                        # 存储符合条件的stage2结束索引，用于后续提取stage3
                        if valid_stage2 and is_qualified_stage2:
                            stage2_ends_for_stage3.append({
                                'end_index': i-1,
                                'end_date': stage2_end_date,
                                'start_date': stage2_start_date
                            })
                        
                        # 提取stage1的均线数据
                        if valid_stage1:
                            # 使用stage1对应的阈值判断长短周期
                            duration_type = "long" if stage1_days > duration_thresholds['stage1'] else "short"

                            # 筛选stage1期间的数据（使用trade_date）
                            stage1_mask = (full_data['trade_date'] >= stage1_start_date) & \
                                         (full_data['trade_date'] <= stage1_end_date)
                            stage1_data = full_data.loc[stage1_mask, ['trade_date'] + [f'MA{period}' for period in ma_periods]]
                            
                            # 转换为字典列表以便存储
                            ma_values = stage1_data.to_dict('records')
                            
                            # 提取特征用于相似度计算
                            feature_vector = extract_stage_features(stage1_data, ma_periods)
                            
                            # 预测stage1的多空类型
                            stage1_type, stage1_prob = predict_stage_type(
                                stock_code, 
                                stage1_start_date, 
                                stage1_end_date, 
                                'stage1',
                                duration_type,
                                model_base_path,
                                data_folder
                            )
                            
                            stock_stages['stage1'].append({
                                'stock_code': stock_code,
                                'start': stage1_start_date.strftime('%Y-%m-%d'),
                                'end': stage1_end_date.strftime('%Y-%m-%d'),
                                'duration_days': stage1_days,
                                'duration_type': duration_type,
                                'threshold_used': duration_thresholds['stage1'],
                                'ma_data': ma_values,
                                'feature_vector': feature_vector,
                                'prediction': stage1_type,
                                'probability': stage1_prob
                            })
                        
                        # 提取stage2的均线数据
                        if valid_stage2:
                            # 使用stage2对应的阈值判断长短周期
                            duration_type = "long" if stage2_days > duration_thresholds['stage2'] else "short"
                            
                            # 筛选stage2期间的数据（使用trade_date）
                            stage2_mask = (full_data['trade_date'] >= stage2_start_date) & \
                                         (full_data['trade_date'] <= stage2_end_date)
                            stage2_data = full_data.loc[stage2_mask, ['trade_date'] + [f'MA{period}' for period in ma_periods]]
                            
                            # 转换为字典列表以便存储
                            ma_values = stage2_data.to_dict('records')
                            
                            # 提取特征用于相似度计算
                            feature_vector = extract_stage_features(stage2_data, ma_periods)
                            
                            # 预测stage2的多空类型
                            stage2_type, stage2_prob = predict_stage_type(
                                stock_code, 
                                stage2_start_date, 
                                stage2_end_date, 
                                'stage2',
                                duration_type,
                                model_base_path,
                                data_folder
                            )
                            
                            stock_stages['stage2'].append({
                                'stock_code': stock_code,
                                'start': stage2_start_date.strftime('%Y-%m-%d'),
                                'end': stage2_end_date.strftime('%Y-%m-%d'),
                                'duration_days': stage2_days,
                                'duration_type': duration_type,
                                'threshold_used': duration_thresholds['stage2'],
                                'end_condition': f"结束时下降的均线: {dropping_ma}",
                                'ma_data': ma_values,
                                'feature_vector': feature_vector,
                                'prediction': stage2_type,
                                'probability': stage2_prob
                            })
                        
                        current_stage2_start = None
            
            # 处理结束时仍处于stage2的情况
            if current_stage2_start is not None:
                # 计算stage2区间长度
                stage2_days = len(full_data) - current_stage2_start
                
                # 分析stage2结束时的均线状态
                last_idx = len(full_data) - 1
                dropping_ma = []
                for period in ma_periods:
                    ma_col = f'MA{period}'
                    if full_data.loc[last_idx, ma_col] <= full_data.loc[last_idx-1, ma_col]:
                        dropping_ma.append(period)
                
                # 判断是否是"结束时只有MA4下降，其余均线不下降"的stage2
                is_qualified_stage2 = (len(dropping_ma) == 1 and 4 in dropping_ma)
                
                # 寻找stage1的起始点
                stage1_start_idx = None
                for j in range(current_stage2_start - 1, 0, -1):
                    rising_count = 0
                    for period in ma_periods:
                        ma_col = f'MA{period}'
                        if full_data.loc[j, ma_col] > full_data.loc[j-1, ma_col]:
                            rising_count += 1
                    
                    if rising_count < 4:
                        stage1_start_idx = j + 1
                        break
                
                if stage1_start_idx is None:
                    stage1_start_idx = 1
                
                # 计算stage1持续天数
                stage1_days = (current_stage2_start - 1) - stage1_start_idx + 1
                
                # 提取日期（使用trade_date保持一致）
                stage1_start_date = full_data.loc[stage1_start_idx, 'trade_date']
                stage1_end_date = full_data.loc[current_stage2_start - 1, 'trade_date']
                stage2_start_date = full_data.loc[current_stage2_start, 'trade_date']
                stage2_end_date = full_data.loc[len(full_data)-1, 'trade_date']
                
                # 检查有效性（保持与原方法相同的阈值）
                valid_stage1 = (stage1_start_idx <= current_stage2_start - 1) and (stage1_days > 3)
                valid_stage2 = (stage2_days > 3)
                
                # 存储符合条件的stage2结束索引，用于后续提取stage3
                if valid_stage2 and is_qualified_stage2:
                    stage2_ends_for_stage3.append({
                        'end_index': len(full_data)-1,
                        'end_date': stage2_end_date,
                        'start_date': stage2_start_date
                    })
                
                # 添加有效的stage1
                if valid_stage1:
                    # 筛选stage1期间的数据（使用trade_date）
                    stage1_mask = (full_data['trade_date'] >= stage1_start_date) & \
                                 (full_data['trade_date'] <= stage1_end_date)
                    stage1_data = full_data.loc[stage1_mask, ['trade_date'] + [f'MA{period}' for period in ma_periods]]
                    
                    ma_values = stage1_data.to_dict('records')
                    
                    # 提取特征用于相似度计算
                    feature_vector = extract_stage_features(stage1_data, ma_periods)
                    
                    # 预测stage1的多空类型
                    stage1_type, stage1_prob = predict_stage_type(
                        stock_code, 
                        stage1_start_date, 
                        stage1_end_date, 
                        'stage1',
                        model_base_path,
                        data_folder
                    )
                    
                    stock_stages['stage1'].append({
                        'stock_code': stock_code,
                        'start': stage1_start_date.strftime('%Y-%m-%d'),
                        'end': stage1_end_date.strftime('%Y-%m-%d'),
                        'duration_days': stage1_days,
                        'ma_data': ma_values,
                        'feature_vector': feature_vector,
                        'prediction': stage1_type,
                        'probability': stage1_prob
                    })
                
                # 添加有效的stage2
                if valid_stage2:
                    # 筛选stage2期间的数据（使用trade_date）
                    stage2_mask = (full_data['trade_date'] >= stage2_start_date) & \
                                 (full_data['trade_date'] <= stage2_end_date)
                    stage2_data = full_data.loc[stage2_mask, ['trade_date'] + [f'MA{period}' for period in ma_periods]]
                    
                    ma_values = stage2_data.to_dict('records')
                    
                    # 提取特征用于相似度计算
                    feature_vector = extract_stage_features(stage2_data, ma_periods)
                    
                    # 预测stage2的多空类型
                    stage2_type, stage2_prob = predict_stage_type(
                        stock_code, 
                        stage2_start_date, 
                        stage2_end_date, 
                        'stage2',
                        model_base_path,
                        data_folder
                    )
                    
                    stock_stages['stage2'].append({
                        'stock_code': stock_code,
                        'start': stage2_start_date.strftime('%Y-%m-%d'),
                        'end': stage2_end_date.strftime('%Y-%m-%d'),
                        'duration_days': stage2_days,
                        'note': '区间持续至检测结束',
                        'end_condition': f"结束时下降的均线: {dropping_ma}",
                        'ma_data': ma_values,
                        'feature_vector': feature_vector,
                        'prediction': stage2_type,
                        'probability': stage2_prob
                    })
            
            # 提取stage3区间
            for stage2_end in stage2_ends_for_stage3:
                stage2_end_idx = stage2_end['end_index']
                stage2_end_date = stage2_end['end_date']
                stage2_start_date = stage2_end['start_date']
                
                # stage3从stage2结束的下一个交易日开始
                stage3_start_idx = stage2_end_idx + 1
                if stage3_start_idx >= len(full_data):
                    continue  # 如果已经是最后一条记录，无法形成stage3
                
                stage3_start_date = full_data.loc[stage3_start_idx, 'trade_date']
                stage3_end_idx = None
                
                # 寻找stage3的结束点（严格按照原方法逻辑）
                for k in range(stage3_start_idx, len(full_data)):
                    # 检查MA4是否下降
                    ma4_col = f'MA4'
                    ma4_dropping = full_data.loc[k, ma4_col] <= full_data.loc[k-1, ma4_col]
                    
                    # 检查是否有其他均线也下降
                    other_dropping = False
                    for period in ma_periods:
                        if period != 4:
                            ma_col = f'MA{period}'
                            if full_data.loc[k, ma_col] <= full_data.loc[k-1, ma_col]:
                                other_dropping = True
                                break
                    
                    if ma4_dropping and other_dropping:
                        stage3_end_idx = k
                        break
                
                # 如果找到结束点，则记录stage3
                if stage3_end_idx is not None:
                    stage3_end_date = full_data.loc[stage3_end_idx, 'trade_date']
                    stage3_days = stage3_end_idx - stage3_start_idx + 1
                    
                    # 筛选stage3期间的均线数据（使用trade_date）
                    stage3_mask = (full_data['trade_date'] >= stage3_start_date) & \
                                 (full_data['trade_date'] <= stage3_end_date)
                    stage3_data = full_data.loc[stage3_mask, ['trade_date'] + [f'MA{period}' for period in ma_periods]]
                    
                    ma_values = stage3_data.to_dict('records')
                    
                    # 提取特征用于相似度计算
                    feature_vector = extract_stage_features(stage3_data, ma_periods)
                    
                    # 使用stage3对应的阈值判断长短周期
                    duration_type = "long" if stage3_days > duration_thresholds['stage3'] else "short"
                    
                    # 预测stage3的多空类型
                    stage3_type, stage3_prob = predict_stage_type(
                        stock_code, 
                        stage3_start_date, 
                        stage3_end_date, 
                        'stage3',
                        duration_type,
                        model_base_path,
                        data_folder
                    )
                    
                    stock_stages['stage3'].append({
                        'stock_code': stock_code,
                        'start': stage3_start_date.strftime('%Y-%m-%d'),
                        'end': stage3_end_date.strftime('%Y-%m-%d'),
                        'duration_days': stage3_days,
                        'duration_type': duration_type,
                        'threshold_used': duration_thresholds['stage3'],
                        'note': f"跟随于 {stage2_end_date.strftime('%Y-%m-%d')} 结束的stage2之后",
                        'ma_data': ma_values,
                        'feature_vector': feature_vector,
                        'prediction': stage3_type,
                        'probability': stage3_prob
                    })
            
            # 将当前股票的各阶段结果合并到总结果中
            for stage_type in ['stage1', 'stage2', 'stage3']:
                result[stage_type].extend(stock_stages[stage_type])
            
            # 打印统计信息
            total_stages = sum(len(stock_stages[st]) for st in ['stage1', 'stage2', 'stage3'])
            print(f"股票 {stock_code} 完成检测，找到 {total_stages} 个符合条件的阶段")
            
        except Exception as e:
            print(f"处理股票 {stock_code} 时出错: {str(e)}")
    
    return result

def determine_stage(stock_code, start_date, end_date, 
                   ma_periods=[4, 8, 12, 16, 20, 47], 
                   data_folder="D:/self/data/kline-data",
                   extended_days=100,
                   model_base_path="./models"):
    """
    判断给定股票在指定时间区间属于哪个阶段（stage1、stage2或stage3）
    并返回对应阶段的多空预测结果及特征向量
    """
    try:
        # 转换原始目标日期格式并保存
        original_start = pd.to_datetime(start_date)
        original_end = pd.to_datetime(end_date)
        
        # 计算扩展区间（仅用于均线计算）
        extended_start = original_start - pd.Timedelta(days=extended_days)
        extended_end = original_end + pd.Timedelta(days=extended_days)
        
        # 1. 获取扩展范围数据用于计算均线（保证均线准确性）
        df = get_stock_data(stock_code, 
                           extended_start.strftime('%Y-%m-%d'), 
                           extended_end.strftime('%Y-%m-%d'), 
                           data_folder)

        if df.empty:
            return 'unknown', {"error": f"股票 {stock_code} 在扩展区间内无有效数据"}, None
        
        # 2. 基于扩展数据计算均线
        df_with_ma = calculate_ma(df.copy(), ma_periods)

        if df_with_ma.empty:
            return 'unknown', {"error": f"股票 {stock_code} 均线计算失败"}, None
        
        # 数据预处理
        df_with_ma = df_with_ma.sort_values('timestamps').reset_index(drop=True)
        df_with_ma['timestamps'] = pd.to_datetime(df_with_ma['timestamps'])
        
        # 3. 严格截取原始目标区间（扩展前的区间）
        mask = (df_with_ma['timestamps'] >= original_start) & (df_with_ma['timestamps'] <= original_end)
        target_data = df_with_ma[mask].copy()
        
        # 验证截取结果
        if len(target_data) < 2:
            actual_dates = [d.strftime('%Y-%m-%d') for d in target_data['timestamps']]
            return 'unknown', {
                "error": f"区间太短，仅找到 {len(target_data)} 个交易日（至少需要2个）",
                "原始目标区间": f"{original_start.strftime('%Y-%m-%d')} 至 {original_end.strftime('%Y-%m-%d')}",
                "实际找到的日期": actual_dates
            }, None
        
        # 首先检查是否符合stage3特征（优先级最高）
        is_stage3 = True
        stage3_evidence = []
        ma4_col = f'MA4'
        other_ma_periods = [p for p in ma_periods if p != 4]
        has_ma4_and_other_dropping_at_end = False
        
        for i in range(1, len(target_data)):
            current_idx = target_data.index[i]
            prev_idx = target_data.index[i-1]
            
            # 检查MA4是否下降
            ma4_dropping = df_with_ma.loc[current_idx, ma4_col] <= df_with_ma.loc[prev_idx, ma4_col]
            
            # 检查其他均线是否有下降
            other_dropping = False
            for period in other_ma_periods:
                ma_col = f'MA{period}'
                if df_with_ma.loc[current_idx, ma_col] <= df_with_ma.loc[prev_idx, ma_col]:
                    other_dropping = True
                    break
            
            # 记录区间末尾是否出现MA4和其他均线同时下降的情况
            if i == len(target_data) - 1 and ma4_dropping and other_dropping:
                has_ma4_and_other_dropping_at_end = True
            
            # stage3中除了最后一个位置外，不应出现"MA4下降且其他均线也下降"的情况
            if i < len(target_data) - 1 and ma4_dropping and other_dropping:
                is_stage3 = False
                stage3_evidence.append(f"在 {df_with_ma.loc[current_idx, 'timestamps'].strftime('%Y-%m-%d')} 出现MA4和其他均线同时下降，不符合stage3特征")
                break
            
            # 检查是否有4条以上均线上升（stage3也需要满足此条件）
            rising_count = 0
            for period in ma_periods:
                ma_col = f'MA{period}'
                if df_with_ma.loc[current_idx, ma_col] > df_with_ma.loc[prev_idx, ma_col]:
                    rising_count += 1
            
            if rising_count < 4:
                is_stage3 = False
                stage3_evidence.append(f"在 {df_with_ma.loc[current_idx, 'timestamps'].strftime('%Y-%m-%d')} 上升均线不足4条，不符合stage3特征")
                break
        
        # 检查stage3是否在末尾有MA4和其他均线同时下降的情况
        if is_stage3 and not has_ma4_and_other_dropping_at_end:
            is_stage3 = False
            stage3_evidence.append(f"区间末尾未出现MA4和其他均线同时下降的情况，不符合stage3特征")
        
        if is_stage3:
            # 提取特征向量
            feature_vector = extract_stage_features(target_data, ma_periods)
            # 计算持续天数
            duration_days = (original_end - original_start).days
            # 确定周期类型
            duration_thresholds = {'stage1': 10, 'stage2': 4, 'stage3': 10}
            duration_type = "long" if duration_days > duration_thresholds['stage3'] else "short"
            # 获取预测结果
            prediction, prob = predict_stage_type(
                stock_code, original_start, original_end, 
                'stage3', duration_type, model_base_path, data_folder
            )
            return 'stage3', {
                "evidence": "符合stage3特征",
                "duration_days": duration_days,
                "duration_type": duration_type,
                "prediction": prediction,
                "probability": prob
            }, feature_vector
        
        # 检查是否符合stage2特征（优先级次之）
        is_stage2 = True
        stage2_evidence = []
        
        for i in range(len(target_data)):
            # 检查均线排列顺序（短周期在上，长周期在下）
            order_valid = True
            for j in range(len(ma_periods) - 1):
                current_ma = f'MA{ma_periods[j]}'
                next_ma = f'MA{ma_periods[j+1]}'
                if target_data.loc[i, current_ma] <= target_data.loc[i, next_ma]:
                    order_valid = False
                    break
            
            # 检查所有均线是否向上（当前值 > 前一天值）
            if i > 0:  # 第一天没有前一天数据，不检查
                all_rising = True
                for period in ma_periods:
                    ma_col = f'MA{period}'
                    if target_data.loc[i, ma_col] <= target_data.loc[i-1, ma_col]:
                        all_rising = False
                        break
            else:
                all_rising = True  # 第一天默认有效
            
            if not order_valid or (i > 0 and not all_rising):
                is_stage2 = False
                date_str = target_data.loc[i, 'timestamps'].strftime('%Y-%m-%d')
                reason = []
                if not order_valid:
                    reason.append("均线排列顺序不符合要求")
                if i > 0 and not all_rising:
                    reason.append("不是所有均线都呈上升趋势")
                stage2_evidence.append(f"在 {date_str}: {', '.join(reason)}")
                break
        
        if is_stage2:
            # 提取特征向量
            feature_vector = extract_stage_features(target_data, ma_periods)
            # 计算持续天数
            duration_days = (original_end - original_start).days
            # 确定周期类型
            duration_thresholds = {'stage1': 10, 'stage2': 4, 'stage3': 10}
            duration_type = "long" if duration_days > duration_thresholds['stage2'] else "short"
            # 获取预测结果
            prediction, prob = predict_stage_type(
                stock_code, original_start, original_end, 
                'stage2', duration_type, model_base_path, data_folder
            )
            return 'stage2', {
                "evidence": "符合stage2特征",
                "duration_days": duration_days,
                "duration_type": duration_type,
                "prediction": prediction,
                "probability": prob
            }, feature_vector
        
        # 检查是否符合stage1特征（优先级最低）
        is_stage1 = True
        stage1_evidence = []
        
        for i in range(1, len(target_data)):
            current_idx = target_data.index[i]
            prev_idx = target_data.index[i-1]
            
            # 检查是否有4条以上均线上升
            rising_count = 0
            for period in ma_periods:
                ma_col = f'MA{period}'
                if target_data.loc[i, ma_col] > target_data.loc[prev_idx, ma_col]:
                    rising_count += 1
            
            if rising_count < 4:
                is_stage1 = False
                date_str = target_data.loc[i, 'timestamps'].strftime('%Y-%m-%d')
                stage1_evidence.append(f"在 {date_str} 上升均线不足4条")
                break
        
        if is_stage1:
            # 提取特征向量
            feature_vector = extract_stage_features(target_data, ma_periods)
            # 计算持续天数
            duration_days = (original_end - original_start).days
            # 确定周期类型
            duration_thresholds = {'stage1': 10, 'stage2': 4, 'stage3': 10}
            duration_type = "long" if duration_days > duration_thresholds['stage1'] else "short"
            # 获取预测结果
            prediction, prob = predict_stage_type(
                stock_code, original_start, original_end, 
                'stage1', duration_type, model_base_path, data_folder
            )
            return 'stage1', {
                "evidence": "符合stage1特征",
                "duration_days": duration_days,
                "duration_type": duration_type,
                "prediction": prediction,
                "probability": prob
            }, feature_vector
        
        # 如果都不符合
        return 'unknown', {
            "stage3_evidence": stage3_evidence,
            "stage2_evidence": stage2_evidence,
            "stage1_evidence": stage1_evidence
        }, None
        
    except Exception as e:
        print(f"判断阶段时出错: {str(e)}")
        return 'unknown', {"error": str(e)}, None

def extract_stage_features(stage_data, ma_periods):
    """提取阶段特征向量用于相似度计算"""
    features = []
    ma_columns = [f'MA{period}' for period in ma_periods]
    
    # 1. 计算各均线的平均斜率
    for ma_col in ma_columns:
        values = stage_data[ma_col].values
        if len(values) < 2:
            slope = 0
        else:
            slope = (values[-1] - values[0]) / len(values)
        features.append(slope)
    
    # 2. 计算各均线的波动率（标准差）
    for ma_col in ma_columns:
        features.append(stage_data[ma_col].std())
    
    # 3. 计算均线之间的平均距离
    for i in range(len(ma_columns) - 1):
        diff = stage_data[ma_columns[i]] - stage_data[ma_columns[i+1]]
        features.append(diff.mean())
    
    # 4. 计算阶段持续天数
    features.append(len(stage_data))
    
    return np.array(features).reshape(1, -1)

def find_similar_stages(test_stock, test_start_date, test_end_date, 
                       stock_pool, pool_start_date, pool_end_date,
                       ma_periods=[4, 8, 12, 16, 20, 47],
                       data_folder="D:/self/data/kline-data",
                       model_base_path="./models",
                       similarity_threshold=0.8):
    """
    查找与测试片段相似的股票池区间
    步骤：
    1. 确定测试片段的阶段
    2. 在股票池中查找相同阶段的区间
    3. 计算相似度并筛选
    4. 输出分类结果
    """
    print("===== 步骤1: 分析测试片段 =====")
    # 1. 确定测试片段的阶段
    test_stage, test_info, test_features = determine_stage(
        stock_code=test_stock,
        start_date=test_start_date,
        end_date=test_end_date,
        ma_periods=ma_periods,
        data_folder=data_folder,
        model_base_path=model_base_path
    )
    
    if test_stage == 'unknown':
        print("无法确定测试片段的阶段，终止查找")
        return {
            "test_stage": test_stage,
            "test_info": test_info,
            "similar_stages": []
        }
    
    print(f"测试片段阶段: {test_stage}")
    print(f"测试片段信息: {test_info}")
    
    print("\n===== 步骤2: 检测股票池区间 =====")
    # 2. 检测股票池中的所有阶段
    pool_stages = detect_bullish_arrangement(
        stock_pool=stock_pool,
        start_date=pool_start_date,
        end_date=pool_end_date,
        ma_periods=ma_periods,
        data_folder=data_folder,
        model_base_path=model_base_path
    )
    
    # 提取与测试片段相同阶段的区间
    same_stage = pool_stages.get(test_stage, [])
    print(f"在股票池中找到 {len(same_stage)} 个 {test_stage} 阶段")
    
    if not same_stage:
        return {
            "test_stage": test_stage,
            "test_info": test_info,
            "similar_stages": []
        }
    
    print("\n===== 步骤3: 计算相似度并筛选 =====")
    # 3. 计算相似度并筛选
    similar_stages = []
    for stage in same_stage:
        # 确保有特征向量
        if 'feature_vector' not in stage or stage['feature_vector'] is None:
            continue
            
        # 计算余弦相似度
        similarity = cosine_similarity(test_features, stage['feature_vector'])[0][0]
        
        if similarity >= similarity_threshold:
            stage['similarity'] = similarity
            similar_stages.append(stage)
    
    # 按相似度排序
    similar_stages.sort(key=lambda x: x['similarity'], reverse=True)
    print(f"找到 {len(similar_stages)} 个与测试片段相似的 {test_stage} 阶段（相似度 >= {similarity_threshold}）")
    
    print("\n===== 步骤4: 输出分类结果 =====")
    # 4. 整理结果
    result = {
        "test_stage": test_stage,
        "test_info": test_info,
        "similar_stages": similar_stages,
        "total_similar": len(similar_stages),
        "similarity_threshold": similarity_threshold
    }
    
    return result

def main():
    # 配置参数
    DATA_FOLDER = "D:/self/data/kline-data"
    MODEL_BASE_PATH = "./models"
    MA_PERIODS = [4, 8, 12, 16, 20, 47]
    TEST_STOCK_POOL = ["000547", "000561", "000830", "001255", "001308", "001309"]
    
    # 日期配置
    # 自定义日期区间（格式：'YYYY-MM-DD'）
    start_str = "2020-01-01"  # 开始日期
    end_str = "2024-12-01"    # 结束日期
    start_date = datetime.strptime(start_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_str, '%Y-%m-%d')
    
    print("="*60)
    print(f"股票阶段分析测试 [{start_str} 至 {end_str}]")
    print("="*60)
    
    try:
        # 1. 单只股票阶段判断
        test_stock = "000021"
        test_start_str = "2024-10-11"
        test_end_str = "2024-10-31"
        test_start = datetime.strptime(test_start_str, '%Y-%m-%d')
        test_end = datetime.strptime(test_end_str, '%Y-%m-%d')

        print(f"\n1. 测试股票 {test_stock} 阶段 [{test_start} 至 {test_end}]")
        stage, info, features = determine_stage(
            stock_code=test_stock,
            start_date=test_start,
            end_date=test_end,
            ma_periods=MA_PERIODS,
            data_folder=DATA_FOLDER,
            model_base_path=MODEL_BASE_PATH
        )
        print(f"→ 阶段判断: {stage} | 持续天数: {info.get('duration_days')} | 预测: {info.get('prediction')}")
        
        # 2. 股票池阶段检测
        pool_start = (end_date - timedelta(days=120)).strftime('%Y-%m-%d')
        print(f"\n2. 股票池检测 [{pool_start} 至 {end_str}] (共 {len(TEST_STOCK_POOL)} 只)")
        stages_result = detect_bullish_arrangement(
            stock_pool=TEST_STOCK_POOL,
            start_date=pool_start,
            end_date=end_str,
            ma_periods=MA_PERIODS,
            data_folder=DATA_FOLDER,
            model_base_path=MODEL_BASE_PATH
        )
        print(f"→ 阶段分布: stage1={len(stages_result['stage1'])} | stage2={len(stages_result['stage2'])} | stage3={len(stages_result['stage3'])}")
        
        # 3. 相似区间查找
        if stage != 'unknown' and features is not None:
            print(f"\n3. 查找与 {test_stock} 相似的 {stage} 阶段")
            similar_results = find_similar_stages(
                test_stock=test_stock,
                test_start_date=test_start,
                test_end_date=test_end,
                stock_pool=TEST_STOCK_POOL,
                pool_start_date=pool_start,
                pool_end_date=end_str,
                ma_periods=MA_PERIODS,
                data_folder=DATA_FOLDER,
                model_base_path=MODEL_BASE_PATH,
                similarity_threshold=0.7
            )
            
            if similar_results['total_similar'] > 0:
                print(f"→ 找到 {similar_results['total_similar']} 个相似区间 (前3名):")
                for i, similar in enumerate(similar_results['similar_stages'][:3]):
                    print(f"  {i+1}. 股票 {similar['stock_code']} | 相似度 {similar['similarity']:.4f} | 预测 {similar['prediction']}")
            
        print("\n" + "="*60)
        print("测试完成")
        
    except Exception as e:
        print(f"\n测试错误: {str(e)}")

if __name__ == "__main__":
    main()
    