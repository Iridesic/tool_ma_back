import pandas as pd
import os
import chardet
import numpy as np
from datetime import datetime
import shutil

from aaa_classify_code.get_stage1_class_stage2copy import predict_from_csv
from aaa_classify_code.get_stage1_class_stage3copy import predict_from_csv as predict_from_csv3
from aaa_classify_code.get_stage1_class import predict_from_csv as predict_from_csv1

################################################
# 功能：
# 1. 检测股票池中的特定阶段片段
# 2. 对每个检测到的区间向前扩充60条数据并保存
# 3. 新增stage列标记不同阶段（0:扩充数据, 1:stage1, 2:stage2, 3:stage3）
################################################

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
        # 按时间排序并重置索引
        return df.sort_values('timestamps').reset_index(drop=True)
    except Exception as e:
        print(f"获取完整股票数据出错: {str(e)}")
        return pd.DataFrame()

def calculate_ma(df, ma_periods):
    """计算均线"""
    df_copy = df.copy()
    for period in ma_periods:
        df_copy[f'MA{period}'] = df_copy['close'].rolling(window=period).mean()
    return df_copy.dropna()

def extract_stage_features(stage_data, ma_periods):
    """提取阶段特征用于相似度计算"""
    features = []
    # 计算各均线的平均斜率
    for period in ma_periods:
        ma_col = f'MA{period}'
        ma_values = stage_data[ma_col].values
        # 计算简单斜率（最后值-初始值）/长度
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
    
    return np.array(features)

def detect_stage_fragments(stock_pool, start_date, end_date, target_stage,
                          ma_periods=[4, 8, 12, 16, 20, 47], 
                          data_folder="D:/self/data/kline-data",
                          duration_thresholds={'stage1': 10, 'stage2': 4, 'stage3': 10}):
    """
    检测股票池中各股票在指定时间区间内符合目标阶段（stage1、stage2、stage3、all）的片段
    当target_stage为'all'时，使用stage1检测算法，结果为从stage1开始到MA4下穿MA20结束的完整区间
    """
    # 验证目标阶段有效性，包含'all'选项
    if target_stage not in ['stage1', 'stage2', 'stage3', 'all']:
        raise ValueError("目标阶段必须是'stage1'、'stage2'、'stage3'或'all'")
    
    # 如果是'all'模式，使用stage1的检测算法
    actual_stage = 'stage1' if target_stage == 'all' else target_stage
    result = []
    
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
            
            # 确保数据按时间排序
            df_with_ma = df_with_ma.sort_values('timestamps').reset_index(drop=True)
            df_with_ma['timestamps'] = pd.to_datetime(df_with_ma['timestamps'])
            full_data = df_with_ma.copy()
            
            # 检测各阶段区间
            current_stage2_start = None
            stage2_ends_for_stage3 = []
            stock_stages = {
                'stage1': [],
                'stage2': [],
                'stage3': [],
                'all': []  # 存储all模式的结果
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
                        current_stage2_start = i  # 记录索引
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
                        
                        # 提取日期
                        stage1_start_date = full_data.loc[stage1_start_idx, 'timestamps']
                        stage1_end_date = full_data.loc[current_stage2_start - 1, 'timestamps']
                        stage2_start_date = full_data.loc[current_stage2_start, 'timestamps']
                        stage2_end_date = full_data.loc[i-1, 'timestamps']
                        
                        # 检查有效性
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
                            duration_type = "long" if stage1_days > duration_thresholds['stage1'] else "short"

                            # 筛选stage1期间的数据
                            stage1_mask = (full_data['timestamps'] >= stage1_start_date) & \
                                         (full_data['timestamps'] <= stage1_end_date)
                            stage1_data = full_data.loc[stage1_mask, ['timestamps'] + [f'MA{period}' for period in ma_periods]]
                            
                            # 转换为字典列表以便存储
                            ma_values = stage1_data.to_dict('records')
                            
                            # 提取特征用于相似度计算
                            feature_vector = extract_stage_features(stage1_data, ma_periods)
                            
                            stock_stages['stage1'].append({
                                'stock_code': stock_code,
                                'start': stage1_start_date.strftime('%Y-%m-%d'),
                                'end': stage1_end_date.strftime('%Y-%m-%d'),
                                'duration_days': stage1_days,
                                'duration_type': duration_type,
                                'threshold_used': duration_thresholds['stage1'],
                                'ma_data': ma_values,
                                'feature_vector': feature_vector,
                                'probabilities': {'class1': 0.69, 'class2': 0.29, 'class3': 0.01, 'class4': 0.01}
                            })
                            
                            # 处理all模式 - 寻找MA4下穿MA20的结束点
                            if target_stage == 'all':
                                # 从stage1结束点开始寻找MA4下穿MA20
                                all_end_idx = None
                                # 从stage1结束点的下一个索引开始查找
                                for k in range(current_stage2_start, len(full_data)):
                                    # 检查MA4是否下穿MA20（当前MA4 <= MA20 且 前一天MA4 > MA20）
                                    if k > 0:
                                        ma4_current = full_data.loc[k, 'MA4']
                                        ma20_current = full_data.loc[k, 'MA20']
                                        ma4_prev = full_data.loc[k-1, 'MA4']
                                        ma20_prev = full_data.loc[k-1, 'MA20']
                                        
                                        if (ma4_current <= ma20_current) and (ma4_prev > ma20_prev):
                                            all_end_idx = k
                                            break
                                
                                # 如果找到MA4下穿MA20的点，则记录完整区间
                                if all_end_idx is not None:
                                    all_end_date = full_data.loc[all_end_idx, 'timestamps']
                                    all_duration_days = (all_end_idx - stage1_start_idx) + 1
                                    
                                    # 筛选完整区间数据（stage1开始到MA4下穿MA20结束）
                                    all_mask = (full_data['timestamps'] >= stage1_start_date) & \
                                              (full_data['timestamps'] <= all_end_date)
                                    all_data = full_data.loc[all_mask, ['timestamps'] + [f'MA{period}' for period in ma_periods]]
                                    
                                    all_ma_values = all_data.to_dict('records')
                                    all_feature_vector = extract_stage_features(all_data, ma_periods)
                                    
                                    stock_stages['all'].append({
                                        'stock_code': stock_code,
                                        'start': stage1_start_date.strftime('%Y-%m-%d'),
                                        'end': all_end_date.strftime('%Y-%m-%d'),
                                        'duration_days': all_duration_days,
                                        'duration_type': "long" if all_duration_days > duration_thresholds['stage1'] else "short",
                                        'threshold_used': duration_thresholds['stage1'],
                                        'ma_data': all_ma_values,
                                        'feature_vector': all_feature_vector,
                                        'note': "包含stage1及后续直到MA4下穿MA20的完整区间",
                                        'probabilities': {'all': 1}
                                    })
                        
                        # 提取stage2的均线数据
                        if valid_stage2:
                            duration_type = "long" if stage2_days > duration_thresholds['stage2'] else "short"
                            
                            # 筛选stage2期间的数据
                            stage2_mask = (full_data['timestamps'] >= stage2_start_date) & \
                                         (full_data['timestamps'] <= stage2_end_date)
                            stage2_data = full_data.loc[stage2_mask, ['timestamps'] + [f'MA{period}' for period in ma_periods]]
                            
                            # 转换为字典列表以便存储
                            ma_values = stage2_data.to_dict('records')
                            
                            # 提取特征用于相似度计算
                            feature_vector = extract_stage_features(stage2_data, ma_periods)
                            
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
                                'probabilities': {'class1': 0.69, 'class2': 0.29, 'class3': 0.01, 'class4': 0.01}
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
                
                # 提取日期
                stage1_start_date = full_data.loc[stage1_start_idx, 'timestamps']
                stage1_end_date = full_data.loc[current_stage2_start - 1, 'timestamps']
                stage2_start_date = full_data.loc[current_stage2_start, 'timestamps']
                stage2_end_date = full_data.loc[len(full_data)-1, 'timestamps']
                
                # 检查有效性
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
                    # 筛选stage1期间的数据
                    stage1_mask = (full_data['timestamps'] >= stage1_start_date) & \
                                 (full_data['timestamps'] <= stage1_end_date)
                    stage1_data = full_data.loc[stage1_mask, ['timestamps'] + [f'MA{period}' for period in ma_periods]]
                    
                    ma_values = stage1_data.to_dict('records')
                    
                    # 提取特征用于相似度计算
                    feature_vector = extract_stage_features(stage1_data, ma_periods)
                    duration_type_stage1 = "long" if stage1_days > duration_thresholds['stage1'] else "short"
                    
                    stock_stages['stage1'].append({
                        'stock_code': stock_code,
                        'start': stage1_start_date.strftime('%Y-%m-%d'),
                        'end': stage1_end_date.strftime('%Y-%m-%d'),
                        'duration_days': stage1_days,
                        'duration_type': duration_type_stage1,
                        'threshold_used': duration_thresholds['stage1'],
                        'ma_data': ma_values,
                        'feature_vector': feature_vector,
                        'probabilities': {'class1': 0.69, 'class2': 0.29, 'class3': 0.01, 'class4': 0.01}
                    })
                    
                    # 处理all模式 - 寻找MA4下穿MA20的结束点
                    if target_stage == 'all':
                        all_end_idx = None
                        # 从stage1结束点的下一个索引开始查找
                        for k in range(current_stage2_start, len(full_data)):
                            if k > 0:
                                ma4_current = full_data.loc[k, 'MA4']
                                ma20_current = full_data.loc[k, 'MA20']
                                ma4_prev = full_data.loc[k-1, 'MA4']
                                ma20_prev = full_data.loc[k-1, 'MA20']
                                
                                if (ma4_current <= ma20_current) and (ma4_prev > ma20_prev):
                                    all_end_idx = k
                                    break
                        
                        # 如果找到结束点
                        if all_end_idx is not None:
                            all_end_date = full_data.loc[all_end_idx, 'timestamps']
                            all_duration_days = (all_end_idx - stage1_start_idx) + 1
                            
                            all_mask = (full_data['timestamps'] >= stage1_start_date) & \
                                      (full_data['timestamps'] <= all_end_date)
                            all_data = full_data.loc[all_mask, ['timestamps'] + [f'MA{period}' for period in ma_periods]]
                            
                            all_ma_values = all_data.to_dict('records')
                            all_feature_vector = extract_stage_features(all_data, ma_periods)
                            
                            stock_stages['all'].append({
                                'stock_code': stock_code,
                                'start': stage1_start_date.strftime('%Y-%m-%d'),
                                'end': all_end_date.strftime('%Y-%m-%d'),
                                'duration_days': all_duration_days,
                                'duration_type': "long" if all_duration_days > duration_thresholds['stage1'] else "short",
                                'threshold_used': duration_thresholds['stage1'],
                                'ma_data': all_ma_values,
                                'feature_vector': all_feature_vector,
                                'note': "包含stage1及后续直到MA4下穿MA20的完整区间",
                                'probabilities': {'all': 1}
                            })
                        # 如果未找到结束点，使用数据最后一条记录作为结束点
                        else:
                            all_end_date = full_data.loc[len(full_data)-1, 'timestamps']
                            all_duration_days = (len(full_data)-1 - stage1_start_idx) + 1
                            
                            all_mask = (full_data['timestamps'] >= stage1_start_date) & \
                                      (full_data['timestamps'] <= all_end_date)
                            all_data = full_data.loc[all_mask, ['timestamps'] + [f'MA{period}' for period in ma_periods]]
                            
                            all_ma_values = all_data.to_dict('records')
                            all_feature_vector = extract_stage_features(all_data, ma_periods)
                            
                            stock_stages['all'].append({
                                'stock_code': stock_code,
                                'start': stage1_start_date.strftime('%Y-%m-%d'),
                                'end': all_end_date.strftime('%Y-%m-%d'),
                                'duration_days': all_duration_days,
                                'duration_type': "long" if all_duration_days > duration_thresholds['stage1'] else "short",
                                'threshold_used': duration_thresholds['stage1'],
                                'ma_data': all_ma_values,
                                'feature_vector': all_feature_vector,
                                'note': "包含stage1及后续数据（未检测到MA4下穿MA20）",
                                'probabilities': {'all': 1}
                            })
                
                # 添加有效的stage2
                if valid_stage2:
                    # 筛选stage2期间的数据
                    stage2_mask = (full_data['timestamps'] >= stage2_start_date) & \
                                 (full_data['timestamps'] <= stage2_end_date)
                    stage2_data = full_data.loc[stage2_mask, ['timestamps'] + [f'MA{period}' for period in ma_periods]]
                    
                    ma_values = stage2_data.to_dict('records')
                    
                    # 提取特征用于相似度计算
                    feature_vector = extract_stage_features(stage2_data, ma_periods)
                    duration_type_stage2 = "long" if stage2_days > duration_thresholds['stage2'] else "short"
                    
                    stock_stages['stage2'].append({
                        'stock_code': stock_code,
                        'start': stage2_start_date.strftime('%Y-%m-%d'),
                        'end': stage2_end_date.strftime('%Y-%m-%d'),
                        'duration_days': stage2_days,
                        'duration_type': duration_type_stage2,
                        'threshold_used': duration_thresholds['stage2'],
                        'note': '区间持续至检测结束',
                        'end_condition': f"结束时下降的均线: {dropping_ma}",
                        'ma_data': ma_values,
                        'feature_vector': feature_vector,
                        'probabilities': {'class1': 0.69, 'class2': 0.29, 'class3': 0.01, 'class4': 0.01}
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
                
                stage3_start_date = full_data.loc[stage3_start_idx, 'timestamps']
                stage3_end_idx = None
                
                # 寻找stage3的结束点
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
                    stage3_end_date = full_data.loc[stage3_end_idx, 'timestamps']
                    stage3_days = stage3_end_idx - stage3_start_idx + 1
                    
                    # 筛选stage3期间的均线数据
                    stage3_mask = (full_data['timestamps'] >= stage3_start_date) & \
                                 (full_data['timestamps'] <= stage3_end_date)
                    stage3_data = full_data.loc[stage3_mask, ['timestamps'] + [f'MA{period}' for period in ma_periods]]
                    
                    ma_values = stage3_data.to_dict('records')
                    
                    # 提取特征用于相似度计算
                    feature_vector = extract_stage_features(stage3_data, ma_periods)
                    
                    # 判断长短周期
                    duration_type = "long" if stage3_days > duration_thresholds['stage3'] else "short"
                    
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
                        'probabilities': {'class1': 0.69, 'class2': 0.29, 'class3': 0.01, 'class4': 0.01}
                    })
            
            # 将当前股票的目标阶段结果添加到总结果
            result.extend(stock_stages[target_stage])
            
            # 打印统计信息
            print(f"股票 {stock_code} 完成检测，找到 {len(stock_stages[target_stage])} 个符合{target_stage}的阶段")
            
        except Exception as e:
            print(f"处理股票 {stock_code} 时出错: {str(e)}")
    
    return result
    
def extend_data_backward(stock_code, start_date, end_date, stage_type, data_folder, extend_rows=60):
    """
    获取指定股票在指定时间段的数据，并向前扩充指定行数
    新增stage列标记不同阶段（0:扩充数据, 1:stage1, 2:stage2, 3:stage3）
    返回扩充后的数据
    """
    # 获取完整数据
    full_data = get_full_stock_data(stock_code, data_folder)
    if full_data.empty:
        print(f"无法获取股票 {stock_code} 的完整数据，无法扩充")
        return None
    
    # 转换日期格式
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # 找到原始区间的起始和结束索引（精确匹配）
    start_mask = full_data['timestamps'] == start_date
    end_mask = full_data['timestamps'] == end_date
    
    if not start_mask.any() or not end_mask.any():
        print(f"在股票 {stock_code} 中未找到精确匹配的日期 {start_date} 至 {end_date}")
        # 尝试找到最接近的日期
        start_idx = full_data[full_data['timestamps'] >= start_date].index.min()
        end_idx = full_data[full_data['timestamps'] <= end_date].index.max()
        
        if pd.isna(start_idx) or pd.isna(end_idx):
            print(f"无法找到有效的日期区间，跳过该片段")
            return None
        else:
            print(f"使用近似日期区间: {full_data.loc[start_idx, 'timestamps'].date()} 至 {full_data.loc[end_idx, 'timestamps'].date()}")
    else:
        start_idx = full_data[start_mask].index[0]
        end_idx = full_data[end_mask].index[0]
    
    # 计算扩充后的起始索引
    extended_start_idx = max(0, start_idx - extend_rows)
    
    # 提取扩充后的数据
    extended_data = full_data.loc[extended_start_idx:end_idx, :].copy()
    
    # 添加stage列并初始化所有值为0（扩充数据）
    extended_data['stage'] = 0
    
    # 根据阶段类型设置主区间的stage值
    stage_value = 0
    if stage_type == 'stage1':
        stage_value = 1
    elif stage_type == 'stage2':
        stage_value = 2
    elif stage_type == 'stage3':
        stage_value = 3
    
    # 精确标记原始区间的stage值（只标记检测到的阶段部分）
    original_period_mask = (extended_data['timestamps'] >= start_date) & (extended_data['timestamps'] <= end_date)
    extended_data.loc[original_period_mask, 'stage'] = stage_value
    
    # 验证标记结果
    stage_count = extended_data[extended_data['stage'] == stage_value].shape[0]
    if stage_count == 0:
        print(f"警告: 未在扩充数据中找到需要标记为 {stage_value} 的阶段数据")
    else:
        print(f"成功标记 {stage_count} 条数据为阶段 {stage_value}")
    
    # 检查扩充数据量
    actual_extend = start_idx - extended_start_idx
    if actual_extend < extend_rows:
        print(f"注意: 股票 {stock_code} 数据不足，实际向前扩充 {actual_extend} 行（目标：{extend_rows}行）")
    
    return extended_data

def save_extended_data(extended_data, stock_code, start_date, end_date, output_folder, fragment_index):
    """保存扩充后的数据到指定文件夹"""
    try:
        # 创建输出文件夹（如果不存在）
        os.makedirs(output_folder, exist_ok=True)
        
        # 生成唯一的文件名
        start_str = pd.to_datetime(start_date).strftime('%Y%m%d')
        end_str = pd.to_datetime(end_date).strftime('%Y%m%d')
        filename = f"{stock_code}_fragment_{fragment_index}_{start_str}_to_{end_str}_extended.csv"
        file_path = os.path.join(output_folder, filename)
        
        # 保存数据
        extended_data.to_csv(file_path, index=False, encoding='utf-8-sig')
        print(f"已保存扩充数据至: {file_path}")
        return file_path  # 返回保存的文件路径
    except Exception as e:
        print(f"保存扩充数据时出错: {str(e)}")
        return None

def process_and_save_fragments(fragments, target_stage, data_folder, output_folder, extend_rows=60):
    """处理所有片段，扩充数据并保存，返回片段与文件路径的映射关系"""
    if not fragments:
        print("没有需要处理的片段")
        return {}
    
    print(f"\n开始处理 {len(fragments)} 个片段，每个片段向前扩充 {extend_rows} 行数据...")
    
    fragment_file_map = {}  # 用于存储片段索引与文件路径的映射
    
    for i, fragment in enumerate(fragments, 1):
        print(f"\n处理片段 {i}/{len(fragments)}: 股票 {fragment['stock_code']}, 区间 {fragment['start']} 至 {fragment['end']}")
        
        # 扩充数据，传入阶段类型用于标记
        extended_data = extend_data_backward(
            stock_code=fragment['stock_code'],
            start_date=fragment['start'],
            end_date=fragment['end'],
            stage_type=target_stage,
            data_folder=data_folder,
            extend_rows=extend_rows
        )
        
        # 保存数据并记录文件路径
        if extended_data is not None:
            file_path = save_extended_data(
                extended_data=extended_data,
                stock_code=fragment['stock_code'],
                start_date=fragment['start'],
                end_date=fragment['end'],
                output_folder=output_folder,
                fragment_index=i
            )
            if file_path:
                fragment_file_map[i-1] = file_path  # 使用索引作为键，与fragments列表对应
    
    return fragment_file_map


# 检测stage2的文件类别【√】
def process_csv_files_stage2(folder_path, fragment_file_map):
    """
    遍历指定的CSV文件并进行分类预测
    返回预测概率结果字典，键为片段索引
    """
    predictions_result = {}
    
    # 遍历每个片段对应的文件
    for fragment_idx, csv_file in fragment_file_map.items():
        if not os.path.exists(csv_file):
            print(f"错误: 文件不存在 - {csv_file}")
            continue
            
        print(f"\n处理文件: {csv_file}")
        
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_file)
            
            # 检查是否包含'stage'列
            if 'stage' not in df.columns:
                print(f"警告: 文件 {os.path.basename(csv_file)} 不包含'stage'列，跳过处理")
                continue
            
            # 统计stage=2的行数
            stage2_count = len(df[df['stage'] == 2])
            print(f"CSV文件中stage=2的行数: {stage2_count}")

            # 根据stage=2的行数选择模型
            if stage2_count > 4:
                model_path = r'D:\self\code\tool_ma_back\models\stage2_long_model.pkl'
                training_feature_path = r'D:\self\code\tool_ma_back\models\stage2_long_features.csv'
                print("使用长模型路径:", model_path)
            else:
                model_path = r'D:\self\code\tool_ma_back\models\stage2_short_model.pkl'
                training_feature_path = r'D:\self\code\tool_ma_back\models\stage2_short_features.csv'
                print("使用短模型路径:", model_path)

            # 执行预测
            predictions, probs = predict_from_csv(csv_file, model_path, training_feature_path)
            
            if predictions is not None and probs is not None:
                print(f"完成预测，共处理 {len(predictions)} 个样本")
                
                # 处理概率结果，转换为字典格式{class1:0.2, class2:0.1...}
                # 假设probs是一个二维数组，每行是一个样本的概率分布
                # 这里取第一个样本的概率分布作为结果
                if len(probs) > 0:
                    class_probs = {}
                    for i, prob in enumerate(probs[0]):
                        class_probs[f"class{i+1}"] = float(prob)
                    predictions_result[fragment_idx] = class_probs
                    
        except FileNotFoundError:
            print(f"错误: 文件未找到 - {csv_file}")
        except Exception as e:
            print(f"处理文件 {os.path.basename(csv_file)} 时发生错误: {e}")
    
    return predictions_result

# 检测stage1的文件类别
def process_csv_files_stage1(folder_path, fragment_file_map):
    """
    遍历指定的CSV文件并进行分类预测
    返回预测概率结果字典，键为片段索引
    """
    predictions_result = {}
    
    # 遍历每个片段对应的文件
    for fragment_idx, csv_file in fragment_file_map.items():
        if not os.path.exists(csv_file):
            print(f"错误: 文件不存在 - {csv_file}")
            continue
            
        print(f"\n处理文件: {csv_file}")
        
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_file)
            
            # 检查是否包含'stage'列
            if 'stage' not in df.columns:
                print(f"警告: 文件 {os.path.basename(csv_file)} 不包含'stage'列，跳过处理")
                continue
            
            # 统计stage=1的行数
            stage1_count = len(df[df['stage'] == 1])
            print(f"CSV文件中stage=1的行数: {stage1_count}")

            # 根据stage=1的行数选择模型
            if stage1_count > 10:
                model_path = r'D:\self\code\tool_ma_back\models\stage1_long_model.pkl'
                training_feature_path = r'D:\self\code\tool_ma_back\models\stage1_long_features.csv'
                print("使用长模型路径:", model_path)
            else:
                model_path = r'D:\self\code\tool_ma_back\models\stage1_short_model.pkl'
                training_feature_path = r'D:\self\code\tool_ma_back\models\stage1_short_features.csv'
                print("使用短模型路径:", model_path)

            # 执行预测
            predictions, probs = predict_from_csv1(csv_file, model_path, training_feature_path)
            
            if predictions is not None and probs is not None:
                print(f"完成预测，共处理 {len(predictions)} 个样本")
                
                # 处理概率结果，转换为字典格式{class1:0.2, class2:0.1...}
                if len(probs) > 0:
                    class_probs = {}
                    for i, prob in enumerate(probs[0]):
                        class_probs[f"class{i+1}"] = float(prob)
                    predictions_result[fragment_idx] = class_probs
                    
        except FileNotFoundError:
            print(f"错误: 文件未找到 - {csv_file}")
        except Exception as e:
            print(f"处理文件 {os.path.basename(csv_file)} 时发生错误: {e}")
    
    return predictions_result

# 检测stage3的文件类别
def process_csv_files_stage3(folder_path, fragment_file_map):
    """
    遍历指定的CSV文件并进行分类预测
    返回预测概率结果字典，键为片段索引
    """
    predictions_result = {}
    
    # 遍历每个片段对应的文件
    for fragment_idx, csv_file in fragment_file_map.items():
        if not os.path.exists(csv_file):
            print(f"错误: 文件不存在 - {csv_file}")
            continue
            
        print(f"\n处理文件: {csv_file}")
        
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_file)
            
            # 检查是否包含'stage'列
            if 'stage' not in df.columns:
                print(f"警告: 文件 {os.path.basename(csv_file)} 不包含'stage'列，跳过处理")
                continue
            
            # 统计stage=3的行数
            stage3_count = len(df[df['stage'] == 3])
            print(f"CSV文件中stage=3的行数: {stage3_count}")

            # 根据stage=3的行数选择模型
            if stage3_count > 10:
                model_path = r'D:\self\code\tool_ma_back\models\stage3_long_model.pkl'
                training_feature_path = r'D:\self\code\tool_ma_back\models\stage3_long_features.csv'
                print("使用长模型路径:", model_path)
            else:
                model_path = r'D:\self\code\tool_ma_back\models\stage3_short_model.pkl'
                training_feature_path = r'D:\self\code\tool_ma_back\models\stage3_short_features.csv'
                print("使用短模型路径:", model_path)

            # 执行预测
            predictions, probs = predict_from_csv3(csv_file, model_path, training_feature_path)
            
            if predictions is not None and probs is not None:
                print(f"完成预测，共处理 {len(predictions)} 个样本")
                
                # 处理概率结果，转换为字典格式{class1:0.2, class2:0.1...}
                if len(probs) > 0:
                    class_probs = {}
                    for i, prob in enumerate(probs[0]):
                        class_probs[f"class{i+1}"] = float(prob)
                    predictions_result[fragment_idx] = class_probs
                    
        except FileNotFoundError:
            print(f"错误: 文件未找到 - {csv_file}")
        except Exception as e:
            print(f"处理文件 {os.path.basename(csv_file)} 时发生错误: {e}")
    
    return predictions_result


def find_stage_segments0914(stock_pool, start_date, end_date, target_stage, ma_periods):
    data_folder = r"D:\self\data\final_data_0821"      # 原始数据文件夹路径
    output_folder = r"D:\self\code\tool_ma_back\aaa_fragments"  # 扩充后数据保存路径
    extend_rows = 60  # 向前扩充的行数
    duration_thresholds = {
        'stage1': 10,
        'stage2': 4,
        'stage3': 10
    }
    
    print(f"开始检测股票池 {stock_pool} 在 {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')} 期间的 {target_stage} 阶段...")
    
    # 执行检测
    results = detect_stage_fragments(
        stock_pool=stock_pool,
        start_date=start_date,
        end_date=end_date,
        target_stage=target_stage,
        ma_periods=ma_periods,
        data_folder=data_folder,
        duration_thresholds=duration_thresholds
    )
    
    # 输出检测结果摘要
    print(f"\n检测完成，共找到 {len(results)} 个符合 {target_stage} 的阶段片段：")
    for i, fragment in enumerate(results, 1):
        print(f"\n片段 {i}:")
        print(f"股票代码: {fragment['stock_code']}")
        print(f"时间区间: {fragment['start']} 至 {fragment['end']}")
        print(f"持续天数: {fragment['duration_days']} 天 ({fragment['duration_type']}周期)")
        if 'end_condition' in fragment:
            print(f"结束条件: {fragment['end_condition']}")
        if 'note' in fragment:
            print(f"备注: {fragment['note']}")
    
    # 处理并保存扩充数据，传入目标阶段用于标记，获取片段与文件路径的映射
    fragment_file_map = process_and_save_fragments(
        fragments=results,
        target_stage=target_stage,
        data_folder=data_folder,
        output_folder=output_folder,
        extend_rows=extend_rows
    )
    
    # 可选：将结果保存为CSV文件
    if results:
        # 转换为DataFrame时排除复杂数据类型
        simple_results = []
        for fragment in results:
            simple_frag = fragment.copy()
            # 移除不适合CSV存储的复杂数据
            if 'ma_data' in simple_frag:
                del simple_frag['ma_data']
            if 'feature_vector' in simple_frag:
                simple_frag['feature_vector'] = ','.join(map(str, simple_frag['feature_vector']))
            simple_results.append(simple_frag)
        
        result_df = pd.DataFrame(simple_results)
        output_file = f"{target_stage}_results_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.csv"
        result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n检测结果汇总已保存至 {output_file}")

    # 计算csv文件的分类结果并获取概率
    predictions_result = {}
    if target_stage == 'stage2' :
        predictions_result = process_csv_files_stage2(output_folder, fragment_file_map)
    elif target_stage == 'stage1':
        predictions_result = process_csv_files_stage1(output_folder, fragment_file_map)
    else :
        predictions_result = process_csv_files_stage3(output_folder, fragment_file_map)
    
    # 构建前端需要的结果格式
    frontend_results = []
    for i, fragment in enumerate(results):
        # 提取均线数据并转换为二维数组
        # 第一维是均线类型(MA4, MA8, MA12, MA16, MA20, MA47)
        # 第二维是对应均线在该区间内的数值
        ma_values = []
        # 遍历每种均线类型
        for period in ma_periods:
            ma_col = f'MA{period}'
            # 收集该均线在区间内的所有值
            ma_data = [entry[ma_col] for entry in fragment['ma_data'] if ma_col in entry]
            ma_values.append(ma_data)
        
        # 基础信息
        result_item = {
            'stock_code': fragment['stock_code'],
            'start_time': fragment['start'],
            'end_time': fragment['end'],
            'length_class': fragment['duration_type'],
            'probabilities': predictions_result.get(i, {}),  # 获取对应片段的概率，如果没有则为空字典
            'ma_values': ma_values  # 添加均线数据，二维数组格式
        }
        frontend_results.append(result_item)
    
    return frontend_results

def main():
    """测试主函数：演示如何使用阶段检测和数据扩充功能"""
    # 配置参数
    stock_pool = ["000021"]  # 示例股票池
    start_date = datetime(2022, 1, 1)            # 检测开始日期
    end_date = datetime(2023, 1, 1)              # 检测结束日期
    target_stage = "stage2"                      # 目标阶段（stage1、stage2或stage3）
    ma_periods = [4, 8, 12, 16, 20, 47]    
    
    # 获取前端需要的结果
    frontend_results = find_stage_segments0914(stock_pool, start_date, end_date, target_stage, ma_periods)
    
    # 打印前端需要的结果
    print("\n前端需要的结果：")
    import json
    print(json.dumps(frontend_results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
