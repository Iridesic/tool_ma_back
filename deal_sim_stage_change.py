import shutil
import joblib
import pandas as pd
import os
from datetime import datetime
from setter import prepare_data_for_gp
import chardet
from utils import get_stock_data, calculate_ma

def detect_bullish_arrangement(stock_pool, start_date, end_date, ma_periods=[4, 8, 12, 16, 20, 47], data_folder="D:/self/data/kline-data"):
    """
    检测股票池中各股票在指定时间区间内均线呈现多头排列的区间，并划分stage1、stage2和stage3
    结果按阶段分类，包含时间区间、股票代码和对应均线数据
    
    阶段定义：
    - stage1: 从"4条以上均线呈现上升趋势"的起始位置到stage2起始位置之前一个时间点
              （持续时间需大于3个交易日）
    - stage2: 符合多头排列的区间（持续时间需大于3个交易日）
    - stage3: 在满足"结束时只有MA4下降，其余均线不下降"的stage2之后，
              从stage2结束的下一个交易日开始，直至检测到"MA4下降且其余均线也下降"的交易日为止
    
    参数:
        stock_pool: 股票代码数组
        start_date: 检测起始时间（YYYY-MM-DD）
        end_date: 检测终止时间（YYYY-MM-DD）
        ma_periods: 均线周期列表，默认[4,8,12,16,20,47]
        data_folder: 股票数据CSV文件存放路径
    
    返回:
        dict: 按阶段分类的结果，包含各阶段的时间区间、股票代码和均线数据
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
            
            # 确保数据按时间排序
            df_with_ma = df_with_ma.sort_values('trade_date').reset_index(drop=True)
            # 确保trade_date为 datetime 类型以便比较
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
                        
                        # 提取日期
                        stage1_start_date = full_data.loc[stage1_start_idx, 'trade_date']
                        stage1_end_date = full_data.loc[current_stage2_start - 1, 'trade_date']
                        stage2_start_date = full_data.loc[current_stage2_start, 'trade_date']
                        stage2_end_date = full_data.loc[i-1, 'trade_date']
                        
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
                            # 筛选stage1期间的数据
                            stage1_mask = (full_data['trade_date'] >= stage1_start_date) & \
                                         (full_data['trade_date'] <= stage1_end_date)
                            stage1_data = full_data.loc[stage1_mask, ['trade_date'] + [f'MA{period}' for period in ma_periods]]
                            
                            # 转换为字典列表以便存储
                            ma_values = stage1_data.to_dict('records')
                            
                            stock_stages['stage1'].append({
                                'stock_code': stock_code,
                                'start': stage1_start_date.strftime('%Y-%m-%d'),
                                'end': stage1_end_date.strftime('%Y-%m-%d'),
                                'duration_days': stage1_days,
                                'ma_data': ma_values
                            })
                        
                        # 提取stage2的均线数据
                        if valid_stage2:
                            # 筛选stage2期间的数据
                            stage2_mask = (full_data['trade_date'] >= stage2_start_date) & \
                                         (full_data['trade_date'] <= stage2_end_date)
                            stage2_data = full_data.loc[stage2_mask, ['trade_date'] + [f'MA{period}' for period in ma_periods]]
                            
                            # 转换为字典列表以便存储
                            ma_values = stage2_data.to_dict('records')
                            
                            stock_stages['stage2'].append({
                                'stock_code': stock_code,
                                'start': stage2_start_date.strftime('%Y-%m-%d'),
                                'end': stage2_end_date.strftime('%Y-%m-%d'),
                                'duration_days': stage2_days,
                                'end_condition': f"结束时下降的均线: {dropping_ma}",
                                'ma_data': ma_values
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
                stage1_start_date = full_data.loc[stage1_start_idx, 'trade_date']
                stage1_end_date = full_data.loc[current_stage2_start - 1, 'trade_date']
                stage2_start_date = full_data.loc[current_stage2_start, 'trade_date']
                stage2_end_date = full_data.loc[len(full_data)-1, 'trade_date']
                
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
                    stage1_mask = (full_data['trade_date'] >= stage1_start_date) & \
                                 (full_data['trade_date'] <= stage1_end_date)
                    stage1_data = full_data.loc[stage1_mask, ['trade_date'] + [f'MA{period}' for period in ma_periods]]
                    
                    ma_values = stage1_data.to_dict('records')
                    
                    stock_stages['stage1'].append({
                        'stock_code': stock_code,
                        'start': stage1_start_date.strftime('%Y-%m-%d'),
                        'end': stage1_end_date.strftime('%Y-%m-%d'),
                        'duration_days': stage1_days,
                        'ma_data': ma_values
                    })
                
                # 添加有效的stage2
                if valid_stage2:
                    # 筛选stage2期间的数据
                    stage2_mask = (full_data['trade_date'] >= stage2_start_date) & \
                                 (full_data['trade_date'] <= stage2_end_date)
                    stage2_data = full_data.loc[stage2_mask, ['trade_date'] + [f'MA{period}' for period in ma_periods]]
                    
                    ma_values = stage2_data.to_dict('records')
                    
                    stock_stages['stage2'].append({
                        'stock_code': stock_code,
                        'start': stage2_start_date.strftime('%Y-%m-%d'),
                        'end': stage2_end_date.strftime('%Y-%m-%d'),
                        'duration_days': stage2_days,
                        'note': '区间持续至检测结束',
                        'end_condition': f"结束时下降的均线: {dropping_ma}",
                        'ma_data': ma_values
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
                    stage3_end_date = full_data.loc[stage3_end_idx, 'trade_date']
                    stage3_days = stage3_end_idx - stage3_start_idx + 1
                    
                    # 筛选stage3期间的均线数据
                    stage3_mask = (full_data['trade_date'] >= stage3_start_date) & \
                                 (full_data['trade_date'] <= stage3_end_date)
                    stage3_data = full_data.loc[stage3_mask, ['trade_date'] + [f'MA{period}' for period in ma_periods]]
                    
                    ma_values = stage3_data.to_dict('records')
                    
                    stock_stages['stage3'].append({
                        'stock_code': stock_code,
                        'start': stage3_start_date.strftime('%Y-%m-%d'),
                        'end': stage3_end_date.strftime('%Y-%m-%d'),
                        'duration_days': stage3_days,
                        'note': f"跟随于 {stage2_end_date.strftime('%Y-%m-%d')} 结束的stage2之后",
                        'ma_data': ma_values
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
                   extended_days=100):
    """
    判断给定股票在指定时间区间属于哪个阶段（stage1、stage2或stage3）
    修正：确保stage3优先级最高，其次是stage2，最后是stage1
    stage3定义：前面有合格的stage2，且区间内MA4可能下降但其他均线未同时下降，
               直到最后出现MA4下降且其他均线也下降的情况
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
            return 'unknown', {"error": f"股票 {stock_code} 在扩展区间内无有效数据"}
        
        # 2. 基于扩展数据计算均线
        df_with_ma = calculate_ma(df.copy(), ma_periods)
        if df_with_ma.empty:
            return 'unknown', {"error": f"股票 {stock_code} 均线计算失败"}
        
        # 数据预处理（使用timestamps列名）
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
            }
        
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
        
        # 查找是否有前置的合格stage2
        has_qualified_stage2 = False
        if is_stage3:
            # 查找目标区间之前是否存在符合条件的stage2
            pre_mask = (df_with_ma['timestamps'] < original_start)
            pre_data = df_with_ma[pre_mask].copy()
            
            if len(pre_data) >= 2:
                # 从pre_data的末尾向前查找可能的stage2结束点
                for i in range(len(pre_data)-1, 0, -1):
                    current_idx = pre_data.index[i]
                    prev_idx = pre_data.index[i-1]
                    
                    # 检查是否是stage2结束时只有MA4下降
                    dropping_ma = []
                    for period in ma_periods:
                        ma_col = f'MA{period}'
                        if df_with_ma.loc[current_idx, ma_col] <= df_with_ma.loc[prev_idx, ma_col]:
                            dropping_ma.append(period)
                    
                    if len(dropping_ma) == 1 and 4 in dropping_ma:
                        # 找到符合条件的stage2结束点
                        has_qualified_stage2 = True
                        break
        
        # 确保stage3必须有前置的合格stage2
        if is_stage3 and not has_qualified_stage2:
            is_stage3 = False
            stage3_evidence.append("未找到符合条件的前置stage2，不符合stage3特征")
        
        if is_stage3 and has_qualified_stage2:
            return 'stage3', {
                "evidence": "区间内符合stage3特征：前面有合格的stage2，MA4可能下降但其他均线未同时下降，且区间末尾出现MA4和其他均线同时下降",
                "duration_days": len(target_data),
                "使用区间": f"{original_start.strftime('%Y-%m-%d')} 至 {original_end.strftime('%Y-%m-%d')}"
            }
        
        # 其次检查是否符合stage2特征
        is_stage2 = True
        stage2_evidence = []
        
        for i in range(1, len(target_data)):
            current_idx = target_data.index[i]
            prev_idx = target_data.index[i-1]
            
            # 检查均线排列顺序
            order_valid = True
            for j in range(len(ma_periods) - 1):
                current_ma = f'MA{ma_periods[j]}'
                next_ma = f'MA{ma_periods[j+1]}'
                if df_with_ma.loc[current_idx, current_ma] <= df_with_ma.loc[current_idx, next_ma]:
                    order_valid = False
                    break
            
            # 检查所有均线是否向上
            all_rising = True
            for period in ma_periods:
                ma_col = f'MA{period}'
                if df_with_ma.loc[current_idx, ma_col] <= df_with_ma.loc[prev_idx, ma_col]:
                    all_rising = False
                    break
            
            if not (order_valid and all_rising):
                is_stage2 = False
                stage2_evidence.append(f"在 {df_with_ma.loc[current_idx, 'timestamps'].strftime('%Y-%m-%d')} 不满足多头排列条件")
                break
        
        if is_stage2:
            last_idx = target_data.index[-1]
            prev_last_idx = target_data.index[-2] if len(target_data) > 1 else target_data.index[-1]
            
            dropping_ma = []
            for period in ma_periods:
                ma_col = f'MA{period}'
                if df_with_ma.loc[last_idx, ma_col] <= df_with_ma.loc[prev_last_idx, ma_col]:
                    dropping_ma.append(period)
            
            return 'stage2', {
                "evidence": "区间内所有交易日均满足多头排列条件",
                "end_condition": f"结束时下降的均线: {dropping_ma}",
                "duration_days": len(target_data),
                "使用区间": f"{original_start.strftime('%Y-%m-%d')} 至 {original_end.strftime('%Y-%m-%d')}"
            }
        
        # 最后检查是否符合stage1特征
        is_stage1 = True
        stage1_evidence = []
        
        for i in range(1, len(target_data)):
            current_idx = target_data.index[i]
            prev_idx = target_data.index[i-1]
            
            # 计算上升均线数量
            rising_count = 0
            for period in ma_periods:
                ma_col = f'MA{period}'
                if df_with_ma.loc[current_idx, ma_col] > df_with_ma.loc[prev_idx, ma_col]:
                    rising_count += 1
            
            if rising_count < 4:
                is_stage1 = False
                stage1_evidence.append(f"在 {df_with_ma.loc[current_idx, 'timestamps'].strftime('%Y-%m-%d')} 上升均线不足4条")
                break
        
        if is_stage1:
            return 'stage1', {
                "evidence": "区间内所有交易日均有4条以上均线上升",
                "duration_days": len(target_data),
                "使用区间": f"{original_start.strftime('%Y-%m-%d')} 至 {original_end.strftime('%Y-%m-%d')}"
            }
        
        # 如果都不符合
        return 'unknown', {
            "evidence": "不符合任何阶段的定义特征",
            "stage1_check": not is_stage1,
            "stage2_check": not is_stage2,
            "stage3_check": not is_stage3 or not has_qualified_stage2,
            "使用区间": f"{original_start.strftime('%Y-%m-%d')} 至 {original_end.strftime('%Y-%m-%d')}"
        }
    
    except Exception as e:
        return 'unknown', {"error": f"判断过程出错: {str(e)}"}
    

# 从 csv 文件中获取历史50日的数据结果
def process_stock_data(folder_path, stock_code, start_date, end_date):
    """
    处理股票数据，提取指定时间区间及历史50条数据
    
    参数:
        folder_path: 存放CSV文件的文件夹路径
        stock_code: 股票代码（如'000021'）
        start_date: 开始日期（字符串，格式'YYYY-MM-DD'）
        end_date: 结束日期（字符串，格式'YYYY-MM-DD'）
    
    返回:
        包含指定时间区间数据和扩展数据集合的字典
    """
    # 构建CSV文件路径
    file_name = f"{stock_code}.csv"
    file_path = os.path.join(folder_path, file_name)
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到股票代码为{stock_code}的文件: {file_path}")
    
    # 读取CSV文件，假设CSV包含'date'列作为日期
    try:
        # 尝试不同的日期解析方式
        df = pd.read_csv(
            file_path,
            parse_dates=['timestamps'],
            date_parser=lambda x: datetime.strptime(x, '%Y-%m-%d')
        )
    except ValueError:
        # 如果解析失败，尝试其他常见格式
        df = pd.read_csv(
            file_path,
            parse_dates=['timestamps'],
            date_parser=lambda x: datetime.strptime(x, '%Y/%m/%d')
        )
    
    # 确保数据按日期排序
    df = df.sort_values('timestamps').reset_index(drop=True)

    # 转换输入日期为datetime对象
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    # 提取指定时间区间的数据
    mask = (df['timestamps'] >= start_dt) & (df['timestamps'] <= end_dt)
    target_data = df.loc[mask].copy()
    
    if target_data.empty:
        raise ValueError(f"在{start_date}至{end_date}区间内没有找到数据")
    
    # 找到目标区间的起始索引
    first_target_index = target_data.index.min()
    
    # 计算历史数据的起始索引（确保不小于0）
    history_start_index = max(0, first_target_index - 50)
    
    # 提取历史数据（目标区间前的50条数据）
    history_data = df.loc[history_start_index:first_target_index - 1].copy()
    
    # 构建包含历史数据和目标区间数据的集合
    extended_data = pd.concat([history_data, target_data]).reset_index(drop=True)
    
    # 构建结果数据结构
    result = {
        'stock_code': stock_code,
        'time_range': {
            'start': start_date,
            'end': end_date
        },
        'target_data': target_data.to_dict('records'),
        'history_data': history_data.to_dict('records'),
        'extended_data': extended_data.to_dict('records'),
        'history_count': len(history_data),
        'target_count': len(target_data),
        'extended_count': len(extended_data)
    }
    
    return result


## ############################################ 分类预测函数 ####################################

training_feature_path = ''

def load_training_feature_columns(column_path=training_feature_path):
    """加载训练时使用的原始特征列名"""
    if not os.path.exists(column_path):
        raise FileNotFoundError(f"训练特征列名文件 {column_path} 不存在，请先运行训练脚本生成")
    return pd.read_csv(column_path)['feature'].tolist()

def predict_from_data(stage1_data, history_data, merged_data, model_path, training_feature_path):
    """
    从外部传入的数据进行预测
    参数:
        stage1_data: stage==1的当前数据（含timestamps列）
        history_data: 历史50天数据
        merged_data: 合并后的数据
        model_path: 模型路径
    返回:
        预测结果和概率
    """
    try:
        # 加载模型管道
        pipeline = joblib.load(model_path)
        print(f"成功加载模型: {model_path}")
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        return None, None

    try:
        # 验证输入数据格式
        for df, name in [(stage1_data, "stage1数据"), (history_data, "历史数据"), (merged_data, "合并数据")]:
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"{name}必须是pandas DataFrame类型")
            if 'timestamps' not in df.columns:
                raise ValueError(f"{name}中缺少'timestamps'列")
            if 'code' not in df.columns:
                raise ValueError(f"{name}中缺少'code'列")

        # 保存临时文件用于特征提取（使用合并数据生成特征）
        temp_folder = 'temp_prediction_data'
        os.makedirs(temp_folder, exist_ok=True)
        
        stock_code = merged_data['code'].iloc[0]
        stock_folder = os.path.join(temp_folder, str(stock_code))
        os.makedirs(stock_folder, exist_ok=True)
        
        temp_file_path = os.path.join(stock_folder, 'data.csv')
        merged_data.to_csv(temp_file_path, index=False)
        
        try:
            # 提取原始特征
            features_df, _ = prepare_data_for_gp(temp_folder)
            
            if features_df is None or features_df.empty:
                print("直接传入文件夹路径未能提取到特征，尝试传入文件所在的父文件夹...")
                features_df, _ = prepare_data_for_gp(stock_folder)
                
                if features_df is None or features_df.empty:
                    print("仍然无法提取有效特征，检查prepare_data_for_gp函数实现")
                    return None, None
        except Exception as e:
            print(f"特征提取过程中出错: {str(e)}")
            return None, None
            
        # 清理临时文件
        try:
            shutil.rmtree(temp_folder)
        except Exception as e:
            print(f"清理临时文件时出错: {str(e)}")
        
        # 加载训练特征列并对齐
        training_feature_columns = load_training_feature_columns(column_path=training_feature_path)
        
        # 补充缺失列
        missing_cols = [col for col in training_feature_columns if col not in features_df.columns]
        if missing_cols:
            # print(f"补充缺失的原始特征: {missing_cols}")
            for col in missing_cols:
                features_df[col] = 0
        
        # 删除多余列
        extra_cols = [col for col in features_df.columns if col not in training_feature_columns]
        if extra_cols:
            # print(f"删除多余的原始特征: {extra_cols}")
            features_df = features_df.drop(columns=extra_cols)
        
        # 筛选与当前stage1数据对应的特征
        if 'timestamps' in features_df.columns:
            final_features = features_df[features_df['timestamps'].isin(stage1_data['timestamps'])]
            final_features = final_features.drop(columns=['timestamps'], errors='ignore')
        elif isinstance(features_df.index, pd.DatetimeIndex):
            final_features = features_df.loc[stage1_data['timestamps'].tolist()]
        else:
            final_features = features_df
            
        if final_features.empty:
            print("筛选后没有剩余有效特征")
            return None, None
            
        # 确保特征顺序一致
        final_features = final_features[training_feature_columns]
        
        print(f"提取的原始特征形状: {features_df.shape}")
        print(f"对齐后用于预测的特征形状: {final_features.shape}")
        
        # 进行预测
        predictions = pipeline.predict(final_features)
        prediction_probs = pipeline.predict_proba(final_features)
        class_labels = pipeline.classes_

        # 输出预测结果
        for date, pred_idx, probs in zip(stage1_data['timestamps'], predictions, prediction_probs):
            class_label = class_labels[pred_idx]
            print(f"日期: {date.date()}")
            print(f"  预测类别: {class_label}")
            for class_name, prob in zip(class_labels, probs):
                print(f"  类别 {class_name} 的概率: {prob:.4f}")
            print()
                
        return predictions, prediction_probs
        
    except Exception as e:
        print(f"预测过程中出错: {str(e)}")
        return None, None















# 示例用法 ##############################
# 【1】基准股票 + 起始时间 + 结束时间
# 【2】股票池 + 起始时间 + 结束时间
# 【3】stage匹配的结果汇总
# ######################################
if __name__ == "__main__":
    # 1. 首先检测目标股票的阶段
    test_stock = '000021'
    test_start = '2023-06-07'
    test_end = '2023-06-15'

    stage, details = determine_stage(test_stock, test_start, test_end)
    
    print(f"股票 {test_stock} 在 {test_start} 至 {test_end} 期间属于: {stage}")
    print("判断依据:")
    for key, value in details.items():
        print(f"  {key}: {value}")

    # 2. 示例检测股票池的所有阶段
    test_stock_pool = ['000561', '000830', '000547', '002057', '002092', '002106', '002119', '002362']
    test_start_date = '2020-01-01'
    test_end_date = '2024-12-31'
    
    results = detect_bullish_arrangement(
        stock_pool=test_stock_pool,
        start_date=test_start_date,
        end_date=test_end_date
    )
    
    # 3. 只输出与目标股票相同阶段的结果
    print("\n===== 检测结果汇总 =====")
    
    # 检查目标股票的阶段是否有效
    if stage in ['stage1', 'stage2', 'stage3']:
        print(f"\n与目标股票相同阶段 ({stage}) 的区间 ({len(results[stage])} 个):")
        if results[stage]:
            for item in results[stage]:
                print(f"  股票 {item['stock_code']}: {item['start']} 至 {item['end']} "
                      f"(持续 {item['duration_days']} 天)")
        else:
            print(f"  股票池中未发现属于 {stage} 的区间")
    else:
        print(f"  目标股票属于未知阶段 ({stage})，无法筛选对应结果")
