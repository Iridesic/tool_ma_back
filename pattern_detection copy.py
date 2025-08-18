import pandas as pd
from datetime import timedelta, datetime
from typing import List, Dict
from utils import get_stock_data, get_stock_data1, calculate_ma, calculate_similarity, get_exchange_index, get_index_ma_values

# 查找相似模式
def find_similar_patterns(
    pool: List[Dict], 
    base_code: str, 
    base_start_date: str, 
    base_end_date: str, 
    start_date: str, 
    end_date: str, 
    ma_list: List[int], 
    min_similarity: float = 0.7
) -> Dict:
    result = {}
    all_five_day_avg_returns = []  # 存储所有相似区间的五日平均收益率
    
    max_ma = max(ma_list)
    base_start_obj = datetime.strptime(base_start_date.split('T')[0], '%Y-%m-%d')
    hist_start_date = (base_start_obj - timedelta(days=max_ma * 2)).strftime('%Y-%m-%d')
    
    # 处理基准股票代码
    ts_base_code = base_code + '.SH' if base_code.startswith('6') else base_code + '.SZ'
    base_data = get_stock_data1(ts_base_code, hist_start_date, base_end_date)
    if base_data.empty:
        print(f"基准股票 {ts_base_code} 历史数据为空")
        return result
    
    base_data = calculate_ma(base_data, ma_list)
    if base_data.empty:
        print(f"基准股票 {ts_base_code} 计算MA后数据为空")
        return result
    
    # 处理日期格式
    base_data['trade_date'] = pd.to_datetime(base_data['trade_date'])
    base_start_date_obj = pd.to_datetime(base_start_date.split('T')[0])
    base_end_date_obj = pd.to_datetime(base_end_date.split('T')[0])
    
    # 提取基准模式的 MA 数据
    base_pattern = base_data[
        (base_data['trade_date'] >= base_start_date_obj) & 
        (base_data['trade_date'] <= base_end_date_obj)
    ]
    if base_pattern.empty:
        print(f"基准时间段内没有MA数据: {base_start_date}~{base_end_date}")
        return result
    
    base_ma_values = base_pattern[[f'MA{ma}' for ma in ma_list]].values
    min_days = max(5, len(base_ma_values) // 2)
    
    if len(base_ma_values) < min_days:
        print(f"基准股票 {ts_base_code} 的数据长度不足: {len(base_ma_values)}")
        return result
    
    # 遍历股票池
    for item in pool:
        print(f"正在处理股票: {item}")
        # 兼容单股票或股票池字典格式
        ts_code = item.get('code') if isinstance(item, dict) else item  
        if not ts_code:
            print(f"股票信息中缺少 'code' 字段: {item}")
            continue
        
        # 处理股票代码后缀
        ts_code_full = ts_code + '.SH' if ts_code.startswith('6') else ts_code + '.SZ'
        
        # 扩展结束日期，用于获取后续 30 天数据
        extended_end_date_obj = pd.to_datetime(end_date.split('T')[0]) + timedelta(days=30)
        extended_end_date = extended_end_date_obj.strftime('%Y-%m-%d')
        
        # 获取股票数据
        stock_data = get_stock_data1(ts_code_full, start_date, extended_end_date)
        if stock_data.empty:
            print(f"股票 {ts_code_full} 数据为空")
            continue
        
        stock_data = calculate_ma(stock_data, ma_list)
        if stock_data.empty:
            print(f"股票 {ts_code_full} 计算MA后数据为空")
            continue
            
        stock_data['trade_date'] = pd.to_datetime(stock_data['trade_date'])
        
        # 检查数据长度
        if len(stock_data) < len(base_ma_values):
            print(f"股票 {ts_code_full} 的数据长度不足: {len(stock_data)} < {len(base_ma_values)}")
            continue
            
        similar_intervals = []
        # 滑动窗口查找相似区间
        for i in range(0, len(stock_data) - len(base_ma_values) + 1, 2):
            current_ma_values = stock_data[[f'MA{ma}' for ma in ma_list]].iloc[i:i+len(base_ma_values)].values
            similarity = calculate_similarity(base_ma_values.flatten(), current_ma_values.flatten())
            
            if similarity >= min_similarity:
                # 记录区间日期
                start = stock_data.iloc[i]['trade_date'].strftime('%Y-%m-%d')
                end = stock_data.iloc[i+len(base_ma_values)-1]['trade_date'].strftime('%Y-%m-%d')
                
                # 记录 MA 值和日期
                ma_values_in_interval = stock_data[[f'MA{ma}' for ma in ma_list]].iloc[i:i+len(base_ma_values)].values.tolist()
                x_axis_dates = stock_data['trade_date'].iloc[i:i+len(base_ma_values)].dt.strftime('%m%d').tolist()
                
                # 获取区间后 30 天数据（用于后续分析）
                post_start_idx = i + len(base_ma_values)
                post_end_idx = post_start_idx + 30
                post_end_idx = min(post_end_idx, len(stock_data))
                
                post_ma_values = []
                post_dates = []
                if post_start_idx < len(stock_data):
                    post_data = stock_data.iloc[post_start_idx:post_end_idx]
                    post_ma_values = post_data[[f'MA{ma}' for ma in ma_list]].values.tolist()
                    post_dates = post_data['trade_date'].dt.strftime('%m%d').tolist()
                
                # -----------------------
                # 关键修改：计算五日平均收益率（以模式出现后第一日为基准）
                # -----------------------
                five_days_end_idx = min(post_start_idx + 5, len(stock_data))
                five_days_data = stock_data.iloc[post_start_idx:five_days_end_idx]
                
                if not five_days_data.empty:
                    P0 = five_days_data.iloc[0]['close']  # 模式出现后第一日的收盘价
                    returns = []
                    for j in range(1, len(five_days_data)):
                        Pi = five_days_data.iloc[j]['close']
                        Ri = (Pi - P0) / P0 * 100  # 计算第j日相对于P0的收益率（百分比）
                        returns.append(Ri)
                    
                    # 计算平均收益率（不足5天按实际天数）
                    five_day_avg_return = sum(returns) / len(returns) if returns else 0
                else:
                    five_day_avg_return = 0
                
                all_five_day_avg_returns.append(five_day_avg_return)
                
                # 存入相似区间结果
                similar_intervals.append({
                    'start_date': start,
                    'end_date': end,
                    'similarity': float(similarity),
                    'ma_values': ma_values_in_interval,
                    'x_axis_dates': x_axis_dates,
                    'post_ma_values': post_ma_values,  
                    'post_dates': post_dates,  
                    'five_day_avg_return': five_day_avg_return  
                })
        
        # 筛选 top10 相似区间
        if similar_intervals:
            similar_intervals.sort(key=lambda x: x['similarity'], reverse=True)
            result[ts_code_full] = similar_intervals[:10]

    # 计算整体平均收益率
    overall_five_day_avg_return = (
        sum(all_five_day_avg_returns) / len(all_five_day_avg_returns) 
        if all_five_day_avg_returns 
        else 0
    )
    
    # 组装最终结果
    final_result = {
        'stock_patterns': result,
        'overall_five_day_avg_return': overall_five_day_avg_return
    }
    
    print(f"找到相似模式的股票数量: {len(result)}")
    return final_result



# 检测金叉的函数
def is_golden_cross(data, short_ma, long_ma):
    if len(data) < 2:
        return False
    ma_short = f'MA{short_ma}'
    ma_long = f'MA{long_ma}'
    prev_short = data[ma_short].iloc[-2]
    prev_long = data[ma_long].iloc[-2]
    curr_short = data[ma_short].iloc[-1]
    curr_long = data[ma_long].iloc[-1]
    return prev_short <= prev_long and curr_short > curr_long

# 检测均线多头排列的函数
def is_bullish_arrangement(data_segment, ma_list):
    """均线多头排列：短期均线在长期均线之上"""
    ma_columns = [f'MA{ma}' for ma in ma_list]
    for i in range(len(ma_columns) - 1):
        if data_segment[ma_columns[i]].iloc[-1] <= data_segment[ma_columns[i+1]].iloc[-1]:
            return False
    return True


def find_pattern_segments(pool, start_date, end_date, ma_list, pattern_func, *args, min_days=1, extend_days=3):
    result = {}
    all_five_day_avg_returns = []  # 用于存储所有金叉结果的后续五日平均收益率
    for item in pool:
        ts_code = item.get('code')
        if not ts_code:
            print(f"股票信息中缺少 'code' 字段: {item}")
            continue
        ts_code = ts_code + '.SH' if ts_code.startswith('6') else ts_code + '.SZ'
        data = get_stock_data1(ts_code, start_date, end_date)
        if data.empty:
            continue
        data = calculate_ma(data, ma_list)
        data['trade_date'] = pd.to_datetime(data['trade_date'])
        pattern_dates = []

        for i in range(len(data)):
            if i < min_days - 1:
                continue
            current_data = data.iloc[:i + 1]
            if args:
                match = pattern_func(current_data, *args)
            else:
                match = pattern_func(current_data, ma_list)
            if match:
                pattern_dates.append(data.iloc[i]['trade_date'])

        intervals = []
        if len(pattern_dates) >= 1:
            current_sequence = [pattern_dates[0]]
            for date in pattern_dates[1:]:
                if (date - current_sequence[-1]).days <= 1:
                    current_sequence.append(date)
                else:
                    if len(current_sequence) >= min_days:
                        intervals.append([
                            current_sequence[0].strftime('%Y%m%d'),
                            current_sequence[-1].strftime('%Y%m%d')
                        ])
                    current_sequence = [date]
            if len(current_sequence) >= min_days:
                intervals.append([
                    current_sequence[0].strftime('%Y%m%d'),
                    current_sequence[-1].strftime('%Y%m%d')
                ])

        filtered_intervals = []
        for interval in intervals:
            start = pd.to_datetime(interval[0])
            end = pd.to_datetime(interval[1])
            if (end - start).days + 1 >= min_days:
                filtered_intervals.append(interval)

        if filtered_intervals:
            # 提取区间内的均线数据，并扩展前后各extend_days天
            interval_data = []
            index_code = get_exchange_index(ts_code.split('.')[0])
            for interval in filtered_intervals:
                # 将字符串日期转换为datetime对象
                start_date_dt = pd.to_datetime(interval[0])
                end_date_dt = pd.to_datetime(interval[1])
                
                # 计算扩展后的日期范围
                extended_start = (start_date_dt - timedelta(days=extend_days)).strftime('%Y%m%d')
                extended_end = (end_date_dt + timedelta(days=extend_days)).strftime('%Y%m%d')
                
                # 提取扩展后的区间数据
                interval_df = data[(data['trade_date'] >= extended_start) & (data['trade_date'] <= extended_end)]
                
                # 确保数据不为空
                if not interval_df.empty:
                    ma_data = interval_df[['trade_date', 'MA4', 'MA12', 'MA20']].copy()
                    # 将日期转换为字符串格式
                    ma_data['trade_date'] = ma_data['trade_date'].dt.strftime('%Y%m%d')
                    ma_data = ma_data.values.tolist()
                    
                    # 获取区间内的指数MA值
                    index_ma_values = get_index_ma_values(index_code, extended_start, extended_end, ma_list)
                    
                    # 获取区间后30天的数据
                    post_start_date = (end_date_dt + timedelta(days=1)).strftime('%Y-%m-%d')
                    post_end_date = (end_date_dt + timedelta(days=30)).strftime('%Y-%m-%d')
                    post_index_ma_values = get_index_ma_values(index_code, post_start_date, post_end_date, ma_list)
                    
                    # 获取未来30天的股票均线数据
                    post_stock_data = data[(data['trade_date'] > end_date_dt) & (data['trade_date'] <= (end_date_dt + timedelta(days=30)))]
                    if not post_stock_data.empty:
                        post_ma_data = post_stock_data[['trade_date', 'MA4', 'MA12', 'MA20']].copy()
                        post_ma_data['trade_date'] = post_ma_data['trade_date'].dt.strftime('%Y%m%d')
                        post_ma_data = post_ma_data.values.tolist()
                        post_dates = [row[0] for row in post_ma_data]
                    else:
                        post_ma_data = []
                        post_dates = []
                    
                    # 获取该股票在查找结果后五天内的收盘价数据
                    five_days_end_idx = min(data.index[data['trade_date'] == end_date_dt].tolist()[0] + 5, len(data))
                    five_days_data = data.iloc[data.index[data['trade_date'] == end_date_dt].tolist()[0]:five_days_end_idx]
                    if len(five_days_data) > 1:
                        # 计算后续五日的平均收益率
                        close_prices = five_days_data['close'].values
                        returns = [(close_prices[i] - close_prices[i-1]) / close_prices[i-1] for i in range(1, len(close_prices))]
                        five_day_avg_return = sum(returns) / len(returns) if returns else 0
                        all_five_day_avg_returns.append(five_day_avg_return)
                    else:
                        five_day_avg_return = 0
                    
                    interval_data.append({
                        "interval": [extended_start, extended_end],
                        "original_interval": interval,  # 保留原始检测到的区间
                        "ma_data": ma_data,
                        "index_code": index_code,
                        "index_ma_values": index_ma_values,
                        "index_name": "上证指数" if index_code == '000001' else "深证成指",
                        "post_index_ma_values": post_index_ma_values,
                        "post_stock_ma_data": post_ma_data,  # 新增：未来30天的股票均线数据
                        "post_stock_dates": post_dates,  # 新增：未来30天的股票日期
                        "five_day_avg_return": five_day_avg_return  # 新增：后续五日平均收益率
                    })

            if interval_data:
                result[ts_code] = interval_data[:5]

    # 计算整体查找结果后续五日平均收益率
    overall_five_day_avg_return = sum(all_five_day_avg_returns) / len(all_five_day_avg_returns) if all_five_day_avg_returns else 0
    
    # 将整体查找结果后续五日平均收益率加入到返回结果中
    final_result = {
        'stock_patterns': result,
        'overall_five_day_avg_return': overall_five_day_avg_return
    }
    
    return final_result