import pandas as pd
from datetime import timedelta
import os
import matplotlib.pyplot as plt
from utils import get_stock_data, calculate_ma

plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 检测均线多头排列的函数（保持不变）
def is_bullish_arrangement(data_segment, ma_list):
    """检查整个区间内是否完全符合多头排列定义：
    1. 短期均线始终在长期均线之上
    2. 所有均线在区间内持续上升
    3. 无交叉情况
    """
    ma_columns = [f'MA{ma}' for ma in ma_list]
    
    # 检查区间长度
    if len(data_segment) < 2:
        return False
    
    # 检查每一天的均线顺序
    for i in range(len(data_segment)):
        for j in range(len(ma_columns) - 1):
            if data_segment[ma_columns[j]].iloc[i] <= data_segment[ma_columns[j + 1]].iloc[i]:
                return False
    
    # 检查每一天的均线趋势（是否持续上升）
    for i in range(1, len(data_segment)):
        for ma_col in ma_columns:
            if data_segment[ma_col].iloc[i] <= data_segment[ma_col].iloc[i-1]:
                return False
    return True

# 检测4日均线下穿20日均线的函数
def find_ma4_cross_ma20(data, start_idx):
    """从指定位置开始检测，直到4日均线下穿20日均线"""
    if start_idx >= len(data) - 1:
        return None
    
    for i in range(start_idx, len(data) - 1):
        # 检查是否下穿
        if data['MA4'].iloc[i] > data['MA20'].iloc[i] and data['MA4'].iloc[i+1] < data['MA20'].iloc[i+1]:
            return i + 1  # 返回下穿发生的索引
    return None  # 未找到下穿点

# 提取多头排列经典区间的函数（修改版）
def find_pattern_segments(pool, start_date, end_date, ma_list=[4, 8, 12, 16, 20, 47], min_days=1, extend_days=10):
    result = {}
    # 创建 plots1 文件夹（多头排列区间）和 plots2 文件夹（包含结束阶段）
    for folder in ['plots1', 'plots2', 'csv_data']:
        if not os.path.exists(folder):
            os.makedirs(folder)

    for item in pool:
        ts_code = item.get('code')
        if not ts_code:
            print(f"股票信息中缺少 'code' 字段: {item}")
            continue
        ts_code = ts_code + '.SH' if ts_code.startswith('6') else ts_code + '.SZ'
        data = get_stock_data(ts_code, start_date, end_date)
        if data.empty:
            continue
        data = calculate_ma(data, ma_list)
        data['trade_date'] = pd.to_datetime(data['trade_date'])
        
        # 优化区间检测逻辑，避免重复检测
        intervals = []
        i = 0
        while i < len(data):
            # 找到区间起始点
            current_window = max(min_days, 2)  # 最小窗口长度
            if i + current_window > len(data):
                break
                
            # 检查当前窗口是否满足多头排列
            window_data = data.iloc[i:i+current_window]
            if not is_bullish_arrangement(window_data, ma_list):
                i += 1  # 若不满足，向前移动一个位置
                continue
            
            # 尝试扩展区间至最大
            j = i + current_window
            while j < len(data):
                extended_data = data.iloc[i:j+1]
                if is_bullish_arrangement(extended_data, ma_list):
                    j += 1
                else:
                    break
            
            # 确认区间结束后的第一个交易日不满足条件
            if j < len(data):
                next_day_data = data.iloc[i:j+1]  # 包含区间结束后的第一个交易日
                if is_bullish_arrangement(next_day_data, ma_list):
                    # 如果下一天仍满足，继续扩展
                    j += 1
                    continue
            
            # 记录有效区间（至少需要min_days天）
            if j - i >= min_days:
                intervals.append([
                    data.iloc[i]['trade_date'].strftime('%Y%m%d'),
                    data.iloc[j-1]['trade_date'].strftime('%Y%m%d'),
                    i,  # 保存起始索引
                    j-1  # 保存结束索引
                ])
            
            # 关键修改：从区间结束后继续搜索，避免重复检测
            i = j  # 直接跳到区间结束后的下一个位置
        
        # 过滤区间（保持不变）
        filtered_intervals = []
        for interval in intervals:
            start = pd.to_datetime(interval[0])
            end = pd.to_datetime(interval[1])
            if (end - start).days + 1 >= min_days:
                filtered_intervals.append(interval)

        if filtered_intervals:
            # 提取区间内的均线数据，并扩展前后各extend_days天
            interval_data = []
            for interval in filtered_intervals:
                # 解析区间信息
                start_str, end_str, start_idx, end_idx = interval
                
                # 计算扩展后的日期范围（多头排列区间）
                start_date_dt = pd.to_datetime(start_str)
                end_date_dt = pd.to_datetime(end_str)
                extended_start = (start_date_dt - timedelta(days=extend_days)).strftime('%Y%m%d')
                extended_end = (end_date_dt + timedelta(days=extend_days)).strftime('%Y%m%d')
                
                # 提取多头排列区间数据
                bullish_df = data[(data['trade_date'] >= extended_start) & (data['trade_date'] <= extended_end)]
                
                # 寻找结束阶段（从多头排列结束点开始，到4日均线下穿20日均线）
                # 确保从多头排列结束点之后开始检测
                end_phase_end_idx = find_ma4_cross_ma20(data, end_idx + 1)
                
                # 如果找到结束阶段，扩展图表范围
                if end_phase_end_idx is not None:
                    # 计算结束阶段的扩展日期
                    end_phase_extended_end = (data.iloc[end_phase_end_idx]['trade_date'] + timedelta(days=extend_days)).strftime('%Y%m%d')
                    
                    # 提取包含结束阶段的完整数据
                    full_df = data[(data['trade_date'] >= extended_start) & (data['trade_date'] <= end_phase_extended_end)]
                    
                    # 提取结束阶段数据
                    end_phase_df = data[(data['trade_date'] > end_str) & (data['trade_date'] <= end_phase_extended_end)]
                    
                    # 为包含结束阶段的数据绘图
                    plot_full_data(full_df, ts_code, extended_start, end_phase_extended_end, end_str, 'plots2')
                    
                    # 标记 stage 列
                    full_df['stage'] = 0
                    full_df.loc[(full_df['trade_date'] >= start_str) & (full_df['trade_date'] <= end_str), 'stage'] = 2
                    full_df.loc[(full_df['trade_date'] > end_str) & (full_df['trade_date'] <= data.iloc[end_phase_end_idx]['trade_date']), 'stage'] = 4
                    
                    # 保存为 CSV 文件
                    csv_filename = f'csv_data/{ts_code}_{extended_start}_{end_phase_extended_end}.csv'
                    csv_df = full_df[['trade_date', 'MA4', 'MA8', 'MA12', 'MA16', 'MA20', 'MA47', 'stage']].copy()
                    csv_df.rename(columns={'trade_date': 'timestamps'}, inplace=True)
                    csv_df['code'] = ts_code
                    csv_df = csv_df[['timestamps', 'code', 'MA4', 'MA8', 'MA12', 'MA16', 'MA20', 'MA47', 'stage']]
                    csv_df.to_csv(csv_filename, index=False)
                
                # 为多头排列区间绘图（保持不变）
                if not bullish_df.empty:
                    ma_data = bullish_df[['trade_date', 'MA4', 'MA8', 'MA12', 'MA16', 'MA20', 'MA47']].copy()
                    ma_data['trade_date'] = ma_data['trade_date'].dt.strftime('%Y%m%d')
                    ma_data = ma_data.values.tolist()

                    interval_data.append({
                        "interval": [extended_start, extended_end],
                        "original_interval": [start_str, end_str],
                        "ma_data": ma_data
                    })

                    # 绘图并保存（多头排列区间）
                    ma_df = pd.DataFrame(ma_data, columns=['trade_date', 'MA4', 'MA8', 'MA12', 'MA16', 'MA20', 'MA47'])
                    ma_df['trade_date'] = pd.to_datetime(ma_df['trade_date'])
                    plt.figure(figsize=(12, 6))
                    for ma in ma_list:
                        plt.plot(ma_df['trade_date'], ma_df[f'MA{ma}'], label=f'MA{ma}')
                    plt.title(f'{ts_code} 多头排列区间 {extended_start} - {extended_end} 均线图')
                    plt.xlabel('日期')
                    plt.ylabel('均线值')
                    plt.legend()
                    filename = f'plots1/{ts_code}_{extended_start}_{extended_end}.png'
                    plt.savefig(filename)
                    plt.close()

            if interval_data:
                result[ts_code] = interval_data[:5]

    return result

# 绘制包含结束阶段的完整数据图表
def plot_full_data(data, ts_code, start_date, end_date, bullish_end_date, folder):
    """绘制包含多头排列区间和结束阶段的完整图表"""
    ma_columns = ['MA4', 'MA8', 'MA12', 'MA16', 'MA20', 'MA47']
    
    plt.figure(figsize=(14, 7))
    
    # 绘制所有均线
    for ma in ma_columns:
        plt.plot(data['trade_date'], data[ma], label=ma)
    
    # 标记多头排列结束点
    end_point = data[data['trade_date'] == bullish_end_date].iloc[0]
    plt.axvline(x=end_point['trade_date'], color='r', linestyle='--', label='多头排列结束')
    
    # 标记4日均线下穿20日均线的点
    cross_idx = find_ma4_cross_ma20(data, data[data['trade_date'] > bullish_end_date].index.min())
    if cross_idx is not None:
        cross_point = data.iloc[cross_idx]
        plt.scatter(cross_point['trade_date'], cross_point['MA4'], color='black', s=100, zorder=5)
        plt.axvline(x=cross_point['trade_date'], color='g', linestyle='--', label='4日均线下穿20日均线')
    
    plt.title(f'{ts_code} 多头排列与结束阶段 {start_date} - {end_date}')
    plt.xlabel('日期')
    plt.ylabel('均线值')
    plt.legend()
    plt.grid(True)
    
    filename = f'{folder}/{ts_code}_{start_date}_{end_date}_with_end_phase.png'
    plt.savefig(filename)
    plt.close()

# 主程序，用于调试
if __name__ == "__main__":
    # 示例股票池
    pool = [{'code': '002919'}, {'code': '002119'}, {'code': '002469'},{'code': '002915'},{'code': '002913'},{'code': '002362'},{'code': '002106'},{'code': '002161'},{'code': '003042'},{'code': '002236'}]
    start_date = '20200101'
    end_date = '20201001'
    base = '000021'
    base_start_date = '20241008'
    base_end_date = '20241031'
    ma_list = [4, 8, 12, 16, 20, 47]

    result = find_pattern_segments(pool, start_date, end_date, ma_list)
    print(result)