import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from typing import Dict, Optional, List, Tuple

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def load_recent_stock_data(folder_path: str, rows_to_keep: int = 40) -> Dict[str, pd.DataFrame]:
    """
    从指定文件夹中读取所有CSV文件，并获取每个文件的最后N行数据。
    
    参数:
        folder_path (str): 包含CSV文件的文件夹路径
        rows_to_keep (int): 每个文件要保留的最后行数，默认为40
    
    返回:
        Dict[str, pd.DataFrame]: 字典，键为股票代码（不含.csv扩展名），值为对应的DataFrame
    """
    stock_data = {}
    
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"指定的文件夹不存在: {folder_path}")
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件是否为CSV文件
        if filename.endswith('.csv'):
            # 获取股票代码（文件名去掉.csv扩展名）
            stock_code = os.path.splitext(filename)[0]
            file_path = os.path.join(folder_path, filename)
            
            try:
                # 读取CSV文件的最后N行
                df = pd.read_csv(file_path, skipfooter=0, nrows=rows_to_keep, engine='python')
                stock_data[stock_code] = df
                print(f"成功读取 {stock_code} 的数据，共 {len(df)} 行")
            except Exception as e:
                print(f"读取文件 {filename} 时出错: {str(e)}")
    
    return stock_data

def find_bullish_ma_pattern(stock_data: Dict[str, pd.DataFrame], 
                           ma_columns: List[str] = ['MA4', 'MA8', 'MA12', 'MA16', 'MA20', 'MA47']) -> Dict[str, List[Tuple[int, int]]]:
    """
    从股票数据中寻找多头排列模式（无交叉、全上升、均线依次排列），返回连续的区间
    
    参数:
        stock_data (Dict[str, pd.DataFrame]): 股票数据字典，键为股票代码，值为对应的DataFrame
        ma_columns (List[str]): 要检查的均线列名，默认为MA4, MA8, MA12, MA16, MA20, MA47
    
    返回:
        Dict[str, List[Tuple[int, int]]]: 字典，键为股票代码，值为符合条件的时间区间列表（元组形式：(起始日, 终止日)）
    """
    bullish_patterns = {}
    
    # 检查均线列是否按周期递增排列
    ma_periods = [int(col.replace('MA', '')) for col in ma_columns]
    if ma_periods != sorted(ma_periods):
        raise ValueError("均线列名必须按周期递增排列")
    
    for stock_code, df in stock_data.items():
        # 检查DataFrame是否包含所有需要的均线列
        missing_columns = [col for col in ma_columns if col not in df.columns]
        if missing_columns:
            print(f"警告: {stock_code} 的数据中缺少均线列: {', '.join(missing_columns)}")
            continue
        
        # 初始化符合条件的区间列表
        bullish_ranges = []
        current_start = None
        
        # 检查每个交易日是否符合多头排列条件
        for i in range(len(df)):
            current_row = df.iloc[i]
            
            # 条件1: 均线依次排列 (MA4 > MA8 > MA12 > ... > MA47)
            is_order_correct = all(current_row[ma_columns[j]] > current_row[ma_columns[j+1]] 
                                  for j in range(len(ma_columns)-1))
            
            # 条件2: 所有均线呈上升趋势
            is_rising = True
            if i > 0:  # 除了第一天，需要比较前一天的数据
                prev_row = df.iloc[i-1]
                is_rising = all(current_row[col] > prev_row[col] for col in ma_columns)
            
            # 如果符合条件
            if is_order_correct and is_rising:
                # 如果还没有开始区间，则标记为区间开始
                if current_start is None:
                    current_start = i
            # 如果不符合条件且已经有开始区间，则标记为区间结束
            elif current_start is not None:
                bullish_ranges.append((current_start, i-1))
                current_start = None
        
        # 处理最后一个区间（如果循环结束时仍有未关闭的区间）
        if current_start is not None:
            bullish_ranges.append((current_start, len(df)-1))
        
        # 如果找到符合条件的区间，添加到结果字典中
        if bullish_ranges:
            bullish_patterns[stock_code] = bullish_ranges
    
    return bullish_patterns

def validate_bullish_patterns(stock_data: Dict[str, pd.DataFrame], 
                             bullish_patterns: Dict[str, List[Tuple[int, int]]],
                             ma_short: str = 'MA4', 
                             ma_long: str = 'MA20') -> Dict[str, List[Tuple[int, int]]]:
    """
    验证多头排列区间，排除后续出现短期均线下穿长期均线的情况
    
    参数:
        stock_data (Dict[str, pd.DataFrame]): 股票数据字典
        bullish_patterns (Dict[str, List[Tuple[int, int]]]): 多头排列区间字典
        ma_short (str): 短期均线列名，默认为MA4
        ma_long (str): 长期均线列名，默认为MA20
    
    返回:
        Dict[str, List[Tuple[int, int]]]: 验证后的多头排列区间字典
    """
    validated_patterns = {}
    
    for stock_code, ranges in bullish_patterns.items():
        if stock_code not in stock_data:
            continue
        
        df = stock_data[stock_code]
        valid_ranges = []
        
        # 检查每个区间
        for start, end in ranges:
            # 从区间结束日的下一个交易日开始检查
            check_start = end + 1
            
            # 如果区间结束日是最后一个交易日，则无需检查
            if check_start >= len(df):
                valid_ranges.append((start, end))
                continue
            
            # 检查后续是否出现短期均线下穿长期均线的情况
            has_cross = False
            prev_ma_short = df.iloc[check_start-1][ma_short]
            prev_ma_long = df.iloc[check_start-1][ma_long]
            
            for i in range(check_start, len(df)):
                current_ma_short = df.iloc[i][ma_short]
                current_ma_long = df.iloc[i][ma_long]
                
                # 检查下穿条件：前一天MA4 > MA20，当前MA4 < MA20
                if prev_ma_short > prev_ma_long and current_ma_short < current_ma_long:
                    has_cross = True
                    break
                
                prev_ma_short, prev_ma_long = current_ma_short, current_ma_long
            
            # 如果没有出现下穿，保留这个区间
            if not has_cross:
                valid_ranges.append((start, end))
        
        # 如果有有效区间，添加到结果中
        if valid_ranges:
            validated_patterns[stock_code] = valid_ranges
    
    return validated_patterns

def get_last_bullish_pattern(validated_patterns: Dict[str, List[Tuple[int, int]]]) -> Dict[str, Tuple[int, int]]:
    """
    获取每个股票最后一个多头排列区间
    
    参数:
        validated_patterns (Dict[str, List[Tuple[int, int]]]): 验证后的多头排列区间字典
    
    返回:
        Dict[str, Tuple[int, int]]: 字典，键为股票代码，值为最后一个多头排列区间（元组形式：(起始日, 终止日)）
    """
    last_patterns = {}
    
    for stock_code, ranges in validated_patterns.items():
        if ranges:
            # 取最后一个区间（按交易日排序）
            last_patterns[stock_code] = ranges[-1]
    
    return last_patterns

def plot_validated_patterns(stock_data: Dict[str, pd.DataFrame],
                           last_patterns: Dict[str, Tuple[int, int]],
                           ma_columns: List[str] = ['MA4', 'MA8', 'MA12', 'MA16', 'MA20', 'MA47'],
                           output_dir: str = 'plots') -> None:
    """
    为每个股票的最后一个多头排列区间绘制均线走势图
    
    参数:
        stock_data (Dict[str, pd.DataFrame]): 股票数据字典
        last_patterns (Dict[str, Tuple[int, int]]): 每个股票的最后一个多头排列区间字典
        ma_columns (List[str]): 均线列名列表
        output_dir (str): 图表保存目录
    """
    # 创建保存图表的目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用matplotlib内置字体支持中文
    try:
        # 尝试使用SimHei字体
        plt.rcParams["font.family"] = ["SimHei", "sans-serif"]
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    except:
        # 如果SimHei不可用，使用FontProperties方式
        print("警告: 无法设置全局字体，将尝试为每个文本元素单独设置字体")
    
    # 定义不同均线的颜色
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']
    
    for stock_code, (start, end) in last_patterns.items():
        if stock_code not in stock_data:
            continue
        
        df = stock_data[stock_code]
        
        # 跳过一日的区间
        if end - start < 1:
            print(f"股票 {stock_code} 的多头排列区间过短（少于2个交易日）")
            continue
        
        # 创建图表
        plt.figure(figsize=(14, 7))
        
        # 绘制所有均线
        for i, col in enumerate(ma_columns):
            plt.plot(df.index, df[col], label=col, color=colors[i], linewidth=1.5)
        
        # 标记多头排列区间
        plt.axvspan(start, end, color='gray', alpha=0.3)
        
        try:
            plt.text((start + end) / 2, plt.ylim()[1] * 0.95, 
                    f'多头排列\n{start}-{end}', 
                    ha='center', va='top', 
                    bbox=dict(boxstyle='round,pad=0.3', fc='gray', alpha=0.5),
                    fontproperties=FontProperties(family=['SimHei', 'sans-serif']))
        except:
            plt.text((start + end) / 2, plt.ylim()[1] * 0.95, 
                    f'多头排列\n{start}-{end}', 
                    ha='center', va='top', 
                    bbox=dict(boxstyle='round,pad=0.3', fc='gray', alpha=0.5))
        
        # 设置标题和标签
        try:
            plt.title(f'{stock_code} 最后一个多头排列区间均线走势图', fontproperties=FontProperties(family=['SimHei', 'sans-serif']))
            plt.xlabel('交易日', fontproperties=FontProperties(family=['SimHei', 'sans-serif']))
            plt.ylabel('均线值', fontproperties=FontProperties(family=['SimHei', 'sans-serif']))
        except:
            plt.title(f'{stock_code} 最后一个多头排列区间均线走势图')
            plt.xlabel('交易日')
            plt.ylabel('均线值')
        
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 设置图例
        try:
            legend = plt.legend(prop=FontProperties(family=['SimHei', 'sans-serif']))
        except:
            legend = plt.legend()
        
        # 保存图表
        filename = os.path.join(output_dir, f'{stock_code}_last_bullish_pattern.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"已生成 {stock_code} 的图表: {filename}")

# 示例用法
if __name__ == "__main__":
    folder_path = r"D:\self\data\kline-data"  # 替换为实际的文件夹路径
    output_dir = "plots"  # 图表保存目录
    
    # 加载数据
    stock_data = load_recent_stock_data(folder_path)
    
    # 定义要检查的均线列
    ma_columns = ['MA4', 'MA8', 'MA12', 'MA16', 'MA20', 'MA47']
    
    # 寻找多头排列模式
    bullish_patterns = find_bullish_ma_pattern(stock_data, ma_columns)
    
    # 验证多头排列模式
    validated_patterns = validate_bullish_patterns(stock_data, bullish_patterns)
    
    # 获取每个股票的最后一个多头排列区间
    last_patterns = get_last_bullish_pattern(validated_patterns)
    
    # 绘制并保存图表
    plot_validated_patterns(stock_data, last_patterns, ma_columns, output_dir)
    
    # 打印最终结果
    print("\n最终多头排列模式结果（每个股票取最后一个区间）:")
    for stock_code, (start, end) in last_patterns.items():
        print(f"股票 {stock_code}:")
        print(f"  从第 {start} 日到第 {end} 日符合多头排列模式")
        print(f"  区间长度: {end - start + 1} 个交易日")
        print(f"  起始日均线值:")
        for col in ma_columns:
            print(f"    {col}: {stock_data[stock_code].iloc[start][col]:.2f}")
        print(f"  终止日均线值:")
        for col in ma_columns:
            print(f"    {col}: {stock_data[stock_code].iloc[end][col]:.2f}")
        print()    