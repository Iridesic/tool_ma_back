import pandas as pd
import os
from typing import List, Dict
import shutil

from deal_stock_choose0919 import get_stage_class_from_csv0919

def load_stock_data(stock_code: str, data_folder: str = "D:\\self\\data\\final_data_0821") -> pd.DataFrame:

    """加载股票数据从指定目录的CSV文件"""
    file_path = os.path.join(data_folder, f"{stock_code}.csv")
    try:
        df = pd.read_csv(file_path, parse_dates=['timestamps'], index_col='timestamps')
        return df.sort_index()  # 确保按日期升序排列（最新数据在最后）
    except FileNotFoundError:
        print(f"警告: 股票 {stock_code} 的数据文件未找到，路径: {file_path}")
        return None
    except Exception as e:
        print(f"加载股票 {stock_code} 数据时出错: {str(e)}")
        return None

def is_ma_sequential(ma4: float, ma8: float, ma12: float, ma16: float, ma20: float, ma47: float) -> bool:
    """检查单天均线是否严格顺次排列（允许微小误差）"""
    eps = 1e-6  # 微小误差阈值
    return (ma4 > ma8 + eps and 
            ma8 > ma12 + eps and 
            ma12 > ma16 + eps and 
            ma16 > ma20 + eps and 
            ma20 > ma47 + eps)

def is_ma_rising(window: pd.DataFrame) -> bool:
    """检查窗口内所有均线是否呈现上升趋势（允许微小回调）"""
    ma_columns = ['MA4', 'MA8', 'MA12', 'MA16', 'MA20', 'MA47']
    for ma in ma_columns:
        changes = window[ma].diff().dropna()
        if len(changes) == 0:
            return False
        up_days_ratio = (changes > 0).mean()
        overall_trend = window[ma].iloc[-1] > window[ma].iloc[0]
        if up_days_ratio < 0.6 or not overall_trend:
            return False
    return True

def has_no_crossovers(window: pd.DataFrame) -> bool:
    """检查窗口内是否无任何均线交叉"""
    ma_columns = ['MA4', 'MA8', 'MA12', 'MA16', 'MA20', 'MA47']
    for i in range(len(ma_columns)):
        for j in range(i + 1, len(ma_columns)):
            upper_ma = ma_columns[i]
            lower_ma = ma_columns[j]
            if (window[upper_ma] < window[lower_ma]).any():
                return False
    return True

def find_stage2_fragments(df: pd.DataFrame, n_days: int, min_fragment_length: int = 5) -> List[pd.DataFrame]:
    """在近N天内查找所有符合条件的stage2模式片段"""
    if len(df) < n_days:
        print(f"数据不足：需要至少{n_days}天，实际有{len(df)}天")
        return []
    
    recent_data = df.tail(n_days).copy()
    stage2_fragments = []
    
    for window_length in range(min(n_days, len(recent_data)), min_fragment_length - 1, -1):
        for i in range(len(recent_data) - window_length + 1):
            window = recent_data.iloc[i:i+window_length]
            
            # 检查窗口内每天均线是否顺次排列
            sequential = True
            for _, day_data in window.iterrows():
                if not is_ma_sequential(
                    day_data.MA4, day_data.MA8, day_data.MA12,
                    day_data.MA16, day_data.MA20, day_data.MA47
                ):
                    sequential = False
                    break
            if not sequential:
                continue
            
            # 检查均线上升趋势
            if not is_ma_rising(window):
                continue
            
            # 检查无交叉
            if not has_no_crossovers(window):
                continue
            
            stage2_fragments.append(window)
    
    # 去重（保留最长片段）
    if stage2_fragments:
        stage2_fragments.sort(key=lambda x: len(x), reverse=True)
        unique_fragments = []
        for frag in stage2_fragments:
            is_contained = False
            for existing in unique_fragments:
                if frag.index[0] >= existing.index[0] and frag.index[-1] <= existing.index[-1]:
                    is_contained = True
                    break
            if not is_contained:
                unique_fragments.append(frag)
        return unique_fragments
    
    return []

def has_ma4_cross_below_ma20(df: pd.DataFrame) -> bool:
    """检查片段结束后是否出现MA4下穿MA20"""
    if len(df) < 3:
        return False
    
    # 检查最后3天内是否有下穿信号
    for i in range(len(df)-2, len(df)):
        if i == 0:
            continue
        prev_day = df.iloc[i-1]
        curr_day = df.iloc[i]
        if (prev_day.MA4 > prev_day.MA20) and (curr_day.MA4 < curr_day.MA20):
            print(f"在位置 {i} 发现MA4下穿MA20")
            return True
    return False

def determine_stage(df: pd.DataFrame, fragment: pd.DataFrame) -> str:
    """确定股票当前的阶段"""
    latest_date = df.index[-1]
    
    if latest_date in fragment.index:
        latest_data = df.loc[latest_date]
        if is_ma_sequential(
            latest_data.MA4, latest_data.MA8, latest_data.MA12,
            latest_data.MA16, latest_data.MA20, latest_data.MA47
        ):
            if len(df) >= 2 and latest_data.MA4 < df.iloc[-2].MA4:
                return "stage3"
            return "stage2"
    
    # 检查是否出现MA4下降或交叉
    ma_columns = ['MA4', 'MA8', 'MA12', 'MA16', 'MA20', 'MA47']
    if len(df) >= 2 and df.iloc[-1].MA4 < df.iloc[-2].MA4:
        return "stage3"
    
    for i in range(len(ma_columns)):
        for j in range(i + 1, len(ma_columns)):
            upper_ma = ma_columns[i]
            lower_ma = ma_columns[j]
            if df.iloc[-1][upper_ma] < df.iloc[-1][lower_ma]:
                return "stage3"
    
    return "stage3"

def find_stage1_fragments(df: pd.DataFrame, lookback_days: int = 60, min_length: int = 3) -> List[pd.DataFrame]:
    """查找符合stage1条件的片段"""
    if len(df) < lookback_days:
        return []
    
    lookback_data = df.tail(lookback_days).copy()
    ma_columns = ['MA4', 'MA8', 'MA12', 'MA16', 'MA20', 'MA47']
    stage1_fragments = []
    
    for window_length in range(min(lookback_days, len(lookback_data)), min_length - 1, -1):
        for i in range(len(lookback_data) - window_length + 1):
            window = lookback_data.iloc[i:i+window_length]
            
            # 检查至少3条均线呈上升趋势
            rising_ma_count = 0
            for ma in ma_columns:
                changes = window[ma].diff().dropna()
                if len(changes) == 0:
                    continue
                up_ratio = (changes > 0).mean()
                overall_up = window[ma].iloc[-1] > window[ma].iloc[0]
                if up_ratio >= 0.5 and overall_up:
                    rising_ma_count += 1
            
            if rising_ma_count < 3:
                continue
            
            # 检查是否出现连续金叉
            cross_count = 0
            for j in range(1, len(window)):
                curr = window.iloc[j]
                prev = window.iloc[j-1]
                
                day_cross = False
                for a in range(len(ma_columns)):
                    for b in range(a + 1, len(ma_columns)):
                        short_ma = ma_columns[a]
                        long_ma = ma_columns[b]
                        if prev[short_ma] < prev[long_ma] and curr[short_ma] > curr[long_ma]:
                            day_cross = True
                            cross_count += 1
                            break
                    if day_cross:
                        break
            
            if cross_count < 2:
                continue
            
            # 检查均线由聚集到发散
            start_std = window.iloc[0][ma_columns].std()
            end_std = window.iloc[-1][ma_columns].std()
            std_increase = end_std > start_std * 1.2
            
            if not std_increase:
                continue
            
            stage1_fragments.append(window)
    
    # 去重，保留最长片段
    if stage1_fragments:
        stage1_fragments.sort(key=lambda x: len(x), reverse=True)
        unique_fragments = []
        for frag in stage1_fragments:
            is_contained = False
            for existing in unique_fragments:
                if frag.index[0] >= existing.index[0] and frag.index[-1] <= existing.index[-1]:
                    is_contained = True
                    break
            if not is_contained:
                unique_fragments.append(frag)
        return unique_fragments
    
    return []

def get_latest_fragment(fragments: List[Dict]) -> Dict:
    """从多个片段中获取最新的一个"""
    if not fragments:
        return None
    return sorted(fragments, key=lambda x: x['end_date'], reverse=True)[0]

def export_stage_data(stock_code: str, df: pd.DataFrame, fragment: Dict, target_folder: str) -> None:
    """导出包含阶段标记的数据到CSV文件"""
    os.makedirs(target_folder, exist_ok=True)
    
    start_date = fragment['start_date']
    end_date = fragment['end_date']
    stage = fragment['stage']
    stage_value = 1 if stage == 'stage1' else 2 if stage == 'stage2' else 3
    
    start_idx = df.index.get_indexer([pd.Timestamp(start_date)], method='pad')[0]
    if start_idx == -1:
        start_idx = 0
    
    history_start_idx = max(0, start_idx - 60)
    combined_data = df.iloc[history_start_idx:].copy()
    combined_data['stage'] = 0
    mask = (combined_data.index.date >= start_date) & (combined_data.index.date <= end_date)
    combined_data.loc[mask, 'stage'] = stage_value
    
    output_path = os.path.join(target_folder, f"{stock_code['code']}_with_stage.csv")
    combined_data.reset_index().to_csv(output_path, index=False)
    print(f"已导出带阶段标记的数据到: {output_path}")

def analyze_stock(stock_code: str, n_days: int, min_fragment_length: int = 5, 
                 data_folder: str = "D:\\self\\data\\final_data_0821", stage1_lookback: int = 60,
                 target_folder: str = "D:\\self\\stage_data") -> Dict:
    """分析单只股票，查找近N天内的阶段片段"""
    print("执行：分析单只股票，查找近N天内的阶段片段，并导出csv文件")
    print(f"\n开始分析股票: {stock_code}")

    df = load_stock_data(stock_code['code'], data_folder)
    if df is None or len(df) == 0:
        return None
    
    print(f"股票 {stock_code['code']} 数据量: {len(df)} 条，检查近 {n_days} 天")
    print(f"数据日期范围: {df.index.min().date()} 至 {df.index.max().date()}")
    
    # 先查找stage2片段（优先检测）
    stage2_fragments = find_stage2_fragments(df, n_days, min_fragment_length)
    valid_fragments = []
    
    for fragment in stage2_fragments:
        # 对stage2片段仅检查MA4下穿MA20
        if has_ma4_cross_below_ma20(fragment):
            print(f"股票 {stock_code} 的stage2片段出现MA4下穿MA20，已舍弃")
            continue
            
        stage = determine_stage(df, fragment)
        # 只保留确实是stage2的片段
        if stage == "stage2":
            valid_fragments.append({
                "start_date": fragment.index[0].date(),
                "end_date": fragment.index[-1].date(),
                "length_days": len(fragment),
                "ma_data": fragment[['MA4', 'MA8', 'MA12', 'MA16', 'MA20', 'MA47']].reset_index().to_dict('records'),
                "stage": stage
            })
    
    # 如果有有效的stage2结果，直接返回
    if valid_fragments:
        latest_fragment = get_latest_fragment(valid_fragments)
        print(f"股票 {stock_code} 找到最新的stage2片段 ({latest_fragment['end_date']})，并导出到csv文件---------------")
        export_stage_data(stock_code, df, latest_fragment, target_folder)
        return {
            "stock_code": stock_code,
            "latest_fragment": latest_fragment
        }
    
    # 若无stage2结果，则查找stage1片段
    print(f"股票 {stock_code} 未找到有效stage2片段，尝试查找stage1片段...")
    stage1_fragments = find_stage1_fragments(df, stage1_lookback, min_fragment_length)
    
    if not stage1_fragments:
        print(f"股票 {stock_code} 未找到符合条件的stage1片段")
        return None
    
    # 处理stage1结果
    stage1_results = []
    for fragment in stage1_fragments:
        if has_ma4_cross_below_ma20(fragment):
            print(f"股票 {stock_code} 的stage1片段出现MA4下穿MA20，已舍弃")
            continue
            
        stage1_results.append({
            "start_date": fragment.index[0].date(),
            "end_date": fragment.index[-1].date(),
            "length_days": len(fragment),
            "ma_data": fragment[['MA4', 'MA8', 'MA12', 'MA16', 'MA20', 'MA47']].reset_index().to_dict('records'),
            "stage": "stage1"
        })
    
    if not stage1_results:
        print(f"股票 {stock_code} 所有stage1片段均出现MA4下穿MA20，无有效片段")
        return None
    
    latest_fragment = get_latest_fragment(stage1_results)
    print(f"股票 {stock_code} 找到最新的stage1片段 ({latest_fragment['end_date']})，并导出到csv文件---------------")
    export_stage_data(stock_code, df, latest_fragment, target_folder)
    
    return {
        "stock_code": stock_code,
        "latest_fragment": latest_fragment
    }

def analyze_stock_pool(stock_codes: List[str], n_days: int, min_fragment_length: int = 5,
                      data_folder: str = "D:\\self\\data\\final_data_0821", stage1_lookback: int = 60,
                      target_folder: str = "D:\\self\\stage_data") -> List[Dict]:
    """分析股票池，返回所有找到阶段片段的股票结果"""
    print("执行analyze_stock_pool")
    results = []
    for code in stock_codes:
        stock_result = analyze_stock(
            code, n_days, min_fragment_length, 
            data_folder, stage1_lookback,
            target_folder
        )
        if stock_result:
            results.append(stock_result)
    return results

def find_N_days_bullish_0919(stock_pool: List[str], n_days: int, ma_periods: List[str], min_fragment_length: int = 5, 
                            stage1_lookback: int = 60, target_folder: str = "D:\\self\\stage_data") -> List[Dict]:
    """处理前端请求，返回分析结果"""
    print("执行find_N_days_bullish_0919")
    result = analyze_stock_pool(stock_pool, n_days, min_fragment_length, 
                             data_folder="D:\\self\\data\\final_data_0821", 
                             stage1_lookback=stage1_lookback,
                             target_folder=target_folder)
    print(len(result))

    all_results = get_stage_class_from_csv0919(ma_periods=ma_periods, recent_days=n_days)
    # print(all_results)
    print(type(all_results))
    return all_results


if __name__ == "__main__":
    stock_pool = ["000021", "000830", "300516", "001255", "001308", "001309", "001333", "001339", "002025", "002052", "002057", "002092", "002106", "002119"]
    n_days = 30
    min_fragment_length = 3
    stage1_lookback = 30
    target_folder = r"D:\self\code\tool_ma_back\bbb_fragments"
    ma_periods = [4, 8, 12, 16, 20, 47]
    
    results = find_N_days_bullish_0919(stock_pool, n_days, ma_periods, min_fragment_length, stage1_lookback, target_folder)
    
    print("\n===== 最终符合条件的股票 =====")
    for result in results:
        print(f"股票代码: {result['stock_code']}")
        frag = result['latest_fragment']
        print(f"最新片段: {frag['start_date']} 至 {frag['end_date']}, 共 {frag['length_days']} 天, 阶段: {frag['stage']}")
        print("---")
    