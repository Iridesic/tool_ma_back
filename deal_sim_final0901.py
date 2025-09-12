# 示例用法 ##############################
# 【1】基准股票 + 起始时间 + 结束时间
# 【2】股票池 + 起始时间 + 结束时间
# 【3】stage匹配的结果汇总
# ######################################
import os
from deal_sim_stage_change import detect_bullish_arrangement, determine_stage, predict_from_data, process_stock_data
import pandas as pd

def analyze_stock_data(test_stock, test_start, test_end, test_stock_pool, test_start_date, test_end_date):
    """
    分析股票数据并返回结构化结果
    
    参数:
        test_stock: 测试股票代码
        test_start: 测试股票起始时间
        test_end: 测试股票结束时间
        test_stock_pool: 股票池列表
        test_start_date: 股票池起始时间
        test_end_date: 股票池结束时间
    
    返回:
        包含分析结果的字典
    """
    result = {
        "test_stock_info": {},          # 测试股票的stage与分类
        "same_stage_same_category": [], # 同阶段同类别的结果
        "same_stage_diff_category": []  # 同阶段不同类别的结果
    }
    
    # 1. 处理测试股票
    # 1.1 确定测试股票阶段
    stage, details = determine_stage(test_stock, test_start, test_end)
    
    # 1.2 处理测试股票数据
    base_folder = r'D:/self/data/final_data_0821'
    stock_info = process_stock_data(base_folder, test_stock, test_start, test_end)
    
    # 1.3 准备数据框
    cur_stage = pd.DataFrame(stock_info['target_data'])
    history_stage = pd.DataFrame(stock_info['history_data'])
    extended_stage = pd.DataFrame(stock_info['extended_data'])
    
    # 1.4 确定测试股票的分类（长/短）
    stage_count = len(cur_stage)
    stage_suffix = stage if stage in ['stage1', 'stage2', 'stage3'] else 'stage3'
    length_suffix = 'long' if stage_count > (4 if stage_suffix == 'stage2' else 10) else 'short'
    
    # 1.5 保存测试股票信息
    result["test_stock_info"] = {
        "stock_code": test_stock,
        "time_range": {"start": test_start, "end": test_end},
        "stage": stage,
        "category": length_suffix,
        "details": details
    }
    
    # 2. 处理股票池
    if stage in ['stage1', 'stage2', 'stage3']:
        # 2.1 获取股票池检测结果
        results = detect_bullish_arrangement(
            stock_pool=test_stock_pool,
            start_date=test_start_date,
            end_date=test_end_date
        )
        
        # 2.2 处理同阶段结果
        same_stage_results = results.get(stage, [])
        for item in same_stage_results:
            # 获取股票详细数据
            item_info = process_stock_data(
                base_folder, 
                item['stock_code'], 
                item['start'], 
                item['end']
            )
            
            # 准备数据框
            item_cur_stage = pd.DataFrame(item_info['target_data'])
            
            # 确定该股票的分类
            item_stage_count = len(item_cur_stage)
            item_stage_suffix = stage
            item_length_suffix = 'long' if item_stage_count > (4 if item_stage_suffix == 'stage2' else 10) else 'short'
            
            # 提取MA数据
            ma_columns = ['MA4', 'MA8', 'MA12', 'MA16', 'MA20', 'MA47']
            ma_data = {}
            if not item_cur_stage.empty and all(ma in item_cur_stage.columns for ma in ma_columns):
                ma_data = {
                    ma: item_cur_stage[ma].tolist()
                    for ma in ma_columns
                }
            
            # 构建结果项
            result_item = {
                "stock_code": item['stock_code'],
                "time_range": {
                    "start": item['start'],
                    "end": item['end']
                },
                "duration_days": item['duration_days'],
                "category": item_length_suffix,
                "ma_data": ma_data
            }
            
            # 分类存入结果
            if item_length_suffix == length_suffix:
                result["same_stage_same_category"].append(result_item)
            else:
                result["same_stage_diff_category"].append(result_item)
        result["base_data"] = cur_stage.to_dict(orient='records')
    
    return result

# 示例用法
if __name__ == "__main__":
    # 测试参数
    test_stock = '000021'
    test_start = '2023-06-07'
    test_end = '2023-06-15'
    test_stock_pool = ['000561', '000830', '000547', '002057', '002092', '002106', '002119', '002362']
    test_start_date = '2020-01-01'
    test_end_date = '2024-12-31'
    
    # 执行分析
    analysis_result = analyze_stock_data(
        test_stock, 
        test_start, 
        test_end, 
        test_stock_pool, 
        test_start_date, 
        test_end_date
    )
