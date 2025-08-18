import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from dtw import dtw
from scipy.spatial.distance import euclidean
from utils import get_stock_data1, calculate_ma
from pattern_detection import is_golden_cross  # 复用金叉检测函数

def find_similar_stocks(target_code, stock_data_dict, n_days, ma_list):
    """
    找到与目标股票最相似的前10只股票
    
    参数:
    target_code (str): 目标股票代码
    stock_data_dict (dict): 包含股票数据的字典
    n_days (int): 近N天数据
    ma_list (list): 均线周期列表
    
    返回:
    list: 包含最相似股票信息的列表，每个元素是(股票代码, 相似度)
    """
    # 处理目标股票代码格式
    if not target_code.endswith(('.SH', '.SZ')):
        target_code = f"{target_code}.SH" if target_code.startswith('6') else f"{target_code}.SZ"
    
    # 检查目标股票是否在已获取的数据中
    if target_code not in stock_data_dict:
        # 如果不在，尝试单独获取目标股票数据
        print(f"目标股票 {target_code} 不在股票列表中，尝试单独获取数据...")
        end_date = datetime.now().strftime('%Y-%m-%d')
        max_ma_period = max(ma_list) if ma_list else 0
        required_days = max_ma_period + 30
        start_date = (datetime.now() - timedelta(days=required_days)).strftime('%Y-%m-%d')
        
        target_data = get_stock_data1(target_code, start_date, end_date)
        if target_data.empty:
            print(f"无法获取目标股票 {target_code} 的数据")
            return []
        
        target_data = calculate_ma(target_data, ma_list)
        if target_data.empty:
            print(f"目标股票 {target_code} 均线计算失败")
            return []
        
        try:
            target_data['trade_date'] = pd.to_datetime(target_data['trade_date'])
        except Exception as e:
            print(f"目标股票 {target_code} 日期格式转换失败: {e}")
            return []
        
        recent_cutoff = datetime.now() - timedelta(days=n_days)
        target_data = target_data[target_data['trade_date'] >= recent_cutoff]
        
        if target_data.empty:
            print(f"目标股票 {target_code} 近{n_days}日无交易数据")
            return []
        
        target_data = target_data.sort_values('trade_date').reset_index(drop=True)
    else:
        target_data = stock_data_dict[target_code]
        # 从字典中移除目标股票自身，避免与自身比较
        del stock_data_dict[target_code]
    
    # 确保所有股票数据长度一致
    target_length = len(target_data)
    valid_stocks = {}
    
    for code, data in stock_data_dict.items():
        if len(data) == target_length:
            valid_stocks[code] = data
        else:
            print(f"股票 {code} 数据长度与目标股票不一致，跳过比较")
    
    if not valid_stocks:
        print("没有找到可比较的有效股票数据")
        return []
    
    # 计算目标股票的均线组
    target_ma_group = {f'MA{period}': target_data[f'MA{period}'].values for period in ma_list}
    
    # 计算与每只股票的相似度
    similarity_results = []
    for code, data in valid_stocks.items():
        try:
            ma_group = {f'MA{period}': data[f'MA{period}'].values for period in ma_list}
            similarity = calculate_ma_group_similarity(target_ma_group, ma_group, ma_list)
            similarity_results.append((code, similarity['overall_similarity']))
            print(f"计算 {code} 相似度完成: {similarity['overall_similarity']:.4f}")
        except Exception as e:
            print(f"计算 {code} 相似度时出错: {e}")
            continue
    
    # 按相似度降序排序并返回前10
    similarity_results.sort(key=lambda x: x[1], reverse=True)
    return similarity_results[:10]


# ==================== 相似度计算函数 ====================
def identify_crossovers(ma_short, ma_long):
    """识别短期均线和长期均线之间的金叉和死叉（复用is_golden_cross逻辑）"""
    if len(ma_short) != len(ma_long):
        raise ValueError("短期均线和长期均线长度必须一致")
    
    golden_cross = []  # 金叉：短期均线上穿长期均线
    death_cross = []   # 死叉：短期均线下穿长期均线
    
    # 构造临时DataFrame用于复用is_golden_cross
    for i in range(1, len(ma_short)):
        temp_data = pd.DataFrame({
            f'MA{len(ma_short)}': [ma_short[i-1], ma_short[i]],
            f'MA{len(ma_long)}': [ma_long[i-1], ma_long[i]]
        })
        # 复用金叉检测函数
        if is_golden_cross(temp_data, len(ma_short), len(ma_long)):
            golden_cross.append(i)
        # 死叉检测（金叉的反向逻辑）
        if (ma_short[i-1] >= ma_long[i-1] and ma_short[i] < ma_long[i]):
            death_cross.append(i)
    
    return golden_cross, death_cross

def calculate_crossover_similarity(ma_group1, ma_group2, ma_periods):
    """计算两组均线之间交叉模式的相似性"""
    cross_pairs = []
    for i in range(len(ma_periods)):
        for j in range(i+1, len(ma_periods)):
            cross_pairs.append((ma_periods[i], ma_periods[j]))
    
    total_golden_similarity = 0
    total_death_similarity = 0
    cross_count = len(cross_pairs)
    
    for short_period, long_period in cross_pairs:
        ma1_short = ma_group1[f'MA{short_period}']
        ma1_long = ma_group1[f'MA{long_period}']
        ma2_short = ma_group2[f'MA{short_period}']
        ma2_long = ma_group2[f'MA{long_period}']
        
        g1_golden, g1_death = identify_crossovers(ma1_short, ma1_long)
        g2_golden, g2_death = identify_crossovers(ma2_short, ma2_long)
        
        count_golden1, count_golden2 = len(g1_golden), len(g2_golden)
        count_death1, count_death2 = len(g1_death), len(g2_death)
        
        golden_count_sim = 1 - abs(count_golden1 - count_golden2) / (max(count_golden1, count_golden2, 1) + 1e-10)
        death_count_sim = 1 - abs(count_death1 - count_death2) / (max(count_death1, count_death2, 1) + 1e-10)
        
        if count_golden1 > 0 and count_golden2 > 0:
            # 修复 DTW 结果访问方式
            dtw_dist = dtw(np.array(g1_golden), np.array(g2_golden)).distance
            golden_pos_sim = 1 - dtw_dist / (len(ma1_short) * max(count_golden1, count_golden2))
        else:
            golden_pos_sim = 1.0 if count_golden1 == count_golden2 else 0.0
            
        if count_death1 > 0 and count_death2 > 0:
            # 修复 DTW 结果访问方式
            dtw_dist = dtw(np.array(g1_death), np.array(g2_death)).distance
            death_pos_sim = 1 - dtw_dist / (len(ma1_short) * max(count_death1, count_death2))
        else:
            death_pos_sim = 1.0 if count_death1 == count_death2 else 0.0
        
        golden_sim = (golden_count_sim + golden_pos_sim) / 2
        death_sim = (death_count_sim + death_pos_sim) / 2
        
        total_golden_similarity += golden_sim
        total_death_similarity += death_sim
    
    avg_golden_similarity = total_golden_similarity / cross_count if cross_count > 0 else 0
    avg_death_similarity = total_death_similarity / cross_count if cross_count > 0 else 0
    overall_crossover_similarity = (avg_golden_similarity + avg_death_similarity) / 2
    
    return {
        'golden_cross_similarity': avg_golden_similarity,
        'death_cross_similarity': avg_death_similarity,
        'overall_crossover_similarity': overall_crossover_similarity
    }

def calculate_single_ma_similarity(series1, series2):
    """计算单条均线的相似性指标"""
    s1 = np.array(series1)
    s2 = np.array(series2)
    n = len(s1)
    
    # 趋势方向一致性
    diff1 = np.diff(s1)
    diff2 = np.diff(s2)
    direction_agreement = np.mean(np.sign(diff1) == np.sign(diff2))
    
    # 趋势强度相似度
    x = np.arange(n).reshape(-1, 1)
    model1 = LinearRegression().fit(x, s1)
    model2 = LinearRegression().fit(x, s2)
    slope1, slope2 = model1.coef_[0], model2.coef_[0]
    slope_similarity = 1 - abs(slope1 - slope2) / (abs(slope1) + abs(slope2) + 1e-10)
    slope_similarity = max(0, min(1, slope_similarity))
    
    # 波动特性相似度
    std1, std2 = np.std(s1), np.std(s2)
    volatility_similarity = 1 - abs(std1 - std2) / (std1 + std2 + 1e-10)
    
    # 皮尔逊相关系数
    pearson_corr, _ = pearsonr(s1, s2)
    pearson_similarity = (pearson_corr + 1) / 2  # 转换到0-1范围
    
    # 斯皮尔曼等级相关系数
    spearman_corr, _ = spearmanr(s1, s2)
    spearman_similarity = (spearman_corr + 1) / 2  # 转换到0-1范围
    
    # 动态时间规整(DTW)距离
    dtw_result = dtw(s1, s2)
    d = dtw_result.distance

    max_possible = np.sqrt(sum((np.max([s1, s2]) - np.min([s1, s2]))**2 for _ in range(n)))    
    dtw_similarity = 1 - min(d / max_possible, 1.0)
    
    # 均值水平相似度
    mean1, mean2 = np.mean(s1), np.mean(s2)
    mean_similarity = 1 - abs(mean1 - mean2) / (mean1 + mean2 + 1e-10)
    mean_similarity = max(0, min(1, mean_similarity))
    
    return {
        'trend_direction_agreement': direction_agreement,
        'trend_strength_similarity': slope_similarity,
        'volatility_similarity': volatility_similarity,
        'pearson_similarity': pearson_similarity,
        'spearman_similarity': spearman_similarity,
        'dtw_similarity': dtw_similarity,
        'mean_level_similarity': mean_similarity
    }

def calculate_ma_group_similarity(ma_group1, ma_group2, ma_periods=None, weights=None):
    """综合评估两组均线（包含多条均线）的相似性，考虑金叉和死叉"""
    if ma_periods is None:
        ma_periods = [4, 8, 12, 16, 20, 47]
    
    # 验证输入
    for period in ma_periods:
        ma_key = f'MA{period}'
        if ma_key not in ma_group1 or ma_key not in ma_group2:
            raise ValueError(f"两组均线都必须包含 {ma_key}")
        
        if len(ma_group1[ma_key]) != len(ma_group2[ma_key]):
            raise ValueError(f"{ma_key} 的长度在两组均线中必须一致")
    
    # 处理可能的NaN值
    for period in ma_periods:
        ma_key = f'MA{period}'
        if np.isnan(ma_group1[ma_key]).any() or np.isnan(ma_group2[ma_key]).any():
            raise ValueError(f"{ma_key} 中不能包含NaN值")
    
    # 默认权重
    if weights is None:
        weights = {
            'single_ma_features': 0.6,  # 单条均线特征的权重
            'crossover_features': 0.4   # 交叉特征的权重
        }
    
    # 计算每条均线的相似性并取平均
    single_ma_metrics = {}
    for metric in ['trend_direction_agreement', 'trend_strength_similarity', 
                   'volatility_similarity', 'pearson_similarity', 
                   'spearman_similarity', 'dtw_similarity', 'mean_level_similarity']:
        single_ma_metrics[metric] = 0.0
    
    for period in ma_periods:
        ma_key = f'MA{period}'
        ma_similarity = calculate_single_ma_similarity(
            ma_group1[ma_key], ma_group2[ma_key]
        )
        for metric, value in ma_similarity.items():
            single_ma_metrics[metric] += value / len(ma_periods)
    
    # 计算交叉（金叉/死叉）模式的相似性
    crossover_metrics = calculate_crossover_similarity(ma_group1, ma_group2, ma_periods)
    
    # 计算单条均线特征的综合相似度
    single_ma_weights = {
        'trend_direction_agreement': 0.15,
        'trend_strength_similarity': 0.1,
        'volatility_similarity': 0.15,
        'pearson_similarity': 0.2,
        'spearman_similarity': 0.15,
        'dtw_similarity': 0.15,
        'mean_level_similarity': 0.1
    }
    
    single_ma_overall = 0.0
    for metric, weight in single_ma_weights.items():
        single_ma_overall += single_ma_metrics[metric] * weight
    
    # 计算总体综合相似度
    overall_similarity = (
        single_ma_overall * weights['single_ma_features'] +
        crossover_metrics['overall_crossover_similarity'] * weights['crossover_features']
    )
    
    # 整合所有结果
    result = {**single_ma_metrics,** crossover_metrics,
              'single_ma_overall': single_ma_overall,
              'overall_similarity': overall_similarity}
    
    return result


def extract_stock_data_from_txt(txt_path, n_days, ma_list):
    """
    从指定路径的txt文件提取股票代码，用所有历史数据计算均线后，截取近N日的行情与均线数据
    
    参数:
    txt_path (str): 包含股票代码的txt文件路径
    n_days (int): 需要提取的近N日数据
    ma_list (list): 需要计算的均线周期列表，如[4,8,12,16,20,47]
    
    返回:
    dict: 键为股票代码，值为包含行情和均线数据的DataFrame
    """
    # 读取股票代码
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            # 过滤注释行（#开头）和空行
            stock_codes = [
                line.strip() for line in f 
                if line.strip() and not line.strip().startswith('#')
            ]
    except Exception as e:
        print(f"读取股票代码文件出错: {e}")
        return {}
    
    if not stock_codes:
        print("未提取到有效股票代码")
        return {}
    
    # 关键修正：获取足够早的历史数据（而非限制为2*N天）
    # 例如：获取过去5年的数据（确保覆盖最长均线周期，如47天）
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')  # 5年历史数据
    
    # 确定最长均线周期，确保历史数据至少覆盖该周期
    max_ma_period = max(ma_list) if ma_list else 0
    required_days = max_ma_period + 30  # 额外加30天缓冲（应对非交易日）
    required_start_date = (datetime.now() - timedelta(days=required_days)).strftime('%Y-%m-%d')
    # 取更早的日期作为起始点
    start_date = min(start_date, required_start_date)
    
    result = {}
    
    for code in stock_codes:
        # 处理股票代码后缀
        if code.endswith(('.SH', '.SZ')):
            ts_code = code
        else:
            ts_code = f"{code}.SH" if code.startswith('6') else f"{code}.SZ"
        
        # 获取完整历史数据（用于计算均线）
        stock_data = get_stock_data1(ts_code, start_date, end_date)
        if stock_data.empty:
            print(f"无法获取 {ts_code} 的历史数据，跳过")
            continue
        
        # 检查历史数据量是否满足最长均线需求
        if len(stock_data) < max_ma_period:
            print(f"{ts_code} 历史数据不足（{len(stock_data)}天），无法计算 {max_ma_period} 天均线，跳过")
            continue
        
        # 用完整历史数据计算均线
        stock_data = calculate_ma(stock_data, ma_list)
        if stock_data.empty:
            print(f"{ts_code} 均线计算失败，跳过")
            continue
        
        # 转换日期格式并筛选近N日数据
        try:
            # 确保trade_date是datetime格式（兼容不同数据源的日期格式）
            stock_data['trade_date'] = pd.to_datetime(stock_data['trade_date'])
        except Exception as e:
            print(f"{ts_code} 日期格式转换失败: {e}，跳过")
            continue
        
        # 截取近N日数据（自然日，而非交易日）
        recent_cutoff = datetime.now() - timedelta(days=n_days)
        recent_data = stock_data[stock_data['trade_date'] >= recent_cutoff]
        
        if recent_data.empty:
            print(f"{ts_code} 近{n_days}日无交易数据，跳过")
            continue
        
        # 按日期排序并返回
        result[ts_code] = recent_data.sort_values('trade_date').reset_index(drop=True)
    
    return result

if __name__ == "__main__":
    path = r'D:\self\code\tool_ma_back\all_stock.txt'
    n_days = 20  # 近20日数据
    ma_list = [4, 8, 12, 16, 20, 47]  # 包含长周期均线
    
    # 提取txt文件中的股票数据
    print("正在提取股票数据...")
    stock_data_dict = extract_stock_data_from_txt(path, n_days, ma_list)
    
    if not stock_data_dict:
        print("未提取到任何有效股票数据，程序退出")
        exit()
    
    # 获取用户输入的目标股票代码
    target_code = input("请输入目标股票代码: ").strip()
    
    # 查找相似股票
    print("正在计算相似度，请稍候...")
    similar_stocks = find_similar_stocks(target_code, stock_data_dict, n_days, ma_list)
    
    # 输出结果
    if similar_stocks:
        print("\n===== 最相似的前10只股票 =====")
        print(f"目标股票: {target_code} (近{n_days}天数据)")
        print("排名 | 股票代码 | 相似度")
        print("-" * 30)
        for i, (code, sim) in enumerate(similar_stocks, 1):
            print(f"{i:4d} | {code:8s} | {sim:.4f}")
    else:
        print("未能找到相似的股票")
