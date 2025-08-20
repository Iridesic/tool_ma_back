import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import os
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from dtw import dtw
from scipy.interpolate import interp1d  # 用于序列长度统一
from utils import calculate_ma
from pattern_detection import is_golden_cross


# 新增：统一序列长度的通用函数
def unify_series_length(series1, series2, target_length=None):
    """
    将两个序列统一为相同长度
    如果未指定目标长度，则使用较长序列的长度
    使用线性插值保持趋势特征，处理长度为1的特殊情况
    """
    s1 = np.array(series1)
    s2 = np.array(series2)
    
    # 确定目标长度
    if target_length is None:
        target_length = max(len(s1), len(s2))
    
    # 处理序列长度为1的特殊情况（避免插值除零）
    def handle_single_element(series, target_len):
        if len(series) == 1:
            return np.full(target_len, series[0])
        x = np.linspace(0, 1, len(series))
        f = interp1d(x, series, kind='linear', fill_value="extrapolate")
        x_new = np.linspace(0, 1, target_len)
        return f(x_new)
    
    # 对两个序列分别处理
    s1_unified = handle_single_element(s1, target_length)
    s2_unified = handle_single_element(s2, target_length)
    
    return s1_unified, s2_unified

def get_stock_data_from_csv(code, start_date=None, end_date=None, data_folder=None, use_full_history=False):
    """从CSV文件获取股票数据，新增返回股票名称"""
    csv_path = os.path.join(data_folder, f"{code}.csv") if data_folder else f"{code}.csv"
    
    if not os.path.exists(csv_path):
        print(f"股票 {code} 的CSV文件不存在: {csv_path}")
        return pd.DataFrame(), ""  # 返回空DataFrame和空名称
    
    try:
        df = pd.read_csv(csv_path)
        required_columns = ['open', 'close', 'high', 'low', 'timestamps']
        # 检查是否包含股票名称列
        has_name_column = 'name' in df.columns
        
        if not set(required_columns).issubset(df.columns):
            print(f"股票 {code} 的CSV文件缺少必要的列，现有列: {df.columns.tolist()}")
            return pd.DataFrame(), ""
        
        # 提取股票名称（取第一个非空值）
        stock_name = ""
        if has_name_column:
            name_series = df['name'].dropna()
            if not name_series.empty:
                stock_name = str(name_series.iloc[0]).strip()
            else:
                print(f"股票 {code} 的CSV文件中没有有效的股票名称数据")
        
        df['trade_date'] = pd.to_datetime(df['timestamps'], errors='coerce')
        if df['trade_date'].isna().any():
            print(f"股票 {code} 的CSV文件中存在无效的时间戳格式")
        
        df_sorted = df.sort_values('trade_date').reset_index(drop=True)
        
        if use_full_history:
            return df_sorted, stock_name
        
        if start_date and end_date:
            try:
                start = pd.to_datetime(start_date)
                end = pd.to_datetime(end_date)
                mask = (df_sorted['trade_date'] >= start) & (df_sorted['trade_date'] <= end)
                filtered_df = df_sorted.loc[mask].reset_index(drop=True)
                print(f"股票 {code} 在指定期间共找到 {len(filtered_df)} 条数据")
                return filtered_df, stock_name
            except Exception as e:
                print(f"筛选 {code} 时间范围时出错: {e}")
        
        return df_sorted, stock_name
    except Exception as e:
        print(f"读取股票 {code} 的CSV文件时出错: {e}")
        return pd.DataFrame(), ""

def find_similar_stocks(target_code, target_start_date, target_end_date, 
                       stock_data_dict, n_days, ma_list,
                       group_weights=None, single_ma_weights=None,
                       crossover_weights=None):
    """找到与目标股票指定时间段数据相似的股票"""
    if target_code.endswith(('.SH', '.SZ')):
        target_code = target_code[:-3]
    
    if '_data_folder' in stock_data_dict:
        data_folder = stock_data_dict['_data_folder']
        # 分离股票数据和名称信息
        stock_pool = {}
        stock_names = {}
        for k, v in stock_data_dict.items():
            if k != '_data_folder' and k != '_stock_names':
                stock_pool[k] = v
        if '_stock_names' in stock_data_dict:
            stock_names = stock_data_dict['_stock_names']
    else:
        print("无法确定数据文件夹路径")
        return []
    
    # 获取目标股票全量数据计算均线
    print(f"正在获取目标股票 {target_code} 的全量历史数据...")
    full_target_data, target_name = get_stock_data_from_csv(
        target_code, 
        data_folder=data_folder,
        use_full_history=True
    )
    
    if full_target_data.empty:
        print(f"无法获取目标股票 {target_code} 的全量历史数据")
        return []
    
    # 计算均线
    full_target_data = calculate_ma(full_target_data, ma_list)
    if full_target_data.empty:
        print(f"目标股票 {target_code} 均线计算失败")
        return []
    
    # 截取目标时间段数据
    start = pd.to_datetime(target_start_date)
    end = pd.to_datetime(target_end_date)
    mask = (full_target_data['trade_date'] >= start) & (full_target_data['trade_date'] <= end)
    target_data = full_target_data.loc[mask].reset_index(drop=True)
    
    if target_data.empty:
        print(f"目标股票 {target_code} 在指定时间段内无数据")
        return []
    
    target_length = len(target_data)
    if target_length < 2:
        print(f"目标股票 {target_code} 在指定时间段内数据不足（仅{target_length}条）")
        return []
    
    print(f"成功获取目标股票数据，共 {target_length} 条记录")
    
    # 筛选有效股票（增加数据长度要求）
    valid_stocks = {}
    for code, data in stock_pool.items():
        if len(data) >= max(2, min(target_length // 2, 5)):  # 确保有足够数据进行比较
            valid_stocks[code] = data
        else:
            print(f"股票 {code} 数据不足（{len(data)}条），跳过比较")
    
    if not valid_stocks:
        print("没有找到可比较的有效股票数据")
        return []
    
    # 计算目标股票的均线组
    target_ma_group = {f'MA{period}': target_data[f'MA{period}'].values for period in ma_list}
    
    # 计算相似度
    similarity_results = []
    for code, data in valid_stocks.items():
        try:
            ma_group = {f'MA{period}': data[f'MA{period}'].values for period in ma_list}
            # 计算相似度时会自动处理长度不匹配
            similarity = calculate_ma_group_similarity(
                target_ma_group, ma_group, ma_list,
                group_weights=group_weights, 
                single_ma_weights=single_ma_weights,
                crossover_weights=crossover_weights
            )
            # 获取股票名称，若无则使用代码
            stock_name = stock_names.get(code, code)
            similarity_results.append((code, stock_name, similarity['overall_similarity']))
            print(f"计算 {code}({stock_name}) 相似度完成: {similarity['overall_similarity']:.4f}")
        except Exception as e:
            print(f"计算 {code} 相似度时出错: {e}")
            continue
    
    similarity_results.sort(key=lambda x: x[2], reverse=True)
    return similarity_results[:20]


# ==================== 相似度计算函数（全部支持长度不匹配） ====================
def identify_crossovers(ma_short, ma_long):
    """识别交叉点，自动处理长度不匹配"""
    # 先统一长度
    ma_short_unified, ma_long_unified = unify_series_length(ma_short, ma_long)
    
    golden_cross = []
    death_cross = []
    
    for i in range(1, len(ma_short_unified)):
        temp_data = pd.DataFrame({
            f'MA{len(ma_short_unified)}': [ma_short_unified[i-1], ma_short_unified[i]],
            f'MA{len(ma_long_unified)}': [ma_long_unified[i-1], ma_long_unified[i]]
        })
        if is_golden_cross(temp_data, len(ma_short_unified), len(ma_long_unified)):
            golden_cross.append(i)
        if (ma_short_unified[i-1] >= ma_long_unified[i-1] and ma_short_unified[i] < ma_long_unified[i]):
            death_cross.append(i)
    
    return golden_cross, death_cross

def calculate_crossover_similarity(ma_group1, ma_group2, ma_periods, crossover_weights=None):
    """计算交叉模式相似度，自动处理长度不匹配，支持金叉死叉特征自定义权重"""
    # 设置交叉特征默认权重
    if crossover_weights is None:
        crossover_weights = {
            'golden_count': 0.25,    # 金叉频率权重
            'golden_position': 0.25, # 金叉位置权重
            'death_count': 0.25,     # 死叉频率权重
            'death_position': 0.25   # 死叉位置权重
        }
    
    # 验证权重是否有效
    weight_sum = sum(crossover_weights.values())
    if not np.isclose(weight_sum, 1.0):
        # 权重之和不为1时进行归一化
        print(f"交叉特征权重之和为{weight_sum}，自动归一化处理")
        for key in crossover_weights:
            crossover_weights[key] /= weight_sum
    
    cross_pairs = []
    for i in range(len(ma_periods)):
        for j in range(i+1, len(ma_periods)):
            cross_pairs.append((ma_periods[i], ma_periods[j]))
    
    total_golden_count = 0
    total_golden_position = 0
    total_death_count = 0
    total_death_position = 0
    cross_count = len(cross_pairs)
    
    for short_period, long_period in cross_pairs:
        # 获取两条均线
        ma1_short = ma_group1[f'MA{short_period}']
        ma1_long = ma_group1[f'MA{long_period}']
        ma2_short = ma_group2[f'MA{short_period}']
        ma2_long = ma_group2[f'MA{long_period}']
        
        # 识别交叉点（内部已处理长度统一）
        g1_golden, g1_death = identify_crossovers(ma1_short, ma1_long)
        g2_golden, g2_death = identify_crossovers(ma2_short, ma2_long)
        
        # 计算交叉点数量相似度（频率）
        count_golden1, count_golden2 = len(g1_golden), len(g2_golden)
        count_death1, count_death2 = len(g1_death), len(g2_death)
        
        golden_count_sim = 1 - abs(count_golden1 - count_golden2) / (max(count_golden1, count_golden2, 1) + 1e-10)
        death_count_sim = 1 - abs(count_death1 - count_death2) / (max(count_death1, count_death2, 1) + 1e-10)
        
        # 计算交叉点位置相似度
        def calculate_position_similarity(pos1, pos2, max_length):
            if len(pos1) == 0 or len(pos2) == 0:
                return 1.0 if len(pos1) == len(pos2) else 0.0
            
            # 统一长度
            pos1_unified, pos2_unified = unify_series_length(pos1, pos2)
            
            # 处理短序列的DTW问题
            if len(pos1_unified) < 2 or len(pos2_unified) < 2:
                # 用均值差异计算相似度
                mean_diff = abs(np.mean(pos1_unified) - np.mean(pos2_unified))
                return 1 - min(mean_diff / max_length, 1.0)
            
            # 正常DTW计算
            try:
                dtw_dist = dtw(pos1_unified, pos2_unified).distance
                return 1 - dtw_dist / (max_length * max(len(pos1), len(pos2)) + 1e-10)
            except:
                # DTW计算失败时的备选方案
                mean_diff = abs(np.mean(pos1_unified) - np.mean(pos2_unified))
                return 1 - min(mean_diff / max_length, 1.0)
        
        golden_pos_sim = calculate_position_similarity(g1_golden, g2_golden, len(ma1_short))
        death_pos_sim = calculate_position_similarity(g1_death, g2_death, len(ma1_short))
        
        # 累加各特征相似度
        total_golden_count += golden_count_sim
        total_golden_position += golden_pos_sim
        total_death_count += death_count_sim
        total_death_position += death_pos_sim
    
    # 计算各特征的平均相似度
    avg_golden_count = total_golden_count / cross_count if cross_count > 0 else 0
    avg_golden_position = total_golden_position / cross_count if cross_count > 0 else 0
    avg_death_count = total_death_count / cross_count if cross_count > 0 else 0
    avg_death_position = total_death_position / cross_count if cross_count > 0 else 0
    
    # 根据自定义权重计算交叉特征整体相似度
    overall_crossover_similarity = (
        avg_golden_count * crossover_weights['golden_count'] +
        avg_golden_position * crossover_weights['golden_position'] +
        avg_death_count * crossover_weights['death_count'] +
        avg_death_position * crossover_weights['death_position']
    )
    
    return {
        'golden_count_similarity': avg_golden_count,
        'golden_position_similarity': avg_golden_position,
        'death_count_similarity': avg_death_count,
        'death_position_similarity': avg_death_position,
        'overall_crossover_similarity': overall_crossover_similarity,
        'crossover_weights_used': crossover_weights  # 返回实际使用的权重（可能经过归一化）
    }

def calculate_single_ma_similarity(series1, series2):
    """计算单条均线相似度，自动处理长度不匹配"""
    # 统一序列长度
    s1, s2 = unify_series_length(series1, series2)
    n = len(s1)
    
    # 趋势方向一致性
    diff1 = np.diff(s1) if n > 1 else np.array([0])
    diff2 = np.diff(s2) if n > 1 else np.array([0])
    direction_agreement = np.mean(np.sign(diff1) == np.sign(diff2)) if len(diff1) > 0 else 0.5
    
    # 趋势强度相似度
    if n >= 2:
        x = np.arange(n).reshape(-1, 1)
        model1 = LinearRegression().fit(x, s1)
        model2 = LinearRegression().fit(x, s2)
        slope1, slope2 = model1.coef_[0], model2.coef_[0]
        slope_similarity = 1 - abs(slope1 - slope2) / (abs(slope1) + abs(slope2) + 1e-10)
        slope_similarity = max(0, min(1, slope_similarity))
    else:
        slope_similarity = 1.0 if abs(s1[0] - s2[0]) < 1e-10 else 0.0
    
    # 波动特性相似度
    std1, std2 = np.std(s1), np.std(s2)
    volatility_similarity = 1 - abs(std1 - std2) / (std1 + std2 + 1e-10)
    
    # 皮尔逊相关系数（现在长度一致）
    if n >= 2:
        pearson_corr, _ = pearsonr(s1, s2)
        pearson_similarity = (pearson_corr + 1) / 2  # 转换到0-1范围
    else:
        pearson_similarity = 1.0 if abs(s1[0] - s2[0]) < 1e-10 else 0.0
    
    # 斯皮尔曼等级相关系数（现在长度一致）
    if n >= 2:
        spearman_corr, _ = spearmanr(s1, s2)
        spearman_similarity = (spearman_corr + 1) / 2  # 转换到0-1范围
    else:
        spearman_similarity = 1.0 if abs(s1[0] - s2[0]) < 1e-10 else 0.0
    
    # 动态时间规整距离（增加短序列处理）
    if n < 2:
        # 短序列直接计算值差异
        diff = abs(s1[0] - s2[0])
        max_val = max(np.max(s1), np.max(s2))
        min_val = min(np.min(s1), np.min(s2))
        max_possible = max_val - min_val if max_val != min_val else 1e-10
        dtw_similarity = 1 - min(diff / max_possible, 1.0)
    else:
        try:
            dtw_result = dtw(s1, s2)
            d = dtw_result.distance
            max_possible = np.sqrt(sum((np.max([s1, s2]) - np.min([s1, s2]))**2 for _ in range(n)))    
            dtw_similarity = 1 - min(d / max_possible, 1.0)
        except:
            # DTW计算失败时的备选方案
            diff = np.mean(np.abs(s1 - s2))
            max_val = max(np.max(s1), np.max(s2))
            min_val = min(np.min(s1), np.min(s2))
            max_possible = max_val - min_val if max_val != min_val else 1e-10
            dtw_similarity = 1 - min(diff / max_possible, 1.0)
    
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

def calculate_ma_group_similarity(ma_group1, ma_group2, ma_periods=None,
                                 group_weights=None, single_ma_weights=None,
                                 crossover_weights=None):
    """综合评估均线组相似度，自动处理长度不匹配，支持交叉特征自定义权重"""
    if ma_periods is None:
        ma_periods = [4, 8, 12, 16, 20, 47]
    
    # 验证输入
    for period in ma_periods:
        ma_key = f'MA{period}'
        if ma_key not in ma_group1 or ma_key not in ma_group2:
            raise ValueError(f"两组均线都必须包含 {ma_key}")
    
    # 默认权重 - 组间权重
    if group_weights is None:
        group_weights = {
            'single_ma_features': 0.6,
            'crossover_features': 0.4
        }
    
    # 验证组间权重
    group_weight_sum = sum(group_weights.values())
    if not np.isclose(group_weight_sum, 1.0):
        print(f"组间权重之和为{group_weight_sum}，自动归一化处理")
        for key in group_weights:
            group_weights[key] /= group_weight_sum
    
    # 默认权重 - 单均线特征权重
    if single_ma_weights is None:
        single_ma_weights = {
            'trend_direction_agreement': 0.15,
            'trend_strength_similarity': 0.1,
            'volatility_similarity': 0.15,
            'pearson_similarity': 0.2,
            'spearman_similarity': 0.15,
            'dtw_similarity': 0.15,
            'mean_level_similarity': 0.1
        }
    
    # 验证单均线权重
    single_ma_weight_sum = sum(single_ma_weights.values())
    if not np.isclose(single_ma_weight_sum, 1.0):
        print(f"单均线特征权重之和为{single_ma_weight_sum}，自动归一化处理")
        for key in single_ma_weights:
            single_ma_weights[key] /= single_ma_weight_sum
    
    # 计算每条均线的相似性并取平均
    single_ma_metrics = {}
    for metric in single_ma_weights.keys():
        single_ma_metrics[metric] = 0.0
    
    for period in ma_periods:
        ma_key = f'MA{period}'
        # 计算单条均线相似度（内部已处理长度统一）
        ma_similarity = calculate_single_ma_similarity(
            ma_group1[ma_key], ma_group2[ma_key]
        )
        for metric, value in ma_similarity.items():
            single_ma_metrics[metric] += value / len(ma_periods)
    
    # 计算交叉模式相似度（内部已处理长度统一）
    crossover_metrics = calculate_crossover_similarity(
        ma_group1, ma_group2, ma_periods, 
        crossover_weights=crossover_weights
    )
    
    # 计算单条均线特征的综合相似度
    single_ma_overall = 0.0
    for metric, weight in single_ma_weights.items():
        single_ma_overall += single_ma_metrics[metric] * weight
    
    # 计算总体综合相似度
    overall_similarity = (
        single_ma_overall * group_weights['single_ma_features'] +
        crossover_metrics['overall_crossover_similarity'] * group_weights['crossover_features']
    )
    
    return {**single_ma_metrics,** crossover_metrics,
              'single_ma_overall': single_ma_overall,
              'overall_similarity': overall_similarity,
              'group_weights_used': group_weights,
              'single_ma_weights_used': single_ma_weights}


# 使用前端传来的stock_pool股票池数据进行候选查找
def get_candidate_stocks(stock_pool, data_folder, n_days, ma_list):
    """
    获取候选股票数据
    参数:
        stock_pool: 前端传入的股票池列表，为空则读取全部股票
        data_folder: 股票数据文件夹路径
        n_days: 需要提取的最近天数
        ma_list: 均线周期列表

    返回:
        包含候选股票数据的字典，结构同extract_stock_data_from_folder的返回值
    """
    stock_codes = stock_pool
    print(f"使用前端传入的股票池，共{len(stock_codes)}只股票")
    
    # 初始化结果字典，包含数据文件夹路径和股票名称存储
    result = {'_data_folder': data_folder, '_stock_names': {}}
    
    for code in stock_codes:
        print(f"\n正在处理股票 {code}...")
        # 获取股票全量数据和名称
        full_data, stock_name = get_stock_data_from_csv(
            code, 
            data_folder=data_folder,
            use_full_history=True
        )
        if full_data.empty:
            print(f"股票 {code} 全量数据为空，跳过")
            continue
        
        # 存储股票名称
        result['_stock_names'][code] = stock_name if stock_name else code
        
        # 检查数据是否足够计算均线
        max_ma_period = max(ma_list) if ma_list else 0
        if len(full_data) < max_ma_period:
            print(f"股票 {code}({stock_name}) 数据不足 {max_ma_period} 条，无法计算均线，跳过")
            continue
        
        # 计算均线
        full_data = calculate_ma(full_data, ma_list)
        if full_data.empty:
            print(f"股票 {code}({stock_name}) 均线计算失败，跳过")
            continue
        
        # 提取最近n天数据
        if len(full_data) >= n_days:
            recent_data = full_data.sort_values('trade_date').tail(n_days).reset_index(drop=True)
            print(f"股票 {code}({stock_name}) 提取了最后 {n_days} 条数据")
        else:
            recent_data = full_data.sort_values('trade_date').reset_index(drop=True)
            print(f"股票 {code}({stock_name}) 数据不足 {n_days} 条，使用全部数据")
        
        result[code] = recent_data
    
    return result


# 默认使用全部候选股票进行查找
def extract_stock_data_from_folder(data_folder, n_days, ma_list):
    """从文件夹提取股票池数据，新增存储股票名称"""
    try:
        files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
        print(f"在数据文件夹中找到 {len(files)} 个CSV文件")
    except Exception as e:
        print(f"读取数据文件夹出错: {e}")
        return {}
    
    if not files:
        print("数据文件夹中未找到任何CSV文件")
        return {}
    
    stock_codes = [os.path.splitext(f)[0] for f in files]
    result = {'_data_folder': data_folder, '_stock_names': {}}  # 新增_stock_names键存储名称
    
    for code in stock_codes:
        print(f"\n正在处理股票 {code}...")
        full_data, stock_name = get_stock_data_from_csv(code, data_folder=data_folder, use_full_history=True)
        if full_data.empty:
            print(f"股票 {code} 全量数据为空，跳过")
            continue
        
        # 存储股票名称
        result['_stock_names'][code] = stock_name if stock_name else code
        
        first_date = full_data['trade_date'].min()
        last_date = full_data['trade_date'].max()
        print(f"股票 {code}({stock_name}) 全量数据包含 {len(full_data)} 条记录，时间范围: {first_date.date()} 至 {last_date.date()}")
        
        max_ma_period = max(ma_list) if ma_list else 0
        if len(full_data) < max_ma_period:
            print(f"股票 {code}({stock_name}) 数据不足 {max_ma_period} 条，无法计算均线，跳过")
            continue
        
        full_data = calculate_ma(full_data, ma_list)
        if full_data.empty:
            print(f"股票 {code}({stock_name}) 均线计算失败，跳过")
            continue
        
        if len(full_data) >= n_days:
            recent_data = full_data.sort_values('trade_date').tail(n_days).reset_index(drop=True)
            print(f"股票 {code}({stock_name}) 提取了最后 {n_days} 条数据")
        else:
            recent_data = full_data.sort_values('trade_date').reset_index(drop=True)
            print(f"股票 {code}({stock_name}) 数据不足 {n_days} 条，使用全部数据")
        
        result[code] = recent_data
    
    return result

if __name__ == "__main__":
    data_folder = r'D:\self\data\kline-data'
    n_days = 20
    ma_list = [4, 8, 12, 16, 20, 47]
    
    # 默认权重配置 ################################################
    default_group_weights = {
        'single_ma_features': 0.6,
        'crossover_features': 0.4
    }
    
    default_single_ma_weights = {
        'trend_direction_agreement': 0.15, 
        'trend_strength_similarity': 0.1,
        'volatility_similarity': 0.15,
        'pearson_similarity': 0.2,
        'spearman_similarity': 0.15,
        'dtw_similarity': 0.15,
        'mean_level_similarity': 0.1
    }
    
    # 新增：交叉特征默认权重（金叉、死叉的位置和频率）
    default_crossover_weights = {
        'golden_count': 0.25,    # 金叉频率权重
        'golden_position': 0.25, # 金叉位置权重
        'death_count': 0.25,     # 死叉频率权重
        'death_position': 0.25   # 死叉位置权重
    }
    ##############################################################

    print("正在提取股票池数据...")
    stock_data_dict = extract_stock_data_from_folder(data_folder, n_days, ma_list)
    
    if not stock_data_dict or len(stock_data_dict) <= 2:  # 因为包含了_data_folder和_stock_names
        print("未提取到任何有效股票数据，程序退出")
        exit()
    
    target_code = input("请输入目标股票代码: ").strip()
    target_start_date = input("请输入起始日期 (格式: YYYY-MM-DD): ").strip()
    target_end_date = input("请输入终止日期 (格式: YYYY-MM-DD): ").strip()
    
    try:
        pd.to_datetime(target_start_date)
        pd.to_datetime(target_end_date)
    except ValueError:
        print("日期格式错误，请使用YYYY-MM-DD格式")
        exit()
    
    print("正在计算相似度，请稍候...")
    similar_stocks = find_similar_stocks(
        target_code, target_start_date, target_end_date,
        stock_data_dict, n_days, ma_list,
        group_weights=default_group_weights,
        single_ma_weights=default_single_ma_weights,
        crossover_weights=default_crossover_weights  # 传入交叉特征权重
    )
    
    if similar_stocks:
        # 获取目标股票名称
        target_name = stock_data_dict['_stock_names'].get(target_code, target_code)
        print(f"\n===== 与 {target_code}({target_name}) 在 {target_start_date} 至 {target_end_date} 期间走势最相似的股票 =====")
        print(f"对比基准: 股票池从CSV末尾向前取{n_days}条数据")
        print("排名 | 股票代码 | 股票名称 | 相似度")
        print("-" * 50)
        for i, (code, name, sim) in enumerate(similar_stocks, 1):
            print(f"{i:4d} | {code:8s} | {name:10s} | {sim:.4f}")
    else:
        print("未能找到相似的股票")
