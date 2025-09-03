import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from gplearn.genetic import SymbolicTransformer
import os
from typing import List, Dict, Any, Tuple
import sympy as sp
try:
    from sympy.parsing.sympy_parser import parse_expr
except ImportError:
    # 旧版本SymPy使用sympify替代parse_expr
    from sympy import sympify as parse_expr

# 定义均线列表
mas = ['MA4', 'MA8', 'MA12', 'MA16', 'MA20', 'MA47']

def program_to_readable(program, feature_names):
    """递归解析gplearn表达式树，生成可读字符串"""
    # 处理常数节点
    if isinstance(program, (int, float)):
        return str(program)
    
    # 处理特征索引节点
    if isinstance(program, int) and program < len(feature_names):
        return feature_names[program]
    
    # 处理函数节点（gplearn使用元组表示）
    if isinstance(program, tuple) and len(program) >= 1:
        func = program[0]
        
        # 获取函数名称
        func_name = getattr(func, 'name', None) or getattr(func, '__name__', str(func))
        
        # 递归处理所有参数
        args = [program_to_readable(arg, feature_names) for arg in program[1:]]
        
        # 处理二元运算符
        binary_ops = {
            'add': '+',
            'sub': '-',
            'mul': '*',
            'div': '/',
            'protected_div': '/'
        }
        
        if func_name in binary_ops and len(args) == 2:
            return f"({args[0]} {binary_ops[func_name]} {args[1]})"
        
        # 处理一元运算符
        unary_ops = {
            'neg': '-',
            'sqrt': 'sqrt',
            'log': 'log',
            'abs': 'abs',
            'sin': 'sin',
            'cos': 'cos',
            'tan': 'tan'
        }
        
        if func_name in unary_ops and len(args) == 1:
            if func_name == 'neg':
                return f"-({args[0]})"
            return f"{func_name}({args[0]})"
        
        # 处理三元运算符（如max, min）
        if func_name in ['max', 'min'] and len(args) >= 2:
            return f"{func_name}({', '.join(args)})"
        
        # 默认函数调用格式
        return f"{func_name}({', '.join(args)})"
    
    # 处理原始gplearn函数对象
    if hasattr(program, 'function'):
        return program_to_readable(program.function, feature_names)
    
    # 处理Program对象
    if hasattr(program, 'program'):
        return program_to_readable(program.program, feature_names)
    
    # 默认返回原始表示
    return str(program)

class StockFeatureGP(BaseEstimator, TransformerMixin):
    """使用遗传编程生成股票特征的转换器"""
    
    def __init__(self, generations=10, population_size=1000, n_components=10, 
                 function_set=('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'max', 'min')):
        """
        初始化遗传编程特征生成器
        
        参数:
            generations: 遗传算法迭代次数
            population_size: 种群大小
            n_components: 要生成的特征数量
            function_set: 允许的数学函数集合
        """
        self.generations = generations
        self.population_size = population_size
        self.n_components = n_components
        self.function_set = function_set
        self.transformer = None
        self.selected_features = None
    
    def simplify_expression(self, expr_str):
        """简化数学表达式"""
        try:
            # 创建符号变量
            symbols_map = {f'x{i}': sp.symbols(f'x{i}') for i in range(100)}
            
            # 替换常见数学函数为SymPy兼容版本
            replacements = {
                'sqrt': 'sqrt',
                'abs': 'Abs',
                'log': 'log',
                'max': 'Max',
                'min': 'Min'
            }
            
            # 尝试解析表达式
            expr = sp.sympify(expr_str, locals=symbols_map)
            simplified = sp.simplify(expr)
            
            # 转换为字符串并恢复原始函数名
            simplified_str = str(simplified)
            for sympy_name, orig_name in replacements.items():
                simplified_str = simplified_str.replace(sympy_name, orig_name)
            
            return simplified_str
            
        except Exception as e:
            return f"无法简化: {type(e).__name__} - {str(e)}"
    
    def get_feature_expressions(self, feature_names=None, simplify=True):
        """获取生成的特征表达式，并尝试简化"""
        if self.transformer is None:
            return []
        
        expressions = []
        last_generation = self.transformer._programs[-1] if self.transformer._programs else []
        
        # 如果没有提供特征名，使用默认占位符
        if feature_names is None:
            feature_names = [f'X{i}' for i in range(100)]
        
        for i, program in enumerate(last_generation):
            if program is None:
                expressions.append(f"特征 {i+1}: 无效程序")
                continue
                
            try:
                # 解析为可读表达式
                raw_expression = program_to_readable(program, feature_names)
                
                # 移除多余的括号
                if raw_expression.startswith('(') and raw_expression.endswith(')'):
                    raw_expression = raw_expression[1:-1]
                
                if simplify:
                    # 尝试简化表达式
                    simplified = self.simplify_expression(raw_expression)
                    
                    # 仅当简化成功且结果不同时才显示
                    if not simplified.startswith("无法简化") and simplified != raw_expression:
                        expressions.append(f"特征 {i+1}: {simplified} (原始: {raw_expression})")
                    else:
                        expressions.append(f"特征 {i+1}: {raw_expression}")
                else:
                    expressions.append(f"特征 {i+1}: {raw_expression}")
                    
            except Exception as e:
                # 尝试直接输出原始表达式
                try:
                    expr_str = str(program)
                    expressions.append(f"特征 {i+1}: {expr_str} (解析失败: {type(e).__name__})")
                except:
                    expressions.append(f"特征 {i+1}: 解析完全失败 - {type(e).__name__}: {str(e)}")
        
        return expressions
    
    def fit(self, X, y=None):
        """训练遗传编程模型以发现最佳特征"""
        print("开始遗传编程特征发现...")
        
        # 创建遗传编程特征转换器
        self.transformer = SymbolicTransformer(
            generations=self.generations,
            population_size=self.population_size,
            hall_of_fame=100,
            n_components=self.n_components,
            function_set=self.function_set,
            parsimony_coefficient=0.001,
            max_samples=0.9,
            verbose=1,
            random_state=42,
            n_jobs=1
        )
        
        # 训练遗传编程模型
        self.transformer.fit(X, y)
        
        # 记录生成的特征表达式
        self.selected_features = self.transformer._programs
        
        print(f"遗传编程已完成，生成了 {self.n_components} 个特征")
        return self
    
    def transform(self, X):
        """应用遗传编程生成的特征转换数据"""
        if self.transformer is None:
            raise ValueError("请先调用fit方法训练遗传编程模型")
        
        # 生成新特征
        gp_features = self.transformer.transform(X)
        
        # 将新特征与原始特征合并
        if isinstance(X, pd.DataFrame):
            feature_names = [f"gp_feature_{i}" for i in range(gp_features.shape[1])]
            gp_df = pd.DataFrame(gp_features, columns=feature_names, index=X.index)
            return pd.concat([X, gp_df], axis=1)
        else:
            return np.hstack([X, gp_features])
    
    def get_feature_names(self):
        """获取生成的特征名称"""
        if self.transformer is None:
            return []
        
        feature_names = [f"gp_feature_{i}" for i in range(self.n_components)]
        return feature_names

def run_genetic_programming_for_features(train_data: pd.DataFrame, train_labels: np.ndarray, 
                                         feature_columns: List[str], target_column: str,
                                         generations: int = 10, population_size: int = 1000,
                                         n_components: int = 10) -> Tuple[pd.DataFrame, StockFeatureGP]:
    """
    运行遗传编程来发现最佳特征
    
    参数:
        train_data: 训练数据
        train_labels: 训练标签
        feature_columns: 要使用的特征列名
        target_column: 目标列名
        generations: 遗传算法迭代次数
        population_size: 种群大小
        n_components: 要生成的特征数量
        
    返回:
        包含原始特征和新生成特征的DataFrame
        训练好的遗传编程特征生成器
    """
    # 准备训练数据
    X_train = train_data[feature_columns]
    y_train = train_labels
    
    # 创建并训练遗传编程特征生成器
    gp_feature_generator = StockFeatureGP(
        generations=generations,
        population_size=population_size,
        n_components=n_components
    )
    
    # 训练遗传编程模型
    gp_feature_generator.fit(X_train, y_train)
    
    # 生成新特征
    X_train_gp = gp_feature_generator.transform(X_train)
    
    # 输出简化后的特征表达式
    # print("\n最后一代的遗传编程特征表达式:")
    # expressions = gp_feature_generator.get_feature_expressions(feature_names=feature_columns, simplify=True)
    # for expr in expressions:
    #     print(expr)
    
    return X_train_gp, gp_feature_generator

# ... 文件其余部分保持不变 ...

def prepare_data_for_gp(root_folder: str, feature_extraction_func=None) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    准备用于遗传编程的训练数据
    
    参数:
        root_folder: 根数据文件夹
        feature_extraction_func: 特征提取函数
        
    返回:
        合并后的特征DataFrame
        标签数组
    """
    all_features = []
    all_labels = []
    
    # 定义基本特征提取函数（如果未提供）
    if feature_extraction_func is None:
        def feature_extraction_func(df):
            """提取基本特征用于遗传编程"""
            features = {}
            
            # 提取价格相关特征
            features['close'] = df['close'].iloc[-1]
            features['open'] = df['open'].iloc[-1]
            features['high'] = df['high'].iloc[-1]
            features['low'] = df['low'].iloc[-1]

            # 提取成交量特征（注意列名是vol而不是volume）
            if 'vol' in df.columns:
                features['volume'] = df['vol'].iloc[-1]

            # 均线交叉特征
            if len(df) > 10 and all(ma in df.columns for ma in ['MA4', 'MA8', 'MA12', 'MA16', 'MA20', 'MA47']):
                # 短期均线与长期均线的比值
                features['ma4_ma47_ratio'] = df['MA4'].iloc[-1] / df['MA47'].iloc[-1]
                features['ma8_ma47_ratio'] = df['MA8'].iloc[-1] / df['MA47'].iloc[-1]
                features['ma20_ma47_ratio'] = df['MA20'].iloc[-1] / df['MA47'].iloc[-1]
                
                # 均线之间的差距
                features['ma4_ma47_diff'] = df['MA4'].iloc[-1] - df['MA47'].iloc[-1]
                features['ma8_ma47_diff'] = df['MA8'].iloc[-1] - df['MA47'].iloc[-1]
                features['ma20_ma47_diff'] = df['MA20'].iloc[-1] - df['MA47'].iloc[-1]
                
                # 均线排列状态
                features['bullish_alignment'] = 1 if (df['MA4'].iloc[-1] > df['MA8'].iloc[-1] > 
                                                    df['MA12'].iloc[-1] > df['MA16'].iloc[-1] > 
                                                    df['MA20'].iloc[-1] > df['MA47'].iloc[-1]) else 0
                features['bearish_alignment'] = 1 if (df['MA4'].iloc[-1] < df['MA8'].iloc[-1] < 
                                                    df['MA12'].iloc[-1] < df['MA16'].iloc[-1] < 
                                                    df['MA20'].iloc[-1] < df['MA47'].iloc[-1]) else 0

            # 均线斜率特征
            if len(df) > 10 and all(ma in df.columns for ma in ['MA4', 'MA8', 'MA12', 'MA16', 'MA20', 'MA47']):
                # 计算均线的斜率（变化率）
                features['ma4_slope'] = df['MA4'].iloc[-1] / df['MA4'].iloc[-5] - 1
                features['ma8_slope'] = df['MA8'].iloc[-1] / df['MA8'].iloc[-5] - 1
                features['ma12_slope'] = df['MA12'].iloc[-1] / df['MA12'].iloc[-5] - 1
                features['ma16_slope'] = df['MA16'].iloc[-1] / df['MA16'].iloc[-5] - 1
                features['ma20_slope'] = df['MA20'].iloc[-1] / df['MA20'].iloc[-5] - 1
                features['ma47_slope'] = df['MA47'].iloc[-1] / df['MA47'].iloc[-5] - 1
                
                # 均线方向变化
                features['ma4_direction_change'] = 1 if (df['MA4'].iloc[-1] > df['MA4'].iloc[-2] and 
                                                    df['MA4'].iloc[-2] < df['MA4'].iloc[-3]) else 0
                features['ma20_direction_change'] = 1 if (df['MA20'].iloc[-1] > df['MA20'].iloc[-2] and 
                                                    df['MA20'].iloc[-2] < df['MA20'].iloc[-3]) else 0
                
                # 均线斜率的加速度
                features['ma4_acceleration'] = (df['MA4'].iloc[-1] - df['MA4'].iloc[-2]) - (df['MA4'].iloc[-2] - df['MA4'].iloc[-3])
                features['ma20_acceleration'] = (df['MA20'].iloc[-1] - df['MA20'].iloc[-2]) - (df['MA20'].iloc[-2] - df['MA20'].iloc[-3])

            # 价格与均线的关系特征
            if len(df) > 5 and all(ma in df.columns for ma in ['MA4', 'MA8', 'MA12', 'MA16', 'MA20', 'MA47']):
                # 价格相对均线的位置
                features['price_ma4_ratio'] = df['close'].iloc[-1] / df['MA4'].iloc[-1]
                features['price_ma8_ratio'] = df['close'].iloc[-1] / df['MA8'].iloc[-1]
                features['price_ma20_ratio'] = df['close'].iloc[-1] / df['MA20'].iloc[-1]
                features['price_ma47_ratio'] = df['close'].iloc[-1] / df['MA47'].iloc[-1]
                
                # 价格在均线之上/之下
                features['price_above_ma4'] = 1 if df['close'].iloc[-1] > df['MA4'].iloc[-1] else 0
                features['price_above_ma8'] = 1 if df['close'].iloc[-1] > df['MA8'].iloc[-1] else 0
                features['price_above_ma20'] = 1 if df['close'].iloc[-1] > df['MA20'].iloc[-1] else 0
                features['price_above_ma47'] = 1 if df['close'].iloc[-1] > df['MA47'].iloc[-1] else 0
                
                # 价格穿越均线
                features['price_cross_ma4'] = 1 if (df['close'].iloc[-2] <= df['MA4'].iloc[-2] and 
                                                df['close'].iloc[-1] > df['MA4'].iloc[-1]) else 0
                features['price_cross_ma20'] = 1 if (df['close'].iloc[-2] <= df['MA20'].iloc[-2] and 
                                                df['close'].iloc[-1] > df['MA20'].iloc[-1]) else 0

            # 成交量与均线结合特征
            if len(df) > 5 and 'vol' in df.columns and all(ma in df.columns for ma in ['MA4', 'MA8', 'MA12', 'MA16', 'MA20', 'MA47']):
                # 价格上涨但成交量下降（可能是弱趋势）
                features['price_up_volume_down'] = 1 if (df['close'].iloc[-1] > df['close'].iloc[-2] and 
                                                    df['vol'].iloc[-1] < df['vol'].iloc[-2]) else 0
                
                # 价格下跌但成交量下降（可能是弱趋势）
                features['price_down_volume_down'] = 1 if (df['close'].iloc[-1] < df['close'].iloc[-2] and 
                                                    df['vol'].iloc[-1] < df['vol'].iloc[-2]) else 0
                
                # 均线向上且成交量增加（可能是强趋势）
                features['ma_up_volume_up'] = 1 if (df['MA20'].iloc[-1] > df['MA20'].iloc[-2] and 
                                                df['vol'].iloc[-1] > df['vol'].iloc[-2]) else 0
                
                # 均线向下且成交量增加（可能是强趋势）
                features['ma_down_volume_up'] = 1 if (df['MA20'].iloc[-1] < df['MA20'].iloc[-2] and 
                                                df['vol'].iloc[-1] > df['vol'].iloc[-2]) else 0
    
            # 均线波动性特征
            if len(df) > 5 and all(ma in df.columns for ma in ['MA4', 'MA8', 'MA12', 'MA16', 'MA20', 'MA47']):
                # 均线的标准差（波动性）
                features['ma4_std'] = df['MA4'].iloc[-5:].std()
                features['ma8_std'] = df['MA8'].iloc[-5:].std()
                features['ma20_std'] = df['MA20'].iloc[-5:].std()
                
                # 均线标准差的比值（相对波动性）
                features['ma4_ma20_std_ratio'] = features['ma4_std'] / features['ma20_std'] if features['ma20_std'] > 0 else 0
                
                # 均线间距的变化（收敛/发散）
                features['ma4_ma20_spread'] = abs(df['MA4'].iloc[-1] - df['MA20'].iloc[-1])
                features['ma4_ma20_spread_change'] = features['ma4_ma20_spread'] / abs(df['MA4'].iloc[-2] - df['MA20'].iloc[-2]) - 1

            # 综合趋势特征
            if len(df) > 5 and all(ma in df.columns for ma in ['MA4', 'MA8', 'MA12', 'MA16', 'MA20', 'MA47']):
                # 多头趋势强度（0-1之间）
                features['bullish_strength'] = (
                    (1 if df['MA4'].iloc[-1] > df['MA8'].iloc[-1] else 0) +
                    (1 if df['MA8'].iloc[-1] > df['MA12'].iloc[-1] else 0) +
                    (1 if df['MA12'].iloc[-1] > df['MA16'].iloc[-1] else 0) +
                    (1 if df['MA16'].iloc[-1] > df['MA20'].iloc[-1] else 0) +
                    (1 if df['MA20'].iloc[-1] > df['MA47'].iloc[-1] else 0)
                ) / 5.0
                
                # 空头趋势强度（0-1之间）
                features['bearish_strength'] = (
                    (1 if df['MA4'].iloc[-1] < df['MA8'].iloc[-1] else 0) +
                    (1 if df['MA8'].iloc[-1] < df['MA12'].iloc[-1] else 0) +
                    (1 if df['MA12'].iloc[-1] < df['MA16'].iloc[-1] else 0) +
                    (1 if df['MA16'].iloc[-1] < df['MA20'].iloc[-1] else 0) +
                    (1 if df['MA20'].iloc[-1] < df['MA47'].iloc[-1] else 0)
                ) / 5.0
                
                # 趋势一致性（0-1之间，值越高表示均线方向越一致）
                features['trend_consistency'] = max(features['bullish_strength'], features['bearish_strength'])

            # 提取均线相关特征
            for ma in mas:
                if ma in df.columns:
                    features[ma] = df[ma].iloc[-1]
            
            # 提取价格变化特征
            if len(df) > 1:
                features['price_change'] = df['close'].iloc[-1] / df['close'].iloc[-2] - 1
                features['volume_change'] = df['vol'].iloc[-1] / df['vol'].iloc[-2] - 1 if 'vol' in df.columns else 0
            
            # 提取统计特征
            if len(df) > 5:
                features['price_std_5'] = df['close'].iloc[-5:].std()
                features['volume_std_5'] = df['vol'].iloc[-5:].std() if 'vol' in df.columns else 0
            
            return features
    
    # 遍历所有子文件夹
    for label, sub_folder in enumerate(os.listdir(root_folder)):
        sub_folder_path = os.path.join(root_folder, sub_folder)
        if os.path.isdir(sub_folder_path):
            # 处理每个子文件夹中的数据
            for filename in os.listdir(sub_folder_path):
                if filename.endswith('.csv'):
                    file_path = os.path.join(sub_folder_path, filename)
                    try:
                        df = pd.read_csv(file_path, parse_dates=['timestamps'], index_col='timestamps')
                        # 筛选出stage列值为1的行
                        # df = df[df['stage'] == 1]
                        if not df.empty:
                            features = feature_extraction_func(df)
                            if features:
                                all_features.append(features)
                                all_labels.append(label)
                    except Exception as e:
                        print(f"处理文件 {filename} 时出错: {str(e)}")
    
    # 转换为DataFrame
    if all_features:
        df_features = pd.DataFrame(all_features).fillna(0)
        return df_features, np.array(all_labels)
    else:
        print("没有找到有效的特征数据")
        return None, None

def split_data_by_ratio(features_df: pd.DataFrame, labels: np.ndarray, test_size: float = 0.3, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    按指定比例将数据划分为训练集和测试集
    
    参数:
        features_df: 特征DataFrame
        labels: 标签数组
        test_size: 测试集比例
        random_state: 随机种子
        
    返回:
        X_train: 训练特征
        X_test: 测试特征
        y_train: 训练标签
        y_test: 测试标签
    """
    # 使用sklearn的train_test_split函数按比例划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    return X_train, X_test, y_train, y_test

def evaluate_generated_features(X_train: pd.DataFrame, y_train: np.ndarray, 
                               X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, Any]:
    """
    评估遗传编程生成的特征性能
    
    参数:
        X_train: 训练特征
        y_train: 训练标签
        X_test: 测试特征
        y_test: 测试标签
        
    返回:
        包含评估结果的字典
    """
    # 训练随机森林模型
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # 在测试集上评估
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # 获取特征重要性
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    # 评估结果
    results = {
        'accuracy': accuracy,
        'feature_importance': feature_importance
    }
    
    return results