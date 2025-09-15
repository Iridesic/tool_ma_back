import joblib
import pandas as pd
import os
import chardet
import shutil

from aaa_classify_code.setter_stage2 import prepare_data_for_gp

# training_feature_path = ''

# 新增：加载训练时的原始特征列名
def load_training_feature_columns(column_path):
    """加载训练时使用的原始特征列名"""
    if not os.path.exists(column_path):
        raise FileNotFoundError(f"训练特征列名文件 {column_path} 不存在，请先运行训练脚本生成")
    return pd.read_csv(column_path)['feature'].tolist()

def detect_encoding(file_path):
    """检测文件编码"""
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

def predict_from_csv(csv_file_path, model_path, training_feature_path, history_days=50):
    """从CSV文件加载数据并使用已训练的模型进行预测"""
    # 加载模型管道
    try:
        pipeline = joblib.load(model_path)
        print(f"成功加载模型: {model_path}")
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        return None, None
    
    # 读取并准备CSV数据
    try:
        # 检测文件编码
        current_encoding = detect_encoding(csv_file_path)
        current_df = pd.read_csv(csv_file_path, parse_dates=['timestamps'], encoding=current_encoding)
        
        if 'timestamps' not in current_df.columns:
            raise ValueError("CSV文件中缺少'timestamps'列")
        
        stage2_df = current_df[current_df['stage'] == 2].copy()
        if stage2_df.empty:
            raise ValueError("CSV文件中无stage=2的数据，无需预测")
        
        # 2. 以stage=2的最小日期作为基准，向前推50天（而非整个CSV的最小日期）
        stage2_min_date = stage2_df['timestamps'].min()  # stage=2数据的起始日期
        end_date = current_df['timestamps'].max()        # 整个CSV的结束日期（不变）
        history_start_date = stage2_min_date - pd.Timedelta(days=history_days)  # 仅为stage=2推历史

        source_csv_path = find_source_csv_with_history(history_start_date, end_date, csv_file_path)
        
        if not source_csv_path:
            raise FileNotFoundError(f"找不到包含 {history_start_date} 到 {end_date} 日期范围的源CSV文件")
            
        source_encoding = detect_encoding(source_csv_path)
        source_df = pd.read_csv(source_csv_path, parse_dates=['timestamps'], encoding=source_encoding)
        required_df = source_df[(source_df['timestamps'] >= history_start_date) & 
                                (source_df['timestamps'] <= end_date)]
        
        if required_df.empty:
            raise ValueError(f"源CSV文件中不包含 {history_start_date} 到 {end_date} 的数据")
            
        # 保存临时文件用于特征提取
        temp_folder = 'temp_prediction_data'
        os.makedirs(temp_folder, exist_ok=True)
        
        # 创建子文件夹结构
        stock_code = required_df['code'].iloc[0]
        stock_folder = os.path.join(temp_folder, str(stock_code))
        os.makedirs(stock_folder, exist_ok=True)
        
        temp_file_path = os.path.join(stock_folder, 'data.csv')
        required_df.to_csv(temp_file_path, index=False)
        
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
        
        # 关键修改1：加载训练时的原始特征列名，确保特征一致性
        training_feature_columns = load_training_feature_columns(column_path=training_feature_path)
        
        # 关键修改2：对齐原始特征（补充缺失列，删除多余列）
        # 补充缺失的特征列（用训练时的中位数填充）
        missing_cols = [col for col in training_feature_columns if col not in features_df.columns]
        if missing_cols:
            print(f"补充缺失的原始特征: {missing_cols}")
            # 这里假设训练时的特征中位数已保存，实际使用时需要提前保存
            # 简化处理：用0填充（实际应替换为训练集的中位数）
            for col in missing_cols:
                features_df[col] = 0
        
        # 删除多余的特征列
        extra_cols = [col for col in features_df.columns if col not in training_feature_columns]
        if extra_cols:
            print(f"删除多余的原始特征: {extra_cols}")
            features_df = features_df.drop(columns=extra_cols)
        
        # 筛选与当前CSV对应的特征
        if 'timestamps' in features_df.columns:
            final_features = features_df[features_df['timestamps'].isin(current_df['timestamps'])]
            # 移除timestamps列（训练时没有该列）
            final_features = final_features.drop(columns=['timestamps'], errors='ignore')
        elif isinstance(features_df.index, pd.DatetimeIndex):
            final_features = features_df.loc[current_df['timestamps'].tolist()]
        else:
            final_features = features_df
            
        if final_features.empty:
            print("筛选后没有剩余有效特征")
            return None, None
            
        # 关键修改3：确保特征顺序与训练时一致
        final_features = final_features[training_feature_columns]
        
        print(f"提取的原始特征形状: {features_df.shape}")
        print(f"对齐后用于生成特征的原始特征形状: {final_features.shape}")
        
        # 进行预测（Pipeline会自动调用gp_feature_generator生成新特征）
        predictions = pipeline.predict(final_features)
        prediction_probs = pipeline.predict_proba(final_features)

        # 获取模型的实际类别标签
        class_labels = pipeline.classes_

        # 输出预测结果
        for date, pred_idx, probs in zip(current_df['timestamps'], predictions, prediction_probs):
            class_label = class_labels[pred_idx]
            print(f"日期: {date.date()}")
            print(f"  预测类别: {class_label}")
            for class_name, prob in zip(class_labels, probs):
                print(f"  类别 {class_name} 的概率: {prob:.4f}")
            print()  # 每条记录之间空一行
                
        return predictions, prediction_probs
        
    except Exception as e:
        print(f"预测过程中出错: {str(e)}")
        return None, None

# 其他函数（find_source_csv_with_history等）保持不变
def find_source_csv_with_history(start_date, end_date, input_csv_path):
    """
    查找包含指定时间范围的源CSV文件
    :param start_date: 历史起始日期（需覆盖）
    :param end_date: 结束日期（输入CSV的最大日期）
    :param input_csv_path: 输入CSV的路径（优先检查其是否包含历史数据）
    :return: 包含所需数据的CSV路径，无则返回None
    """
    # 第一步：优先检查输入CSV本身是否包含足够历史数据（避免依赖其他文件）
    if os.path.exists(input_csv_path):
        input_encoding = detect_encoding(input_csv_path)
        input_df = pd.read_csv(input_csv_path, parse_dates=['timestamps'], encoding=input_encoding)
        # 检查输入CSV是否覆盖 [start_date, end_date] 范围
        if (input_df['timestamps'].min() <= start_date) and (input_df['timestamps'].max() >= end_date):
            print(f"使用输入CSV作为源文件（已包含所需历史数据）: {input_csv_path}")
            return input_csv_path
    
    # 第二步：若输入CSV不满足，再检查默认路径（保留原逻辑）
    default_path = r'D:\self\code\ma_final\data\stage2_class_predict.csv'
    if os.path.exists(default_path):
        default_encoding = detect_encoding(default_path)
        default_df = pd.read_csv(default_path, parse_dates=['timestamps'], encoding=default_encoding)
        if (default_df['timestamps'].min() <= start_date) and (default_df['timestamps'].max() >= end_date):
            print(f"使用默认源文件: {default_path}")
            return default_path
    
    # 第三步：均不满足时返回None
    print(f"输入CSV和默认文件均不包含 {start_date} 到 {end_date} 的数据")
    return None


# 使用的时候需要修改setter中stage的值！！！！####################################################
##### 这个可以用0913 ##########################################################################
if __name__ == "__main__":
    # 短数据检测
    csv_file = r'D:\self\code\ma_final\data\000021_fragment_1_20220627_to_20220630_extended.csv'
    # 长数据检测
    # csv_file = r'D:\self\code\ma_final\data\stage2_class_predict.csv'
    try:
        df = pd.read_csv(csv_file)
        stage2_count = len(df[df['stage'] == 2])
        print(f"CSV文件中stage=2的行数: {stage2_count}")

        if stage2_count > 4:
            model_path = r'D:\self\code\ma_final\model_new\stage2_long\stock_classifier_pipeline.pkl'
            training_feature_path = r'D:\self\code\ma_final\model_new\stage2_long\training_feature_columns.csv'
            print("使用长模型路径:", model_path)
        else:
            model_path = r'D:\self\code\ma_final\model_new\stage2_short\stock_classifier_pipeline.pkl'
            training_feature_path = r'D:\self\code\ma_final\model_new\stage2_short\training_feature_columns.csv'
            print("使用短模型路径:", model_path)

        # 训练时需提前保存原始特征列名（在gp.py中添加）
        # 这里假设已运行gp.py生成training_feature_columns.csv
        predictions, probs = predict_from_csv(csv_file, model_path, training_feature_path)
        
        if predictions is not None:
            print(f"完成预测，共处理 {len(predictions)} 个样本")
    except FileNotFoundError:
        print(f"错误: CSV文件未找到 - {csv_file}")
    except Exception as e:
        print(f"发生未知错误: {e}")