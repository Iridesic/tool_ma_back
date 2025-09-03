import joblib
import pandas as pd
import os
from setter import prepare_data_for_gp
import chardet
import shutil

training_feature_path = ''

def load_training_feature_columns(column_path=training_feature_path):
    """加载训练时使用的原始特征列名"""
    if not os.path.exists(column_path):
        raise FileNotFoundError(f"训练特征列名文件 {column_path} 不存在，请先运行训练脚本生成")
    return pd.read_csv(column_path)['feature'].tolist()

def detect_encoding(file_path):
    """检测文件编码"""
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

def predict_from_data(stage1_data, history_data, merged_data, model_path):
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
            print(f"补充缺失的原始特征: {missing_cols}")
            for col in missing_cols:
                features_df[col] = 0
        
        # 删除多余列
        extra_cols = [col for col in features_df.columns if col not in training_feature_columns]
        if extra_cols:
            print(f"删除多余的原始特征: {extra_cols}")
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

if __name__ == "__main__":
    # 测试流程：加载数据 -> 拆分三种数据 -> 调用预测函数
    try:
        # 1. 加载原始数据文件（模拟外部数据来源）
        test_csv = r'D:\self\code\ma_final\data\stage1_class_predict copy.csv'
        encoding = detect_encoding(test_csv)
        raw_data = pd.read_csv(test_csv, parse_dates=['timestamps'], encoding=encoding)
        
        # 2. 拆分三种输入数据
        # 2.1 当前stage==1的数据
        stage1_data = raw_data[raw_data['stage'] == 1].copy()
        # 2.2 历史50天数据（计算时间范围并筛选）
        if not stage1_data.empty:
            start_date = stage1_data['timestamps'].min()
            history_start_date = start_date - pd.Timedelta(days=50)
            history_data = raw_data[
                (raw_data['timestamps'] >= history_start_date) & 
                (raw_data['timestamps'] < start_date)
            ].copy()
        else:
            history_data = pd.DataFrame(columns=raw_data.columns)
        # 2.3 合并后的数据（历史+当前stage1数据）
        merged_data = pd.concat([history_data, stage1_data], ignore_index=True)
        
        print(f"测试数据拆分完成:")
        print(f"  stage1数据行数: {len(stage1_data)}")
        print(f"  历史50天数据行数: {len(history_data)}")
        print(f"  合并后数据行数: {len(merged_data)}")
        
        # 3. 确定模型路径
        stage1_count = len(stage1_data)
        if stage1_count > 10:
            model_path = r'D:\self\code\ma_final\model_new\stage1_long\stage1_long.pkl'
            training_feature_path = r'D:\self\code\ma_final\model_new\stage1_long\training_feature_columns.csv'
        else:
            model_path = r'D:\self\code\ma_final\model_new\stage1_short\stock_classifier_pipeline.pkl'
            training_feature_path = r'D:\self\code\ma_final\model_new\stage1_short\training_feature_columns.csv'
        print(f"使用模型路径: {model_path}")
        
        # 4. 调用预测函数
        predictions, probs = predict_from_data(
            stage1_data=stage1_data,
            history_data=history_data,
            merged_data=merged_data,
            model_path=model_path
        )
        
        if predictions is not None:
            print(f"测试完成，共预测 {len(predictions)} 个样本")
            
    except FileNotFoundError:
        print(f"错误: 测试文件未找到 - {test_csv}")
    except Exception as e:
        print(f"测试过程中发生错误: {str(e)}")