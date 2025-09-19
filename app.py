import shutil
from flask import Flask, request, jsonify
from flask_cors import CORS
from config import Config
from deal_stock_choose0918 import find_N_days_bullish_0919
from json_config import move_images_to_new_folder, read_json_data, write_json_data
from pattern_detection import find_similar_patterns, find_pattern_segments, is_golden_cross, is_bullish_arrangement
from yolo_utils import is_bullish_arrangement as yolo_is_bullish_arrangement
from utils import calculate_ma, get_exchange_index, get_index_ma_values, get_stock_data
from deal_sim_final0901 import analyze_stock_data, find_stage_fragments
from datetime import datetime, timedelta
import os
from datetime import datetime
# 首先添加必要的导入
from deal_sim_time_range import find_similar_stocks, extract_stock_data_from_folder, get_candidate_stocks
import pandas as pd
from sim_class0914 import find_stage_segments0914


app = Flask(__name__)
# 将会话有效期延长
app.permanent_session_lifetime = timedelta(minutes=30)


# 离线查找指定stage的路由 #######################################################
CORS(app, resources={r"/offline_stage": {
    "origins": Config.CORS_ORIGINS,
    "methods": ["POST", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization"],
    "expose_headers": ["Content-Type"],
    "supports_credentials": True
}})

# 在线查找近N天指定stage的路由 ##################################################
CORS(app, resources={r"/online_recent_N_stage": {
    "origins": Config.CORS_ORIGINS,
    "methods": ["POST", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization"],
    "expose_headers": ["Content-Type"],
    "supports_credentials": True
}})


# 检测自定义模式的路由 #######################################################
CORS(app, resources={r"/detect_stock_mode": {
    "origins": Config.CORS_ORIGINS,
    "methods": ["POST", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization"],
    "expose_headers": ["Content-Type"],
    "supports_credentials": True
}})

# 阶段转移找相似片段（只能横向）################################################
# 用户输入：一只股票 + 时间区间 + 股票池
# 系统计算：用户股票时间区间的stage + 股票池相同stage的结果 + 同类结果 + 不同类结果
# 系统输出：同类结果（超强相似） + 不同类结果（一般相似）
CORS(app, resources={r"/detect_stock_mode_stage": {
    "origins": Config.CORS_ORIGINS,
    "methods": ["POST", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization"],
    "expose_headers": ["Content-Type"],
    "supports_credentials": True
}})


# 检测金叉形态的路由（规则匹配） ###############################################
CORS(app, resources={r"/detect_golden_cross": {
    "origins": Config.CORS_ORIGINS,
    "methods": ["POST", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization"],
    "expose_headers": ["Content-Type"],
    "supports_credentials": True
}})

# YOLO检测多头排列的路由 ######################################################
CORS(app, resources={r"/detect_bullish_arrangement": {
    "origins": Config.CORS_ORIGINS,
    "methods": ["POST", "OPTIONS"],
    "allow_headers": ["Content-Type"],
    "supports_credentials": True
}})


# 根据模式选股（历史模式查找的路由） ############################################
CORS(app, resources={r"/detect_sim_history": {
    "origins": Config.CORS_ORIGINS,
    "methods": ["POST", "OPTIONS"],
    "allow_headers": ["Content-Type"],
    "supports_credentials": True
}})


@app.route('/detect_sim_history', methods=['POST'])
def detect_sim_history():
    data = request.get_json(force=True)
    pool = data.get('pool', [])
    base_code = data.get('base_code', '')
    base_start_date = data.get('base_start_date', '')
    base_end_date = data.get('base_end_date', '')
    start_date = data.get('start_date', '')
    end_date = data.get('end_date', '')
    ma_list = data.get('ma_list', [])
    
    print(f"接收到请求: 基准股票={base_code}, 基准时间={base_start_date}~{base_end_date}, MA={ma_list}, 股票池大小={len(pool)}")
    
    if not base_code or not base_start_date or not base_end_date or not ma_list:
        return jsonify({"error": "缺少必要参数"}), 400
    
    # 调用模式查找函数
    stock_mode = find_similar_patterns(pool, base_code, base_start_date, base_end_date, start_date, end_date, ma_list)
    
    patterns = stock_mode.get('stock_patterns', {})
    overall_return = stock_mode.get('overall_five_day_avg_return', 0)
    stock_mode = patterns if patterns else {}

    base_index_code = get_exchange_index(base_code)
    
    for stock_code, intervals in stock_mode.items():
        index_code = get_exchange_index(stock_code.split('.')[0])
        
        for interval in intervals:
            start_date = interval['start_date']
            end_date = interval['end_date']
            
            # 获取该区间内的指数MA值（使用改进方法）
            index_ma_values = get_index_ma_values(index_code, start_date, end_date, ma_list)
            
            if index_ma_values:
                interval['index_code'] = index_code
                interval['index_ma_values'] = index_ma_values
                interval['index_name'] = "上证指数" if index_code == '000001' else "深证成指"
            else:
                # 创建与股票MA值相同结构的全0数组
                dummy_index_ma = [[0.0] * len(ma_list) for _ in range(5)]  # 固定为5个点
                interval['index_code'] = index_code
                interval['index_ma_values'] = dummy_index_ma
                interval['index_name'] = "上证指数" if index_code == '000001' else "深证成指"
                interval['index_error'] = "获取指数数据失败"

    response = {
        "result": stock_mode,
        "overall_return": overall_return,
        "debug": {
            "base_code": base_code,
            "base_date_range": f"{base_start_date}~{base_end_date}",
            "ma_list": ma_list,
            "pool_size": len(pool),
            "search_date_range": f"{start_date}~{end_date}",
            "base_index_code": base_index_code,
            "base_index_name": "上证指数" if base_index_code == '000001' else "深证成指",
        }
    }
    return jsonify(response)

# 检测自定义模式的路由
# @app.route('/detect_stock_mode', methods=['POST'])
# def detect_stock_mode():
#     data = request.get_json(force=True)
#     pool = data.get('pool', [])
#     base_code = data.get('base_code', '')
#     base_start_date = data.get('base_start_date', '')
#     base_end_date = data.get('base_end_date', '')
#     start_date = data.get('start_date', '')
#     end_date = data.get('end_date', '')
#     ma_list = data.get('ma_list', [])
    
#     print(f"接收到请求: 基准股票={base_code}, 基准时间={base_start_date}~{base_end_date}, MA={ma_list}, 股票池大小={len(pool)}")
    
#     if not base_code or not base_start_date or not base_end_date or not ma_list:
#         return jsonify({"error": "缺少必要参数"}), 400
    
#     stock_mode = find_similar_patterns(pool, base_code, base_start_date, base_end_date, start_date, end_date, ma_list)
    
#     patterns = stock_mode.get('stock_patterns', {})
#     overall_return = stock_mode.get('overall_five_day_avg_return', 0)
#     stock_mode = patterns if patterns else {}

#     base_index_code = get_exchange_index(base_code)
    
#     for stock_code, intervals in stock_mode.items():
#         index_code = get_exchange_index(stock_code.split('.')[0])
        
#         for interval in intervals:
#             start_date = interval['start_date']
#             end_date = interval['end_date']
            
#             # 获取该区间内的指数MA值（使用改进方法）
#             index_ma_values = get_index_ma_values(index_code, start_date, end_date, ma_list)
            
#             if index_ma_values:
#                 interval['index_code'] = index_code
#                 interval['index_ma_values'] = index_ma_values
#                 interval['index_name'] = "上证指数" if index_code == '000001' else "深证成指"
#             else:
#                 # 创建与股票MA值相同结构的全0数组
#                 dummy_index_ma = [[0.0] * len(ma_list) for _ in range(5)]  # 固定为5个点
#                 interval['index_code'] = index_code
#                 interval['index_ma_values'] = dummy_index_ma
#                 interval['index_name'] = "上证指数" if index_code == '000001' else "深证成指"
#                 interval['index_error'] = "获取指数数据失败"

#     response = {
#         "result": stock_mode,
#         "overall_return": overall_return,
#         "debug": {
#             "base_code": base_code,
#             "base_date_range": f"{base_start_date}~{base_end_date}",
#             "ma_list": ma_list,
#             "pool_size": len(pool),
#             "search_date_range": f"{start_date}~{end_date}",
#             "base_index_code": base_index_code,
#             "base_index_name": "上证指数" if base_index_code == '000001' else "深证成指",
#         }
#     }
#     return jsonify(response)
@app.route('/detect_stock_mode', methods=['POST'])
def detect_stock_mode():
    data = request.get_json(force=True)
    pool = data.get('pool', [])
    base_code = data.get('base_code', '')
    base_start_date = data.get('base_start_date', '')
    base_end_date = data.get('base_end_date', '')
    start_date = data.get('start_date', '')
    end_date = data.get('end_date', '')
    ma_list = data.get('ma_list', [])
    future_days = data.get('future_days', 5)  # 新增：获取前端传入的未来天数参数
    
    print(f"接收到请求: 基准股票={base_code}, 基准时间={base_start_date}~{base_end_date}, MA={ma_list}, 股票池大小={len(pool)}, 未来天数={future_days}")
    
    if not base_code or not base_start_date or not base_end_date or not ma_list:
        return jsonify({"error": "缺少必要参数"}), 400
    
    # 调用修改后的find_similar_patterns函数，传入future_days参数
    stock_mode = find_similar_patterns(
        pool, 
        base_code, 
        base_start_date, 
        base_end_date, 
        start_date, 
        end_date, 
        ma_list,
        future_days=future_days  # 传递未来天数参数
    )
    
    # 从结果中获取扁平化的相似模式列表和整体平均收益率
    patterns = stock_mode.get('stock_patterns', [])  # 现在是列表而不是字典
    overall_return = stock_mode.get('overall_future_avg_return', 0)  # 字段名更新
    
    base_index_code = get_exchange_index(base_code)
    
    # 遍历所有相似区间（现在是列表直接遍历，而非按股票分组遍历）
    for interval in patterns:
        # 从区间数据中获取股票代码
        stock_code = interval['stockCode']
        # 获取对应指数代码
        index_code = get_exchange_index(stock_code.split('.')[0])
        
        start_date = interval['start_date']
        end_date = interval['end_date']
        
        # 获取该区间内的指数MA值
        index_ma_values = get_index_ma_values(index_code, start_date, end_date, ma_list)
        
        if index_ma_values:
            interval['index_code'] = index_code
            interval['index_ma_values'] = index_ma_values
            interval['index_name'] = "上证指数" if index_code == '000001' else "深证成指"
        else:
            # 创建与股票MA值相同结构的全0数组
            # 动态匹配MA值长度而非固定为5
            dummy_index_ma = [[0.0 for _ in ma_list] for _ in range(len(interval['ma_values']))]
            interval['index_code'] = index_code
            interval['index_ma_values'] = dummy_index_ma
            interval['index_name'] = "上证指数" if index_code == '000001' else "深证成指"
            interval['index_error'] = "获取指数数据失败"

    response = {
        "result": patterns,  # 直接返回扁平化列表
        "overall_return": overall_return,
        "debug": {
            "base_code": base_code,
            "base_date_range": f"{base_start_date}~{base_end_date}",
            "ma_list": ma_list,
            "pool_size": len(pool),
            "search_date_range": f"{start_date}~{end_date}",
            "base_index_code": base_index_code,
            "base_index_name": "上证指数" if base_index_code == '000001' else "深证成指",
            "future_days": future_days  # 调试信息中增加未来天数
        }
    }
    return jsonify(response)

# 在线定位近N天stage+输出分类的路由
@app.route('/online_recent_N_stage', methods=['POST'])
def online_recent_N_stage():
    # 前端传入参数
    data = request.get_json(force=True)
    pool = data.get('pool', [])
    n_days = data.get('n_days', '')
    ma_periods = data.get('ma_periods', '')

    # 系统默认参数 
    min_fragment_length = 3
    stage1_lookback = 30
    target_folder = r"D:\self\code\tool_ma_back\bbb_fragments"
    # 生成目标文件夹中csv
    all_results, stage_results = find_N_days_bullish_0919(pool, n_days, ma_periods, min_fragment_length, stage1_lookback, target_folder)
    
    response = {
        "all_results": all_results,
        "stage_results": stage_results,
        "debug": {
            "pool_size": len(pool),
        }
    }
    return jsonify(response)


# 阶段转移定位相似模式查找结果的路由
@app.route('/offline_stage', methods=['POST'])
def offline_stage():
    data = request.get_json(force=True)
    pool = data.get('pool', [])
    start_date = data.get('start_date', '')
    end_date = data.get('end_date', '')
    ma_list = data.get('ma_list', [])
    stage = data.get('stage', '')
    
    print(f"接收到请求: MA={ma_list}, 股票池大小={len(pool)}")

    stock_pool = [item['code'] for item in pool]

    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    result = find_stage_segments0914(
        stock_pool=stock_pool,
        start_date=start_date,
        end_date=end_date,
        target_stage=stage,
        ma_periods=ma_list
    )

    response = {
        "result": result,
        "debug": {
            "ma_list": ma_list,
            "pool_size": len(pool),
            "search_date_range": f"{start_date}~{end_date}",
        }
    }
    return jsonify(response)



# 阶段转移定位相似模式查找结果的路由
@app.route('/detect_stock_mode_stage', methods=['POST'])
def detect_stock_mode_stage():
    data = request.get_json(force=True)
    pool = data.get('pool', [])
    base_code = data.get('base_code', '')
    base_start_date = data.get('base_start_date', '')
    base_end_date = data.get('base_end_date', '')
    start_date = data.get('start_date', '')
    end_date = data.get('end_date', '')
    ma_list = data.get('ma_list', [])
    
    print(f"接收到请求: 基准股票={base_code}, 基准时间={base_start_date}~{base_end_date}, MA={ma_list}, 股票池大小={len(pool)}")
    print(pool)
    if not base_code or not base_start_date or not base_end_date or not ma_list:
        return jsonify({"error": "缺少必要参数"}), 400

    print(pool)
    stock_pool = [item['code'] for item in pool]
    
    result = analyze_stock_data(
        test_stock=base_code,
        test_start=base_start_date,
        test_end=base_end_date,
        test_stock_pool=stock_pool,
        test_start_date=start_date,
        test_end_date=end_date
    )

    response = {
        "result": result,
        "debug": {
            "base_code": base_code,
            "base_date_range": f"{base_start_date}~{base_end_date}",
            "ma_list": ma_list,
            "pool_size": len(pool),
            "search_date_range": f"{start_date}~{end_date}",
        }
    }
    return jsonify(response)


# 检测金叉的路由
@app.route('/detect_golden_cross', methods=['POST'])
def detect_golden_cross():
    data = request.get_json(force=True)
    pool = data.get('pool', [])
    start_date = data.get('start_date', '')
    end_date = data.get('end_date', '')
    ma_list = data.get('ma_list', [])
    # 添加extend_days参数，默认为5
    extend_days = data.get('extend_days', 3)
    golden_cross = find_pattern_segments(pool, start_date, end_date, ma_list, is_golden_cross, 4, 12, extend_days=extend_days)
    
    patterns = golden_cross.get('stock_patterns', {})
    overall_return = golden_cross.get('overall_five_day_avg_return', 0)
    
    response = {
        "result": patterns,
        "overall_return": overall_return,
        "debug": {
            "start_date": start_date,
            "end_date": end_date,
            "ma_list": ma_list,
            "pool_size": len(pool),
            "extend_days": extend_days
        }
    }
    return jsonify(response)

# 检测均线多头排列的路由YOLO
@app.route('/detect_bullish_arrangement', methods=['POST'])
def detect_bullish_arrangement():
    data = request.get_json()
    stock_pool = data.get('pool', [])
    start_date = data['start_date'].replace('-', '')
    end_date = data['end_date'].replace('-', '')
    ma_list = data.get('ma_list', [4, 8, 12, 16, 20, 47])
    extend_days = data.get('extend_days', 3)
    window_size = data.get('window_size', 21)  # 窗口大小
    step_size = data.get('step_size', 21)      # 步长大小
    data_folder = data.get('data_folder', r'D:\self\data\kline-data')  # 默认数据文件夹
    result = {}
    for stock in stock_pool:
        ts_code = f"{stock['code']}.SH" if stock['code'].startswith('6') else f"{stock['code']}.SZ"
        # 获取股票数据
        df = get_stock_data(ts_code, start_date, end_date, data_folder=data_folder) 
        if df.empty:
            continue
        
        df = calculate_ma(df, ma_list)
        if df.empty:
            continue

        # 滑动窗口检测多头排列，使用正确的步长
        pattern_intervals = []
        for i in range(0, len(df) - window_size + 1, step_size):
            window = df.iloc[i:i+window_size]
            is_bullish, detection_path = yolo_is_bullish_arrangement(window, ma_list, extend_days)
            if is_bullish:
                pattern_intervals.append({
                    "start": window['trade_date'].iloc[0],
                    "end": window['trade_date'].iloc[-1],
                    "chart_path": detection_path  # 返回检测结果图像路径
                })
        ################## 第二轮检测 ##################
        for i in range(10, len(df) - window_size + 1, step_size):
            window = df.iloc[i:i+window_size]
            is_bullish, detection_path = yolo_is_bullish_arrangement(window, ma_list, extend_days)
            if is_bullish:
                pattern_intervals.append({
                    "start": window['trade_date'].iloc[0],
                    "end": window['trade_date'].iloc[-1],
                    "chart_path": detection_path  # 返回检测结果图像路径
                })
        
        if pattern_intervals:
            result[ts_code] = pattern_intervals
    
    return jsonify({"bullish_arrangements": result})


# 全局CORS配置（可以保留现有配置）
CORS(app, resources={r"/save_screenshot": {
        "origins": Config.CORS_ORIGINS,
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "supports_credentials": True
    }
})

@app.route('/save_screenshot', methods=['POST'])
def save_screenshot():
    try:
        # 获取前端传入的文件夹名称，默认为 'savepng'
        folder_name = request.form.get('folder', 'savepng')
        
        # 检查文件夹名称是否合法（避免路径遍历攻击）
        if not folder_name or any(c in folder_name for c in ['/', '\\', '..']):
            return jsonify({"success": False, "msg": "文件夹名称不合法"}), 400
        
        # 检查是否有文件上传
        if 'file' not in request.files:
            return jsonify({"success": False, "msg": "未找到文件"}), 400
        
        file = request.files['file']
        
        # 检查文件名是否存在
        if file.filename == '':
            return jsonify({"success": False, "msg": "文件名不能为空"}), 400
        
        # 定义保存路径（public/[folder_name]文件夹）
        save_dir = os.path.join(os.getcwd(), 'public', folder_name)
        # 确保文件夹存在
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存文件
        save_path = os.path.join(save_dir, file.filename)
        file.save(save_path)
        
        print(f"截图保存成功: {save_path}")
        return jsonify({
            "success": True, 
            "msg": "截图保存成功", 
            "path": save_path,
            "filename": file.filename,
            "folder": folder_name
        })
    
    except Exception as e:
        print(f"截图保存失败: {str(e)}")
        return jsonify({"success": False, "msg": f"服务器错误: {str(e)}"}), 500



# 完善CORS配置，添加图片访问路由的跨域支持
CORS(app, resources={
    r"/get_all_screenshots": {
        "origins": Config.CORS_ORIGINS,
        "methods": ['GET', "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["Content-Type"],
        "supports_credentials": True
    },
    r"/get_screenshot/<path:filename>": {  # 新增图片访问路由的CORS配置
        "origins": Config.CORS_ORIGINS,
        "methods": ['GET', "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["Content-Type"],
        "supports_credentials": True
    }
})


@app.route('/get_all_screenshots', methods=['GET'])
def get_all_screenshots():
    try:
        # 获取前端传入的文件夹名称，默认为 'savepng'
        folder_name = request.args.get('folder', 'savepng')
        
        # 检查文件夹名称是否合法
        if not folder_name or any(c in folder_name for c in ['/', '\\', '..']):
            return jsonify({"success": False, "msg": "文件夹名称不合法"}), 400
        
        # 构建图片目录路径
        save_dir = os.path.join(os.getcwd(), 'public', folder_name)
        
        if not os.path.exists(save_dir):
            return jsonify({"success": False, "msg": "图片目录不存在"}), 404
        
        if not os.path.isdir(save_dir):
            return jsonify({"success": False, "msg": "无效的图片目录路径"}), 400
        
        all_files = os.listdir(save_dir)
        
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
        image_files = [
            file for file in all_files
            if os.path.isfile(os.path.join(save_dir, file)) and
            os.path.splitext(file)[1].lower() in image_extensions
        ]
        
        image_list = [
            {
                "filename": folder_name + '\\' + file,
                "url": f"/get_screenshot/{folder_name}/{file}", 
                "timestamp": os.path.getmtime(os.path.join(save_dir, file))
            }
            for file in image_files
        ]

        print(image_list)
        
        image_list.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return jsonify({
            "success": True,
            "count": len(image_list),
            "images": image_list,
            "folder": folder_name
        })
    
    except Exception as e:
        print(f"获取图片列表失败: {str(e)}")
        return jsonify({"success": False, "msg": f"服务器错误: {str(e)}"}), 500

# 新增图片访问路由，处理实际的图片文件请求
@app.route('/get_screenshot/<filename>', methods=['GET'])
def get_screenshot(filename):
    print("filename:" , filename)
    try:
        # 图片存储路径（与列表接口保持一致）
        save_dir = os.path.join(os.getcwd(), 'public')
        file_path = os.path.join(save_dir, filename)

        # 添加路径打印
        print(f"尝试访问的图片路径: {file_path}")  # 检查这个路径是否真实存在
        
        # 检查文件是否存在
        if not os.path.exists(file_path) or not os.path.isfile(file_path):
            return jsonify({"success": False, "msg": "图片不存在"}), 404
        
        # 根据文件扩展名设置正确的MIME类型
        ext = os.path.splitext(filename)[1].lower()
        mime_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.webp': 'image/webp'
        }
        mime_type = mime_types.get(ext, 'application/octet-stream')
        
        # 读取图片文件并返回
        with open(file_path, 'rb') as f:
            image_data = f.read()
        
        # 返回图片数据，设置正确的Content-Type
        from flask import Response
        return Response(image_data, mimetype=mime_type)
    
    except Exception as e:
        print(f"获取图片失败: {str(e)}")
        return jsonify({"success": False, "msg": f"服务器错误: {str(e)}"}), 500


# 添加删除文件接口的CORS配置
CORS(app, resources={r"/delete-files": {
    "origins": Config.CORS_ORIGINS,
    "methods": ["POST", "OPTIONS"],
    "allow_headers": ["Content-Type"],
    "supports_credentials": True
}})

@app.route('/delete-files', methods=['POST'])
def delete_folder_contents():
    try:
        data = request.get_json()
        folder_path = data.get('folderPath')
        
        if not folder_path:
            return jsonify({"success": False, "msg": "缺少文件夹路径参数"}), 400
        
        # 拼接实际路径
        base_dir = 'D:/self/code/tool_ma_back'
        full_path = base_dir + folder_path
        
        print(f"尝试删除内容的文件夹路径: {full_path}")
        
        # 安全校验：限制只能操作public目录下的文件夹
        full_path = os.path.abspath(full_path)
        base_dir_abs = os.path.abspath(base_dir)
        
        if not full_path.startswith(base_dir_abs):
            return jsonify({"success": False, "msg": "路径不合法"}), 403
        
        # 检查路径是否存在且是文件夹
        if not os.path.exists(full_path):
            return jsonify({"success": False, "msg": "文件夹不存在"}), 404
        
        if not os.path.isdir(full_path):
            return jsonify({"success": False, "msg": "指定路径不是文件夹"}), 400
        
        # 清空文件夹内容（保留文件夹本身）
        # 遍历文件夹内所有内容
        for item in os.listdir(full_path):
            item_path = os.path.join(full_path, item)
            try:
                # 如果是文件或链接，直接删除
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)
                # 如果是子文件夹，递归删除
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except Exception as e:
                print(f"删除{item_path}失败: {str(e)}")
                return jsonify({"success": False, "msg": f"删除{item}时出错: {str(e)}"}), 500
        
        return jsonify({"success": True, "msg": "文件夹内容已全部清空"})
            
    except Exception as e:
        print(f"清空文件夹操作失败: {str(e)}")
        return jsonify({"success": False, "msg": f"服务器错误: {str(e)}"}), 500


# 全局CORS配置（可以保留现有配置）
CORS(app, resources={r"/deal_sim": {
        "origins": Config.CORS_ORIGINS,
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "supports_credentials": True
    }
})

# 检测自定义模式的路由
@app.route('/deal_sim', methods=['POST'])
def detect_sim_result():
    data = request.get_json(force=True)
    photoOption = data.get('photoOption', [])
    recentNDaysValue = data.get('recentNDaysValue', '')
    savedBrushTimeRanges = data.get('savedBrushTimeRanges', [])
    selectedFactor = data.get('selectedFactor', '')
    selectedFactor2 = data.get('selectedFactor2', '')
    supplementaryOption = data.get('supplementaryOption', '')

    stock_mode = find_similar_patterns(photoOption, recentNDaysValue, savedBrushTimeRanges, selectedFactor, selectedFactor2, supplementaryOption)
    
    for stock_code, intervals in stock_mode.items():
        index_code = get_exchange_index(stock_code.split('.')[0])
        
        for interval in intervals:
            start_date = interval['start_date']
            end_date = interval['end_date']
            
            # 获取该区间内的指数MA值（使用改进方法）
            index_ma_values = get_index_ma_values(index_code, start_date, end_date, ma_list)
            
            if index_ma_values:
                interval['index_code'] = index_code
                interval['index_ma_values'] = index_ma_values
                interval['index_name'] = "上证指数" if index_code == '000001' else "深证成指"
            else:
                # 创建与股票MA值相同结构的全0数组
                dummy_index_ma = [[0.0] * len(ma_list) for _ in range(5)]  # 固定为5个点
                interval['index_code'] = index_code
                interval['index_ma_values'] = dummy_index_ma
                interval['index_name'] = "上证指数" if index_code == '000001' else "深证成指"
                interval['index_error'] = "获取指数数据失败"

    response = {
        "result": stock_mode,
        "overall_return": overall_return,
        "debug": {
            "base_code": base_code,
            "base_date_range": f"{base_start_date}~{base_end_date}",
            "ma_list": ma_list,
            "pool_size": len(pool),
            "search_date_range": f"{start_date}~{end_date}",
            "base_index_code": base_index_code,
            "base_index_name": "上证指数" if base_index_code == '000001' else "深证成指",
        }
    }
    return jsonify(response)


# 添加CORS配置
CORS(app, resources={r"/find_similar_stocks_new": {
    "origins": Config.CORS_ORIGINS,
    "methods": ["POST", "OPTIONS"],
    "allow_headers": ["Content-Type"],
    "supports_credentials": True
}})

# 添加新路由
@app.route('/find_similar_stocks_new', methods=['POST'])
def handle_find_similar_stocks_new():
    try:
        data = request.get_json(force=True)
        
        # 提取请求参数
        target_code = data.get('target_code', '')
        start_date = data.get('start_date', '')
        end_date = data.get('end_date', '')
        stock_pool = data.get('stock_pool', [])
        n_days = data.get('n_days', 20)  # 默认20天
        # 传过来的n_days是字符串'近20天'的形式，将它转换成数字，采用提取指定位置字符的方式
        n_days = n_days[1:-1]
        # 把n_days转换为整数
        n_days = int(n_days) if isinstance(n_days, (int, float, str)) and str(n_days).isdigit() else 20
        ma_list = data.get('ma_list', [4, 8, 12, 16, 20, 47])  # 默认均线列表
        group_weights = data.get('group_weights', None)
        single_ma_weights = data.get('single_ma_weights', None)
        crossover_weights = data.get('crossover_weights', None)

        data_folder = data.get('data_folder', r'D:\self\data\kline-data')  # 默认数据文件夹

        
        # 验证必要参数
        if not target_code or not start_date or not end_date:
            return jsonify({"error": "缺少必要参数：股票代码、起始时间或终止时间"}), 400
        
        # 验证日期格式
        try:
            pd.to_datetime(start_date)
            pd.to_datetime(end_date)
        except ValueError:
            return jsonify({"error": "日期格式错误，请使用YYYY-MM-DD格式"}), 400
        
        # 提取股票池数据（包含股票名称信息）
        print(f"正在提取股票池数据...")
        if(len(stock_pool) == 0):
            stock_data_dict = extract_stock_data_from_folder(data_folder, n_days, ma_list)
        else:
            stock_data_dict = get_candidate_stocks(stock_pool, data_folder, n_days, ma_list)

        if not stock_data_dict or len(stock_data_dict) <= 1:
            return jsonify({"error": "未提取到任何有效股票数据"}), 500
        
        # 查找相似股票（使用deal_sim_time_range.py中的方法）
        print(f"正在计算与 {target_code} 相似的股票...")
        similar_stocks = find_similar_stocks(
            target_code, start_date, end_date,
            stock_data_dict, n_days, ma_list,
            group_weights=group_weights, single_ma_weights=single_ma_weights, crossover_weights=crossover_weights
        )
        
        if not similar_stocks:
            return jsonify({"result": [], "message": "未能找到相似的股票"}), 200
        
        # 整理结果，包含股票名称和近N日的均线数据
        result = []
        for code, stock_name, similarity in similar_stocks:  # 适配返回值包含股票名称的结构
            # 获取该股票的近N日数据
            stock_data = stock_data_dict.get(code)
            if stock_data is None or stock_data.empty:
                continue
            
            # 提取需要返回的均线数据和日期
            ma_columns = [f'MA{period}' for period in ma_list]
            required_columns = ['trade_date', 'open', 'close', 'high', 'low'] + ma_columns
            
            # 筛选存在的列
            available_columns = [col for col in required_columns if col in stock_data.columns]
            stock_data_filtered = stock_data[available_columns].copy()
            
            # 转换日期格式为字符串
            stock_data_filtered['trade_date'] = stock_data_filtered['trade_date'].dt.strftime('%Y-%m-%d')
            print("trade_date--------", stock_data_filtered['trade_date'].tolist())
            
            # 转换为字典列表
            stock_records = stock_data_filtered.to_dict('records')
            
            result.append({
                "stock_code": code,
                "stock_name": stock_name,  # 新增股票名称字段
                "similarity": round(similarity, 4),
                "recent_data": stock_records,
                "ma_list": ma_list,
                "data_length": len(stock_records)  # 新增数据长度信息
            })
        
        return jsonify({
            "result": result,
            "count": len(result),
            "target_code": target_code,
            "date_range": f"{start_date}~{end_date}",
            "n_days": n_days,
            "base_ma_periods": ma_list
        }), 200
        
    except Exception as e:
        print(f"处理相似股票查找请求时出错: {str(e)}")
        return jsonify({"error": f"服务器错误: {str(e)}"}), 500

# json config ##############################################
CORS(app, resources={r"/get_modeListSelf_new": {
    "origins": Config.CORS_ORIGINS,
    "methods": ["POST", "OPTIONS"],
    "allow_headers": ["Content-Type"],
    "supports_credentials": True
}})
@app.route('/get_modeListSelf_new', methods=['GET'])
def get_mode_list():
    """获取所有模式列表"""
    data = read_json_data()
    return jsonify({
        'success': True,
        'data': data
    })

CORS(app, resources={r"/add_modeListSelf_new": {
    "origins": Config.CORS_ORIGINS,
    "methods": ["POST", "OPTIONS"],
    "allow_headers": ["Content-Type"],
    "supports_credentials": True
}})
@app.route('/add_modeListSelf_new', methods=['POST'])
def add_mode():
    """新增模式到JSON文件"""
    try:
        # 获取前端发送的新数据
        new_mode = request.get_json()
        
        # 验证必要字段
        required_fields = ['index', 'name']
        for field in required_fields:
            if field not in new_mode:
                return jsonify({
                    'success': False,
                    'message': f'缺少必要字段: {field}'
                }), 400
        
        # 读取现有数据
        data = read_json_data()
        
        # 检查index是否已存在
        for item in data:
            if item['index'] == new_mode['index']:
                return jsonify({
                    'success': False,
                    'message': f'index已存在: {new_mode["index"]}'
                }), 400
        
        # 添加新数据
        data.append(new_mode)
        # 写入文件
        write_json_data(data)
        move_images_to_new_folder(new_mode['index'])
        return jsonify({
            'success': True,
            'message': '新增成功',
            'data': new_mode
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'服务器错误: {str(e)}'
        }), 500

if __name__ == "__main__":
    app.run(debug=Config.DEBUG)