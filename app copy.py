from flask import Flask, request, jsonify
from flask_cors import CORS
from config import Config
from pattern_detection import find_similar_patterns, find_pattern_segments, is_golden_cross, is_bullish_arrangement
from yolo_utils import is_bullish_arrangement as yolo_is_bullish_arrangement
from utils import calculate_ma, get_exchange_index, get_index_ma_values, get_stock_data
from datetime import timedelta


app = Flask(__name__)
# 将会话有效期延长
app.permanent_session_lifetime = timedelta(minutes=30)

CORS(app, resources={r"/detect_stock_mode": {
    "origins": Config.CORS_ORIGINS,
    "methods": ["POST", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization"],
    "expose_headers": ["Content-Type"],
    "supports_credentials": True
}})

CORS(app, resources={r"/detect_golden_cross": {
    "origins": Config.CORS_ORIGINS,
    "methods": ["POST", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization"],
    "expose_headers": ["Content-Type"],
    "supports_credentials": True
}})

CORS(app, resources={r"/detect_bullish_arrangement": {
    "origins": Config.CORS_ORIGINS,
    "methods": ["POST", "OPTIONS"],
    "allow_headers": ["Content-Type"],
    "supports_credentials": True
}})

# 检测自定义模式的路由
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
    
    print(f"接收到请求: 基准股票={base_code}, 基准时间={base_start_date}~{base_end_date}, MA={ma_list}, 股票池大小={len(pool)}")
    
    if not base_code or not base_start_date or not base_end_date or not ma_list:
        return jsonify({"error": "缺少必要参数"}), 400
    
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
    golden_cross = find_pattern_segments(pool, start_date, end_date, ma_list, is_golden_cross, 4, 20, extend_days=extend_days)
    return jsonify(golden_cross)

# 检测均线多头排列的路由
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
    
    result = {}
    for stock in stock_pool:
        ts_code = f"{stock['code']}.SH" if stock['code'].startswith('6') else f"{stock['code']}.SZ"
        # 获取股票数据
        df = get_stock_data(ts_code, start_date, end_date) 
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

if __name__ == "__main__":
    app.run(debug=Config.DEBUG)