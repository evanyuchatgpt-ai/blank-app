import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time

# --- 1. 網頁設定 ---
st.set_page_config(page_title="HK Market Sniper", layout="wide")
st.title("🚀 HK MARKET SNIPER (手機網頁版)")
st.caption("策略核心: 形態偵測 | 轉勢預警 | 價值通道")

# --- 2. 股票名單 ---
stock_names = {
    # 期貨 & 指數
    "HSI=F": "恒指期貨", "HHI=F": "國指期貨",
    "^HSI": "恒生指數", "^HSCE": "國企指數", "3033.HK": "恒生科指ETF",
    "^DJI": "道瓊斯", "^IXIC": "納斯達克", "^GSPC": "標普500",
    
    # 科技
    "0700.HK": "騰訊", "3690.HK": "美團", "9988.HK": "阿里", "9618.HK": "京東",
    "1810.HK": "小米", "1024.HK": "快手", "9888.HK": "百度", "0981.HK": "中芯",
    "0992.HK": "聯想", "0020.HK": "商湯", "0285.HK": "比亞迪電", "1347.HK": "華虹",
    
    # 金融
    "0005.HK": "匯豐", "1299.HK": "友邦", "0939.HK": "建行", "1398.HK": "工行",
    "3988.HK": "中行", "2318.HK": "平保", "2628.HK": "國壽", "0388.HK": "港交所",
    
    # 其他
    "1211.HK": "比亞迪", "2015.HK": "理想", "9866.HK": "蔚來", "9868.HK": "小鵬",
    "0175.HK": "吉利", "0883.HK": "中海油", "0027.HK": "銀娛", "0066.HK": "港鐵",
    "0016.HK": "新鴻基", "0823.HK": "領展"
}
tickers = list(stock_names.keys())

# --- 3. 計算核心 (保留你的邏輯) ---
def calculate_indicators(df):
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    # KDJ
    low_min = low.rolling(9).min()
    high_max = high.rolling(9).max()
    rsv = (close - low_min) / (high_max - low_min) * 100
    k = rsv.ewm(alpha=1/3, adjust=False).mean()
    d = k.ewm(alpha=1/3, adjust=False).mean()
    j = 3 * k - 2 * d
    
    # MACD
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    
    # RSI
    delta = close.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return k, d, j, macd, signal, rsi

def get_log_channel(close):
    try:
        y = np.log(close.values)
        x = np.arange(len(y))
        m, c = np.polyfit(x, y, 1)
        std = np.std(y - (m*x + c))
        curr_fair = m * (len(y)-1) + c
        upper = np.exp(curr_fair + 2*std)
        lower = np.exp(curr_fair - 2*std)
        return lower, upper
    except:
        return 0, 0

def analyze(code):
    try:
        stock = yf.Ticker(code)
        hist = stock.history(period="1y")
        if len(hist) < 60: return None
        
        close = hist['Close']
        high = hist['High']
        low = hist['Low']
        curr = close.iloc[-1]
        
        k, d, j, macd, sig, rsi = calculate_indicators(hist)
        log_lo, log_hi = get_log_channel(close)
        
        # 簡易 Pivot
        pp = (high.iloc[-1] + low.iloc[-1] + curr) / 3
        r1 = 2*pp - low.iloc[-1]
        s1 = 2*pp - high.iloc[-1]
        
        # 訊號生成
        signals = []
        j_val = j.iloc[-1]
        rsi_val = rsi.iloc[-1]
        m_val = macd.iloc[-1]
        s_val = sig.iloc[-1]
        
        if j_val < 0: signals.append("KDJ超賣")
        if m_val > s_val and macd.iloc[-2] < sig.iloc[-2]: signals.append("MACD金叉")
        if m_val < s_val and macd.iloc[-2] > sig.iloc[-2]: signals.append("MACD死叉")
        if log_lo > 0 and curr <= log_lo * 1.02: signals.append("LOG底")
        if log_lo > 0 and curr >= log_hi * 0.98: signals.append("LOG頂")
        
        # 轉勢預警
        reversal = ""
        if curr > close.iloc[-10:].max() and m_val < macd.iloc[-10:].max():
            reversal = "⚠️頂背馳"
            signals.append("頂背馳")
        elif curr < close.iloc[-10:].min() and m_val > macd.iloc[-10:].min():
            reversal = "🚀底背馳"
            signals.append("底背馳")
            
        # 形態
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        upper_b = ma20 + 2*std20
        if curr > upper_b.iloc[-2]: signals.append("BREAKOUT")
        
        # 評分與方向
        score = 50
        if "MACD金叉" in signals: score += 15
        if "KDJ超賣" in signals: score += 15
        if "LOG底" in signals: score += 20
        if "底背馳" in signals: score += 25
        if "BREAKOUT" in signals: score += 15
        
        if "MACD死叉" in signals: score -= 15
        if "LOG頂" in signals: score -= 20
        if "頂背馳" in signals: score -= 25
        
        score = max(1, min(99, score))
        action = "觀望"
        if score >= 60: action = "看好(Call)"
        if score >= 80: action = "★強力買入"
        if score <= 40: action = "看淡(Put)"
        if score <= 20: action = "▽強力沽出"
        
        target = r1 if score >= 50 else s1
        is_index = code in ["^HSI", "^HSCE", "3033.HK", "^DJI", "^IXIC", "^GSPC", "HSI=F", "HHI=F"]
        
        return {
            "代號": code,
            "名稱": stock_names.get(code, code),
            "現價": round(curr, 2),
            "評分": score,
            "建議": action,
            "預警": reversal,
            "目標價": round(target, 1),
            "訊號": " ".join(signals),
            "is_index": is_index
        }
    except:
        return None

# --- 4. 執行按鈕 ---
if st.button('🔄 按此掃描市場'):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    total = len(tickers)
    
    for i, code in enumerate(tickers):
        status_text.text(f"正在掃描: {stock_names.get(code, code)} ({code})...")
        res = analyze(code)
        if res:
            results.append(res)
        progress_bar.progress((i + 1) / total)
        time.sleep(0.1) # 防止太快被封
        
    status_text.text("掃描完成！")
    progress_bar.empty()
    
    # 整理數據
    if results:
        df_res = pd.DataFrame(results)
        
        # 排序: 指數先 -> 有預警先 -> 分數高先
        df_res['index_sort'] = df_res['is_index'].apply(lambda x: 0 if x else 1)
        df_res['rev_sort'] = df_res['預警'].apply(lambda x: 0 if x else 1)
        df_res = df_res.sort_values(by=['index_sort', 'rev_sort', '評分'], ascending=[True, True, False])
        
        # 美化表格顯示
        st.subheader("📊 掃描結果")
        
        # 用顏色標示建議
        def color_action(val):
            color = 'black'
            if '買入' in val or '看好' in val: color = 'green'
            elif '沽出' in val or '看淡' in val: color = 'red'
            return f'color: {color}; font-weight: bold'
            
        st.dataframe(
            df_res[['代號', '名稱', '現價', '建議', '預警', '目標價', '訊號']].style.applymap(color_action, subset=['建議']),
            use_container_width=True,
            height=800
        )
    else:
        st.error("暫時無法讀取數據，請稍後再試。")
