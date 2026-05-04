import subprocess, sys, os, json, time, requests
from datetime import datetime, timezone

if not os.path.exists("Kronos"):
    subprocess.run(["git", "clone", "https://github.com/shiyu-coder/Kronos.git"], check=True)

sys.path.insert(0, "Kronos")

import yfinance as yf
import pandas as pd
from model import Kronos, KronosTokenizer, KronosPredictor

WATCHLIST = [
    'AAPL', 'NVDA', 'TSLA', 'COIN', 'AMD', 'PYPL', 'MSTR', 'GOOGL', 'AMZN',
    'SPY', 'META', 'MSFT', 'QQQ', 'SLV', 'TSM', 'MU', 'NFLX', 'BABA', 'ABNB',
    'INTC', 'HOOD', 'SNDK', 'CRCL', 'PLTR', 'AVGO', 'BAB',
    'PDD', 'CAR', 'BIRD', 'GME', 'EWY', 'CRWV', 'ORCL', 'RIVN',
    'USAR', 'BMNR', 'URNM', 'LLY', 'RTX', 'DKNG'
]
INTERVAL = '1h'
PERIOD = '60d'
LOOKBACK = 380
PRED_LEN = 24

print("Loading model...")
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)
print("Model loaded.")

def fetch_data(ticker):
    for attempt in range(3):
        try:
            raw = yf.download(ticker, period=PERIOD, interval=INTERVAL, auto_adjust=True, progress=False)
            if len(raw) > 0:
                return raw
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
        time.sleep(5)
    return None

def fetch_latest_price(ticker):
    for attempt in range(3):
        try:
            t = yf.Ticker(ticker)
            info = t.fast_info
            price = info.last_price
            if price and price > 0:
                return float(price)
        except Exception as e:
            print(f"  Latest price attempt {attempt+1} failed: {e}")
        time.sleep(2)
    return None

results = []

for ticker in WATCHLIST:
    try:
        print(f"Predicting {ticker}...")
        raw = fetch_data(ticker)
        if raw is None or len(raw) == 0:
            print(f"  Skip {ticker}: no data")
            continue

        tmp = raw[['Open','High','Low','Close','Volume']].copy()
        tmp.columns = ['open','high','low','close','volume']
        tmp = tmp.reset_index()
        tmp = tmp.rename(columns={'Datetime':'timestamps','Date':'timestamps'})
        tmp['timestamps'] = pd.to_datetime(tmp['timestamps'])

        if len(tmp) < LOOKBACK + PRED_LEN:
            print(f"  Skip {ticker}: only {len(tmp)} rows")
            continue

        total = len(tmp)
        si = total - LOOKBACK - PRED_LEN
        ei = si + LOOKBACK

        x_df = tmp.loc[si:ei-1, ['open','high','low','close','volume']].reset_index(drop=True)
        x_ts = tmp.loc[si:ei-1, 'timestamps'].reset_index(drop=True)
        y_ts = tmp.loc[ei:ei+PRED_LEN-1, 'timestamps'].reset_index(drop=True)

        pred = predictor.predict(df=x_df, x_timestamp=x_ts, y_timestamp=y_ts, pred_len=PRED_LEN, T=1.0, top_p=0.9, sample_count=3)

        # 用最新價格顯示現價
        latest = fetch_latest_price(ticker)
        cur = latest if latest else float(tmp['close'].iloc[-1])
        hist_last = float(tmp['close'].iloc[-1])

        # 計算模型預測嘅方向 (用 history 範圍內嘅相對變化)
        hist_base = float(x_df['close'].iloc[-1])  # 模型 context 最後一個 close
        raw_pred_last = float(pred['close'].iloc[-1])
        raw_chg = (raw_pred_last - hist_base) / hist_base * 100  # 模型預測嘅 % 變化

        # Clamp: 限制最大預測幅度係 ±8%，避免極端值
        MAX_CHG = 8.0
        clamped_chg = max(-MAX_CHG, min(MAX_CHG, raw_chg))

        # 由現價 + clamped % 計算 predicted price
        prd = cur * (1 + clamped_chg / 100)

        # 同樣用 clamped 幅度計算 pred_high / pred_low
        raw_high_chg = (float(pred['high'].max()) - hist_base) / hist_base * 100
        raw_low_chg  = (float(pred['low'].min())  - hist_base) / hist_base * 100
        clamped_high_chg = max(-MAX_CHG, min(MAX_CHG, raw_high_chg))
        clamped_low_chg  = max(-MAX_CHG, min(MAX_CHG, raw_low_chg))
        pred_high = cur * (1 + clamped_high_chg / 100)
        pred_low  = cur * (1 + clamped_low_chg  / 100)

        chg = clamped_chg

        if chg > 1.5:
            sig = 'BUY'
        elif chg < -1.5:
            sig = 'SELL'
        else:
            sig = 'HOLD'

        # 計算 probability：跑多 10 個 samples，計方向一致性
        PROB_SAMPLES = 10
        sample_chgs = []
        for _ in range(PROB_SAMPLES):
            try:
                s_pred = predictor.predict(df=x_df, x_timestamp=x_ts, y_timestamp=y_ts, pred_len=PRED_LEN, T=1.0, top_p=0.9, sample_count=1)
                s_raw = float(s_pred['close'].iloc[-1])
                s_chg = (s_raw - hist_base) / hist_base * 100
                s_chg = max(-MAX_CHG, min(MAX_CHG, s_chg))
                sample_chgs.append(s_chg)
            except:
                pass

        if len(sample_chgs) > 0:
            if sig == 'BUY':
                prob = round(sum(1 for c in sample_chgs if c > 1.5) / len(sample_chgs) * 100)
            elif sig == 'SELL':
                prob = round(sum(1 for c in sample_chgs if c < -1.5) / len(sample_chgs) * 100)
            else:
                prob = round(sum(1 for c in sample_chgs if -1.5 <= c <= 1.5) / len(sample_chgs) * 100)
        else:
            prob = 50

        # history_spark / pred_spark: 用 scale 令 sparkline 視覺上同現價對齊
        scale = (cur / hist_last) if hist_last > 0 else 1.0
        print(f"  {ticker}: cur=${cur:.2f} raw_chg={raw_chg:+.2f}% clamped={clamped_chg:+.2f}% pred=${prd:.2f} -> {sig} ({prob}%)")

        results.append({
            'ticker': ticker,
            'current': round(cur, 2),
            'predicted': round(prd, 2),
            'change_pct': round(chg, 2),
            'signal': sig,
            'probability': prob,
            'pred_high': round(pred_high, 2),
            'pred_low':  round(pred_low, 2),
            'history_spark': [round(v * scale, 2) for v in x_df['close'].tail(24).tolist()],
            'pred_spark':    [round(v * scale, 2) for v in pred['close'].tolist()],
        })

    except Exception as e:
        print(f"  ERROR {ticker}: {e}")

output = {
    'updated_at': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
    'interval': INTERVAL,
    'pred_hours': PRED_LEN,
    'stocks': results
}

os.makedirs('docs', exist_ok=True)
with open('docs/data.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"Done! {len(results)} predictions saved.")
