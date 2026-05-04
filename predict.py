import subprocess, sys, os, json, time, requests
from datetime import datetime, timezone

if not os.path.exists("Kronos"):
    subprocess.run(["git", "clone", "https://github.com/shiyu-coder/Kronos.git"], check=True)

sys.path.insert(0, "Kronos")

import yfinance as yf
import pandas as pd
from model import Kronos, KronosTokenizer, KronosPredictor

WATCHLIST = ['AAPL', 'NVDA', 'TSLA', 'COIN', 'AMD', 'PYPL', 'MSTR', 'GOOGL', 'AMZN', 'SPY', 'META', 'MSFT', 'QQQ', 'SLV', 'TSM', 'MU', 'NFLX', 'BABA', 'ABNB', 'INTC', 'HOOD', 'SNDK', 'CRCL', 'PLTR', 'AVGO', 'BAB']
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

        # 計算 scale factor（現價 vs 模型用嘅歷史最後價）
        scale = (cur / hist_last) if hist_last > 0 and cur != hist_last else 1.0

        # 預測價格按比例調整
        prd = float(pred['close'].iloc[-1]) * scale

        chg = (prd - cur) / cur * 100

        if chg > 1.5:
            sig = 'BUY'
        elif chg < -1.5:
            sig = 'SELL'
        else:
            sig = 'HOLD'

        print(f"  {ticker}: cur=${cur:.2f} hist_last=${hist_last:.2f} scale={scale:.4f} pred=${prd:.2f} {chg:+.2f}% -> {sig}")

        results.append({
            'ticker': ticker,
            'current': round(cur, 2),
            'predicted': round(prd, 2),
            'change_pct': round(chg, 2),
            'signal': sig,
            # FIX: pred_high/pred_low 都用 scale
            'pred_high': round(float(pred['high'].max()) * scale, 2),
            'pred_low':  round(float(pred['low'].min())  * scale, 2),
            # FIX: history_spark 同 pred_spark 都用 scale，令 sparkline 同現價對齊
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
