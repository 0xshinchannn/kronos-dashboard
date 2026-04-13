import subprocess, sys, os, json, time
from datetime import datetime, timezone

if not os.path.exists("Kronos"):
    subprocess.run(["git", "clone", "https://github.com/shiyu-coder/Kronos.git"], check=True)

sys.path.insert(0, "Kronos")

import yfinance as yf
import pandas as pd
from model import Kronos, KronosTokenizer, KronosPredictor

WATCHLIST = ['AAPL', 'NVDA', 'TSLA', 'COIN', 'AMD', 'PYPL', 'MSTR', 'GOOGL', 'AMZN', 'SPY', 'META']
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

        cur = float(x_df['close'].iloc[-1])
        prd = float(pred['close'].iloc[-1])
        chg = (prd - cur) / cur * 100

        if chg > 1.5:
            sig = 'BUY'
        elif chg < -1.5:
            sig = 'SELL'
        else:
            sig = 'HOLD'

        results.append({
            'ticker': ticker,
            'current': round(cur, 2),
            'predicted': round(prd, 2),
            'change_pct': round(chg, 2),
            'signal': sig,
            'pred_high': round(float(pred['high'].max()), 2),
            'pred_low': round(float(pred['low'].min()), 2),
            'history_spark': [round(v, 2) for v in x_df['close'].tail(24).tolist()],
            'pred_spark': [round(v, 2) for v in pred['close'].tolist()],
        })
        print(f"  {ticker}: {chg:+.2f}% -> {sig}")

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
