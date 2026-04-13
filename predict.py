import subprocess, sys, os

# Clone Kronos repo
if not os.path.exists("Kronos"):
    subprocess.run(["git", "clone", "https://github.com/shiyu-coder/Kronos.git"], check=True)

sys.path.insert(0, "Kronos")

import json
from datetime import datetime, timezone
import yfinance as yf
import pandas as pd

from model import Kronos, KronosTokenizer, KronosPredictor

WATCHLIST = ['AAPL', 'NVDA', 'TSLA', 'COIN', 'AMD', 'PYPL', 'MSTR', 'GOOGL', 'AMZN', 'SPY', 'META']
INTERVAL  = '1h'
PERIOD    = '60d'
LOOKBACK  = 380
PRED_LEN  = 24

print("Loading Kronos model...")
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model     = Kronos.from_pretrained("NeoQuasar/Kronos-small")
predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)
print("Model loaded.")

results = []

for ticker in WATCHLIST:
    try:
        print(f"Predicting {ticker}...")
        raw = yf.download(ticker, period=PERIOD, interval=INTERVAL, auto_adjust=True, progress=False)
        raw = raw.dropna()

        tmp = raw[['Open','High','Low','Close','Volume']].copy()
        tmp.columns = ['open','high','low','close','volume']
        tmp = tmp.reset_index()
        tmp = tmp.rename(columns={'Datetime':'timestamps','Date':'timestamps'})
        tmp['timestamps'] = pd.to_datetime(tmp['timestamps'])

        if len(tmp) < LOOKBACK + PRED_LEN:
            print(f"  Skipping {ticker}: not enough data ({len(tmp)} rows)")
            continue

        total     = len(tmp)
        start_idx = total - LOOKBACK - PRED_LEN
        end_hist  = start_idx + LOOKBACK

        x_df        = tmp.loc[start_idx:end_hist-1, ['open','high','low','close','volume']].reset_index(drop=True)
        x_timestamp = tmp.loc[start_idx:end_hist-1, 'timestamps'].reset_index(drop=True)
        y_timestamp = tmp.loc[end_hist:end_hist+PRED_LEN-1, 'timestamps'].reset_index(drop=True)

        pred = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=PRED_LEN,
            T=1.0, top_p=0.9, sample_count=3
        )

        cur_price  = float(x_df['close'].iloc[-1])
        pred_price = float(pred['close'].iloc[-1])
        chg_pct    = (pred_price - cur_price) / cur_price * 100

        if chg_pct > 1.5:
            signal = 'BUY'
        elif chg_pct < -1.5:
            signal = 'SELL'
        else:
            signal = 'HOLD'

        history_spark = x_df['close'].tail(24).tolist()
        pred_spark    = pred['close'].tolist()

        results.append({
            'ticker':        ticker,
            'current':       round(cur_price, 2),
            'predicted':     round(pred_price, 2),
            'change_pct':    round(chg_pct, 2),
            'signal':        signal,
            'pred_high':     round(floa
