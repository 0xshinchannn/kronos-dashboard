import subprocess, sys, os, json, time
from datetime import datetime, timezone

# ── Clone Kronos ──
if not os.path.exists("Kronos"):
    subprocess.run(["git", "clone", "https://github.com/shiyu-coder/Kronos.git"], check=True)
sys.path.insert(0, "Kronos")

# ── Install TimesFM ──
subprocess.run([sys.executable, "-m", "pip", "install", "timesfm", "-q"], check=True)

import yfinance as yf
import pandas as pd
import numpy as np
from model import Kronos, KronosTokenizer, KronosPredictor
import timesfm

# ── CONFIG ──
WATCHLIST = [
    # 大型科技
    'AAPL', 'NVDA', 'TSLA', 'GOOGL', 'AMZN', 'META', 'MSFT', 'TSM',
    # 高波動 / 熱門
    'COIN', 'MSTR', 'GME', 'PLTR', 'HOOD', 'SNDK', 'CRCL', 'CRWV',
    'USAR', 'PDD', 'BABA', 'AMD', 'PYPL', 'MU', 'INTC', 'NFLX', 'ABNB'
]

# Per-ticker 24h max change cap
MAX_CHG_MAP = {
    'AAPL': 5.0, 'MSFT': 5.0, 'GOOGL': 5.0, 'AMZN': 5.0,
    'META': 6.0, 'NVDA': 6.0, 'TSM': 6.0, 'TSLA': 8.0,
    'AMD': 8.0, 'PYPL': 6.0, 'NFLX': 6.0, 'BABA': 7.0,
    'ABNB': 7.0, 'INTC': 7.0, 'PLTR': 8.0, 'MU': 7.0, 'PDD': 8.0,
    'COIN': 12.0, 'MSTR': 15.0, 'GME': 15.0, 'SNDK': 15.0,
    'HOOD': 10.0, 'CRCL': 10.0, 'CRWV': 12.0, 'USAR': 12.0,
}
DEFAULT_MAX_CHG = 8.0

INTERVAL  = '1h'
PERIOD    = '60d'
LOOKBACK  = 380
PRED_LEN  = 24
PROB_SAMPLES = 8   # 減少 samples 節省時間

# ── Load Kronos ──
print("Loading Kronos...")
tokenizer  = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
kmodel     = Kronos.from_pretrained("NeoQuasar/Kronos-small")
kronos     = KronosPredictor(kmodel, tokenizer, device="cpu", max_context=512)
print("Kronos loaded.")

# ── Load TimesFM ──
print("Loading TimesFM...")
tfm = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(
        backend="cpu",
        per_core_batch_size=32,
        horizon_len=PRED_LEN,
        input_patch_len=32,
        output_patch_len=128,
        num_layers=20,
        model_dims=1280,
    ),
    checkpoint=timesfm.TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
    ),
)
print("TimesFM loaded.")

# ── Helpers ──
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
            price = t.fast_info.last_price
            if price and price > 0:
                return float(price)
        except Exception as e:
            print(f"  Latest price attempt {attempt+1} failed: {e}")
        time.sleep(2)
    return None

def clamp(val, cap):
    return max(-cap, min(cap, val))

def signal_from_chg(chg):
    if chg > 1.5:   return 'BUY'
    elif chg < -1.5: return 'SELL'
    else:            return 'HOLD'

# ── Main loop ──
results = []

for ticker in WATCHLIST:
    try:
        print(f"\nPredicting {ticker}...")
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

        cur      = fetch_latest_price(ticker) or float(tmp['close'].iloc[-1])
        hist_last = float(tmp['close'].iloc[-1])
        hist_base = float(x_df['close'].iloc[-1])
        scale     = (cur / hist_last) if hist_last > 0 else 1.0
        MAX_CHG   = MAX_CHG_MAP.get(ticker, DEFAULT_MAX_CHG)

        # ── Kronos prediction ──
        kronos_pred = kronos.predict(
            df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
            pred_len=PRED_LEN, T=1.0, top_p=0.9, sample_count=3
        )
        kronos_raw_chg = (float(kronos_pred['close'].iloc[-1]) - hist_base) / hist_base * 100
        kronos_chg     = clamp(kronos_raw_chg, MAX_CHG)
        kronos_sig     = signal_from_chg(kronos_chg)
        print(f"  Kronos: {kronos_raw_chg:+.2f}% → clamped {kronos_chg:+.2f}% [{kronos_sig}]")

        # ── TimesFM prediction ──
        tfm_chg = None
        tfm_sig = None
        try:
            close_series = x_df['close'].values.astype(float).tolist()
            # TimesFM expects list of arrays
            point_forecast, _ = tfm.forecast(
                inputs=[close_series],
                freq=[0],   # 0 = high-frequency (hourly)
            )
            tfm_last    = float(point_forecast[0][-1])
            tfm_raw_chg = (tfm_last - hist_base) / hist_base * 100
            tfm_chg     = clamp(tfm_raw_chg, MAX_CHG)
            tfm_sig     = signal_from_chg(tfm_chg)
            print(f"  TimesFM: {tfm_raw_chg:+.2f}% → clamped {tfm_chg:+.2f}% [{tfm_sig}]")
        except Exception as e:
            print(f"  TimesFM failed: {e}, using Kronos only")

        # ── Ensemble ──
        if tfm_chg is not None and tfm_sig is not None:
            if kronos_sig == tfm_sig:
                # 兩個一致 → 平均，confidence boost
                final_chg = (kronos_chg + tfm_chg) / 2
                final_sig = kronos_sig
                ensemble_agreement = True
            else:
                # 唔一致 → 保守平均，降級 HOLD 機率高
                final_chg = (kronos_chg + tfm_chg) / 2
                final_sig = signal_from_chg(final_chg)
                ensemble_agreement = False
        else:
            # TimesFM 失敗，單用 Kronos
            final_chg = kronos_chg
            final_sig = kronos_sig
            ensemble_agreement = None

        prd = cur * (1 + final_chg / 100)

        # pred_high / pred_low from Kronos (TimesFM only gives point forecast)
        raw_high_chg = (float(kronos_pred['high'].max()) - hist_base) / hist_base * 100
        raw_low_chg  = (float(kronos_pred['low'].min())  - hist_base) / hist_base * 100
        pred_high = cur * (1 + clamp(raw_high_chg, MAX_CHG) / 100)
        pred_low  = cur * (1 + clamp(raw_low_chg,  MAX_CHG) / 100)

        # ── Probability sampling (Kronos only, faster) ──
        sample_chgs = []
        for _ in range(PROB_SAMPLES):
            try:
                s_pred = kronos.predict(
                    df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
                    pred_len=PRED_LEN, T=1.0, top_p=0.9, sample_count=1
                )
                s_chg = (float(s_pred['close'].iloc[-1]) - hist_base) / hist_base * 100
                sample_chgs.append(clamp(s_chg, MAX_CHG))
            except:
                pass

        if sample_chgs:
            if final_sig == 'BUY':
                prob = round(sum(1 for c in sample_chgs if c > 1.5) / len(sample_chgs) * 100)
            elif final_sig == 'SELL':
                prob = round(sum(1 for c in sample_chgs if c < -1.5) / len(sample_chgs) * 100)
            else:
                prob = round(sum(1 for c in sample_chgs if -1.5 <= c <= 1.5) / len(sample_chgs) * 100)

            # 兩個模型唔一致時，confidence 打折
            if ensemble_agreement is False:
                prob = round(prob * 0.7)
        else:
            prob = 50

        print(f"  Ensemble: {final_chg:+.2f}% [{final_sig}] prob={prob}% agree={ensemble_agreement}")

        results.append({
            'ticker':      ticker,
            'current':     round(cur, 2),
            'predicted':   round(prd, 2),
            'change_pct':  round(final_chg, 2),
            'signal':      final_sig,
            'probability': prob,
            'ensemble_agreement': ensemble_agreement,
            'pred_high':   round(pred_high, 2),
            'pred_low':    round(pred_low, 2),
            'history_spark': [round(v * scale, 2) for v in x_df['close'].tail(24).tolist()],
            'pred_spark':    [round(v * scale, 2) for v in kronos_pred['close'].tolist()],
        })

    except Exception as e:
        print(f"  ERROR {ticker}: {e}")

# ── Save ──
output = {
    'updated_at': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
    'interval':   INTERVAL,
    'pred_hours': PRED_LEN,
    'stocks':     results
}

os.makedirs('docs', exist_ok=True)
with open('docs/data.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nDone! {len(results)} predictions saved.")
