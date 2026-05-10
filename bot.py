#!/usr/bin/env python3
"""
ETH/USDT Trading Bot v3.1 — Version finale
Corrections : Détecteur régime · ATR adaptatif · BTC graduel · Reset heure locale Argentine
"""

import time, json, logging, requests
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

CONFIG = {
    "symbol":           "ETHUSDT",
    "btc_symbol":       "BTCUSDT",
    "capital_initial":  54.0,
    "check_interval":   60,
    "lookback":         100,
    "interval_fast":    "1h",
    "interval_slow":    "4h",
    "max_drawdown_pct": 0.10,       # Pause si -10% dans la journée
    "max_total_loss":   0.15,       # Pause si -15% depuis le départ
    "atr_range_pct":    0.010,      # CORRIGÉ : seuil relevé à 1% pour mieux détecter le range
    "timezone_offset":  -3,         # CORRIGÉ : Argentine = UTC-3
    "paper_trading":    True,

    # Telegram
    "telegram_token":   "8747032701:AAF5gAop7o8U88zEJMNtSJsZrIY8X9kI0YM",
    "telegram_chat_id": "5488337343",

    # Fichiers
    "state_file":       "state.json",
    "log_file":         "eth-trades.log",

    # API Binance
    "api_key":          "",
    "api_secret":       "",
}

# ─── STRATÉGIES ───────────────────────────────────────────────────────────────

STRATEGIES = {
    "HAUSSIER": {
        "conditions_min":  2,
        "rsi_oversold":    50,
        "rsi_overbought":  70,
        "risk_pct":        0.07,
        "trailing_pct":    0.03,
        "take_profit_pct": 0.99,    # Trailing gère la sortie
        "volume_factor":   0.9,
        "atr_multiplier":  1.2,     # CORRIGÉ : réduit de 1.5 → 1.2 pour SL plus serré
        "description":     "Agressif — laisse courir les gains",
    },
    "RANGE": {
        "conditions_min":  99,      # Pause totale en range
        "rsi_oversold":    32,
        "rsi_overbought":  58,
        "risk_pct":        0.03,
        "trailing_pct":    0.015,
        "take_profit_pct": 0.025,
        "volume_factor":   1.3,
        "atr_multiplier":  0.6,
        "description":     "Pause — marché sans direction claire",
    },
    "BAISSIER": {
        "conditions_min":  99,      # Pause totale en baissier
        "rsi_oversold":    25,
        "rsi_overbought":  45,
        "risk_pct":        0.02,
        "trailing_pct":    0.02,
        "take_profit_pct": 0.02,
        "volume_factor":   1.5,
        "atr_multiplier":  0.8,
        "description":     "Pause — marché défavorable",
    },
}

# ─── LOGGING ──────────────────────────────────────────────────────────────────

log_path = Path(__file__).parent / CONFIG["log_file"]
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("ETH_BOT_v3.1")

# ─── HEURE LOCALE ARGENTINE ───────────────────────────────────────────────────

def local_now():
    """Retourne l'heure actuelle en heure locale Argentine (UTC-3)."""
    return datetime.now(timezone(timedelta(hours=CONFIG["timezone_offset"])))

def local_today():
    return str(local_now().date())

# ─── ÉTAT ─────────────────────────────────────────────────────────────────────

state_path = Path(__file__).parent / CONFIG["state_file"]

def load_state():
    if state_path.exists():
        with open(state_path) as f:
            s = json.load(f)
        log.info(f"Reprise — capital: {s['capital']:.2f} USDT | régime: {s.get('regime','?')} | position: {s['in_position']}")
        return s
    return {
        "capital":         CONFIG["capital_initial"],
        "in_position":     False,
        "entry_price":     None,
        "qty":             None,
        "stop_loss":       None,
        "take_profit":     None,
        "highest_price":   None,
        "regime":          "RANGE",
        "total_trades":    0,
        "winning_trades":  0,
        "total_pnl":       0.0,
        "day_pnl":         0.0,
        "last_reset_date": local_today(),   # CORRIGÉ : heure locale Argentine
        "paused_until":    None,
        "last_atr":        None,
        "started_at":      local_now().isoformat(),
    }

def save_state(s):
    with open(state_path, "w") as f:
        json.dump(s, f, indent=2)

# ─── TELEGRAM ─────────────────────────────────────────────────────────────────

def tg(msg: str):
    try:
        url = f"https://api.telegram.org/bot{CONFIG['telegram_token']}/sendMessage"
        requests.post(url, json={
            "chat_id": CONFIG["telegram_chat_id"],
            "text": msg,
            "parse_mode": "HTML"
        }, timeout=5)
    except Exception as e:
        log.warning(f"Telegram: {e}")

# ─── DONNÉES MARCHÉ ───────────────────────────────────────────────────────────

BINANCE = "https://api.binance.com"

def get_klines(symbol, interval, limit=100):
    try:
        r = requests.get(f"{BINANCE}/api/v3/klines",
                         params={"symbol": symbol, "interval": interval, "limit": limit},
                         timeout=10)
        r.raise_for_status()
        df = pd.DataFrame(r.json(), columns=[
            "open_time","open","high","low","close","volume",
            "close_time","quote_volume","trades","tbb","tbq","ignore"])
        for c in ["open","high","low","close","volume"]:
            df[c] = df[c].astype(float)
        return df
    except Exception as e:
        log.error(f"Klines {symbol}: {e}")
        return pd.DataFrame()

def get_price(symbol):
    try:
        r = requests.get(f"{BINANCE}/api/v3/ticker/price",
                         params={"symbol": symbol}, timeout=5)
        p = float(r.json()["price"])
        return p if p > 0 else None
    except:
        return None

# ─── INDICATEURS ──────────────────────────────────────────────────────────────

def ema(s, p): return s.ewm(span=p, adjust=False).mean()

def calc_rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    rs = g / l.replace(0, np.nan)
    val = float((100 - 100 / (1 + rs)).iloc[-1])
    return round(val, 2) if not np.isnan(val) else 50.0

def calc_atr(df, p=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    val = tr.rolling(p).mean().iloc[-1]
    return float(val) if not np.isnan(val) else 1.0

def calc_macd_hist(s):
    macd = ema(s, 12) - ema(s, 26)
    sig  = ema(macd, 9)
    hist = macd - sig
    return float(hist.iloc[-1]), float(hist.iloc[-2]) if len(hist) > 1 else 0.0

# ─── DÉTECTEUR DE MARCHÉ (CORRIGÉ) ───────────────────────────────────────────

def detect_regime(df_4h: pd.DataFrame) -> str:
    """
    CORRECTIONS :
    - ATR seuil relevé à 1% (était 0.8%)
    - Ajout confirmation sur 3 bougies consécutives pour éviter les faux haussiers
    - Slope calculé sur 10 bougies au lieu de 5 pour plus de stabilité
    """
    if df_4h.empty or len(df_4h) < 15:
        return "RANGE"

    c     = df_4h["close"]
    e9    = ema(c, 9)
    e21   = ema(c, 21)
    rsi   = calc_rsi(c)
    atr   = calc_atr(df_4h)
    price = float(c.iloc[-1])

    atr_pct = atr / price

    # Pente sur 10 bougies (plus stable qu'avec 5)
    slope = (float(e21.iloc[-1]) - float(e21.iloc[-10])) / float(e21.iloc[-10]) * 100

    # Tendance confirmée sur 3 bougies consécutives
    trend_up_3   = all(float(e9.iloc[-i]) > float(e21.iloc[-i]) for i in range(1, 4))
    trend_down_3 = all(float(e9.iloc[-i]) < float(e21.iloc[-i]) for i in range(1, 4))

    # Filtre ATR : volatilité trop faible = range
    if atr_pct < CONFIG["atr_range_pct"]:
        regime = "RANGE"
    elif trend_up_3 and rsi > 52 and slope > 0.15:
        regime = "HAUSSIER"
    elif trend_down_3 and rsi < 43 and slope < -0.15:
        regime = "BAISSIER"
    else:
        regime = "RANGE"

    log.info(f"Régime: {regime} | RSI 4H: {rsi} | Pente 10b: {slope:.3f}% | ATR: {atr_pct*100:.2f}%")
    return regime

# ─── CORRÉLATION BTC (CORRIGÉE) ───────────────────────────────────────────────

def btc_score() -> float:
    """
    CORRECTION : Score graduel au lieu d'un seuil binaire.
    Retourne un score entre 0 (BTC très négatif) et 1 (BTC positif).
    Le signal BUY n'est ignoré que si le score < 0.4.
    """
    df = get_klines(CONFIG["btc_symbol"], "1h", 10)
    if df.empty:
        return 1.0  # Assume OK si pas de données

    c = df["close"]
    change_1h = (float(c.iloc[-1]) - float(c.iloc[-2])) / float(c.iloc[-2]) * 100
    change_3h = (float(c.iloc[-1]) - float(c.iloc[-4])) / float(c.iloc[-4]) * 100
    change_6h = (float(c.iloc[-1]) - float(c.iloc[-7])) / float(c.iloc[-7]) * 100

    # Score basé sur les 3 horizons
    score = 1.0
    if change_1h < -1.0: score -= 0.2
    if change_1h < -2.0: score -= 0.2
    if change_3h < -2.0: score -= 0.2
    if change_3h < -4.0: score -= 0.2
    if change_6h < -5.0: score -= 0.2

    score = max(0.0, score)
    log.info(f"BTC score: {score:.1f} | 1H: {change_1h:+.2f}% | 3H: {change_3h:+.2f}% | 6H: {change_6h:+.2f}%")
    return score

# ─── ANALYSE TECHNIQUE ────────────────────────────────────────────────────────

def analyze(df: pd.DataFrame, strategy: dict) -> dict:
    c = df["close"]
    v = df["volume"]

    e9  = ema(c, 9)
    e21 = ema(c, 21)
    rsi = calc_rsi(c)
    macd_h, macd_h_prev = calc_macd_hist(c)
    atr = calc_atr(df)

    vol_avg = float(v.rolling(20).mean().iloc[-1])
    vol_ok  = float(v.iloc[-1]) >= vol_avg * strategy["volume_factor"]

    trend_up  = float(e9.iloc[-1]) > float(e21.iloc[-1])
    cross_up  = trend_up and float(e9.iloc[-2]) <= float(e21.iloc[-2])
    macd_bull = macd_h > 0 and macd_h > macd_h_prev

    conds = sum([trend_up, rsi < strategy["rsi_oversold"], macd_bull, vol_ok, cross_up])

    signal = "HOLD"
    reason = ""

    if conds >= strategy["conditions_min"] and trend_up and rsi < strategy["rsi_overbought"]:
        signal = "BUY"
        parts = []
        if cross_up:  parts.append("EMA crossover")
        if rsi < strategy["rsi_oversold"]: parts.append(f"RSI={rsi}")
        if macd_bull: parts.append("MACD haussier")
        if vol_ok:    parts.append("Volume OK")
        reason = " | ".join(parts) if parts else f"Tendance ({conds}/5)"
    else:
        parts = []
        if not trend_up:  parts.append("EMA bearish")
        if not macd_bull: parts.append("MACD neutre")
        if not vol_ok:    parts.append("Volume faible")
        reason = f"En attente ({conds}/5) | " + " | ".join(parts)

    return {
        "signal":     signal,
        "reason":     reason,
        "rsi":        rsi,
        "ema_fast":   round(float(e9.iloc[-1]), 2),
        "ema_slow":   round(float(e21.iloc[-1]), 2),
        "macd_h":     round(macd_h, 4),
        "vol_ok":     vol_ok,
        "atr":        round(atr, 2),
        "conditions": conds,
    }

# ─── ORDRES ───────────────────────────────────────────────────────────────────

def open_long(state, price, reason, strategy, regime):
    atr  = state.get("last_atr", price * 0.02)
    sl   = round(price - atr * strategy["atr_multiplier"], 2)
    tp   = round(price * (1 + strategy["take_profit_pct"]), 2)
    risk = state["capital"] * strategy["risk_pct"]
    qty  = round(risk / max(price - sl, 0.01), 6)

    state.update({
        "in_position":   True,
        "entry_price":   price,
        "qty":           qty,
        "stop_loss":     sl,
        "take_profit":   tp,
        "highest_price": price,
    })
    state["total_trades"] += 1

    log.info("=" * 55)
    log.info(f"🟢 ACHAT | {CONFIG['symbol']} | Régime: {regime}")
    log.info(f"   Entrée : ${price:,.2f} | SL: ${sl:,.2f} | TP: {'Trailing' if tp > price*1.5 else f'${tp:,.2f}'}")
    log.info(f"   ATR: {atr:.2f} | Qty: {qty} | Capital: ${state['capital']:.2f}")
    log.info(f"   Signal: {reason}")
    log.info("=" * 55)

    tg(f"🟢 <b>ACHAT ETH (PAPER)</b> — {regime}\nPrix: <b>${price:,.2f}</b>\nSL: ${sl:,.2f} | TP: {'Trailing' if tp > price*1.5 else f'${tp:,.2f}'}\nSignal: {reason}")

def update_trailing(state, price, strategy):
    if price > state["highest_price"]:
        state["highest_price"] = price
        new_sl = round(price * (1 - strategy["trailing_pct"]), 2)
        if new_sl > state["stop_loss"]:
            old = state["stop_loss"]
            state["stop_loss"] = new_sl
            log.info(f"📈 Trailing: ${old:,.2f} → ${new_sl:,.2f} (plus haut: ${price:,.2f})")
            tg(f"📈 <b>Trailing stop</b>\nNouveau SL: ${new_sl:,.2f} | Plus haut: ${price:,.2f}")

def close_long(state, price, reason):
    entry   = state["entry_price"]
    qty     = state["qty"]
    pnl     = round((price - entry) * qty, 4)
    pnl_pct = round((price - entry) / entry * 100, 2)

    state["capital"]      = round(state["capital"] + pnl, 4)
    state["total_pnl"]    = round(state["total_pnl"] + pnl, 4)
    state["day_pnl"]      = round(state["day_pnl"] + pnl, 4)
    state["in_position"]  = False
    if pnl > 0:
        state["winning_trades"] += 1

    wr   = round(state["winning_trades"] / state["total_trades"] * 100, 1)
    icon = "✅" if pnl > 0 else "❌"

    log.info("=" * 55)
    log.info(f"{icon} CLÔTURE | Entrée: ${entry:,.2f} | Sortie: ${price:,.2f}")
    log.info(f"   P&L: {'+' if pnl>=0 else ''}{pnl:.4f} USDT ({pnl_pct:+.2f}%)")
    log.info(f"   Capital: ${state['capital']:.2f} | Win rate: {wr}% | {reason}")
    log.info("=" * 55)

    tg(f"{icon} <b>CLÔTURE ETH (PAPER)</b>\nEntrée: ${entry:,.2f} → Sortie: ${price:,.2f}\nP&L: <b>{'+' if pnl>=0 else ''}{pnl:.4f} USDT ({pnl_pct:+.2f}%)</b>\nCapital: ${state['capital']:.2f} | Win rate: {wr}%\n{reason}")

    for k in ["entry_price","qty","stop_loss","take_profit","highest_price"]:
        state[k] = None

# ─── PROTECTION DRAWDOWN ──────────────────────────────────────────────────────

def reset_daily(state):
    """CORRIGÉ : Reset basé sur l'heure locale Argentine, pas UTC."""
    today = local_today()
    if state.get("last_reset_date") != today:
        state["day_pnl"]         = 0.0
        state["last_reset_date"] = today
        state["paused_until"]    = None
        log.info(f"Nouveau jour (heure Argentine) — reset P&L | {local_now().strftime('%d/%m/%Y %H:%M')}")

def is_paused(state) -> bool:
    if not state.get("paused_until"):
        return False
    if local_now().isoformat() < state["paused_until"]:
        log.warning(f"Bot en pause jusqu'à {state['paused_until'][:16]}")
        return True
    state["paused_until"] = None
    return False

def check_drawdown(state) -> bool:
    # 1. Drawdown journalier -10%
    if abs(min(state["day_pnl"], 0)) >= state["capital"] * CONFIG["max_drawdown_pct"]:
        tomorrow = (local_now() + timedelta(days=1)).replace(hour=0, minute=0).isoformat()
        state["paused_until"] = tomorrow
        msg = f"⚠️ DRAWDOWN JOUR -10% | Pause jusqu'à demain\nP&L jour: {state['day_pnl']:.2f} USDT"
        log.warning(msg)
        tg(f"⚠️ <b>DRAWDOWN JOURNALIER</b>\n{msg}")
        return False

    # 2. Perte totale -15%
    total_loss = (state["capital"] - CONFIG["capital_initial"]) / CONFIG["capital_initial"]
    if total_loss <= -CONFIG["max_total_loss"]:
        pause_until = (local_now() + timedelta(hours=48)).isoformat()
        state["paused_until"] = pause_until
        msg = f"🚨 PERTE TOTALE -15% | Capital: ${state['capital']:.2f} | Pause 48h"
        log.warning(msg)
        tg(f"🚨 <b>PERTE TOTALE -15%</b>\nCapital: ${state['capital']:.2f} USDT\nBot en pause 48h")
        return False

    return True

# ─── BOUCLE PRINCIPALE ────────────────────────────────────────────────────────

def run():
    log.info("━" * 55)
    log.info("  ETH/USDT Bot v3.1 — Version finale")
    log.info("  Régime · ATR adaptatif · BTC graduel · Heure locale")
    log.info("━" * 55)
    tg("🚀 <b>Bot v3.1 démarré</b>\nVersion finale · Toutes corrections appliquées")

    state = load_state()
    save_state(state)

    regime_counter = 0

    while True:
        try:
            reset_daily(state)

            if is_paused(state):
                time.sleep(CONFIG["check_interval"])
                continue

            # Détection régime toutes les 4h
            if regime_counter % 240 == 0:
                df_4h = get_klines(CONFIG["symbol"], CONFIG["interval_slow"], CONFIG["lookback"])
                if not df_4h.empty:
                    new_regime = detect_regime(df_4h)
                    if new_regime != state["regime"]:
                        old = state["regime"]
                        state["regime"] = new_regime
                        log.info(f"🔄 Régime: {old} → {new_regime} | {STRATEGIES[new_regime]['description']}")
                        tg(f"🔄 <b>Régime: {old} → {new_regime}</b>\n{STRATEGIES[new_regime]['description']}")
            regime_counter += 1

            strategy = STRATEGIES[state["regime"]]

            # Données 1H
            df = get_klines(CONFIG["symbol"], CONFIG["interval_fast"], CONFIG["lookback"])
            if df.empty:
                log.warning("Données vides, retry 60s...")
                time.sleep(60)
                continue

            state["last_atr"] = calc_atr(df)
            analysis = analyze(df, strategy)
            price    = get_price(CONFIG["symbol"])

            if price is None:
                log.warning("Prix invalide, on ignore ce tick")
                time.sleep(60)
                continue

            log.info(
                f"[{state['regime']}] ${price:,.2f} | "
                f"EMA {analysis['ema_fast']}/{analysis['ema_slow']} | "
                f"RSI: {analysis['rsi']} | MACD: {analysis['macd_h']:+.3f} | "
                f"ATR: {analysis['atr']} | "
                f"Signal: {analysis['signal']} ({analysis['conditions']}/5)"
            )

            # Position ouverte
            if state["in_position"]:
                update_trailing(state, price, strategy)
                sl = state["stop_loss"]
                tp = state["take_profit"]

                if price <= sl:
                    close_long(state, price, f"STOP-LOSS ${sl:,.2f}")
                    check_drawdown(state)
                elif price >= tp:
                    close_long(state, price, f"TAKE-PROFIT ${tp:,.2f}")
                else:
                    pnl_live = round((price - state["entry_price"]) * state["qty"], 4)
                    log.info(f"Position | P&L live: {'+' if pnl_live>=0 else ''}{pnl_live:.4f} | SL: ${sl:,.2f} | Haut: ${state['highest_price']:,.2f}")

            # Cherche un signal
            else:
                if state["regime"] in ("BAISSIER", "RANGE"):
                    log.info(f"Marché {state['regime']} — bot en attente")
                elif analysis["signal"] == "BUY":
                    log.info("Signal BUY — vérification BTC...")
                    btc = btc_score()
                    if btc < 0.4:
                        log.info(f"BTC score trop bas ({btc:.1f}) — signal ignoré")
                    elif not check_drawdown(state):
                        pass
                    else:
                        open_long(state, price, analysis["reason"], strategy, state["regime"])
                else:
                    log.info(f"{analysis['reason']}")

            save_state(state)

        except KeyboardInterrupt:
            log.info("Arrêt manuel. État sauvegardé.")
            tg("🔴 <b>Bot arrêté manuellement</b>")
            save_state(state)
            break
        except Exception as e:
            log.error(f"Erreur inattendue: {e}")

        time.sleep(CONFIG["check_interval"])

if __name__ == "__main__":
    run()
