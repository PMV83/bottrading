#!/usr/bin/env python3
"""
Alpaca Equities Trading Bot — Mean Reversion v3.0

Changelog v3 :
  - capital_initial → 100$ (petit portefeuille, fractions d'actions)
  - calc_position_size retourne un float (4 décimales) — fractions Alpaca
  - Bracket order fractionnaire + fallback market simple si refusé
  - Fuseau Europe/Paris (ZoneInfo) sur tous les timestamps
  - [SCAN] heartbeat verbose à chaque boucle pour chaque symbole
  - state['last_scan'] alimenté → tableau Dernier Scan dans le dashboard
  - Rapport Discord de clôture au passage OPEN → CLOSED
"""

import os
import time
import json
import logging
import requests
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path

import pandas as pd
import numpy as np
from dotenv import load_dotenv

from alpaca.trading.client   import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    TakeProfitRequest,
    StopLossRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests   import StockBarsRequest, StockLatestTradeRequest
from alpaca.data.timeframe  import TimeFrame

load_dotenv()

# ── Fuseau de référence ────────────────────────────────────────────────────────
TZ = ZoneInfo("Europe/Paris")

def now() -> datetime:
    return datetime.now(TZ)

# ── Configuration ──────────────────────────────────────────────────────────────
CONFIG = {
    "symbols":            ["AAPL", "MSFT", "GOOGL", "NVDA"],
    "sma_period":         200,
    "rsi_period":         14,
    "rsi_oversold":       30,
    "lookback_days":      252,
    "risk_pct_per_trade": 0.02,
    "take_profit_pct":    0.04,
    "stop_loss_pct":      0.02,
    "check_interval_sec": 900,
    "state_file":         "alpaca-state.json",
    "market_data_file":   "market_data.json",
    "log_file":           "alpaca-trades.log",
    "capital_initial":    100.0,              # ★ Petit portefeuille
    "alpaca_api_key":     os.getenv("ALPACA_API_KEY",      ""),
    "alpaca_secret_key":  os.getenv("ALPACA_SECRET_KEY",   ""),
    "discord_webhook_url":os.getenv("DISCORD_WEBHOOK_URL", ""),
}

# ── Logging ────────────────────────────────────────────────────────────────────
log_path = Path(__file__).parent / CONFIG["log_file"]
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("ALPACA_BOT")

# ── Clients alpaca-py (globaux) ────────────────────────────────────────────────
trading_client = None
data_client    = None

def init_clients():
    global trading_client, data_client
    trading_client = TradingClient(
        api_key=CONFIG["alpaca_api_key"],
        secret_key=CONFIG["alpaca_secret_key"],
        paper=True,
    )
    data_client = StockHistoricalDataClient(
        api_key=CONFIG["alpaca_api_key"],
        secret_key=CONFIG["alpaca_secret_key"],
    )
    log.info("Clients alpaca-py initialisés")

# ── Fichiers state ─────────────────────────────────────────────────────────────
state_path       = Path(__file__).parent / CONFIG["state_file"]
market_data_path = Path(__file__).parent / CONFIG["market_data_file"]

def load_state() -> dict:
    if state_path.exists():
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                s = json.load(f)
            s.setdefault("trade_history", [])
            s.setdefault("positions",     {})
            s.setdefault("logs",          [])
            s.setdefault("last_scan",     {})
            log.info(f"Reprise — capital: ${s['capital']:.4f} | trades: {s['total_trades']}")
            return s
        except json.JSONDecodeError as e:
            log.error(f"state.json corrompu, réinitialisation : {e}")

    return {
        "capital":          CONFIG["capital_initial"],
        "buying_power":     CONFIG["capital_initial"],
        "in_position":      False,
        "positions":        {},
        "total_trades":     0,
        "winning_trades":   0,
        "total_pnl":        0.0,
        "day_pnl":          0.0,
        "last_reset_date":  str(now().date()),
        "last_signal":      "HOLD",
        "last_symbol":      None,
        "latest_prices":    {},
        "regime":           "CLOSED",
        "started_at":       now().isoformat(),
        "trade_history":    [],
        "logs":             [],
        "last_scan":        {},  # {symbol: {price, sma200, rsi, signal, scanned_at}}
    }

def save_state(s: dict):
    tmp = state_path.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(s, f, indent=2, ensure_ascii=False)
        tmp.replace(state_path)
    except Exception as e:
        log.error(f"Impossible de sauvegarder state.json : {e}")

def push_log(state: dict, message: str, level: str = "info"):
    state["logs"].append({
        "time":  now().strftime("%H:%M:%S"),
        "level": level,
        "msg":   message,
    })
    state["logs"] = state["logs"][-50:]

# ── Market data ────────────────────────────────────────────────────────────────
def fetch_and_save_market_data():
    log.info("Récupération market_data via alpaca-py...")
    result = {"updated_at": now().isoformat(), "symbols": {}}
    end    = now()
    start  = end - timedelta(days=365)

    for symbol in CONFIG["symbols"]:
        try:
            req  = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start, end=end,
                feed="iex", adjustment="split",
            )
            bars = data_client.get_stock_bars(req)
            df   = bars.df
            if df.empty:
                result["symbols"][symbol] = []
                continue
            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(symbol, level="symbol")
            df = df.reset_index().rename(columns={"timestamp": "time"})
            candles = []
            for _, row in df.iterrows():
                ts = row["time"]
                ts_int = int(ts.timestamp()) if hasattr(ts, "timestamp") else int(pd.Timestamp(ts).timestamp())
                candles.append({
                    "time":   ts_int,
                    "open":   round(float(row["open"]),   2),
                    "high":   round(float(row["high"]),   2),
                    "low":    round(float(row["low"]),    2),
                    "close":  round(float(row["close"]),  2),
                    "volume": int(row["volume"]),
                })
            candles.sort(key=lambda c: c["time"])
            result["symbols"][symbol] = candles
            log.info(f"  {symbol} : {len(candles)} bougies")
        except Exception as e:
            log.error(f"fetch_market_data({symbol}) : {e}")
            result["symbols"][symbol] = []

    tmp = market_data_path.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        tmp.replace(market_data_path)
        log.info(f"market_data.json sauvegardé")
    except Exception as e:
        log.error(f"Impossible de sauvegarder market_data.json : {e}")

def should_refresh_market_data() -> bool:
    if not market_data_path.exists():
        return True
    try:
        with open(market_data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        updated_at = datetime.fromisoformat(data.get("updated_at", "2000-01-01T00:00:00+00:00"))
        return updated_at.date() < now().date()
    except Exception:
        return True

# ── Discord ────────────────────────────────────────────────────────────────────
def discord_alert(msg: str, username: str = "Alpaca Bot"):
    webhook_url = CONFIG.get("discord_webhook_url", "")
    if not webhook_url:
        return
    clean = msg.replace("<b>", "**").replace("</b>", "**")
    try:
        requests.post(webhook_url, json={"content": clean, "username": username}, timeout=5)
    except Exception as e:
        log.warning(f"Discord : {e}")

# ── API helpers ────────────────────────────────────────────────────────────────
def get_clock():
    try:
        c = trading_client.get_clock()
        return {
            "is_open":    c.is_open,
            "next_open":  c.next_open.isoformat()  if c.next_open  else "N/A",
            "next_close": c.next_close.isoformat() if c.next_close else "N/A",
        }
    except Exception as e:
        log.error(f"get_clock() : {e}")
        return None

def get_account():
    try:
        a = trading_client.get_account()
        return {
            "buying_power":    float(a.buying_power),
            "portfolio_value": float(a.portfolio_value),
        }
    except Exception as e:
        log.error(f"get_account() : {e}")
        return None

def get_positions_alpaca():
    try:
        return [{
            "symbol":          p.symbol,
            "qty":             float(p.qty),
            "avg_entry_price": float(p.avg_entry_price),
            "current_price":   float(p.current_price) if p.current_price else None,
        } for p in trading_client.get_all_positions()]
    except Exception as e:
        log.error(f"get_positions_alpaca() : {e}")
        return []

def get_bars(symbol: str, limit: int = 252) -> pd.DataFrame:
    try:
        end   = now()
        start = end - timedelta(days=limit + 50)
        req   = StockBarsRequest(
            symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
            start=start, end=end, feed="iex", adjustment="split",
        )
        bars = data_client.get_stock_bars(req)
        df   = bars.df
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level="symbol")
        df = df.reset_index().rename(columns={"timestamp": "time"})
        df = df[["time", "open", "high", "low", "close", "volume"]].sort_values("time")
        return df.tail(limit).reset_index(drop=True)
    except Exception as e:
        log.error(f"get_bars({symbol}) : {e}")
        return pd.DataFrame()

def get_latest_price(symbol: str):
    try:
        req   = StockLatestTradeRequest(symbol_or_symbols=symbol, feed="iex")
        trade = data_client.get_stock_latest_trade(req)
        return float(trade[symbol].price)
    except Exception as e:
        log.error(f"get_latest_price({symbol}) : {e}")
        return None

# ── Indicateurs ───────────────────────────────────────────────────────────────
def calc_sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()

def calc_rsi(series: pd.Series, period: int = 14) -> float:
    if len(series) < period + 1:
        return 50.0
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - (100 / (1 + rs))
    val   = float(rsi.iloc[-1])
    return round(val, 2) if not np.isnan(val) else 50.0

# ── Stratégie ─────────────────────────────────────────────────────────────────
def analyze_symbol(symbol: str) -> dict:
    result = {"symbol": symbol, "signal": "HOLD", "reason": "", "rsi": 50.0, "sma200": None, "price": None}
    df = get_bars(symbol, limit=CONFIG["lookback_days"])
    if df.empty or len(df) < CONFIG["sma_period"] + 5:
        result["reason"] = f"Données insuffisantes ({len(df)} bougies)"
        return result
    close   = df["close"]
    sma200  = calc_sma(close, CONFIG["sma_period"])
    rsi     = calc_rsi(close, CONFIG["rsi_period"])
    price   = float(close.iloc[-1])
    sma_val = float(sma200.iloc[-1])
    result.update({"rsi": rsi, "sma200": round(sma_val, 2), "price": round(price, 2)})
    if price > sma_val and rsi < CONFIG["rsi_oversold"]:
        result["signal"] = "BUY"
        result["reason"] = f"Prix>${sma_val:.2f} | RSI={rsi}<{CONFIG['rsi_oversold']}"
    elif price <= sma_val:
        result["reason"] = f"Prix SOUS SMA200 ({sma_val:.2f}) | RSI={rsi}"
    else:
        result["reason"] = f"RSI={rsi} non oversold | SMA200 OK"
    return result

# ── Sizing fractionnel ★ ───────────────────────────────────────────────────────
def calc_position_size(capital: float, buying_power: float, price: float) -> float:
    """
    ★ v3 : Retourne un float (4 décimales) pour les fractions d'actions.
    Exemple : capital=100$, risk=2$, SL_dist=4$ (2% de 200$) → qty=0.5
    """
    risk_amount = capital * CONFIG["risk_pct_per_trade"]
    sl_distance = price   * CONFIG["stop_loss_pct"]
    qty_risk    = risk_amount / max(sl_distance, 0.0001)

    cost = qty_risk * price
    if cost > buying_power * 0.95:
        qty_bp = (buying_power * 0.95) / price
        log.warning(f"BP limité : {qty_risk:.4f} → {qty_bp:.4f} fractions (BP=${buying_power:.4f})")
        qty_risk = qty_bp

    qty_final = round(qty_risk, 4)
    return qty_final if qty_final >= 0.001 else 0.0

# ── Bracket order avec fallback market simple ★ ───────────────────────────────
def place_bracket_order(symbol: str, qty: float, entry_price: float):
    """
    ★ v3 : qty est un float (fractions d'actions).

    Alpaca supporte les bracket orders fractionnels sur le compte live.
    Sur le paper trading, certains brokers refusent les brackets fractionnels :
    dans ce cas, fallback sur un market order simple (sans SL/TP automatique).
    sync_positions_from_alpaca surveille la clôture et calcule le P&L.
    """
    sl_price = round(entry_price * (1 - CONFIG["stop_loss_pct"]),  2)
    tp_price = round(entry_price * (1 + CONFIG["take_profit_pct"]), 2)

    # Tentative 1 : Bracket order complet
    try:
        order = trading_client.submit_order(MarketOrderRequest(
            symbol=symbol, qty=qty, side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY, order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=tp_price),
            stop_loss=StopLossRequest(stop_price=sl_price),
        ))
        log.info(f"✅ Bracket order | {symbol} | qty:{qty} | SL:${sl_price} | TP:${tp_price} | ID:{order.id}")
        return {"id": str(order.id), "status": str(order.status), "type": "bracket"}
    except Exception as e:
        err = str(e).lower()
        if any(kw in err for kw in ("fractional", "not supported", "invalid", "cannot")):
            log.warning(f"Bracket fractionnel refusé ({e}) — fallback market simple")
        else:
            log.error(f"Bracket order refusé — erreur non récupérable : {e}")
            return None

    # Tentative 2 (Fallback) : Market order simple sans SL/TP automatique
    try:
        order = trading_client.submit_order(MarketOrderRequest(
            symbol=symbol, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
        ))
        log.warning(f"⚠️ Market simple (SANS bracket) | {symbol} | qty:{qty} | ID:{order.id}")
        return {"id": str(order.id), "status": str(order.status), "type": "market_only"}
    except Exception as e2:
        log.error(f"Fallback market refusé ({symbol}) : {e2}")
        return None

# ── Sync positions ─────────────────────────────────────────────────────────────
def sync_positions_from_alpaca(state: dict):
    alpaca_positions = get_positions_alpaca()
    alpaca_symbols   = {p["symbol"] for p in alpaca_positions}

    for sym in set(state["positions"].keys()) - alpaca_symbols:
        pos       = state["positions"][sym]
        entry     = pos["entry_price"]
        qty       = pos["qty"]
        tp_price  = pos["take_profit"]
        exit_price = get_latest_price(sym) or entry
        pnl        = round((exit_price - entry) * qty, 4)
        trade_type = "TP" if exit_price >= (entry * (1 + CONFIG["take_profit_pct"] * 0.9)) else "SL"

        state["capital"]      = round(state["capital"]   + pnl, 4)
        state["total_pnl"]    = round(state["total_pnl"] + pnl, 4)
        state["day_pnl"]      = round(state["day_pnl"]   + pnl, 4)
        state["total_trades"] += 1
        if pnl > 0:
            state["winning_trades"] += 1

        state["trade_history"].append({
            "type": trade_type, "symbol": sym, "pnl": pnl,
            "entry": entry, "exit": round(exit_price, 2),
            "qty": qty, "sl": pos["stop_loss"], "tp": tp_price,
            "date": now().isoformat(),
        })
        state["trade_history"] = state["trade_history"][-10:]
        del state["positions"][sym]

        icon = "✅" if pnl > 0 else "❌"
        msg  = f"{icon} CLÔTURE {sym} | ${entry:.2f}→${exit_price:.2f} | P&L:{'+' if pnl>=0 else ''}{pnl:.4f}$ | {trade_type}"
        log.info(msg)
        push_log(state, msg, "sell")
        discord_alert(f"{icon} **CLÔTURE {sym}**\n${entry:.2f} → ${exit_price:.2f}\nP&L: **{'+' if pnl>=0 else ''}{pnl:.4f}$** | {trade_type}\nCapital: ${state['capital']:.4f}")

    for p in alpaca_positions:
        if p["current_price"]:
            state["latest_prices"][p["symbol"]] = p["current_price"]

    state["in_position"] = len(state["positions"]) > 0

# ── Reset journalier ───────────────────────────────────────────────────────────
def reset_daily_if_needed(state: dict):
    today = str(now().date())
    if state.get("last_reset_date") != today:
        state["day_pnl"]         = 0.0
        state["last_reset_date"] = today
        log.info(f"Nouveau jour Paris — reset P&L journalier ({today})")

# ── Rapport de clôture Discord ★ ──────────────────────────────────────────────
def send_close_report(state: dict):
    day_pnl = state["day_pnl"]
    capital = state["capital"]
    pnl_str = f"{'+' if day_pnl >= 0 else ''}{day_pnl:.4f}$"
    icon    = "📈" if day_pnl >= 0 else "📉"
    msg = (
        f"🏁 **Marché fermé** | {now().strftime('%d/%m %H:%M')} (Paris)\n"
        f"Capital : **{capital:.4f}$** | P&L Jour : **{pnl_str}** {icon}\n"
        f"Positions ouvertes : {len(state['positions'])} | "
        f"Trades totaux : {state['total_trades']}"
    )
    discord_alert(msg, username="Alpaca Bot — Clôture")
    push_log(state, f"Rapport clôture Discord | P&L Jour: {pnl_str}", "info")
    log.info(f"Rapport de clôture envoyé sur Discord")

# ── Heartbeat [SCAN] ★ ────────────────────────────────────────────────────────
def log_scan_result(state: dict, analysis: dict):
    """
    ★ v3 : Log verbose format [SCAN] et alimente state['last_scan']
    pour l'affichage dans le dashboard (tableau Dernier Scan).
    """
    sym    = analysis["symbol"]
    price  = analysis["price"]  or "N/A"
    sma    = analysis["sma200"] or "N/A"
    rsi    = analysis["rsi"]
    signal = analysis["signal"]

    arrow  = "🟢 BUY!" if signal == "BUY" else ("⚪" if signal == "HOLD" else "🔴")
    log.info(f"[SCAN] {sym}: ${price} | SMA200: ${sma} | RSI: {rsi} → {signal} {arrow}")

    state["last_scan"][sym] = {
        "price":      price,
        "sma200":     sma,
        "rsi":        rsi,
        "signal":     signal,
        "reason":     analysis.get("reason", ""),
        "scanned_at": now().strftime("%H:%M:%S"),
    }

# ── Boucle principale ──────────────────────────────────────────────────────────
def run():
    log.info("━" * 60)
    log.info("  Alpaca Equities Bot — Mean Reversion v3.0")
    log.info(f"  Capital : ${CONFIG['capital_initial']:.2f} | Fuseau : Europe/Paris")
    log.info(f"  Univers : {', '.join(CONFIG['symbols'])}")
    log.info("━" * 60)

    if not CONFIG["alpaca_api_key"] or not CONFIG["alpaca_secret_key"]:
        log.critical("ALPACA_API_KEY ou ALPACA_SECRET_KEY manquante — arrêt.")
        return

    init_clients()
    fetch_and_save_market_data()

    state = load_state()
    save_state(state)
    discord_alert(
        f"🚀 **Alpaca Bot v3.0 démarré** | {now().strftime('%d/%m %H:%M')} Paris\n"
        f"Capital: **${CONFIG['capital_initial']:.2f}** | Fractions activées | Paper Trading"
    )

    was_open = False  # Pour détecter la transition OPEN → CLOSED

    while True:
        try:
            reset_daily_if_needed(state)

            if should_refresh_market_data():
                fetch_and_save_market_data()

            clock = get_clock()
            if clock is None:
                log.warning("Market Clock indisponible — retry 60s")
                time.sleep(60)
                continue

            is_open    = clock["is_open"]
            next_open  = clock["next_open"]
            next_close = clock["next_close"]
            state["regime"] = "OPEN" if is_open else "CLOSED"

            # ★ Rapport de clôture Discord : transition OPEN → CLOSED
            if was_open and not is_open:
                send_close_report(state)
            was_open = is_open

            if not is_open:
                log.info(f"Marché FERMÉ — prochain open : {next_open}")
                push_log(state, f"Marché fermé. Prochain open: {next_open}", "warn")
                save_state(state)
                time.sleep(CONFIG["check_interval_sec"])
                continue

            account = get_account()
            if account is None:
                log.warning("Compte Alpaca indisponible — retry 60s")
                time.sleep(60)
                continue

            buying_power    = account["buying_power"]
            portfolio_value = account["portfolio_value"]
            state["capital"]      = round(portfolio_value, 4)
            state["buying_power"] = round(buying_power,    4)

            log.info(
                f"Marché OUVERT | Capital: ${portfolio_value:.4f} | "
                f"BP: ${buying_power:.4f} | Positions: {len(state['positions'])} | "
                f"Clôture: {next_close}"
            )
            push_log(state, f"Scan | BP:${buying_power:.4f} | Pos:{len(state['positions'])}", "info")

            sync_positions_from_alpaca(state)

            for symbol in CONFIG["symbols"]:

                if symbol in state["positions"]:
                    price = get_latest_price(symbol)
                    if price:
                        state["latest_prices"][symbol] = price
                    pos      = state["positions"][symbol]
                    cur      = price or pos["entry_price"]
                    pnl_live = round((cur - pos["entry_price"]) * pos["qty"], 4)
                    log.info(
                        f"[SCAN] {symbol}: ${cur:.2f} | EN POSITION | "
                        f"Entrée:${pos['entry_price']:.2f} | "
                        f"P&L live:{'+' if pnl_live>=0 else ''}{pnl_live:.4f}$"
                    )
                    state["last_scan"][symbol] = {
                        "price":      round(cur, 2),
                        "sma200":     None,
                        "rsi":        None,
                        "signal":     "POSITION",
                        "reason":     f"En position (entrée {pos.get('entry_time','?')[:10]})",
                        "scanned_at": now().strftime("%H:%M:%S"),
                    }
                    continue

                analysis = analyze_symbol(symbol)
                state["latest_prices"][symbol] = analysis["price"] or 0

                # ★ Heartbeat [SCAN] verbose
                log_scan_result(state, analysis)

                if analysis["signal"] != "BUY":
                    continue

                state["last_signal"] = "BUY"
                state["last_symbol"] = symbol
                price = analysis["price"]

                if not price or price <= 0:
                    log.warning(f"[{symbol}] Prix invalide, skip")
                    continue

                # ★ Sizing fractionnel (float)
                qty = calc_position_size(state["capital"], buying_power, price)
                if qty <= 0:
                    msg = f"[{symbol}] Sizing=0 (capital ${state['capital']:.4f} insuffisant)"
                    log.warning(msg)
                    push_log(state, msg, "warn")
                    continue

                cost = round(qty * price, 4)
                log.info(f"[{symbol}] BUY SIGNAL | qty:{qty} fractions | coût:${cost:.4f}")

                order = place_bracket_order(symbol, qty, price)
                if order is None:
                    push_log(state, f"[{symbol}] Ordre refusé par Alpaca", "warn")
                    continue

                sl_price = round(price * (1 - CONFIG["stop_loss_pct"]),  2)
                tp_price = round(price * (1 + CONFIG["take_profit_pct"]), 2)
                type_label = "Bracket" if order["type"] == "bracket" else "Market simple ⚠️"

                state["positions"][symbol] = {
                    "entry_price": price,
                    "qty":         qty,
                    "stop_loss":   sl_price,
                    "take_profit": tp_price,
                    "entry_time":  now().isoformat(),
                    "order_id":    order["id"],
                    "order_type":  order["type"],
                }
                state["in_position"] = True

                msg = f"🟢 ACHAT {symbol} | ${price:.2f} × {qty} fractions | SL:${sl_price} | TP:${tp_price} | ${cost:.4f} | {type_label}"
                log.info("=" * 60)
                log.info(msg)
                log.info("=" * 60)
                push_log(state, msg, "buy")
                discord_alert(
                    f"🟢 **ACHAT {symbol}**\n"
                    f"${price:.2f} × **{qty} fractions** | {type_label}\n"
                    f"SL:${sl_price} | TP:${tp_price} | Coût:${cost:.4f}\n"
                    f"Signal: {analysis['reason']}"
                )

            save_state(state)
            log.info("state.json sauvegardé — fin de cycle")

        except KeyboardInterrupt:
            log.info("Arrêt manuel")
            save_state(state)
            break
        except Exception as e:
            log.error(f"Erreur boucle : {e}", exc_info=True)
            push_log(state, f"Erreur: {e}", "error")
            save_state(state)
            time.sleep(30)

        time.sleep(CONFIG["check_interval_sec"])


if __name__ == "__main__":
    run()