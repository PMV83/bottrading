#!/usr/bin/env python3
"""
Alpaca Equities Trading Bot — Mean Reversion
Architecture miroir du ETH/USDT Bot : backend Python → state.json → dashboard HTML
Stratégie : Mean Reversion sur actions US (SMA200 + RSI oversold)
Paper Trading uniquement via l'API Alpaca

Dépendances : pip install alpaca-py pandas numpy python-dotenv
"""

import os
import time
import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests          # Uniquement pour Telegram (pas d'SDK officiel)
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# ─── SDK OFFICIEL ALPACA-PY ───────────────────────────────────────────────────
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    TakeProfitRequest,
    StopLossRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame

# ─── CHARGEMENT DE L'ENVIRONNEMENT ────────────────────────────────────────────

load_dotenv()

# ─── CONFIGURATION CENTRALE ───────────────────────────────────────────────────

CONFIG = {
    # Univers de trading (mean reversion sur large caps US)
    "symbols": ["AAPL", "MSFT", "GOOGL", "NVDA"],

    # Paramètres de stratégie
    "sma_period":         200,      # Filtre tendance long terme
    "rsi_period":         14,       # RSI standard
    "rsi_oversold":       30,       # Seuil d'entrée oversold
    "lookback_days":      252,      # Bougies historiques à récupérer (> sma_period)

    # Gestion du risque
    "risk_pct_per_trade": 0.02,     # 2% du capital risqué par trade
    "take_profit_pct":    0.04,     # TP à +4% de l'entrée
    "stop_loss_pct":      0.02,     # SL à -2% de l'entrée (ratio R:R = 2:1)

    # Exécution
    "check_interval_sec": 900,      # Vérification toutes les 15 minutes

    # Fichiers
    "state_file":       "state.json",
    "market_data_file": "market_data.json",
    "log_file":         "alpaca-trades.log",

    # Capital de départ (pour calcul P&L affiché, Alpaca gère le vrai solde)
    "capital_initial": 100000.0,

    # Clés API (chargées depuis .env)
    "alpaca_api_key":    os.getenv("ALPACA_API_KEY", ""),
    "alpaca_secret_key": os.getenv("ALPACA_SECRET_KEY", ""),
    "telegram_token":    os.getenv("TELEGRAM_TOKEN", ""),
    "telegram_chat_id":  os.getenv("TELEGRAM_CHAT_ID", ""),
}

# ─── LOGGING ──────────────────────────────────────────────────────────────────

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

# ─── CLIENTS ALPACA-PY (initialisés après vérification des clés) ───────────────

trading_client: TradingClient | None = None
data_client:    StockHistoricalDataClient | None = None

def init_clients():
    """Initialise les clients alpaca-py. Appelé une seule fois au démarrage."""
    global trading_client, data_client
    trading_client = TradingClient(
        api_key=CONFIG["alpaca_api_key"],
        secret_key=CONFIG["alpaca_secret_key"],
        paper=True,       # Paper trading
    )
    # StockHistoricalDataClient fonctionne sans clé (plan gratuit IEX)
    # mais une clé améliore les quotas
    data_client = StockHistoricalDataClient(
        api_key=CONFIG["alpaca_api_key"],
        secret_key=CONFIG["alpaca_secret_key"],
    )
    log.info("Clients alpaca-py initialisés (TradingClient + StockHistoricalDataClient)")

# ─── ÉTAT PERSISTANT (PONT JSON VERS LE DASHBOARD) ───────────────────────────

state_path       = Path(__file__).parent / CONFIG["state_file"]
market_data_path = Path(__file__).parent / CONFIG["market_data_file"]

def load_state() -> dict:
    """
    Charge l'état depuis le fichier JSON.
    Si le fichier n'existe pas, initialise un état vide.
    """
    if state_path.exists():
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                s = json.load(f)
            s.setdefault("trade_history", [])
            s.setdefault("positions", {})
            s.setdefault("logs", [])
            log.info(
                f"Reprise — capital: ${s['capital']:,.2f} | "
                f"trades: {s['total_trades']} | positions ouvertes: {len(s['positions'])}"
            )
            return s
        except json.JSONDecodeError as e:
            log.error(f"state.json corrompu, réinitialisation : {e}")

    return {
        "capital":         CONFIG["capital_initial"],
        "buying_power":    CONFIG["capital_initial"],
        "in_position":     False,
        "positions":       {},
        "total_trades":    0,
        "winning_trades":  0,
        "total_pnl":       0.0,
        "day_pnl":         0.0,
        "last_reset_date": str(datetime.now(timezone.utc).date()),
        "last_signal":     "HOLD",
        "last_symbol":     None,
        "latest_prices":   {},
        "regime":          "OPEN",
        "started_at":      datetime.now(timezone.utc).isoformat(),
        "trade_history":   [],
        "logs":            [],
    }

def save_state(s: dict):
    """Sauvegarde atomique du state.json."""
    tmp_path = state_path.with_suffix(".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(s, f, indent=2, ensure_ascii=False)
        tmp_path.replace(state_path)
    except Exception as e:
        log.error(f"Impossible de sauvegarder state.json : {e}")

def push_log(state: dict, message: str, level: str = "info"):
    """Ajoute une ligne de log horodatée dans state.json pour le dashboard."""
    entry = {
        "time":  datetime.now(timezone.utc).strftime("%H:%M:%S"),
        "level": level,
        "msg":   message,
    }
    state["logs"].append(entry)
    state["logs"] = state["logs"][-50:]

# ─── MARKET DATA — FETCH ET PERSISTANCE ───────────────────────────────────────

def fetch_and_save_market_data():
    """
    Récupère 1 an de bougies quotidiennes pour chaque symbole via alpaca-py
    (StockHistoricalDataClient) et sauvegarde le résultat dans market_data.json.

    Le dashboard HTML lit ce fichier local au lieu d'appeler Yahoo Finance.
    Format de sortie :
    {
      "updated_at": "2024-01-15T18:00:00Z",
      "symbols": {
        "AAPL": [{"time": 1704067200, "open": 185.0, "high": 188.0, "low": 184.0, "close": 187.0, "volume": 123456}, ...],
        ...
      }
    }
    """
    log.info("Récupération des données de marché (alpaca-py)…")
    result = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "symbols":    {},
    }
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=365)

    for symbol in CONFIG["symbols"]:
        try:
            req  = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
                feed="iex",          # IEX = gratuit ; remplacer par "sip" avec abonnement
                adjustment="split",  # Ajustement des splits
            )
            bars = data_client.get_stock_bars(req)
            df   = bars.df

            if df.empty:
                log.warning(f"Aucune donnée market_data pour {symbol}")
                result["symbols"][symbol] = []
                continue

            # Si MultiIndex (symbol, timestamp), ne garder que ce symbole
            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(symbol, level="symbol")

            df = df.reset_index()
            df = df.rename(columns={"timestamp": "time"})

            candles = []
            for _, row in df.iterrows():
                ts = row["time"]
                # Convertir en timestamp UNIX entier (secondes)
                if hasattr(ts, "timestamp"):
                    ts_int = int(ts.timestamp())
                else:
                    ts_int = int(pd.Timestamp(ts).timestamp())
                candles.append({
                    "time":   ts_int,
                    "open":   round(float(row["open"]),   2),
                    "high":   round(float(row["high"]),   2),
                    "low":    round(float(row["low"]),    2),
                    "close":  round(float(row["close"]),  2),
                    "volume": int(row["volume"]),
                })

            # Trier par timestamp croissant (obligatoire pour Lightweight Charts)
            candles.sort(key=lambda c: c["time"])
            result["symbols"][symbol] = candles
            log.info(f"  {symbol} : {len(candles)} bougies récupérées")

        except Exception as e:
            log.error(f"fetch_market_data({symbol}) : {e}")
            result["symbols"][symbol] = []

    # Sauvegarde atomique
    tmp_path = market_data_path.with_suffix(".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        tmp_path.replace(market_data_path)
        log.info(f"market_data.json sauvegardé ({market_data_path})")
    except Exception as e:
        log.error(f"Impossible de sauvegarder market_data.json : {e}")

def should_refresh_market_data() -> bool:
    """
    Retourne True si market_data.json n'existe pas ou a été créé un autre jour
    (mise à jour une fois par jour après la clôture des marchés).
    """
    if not market_data_path.exists():
        return True
    try:
        with open(market_data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        updated_at = datetime.fromisoformat(data.get("updated_at", "2000-01-01T00:00:00+00:00"))
        # Rafraîchir si le fichier date d'avant aujourd'hui (UTC)
        today = datetime.now(timezone.utc).date()
        return updated_at.date() < today
    except Exception:
        return True

# ─── TELEGRAM ─────────────────────────────────────────────────────────────────

def tg(msg: str):
    """Envoie une notification Telegram. Échoue silencieusement si non configuré."""
    token = CONFIG["telegram_token"]
    chat  = CONFIG["telegram_chat_id"]
    if not token or not chat:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat, "text": msg, "parse_mode": "HTML"},
            timeout=5,
        )
    except Exception as e:
        log.warning(f"Telegram : {e}")

# ─── API ALPACA — MARCHÉ (via TradingClient) ──────────────────────────────────

def get_clock() -> dict | None:
    """
    Interroge le Market Clock d'Alpaca via TradingClient.
    Retourne un dict avec les champs is_open, next_open, next_close ou None.
    """
    try:
        clock = trading_client.get_clock()
        return {
            "is_open":    clock.is_open,
            "next_open":  clock.next_open.isoformat() if clock.next_open else "N/A",
            "next_close": clock.next_close.isoformat() if clock.next_close else "N/A",
        }
    except Exception as e:
        log.error(f"get_clock() : {e}")
        return None

def get_account() -> dict | None:
    """
    Récupère les informations du compte Alpaca via TradingClient.
    Retourne un dict avec buying_power, portfolio_value, etc.
    """
    try:
        account = trading_client.get_account()
        return {
            "buying_power":    float(account.buying_power),
            "portfolio_value": float(account.portfolio_value),
            "cash":            float(account.cash),
            "equity":          float(account.equity),
        }
    except Exception as e:
        log.error(f"get_account() : {e}")
        return None

def get_positions_alpaca() -> list:
    """
    Récupère les positions ouvertes depuis Alpaca via TradingClient.
    Retourne une liste de dicts normalisés.
    """
    try:
        positions = trading_client.get_all_positions()
        result = []
        for p in positions:
            result.append({
                "symbol":        p.symbol,
                "qty":           float(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price) if p.current_price else None,
                "unrealized_pl": float(p.unrealized_pl) if p.unrealized_pl else 0.0,
            })
        return result
    except Exception as e:
        log.error(f"get_positions_alpaca() : {e}")
        return []

# ─── API ALPACA — DONNÉES HISTORIQUES (via StockHistoricalDataClient) ──────────

def get_bars(symbol: str, limit: int = 252) -> pd.DataFrame:
    """
    Récupère les bougies quotidiennes OHLCV via StockHistoricalDataClient.
    Utilisé pour le calcul des indicateurs techniques (SMA200, RSI).
    """
    try:
        end   = datetime.now(timezone.utc)
        start = end - timedelta(days=limit + 50)  # Marge pour les jours fériés

        req  = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            feed="iex",
            adjustment="split",
        )
        bars = data_client.get_stock_bars(req)
        df   = bars.df

        if df.empty:
            log.warning(f"Aucune donnée reçue pour {symbol}")
            return pd.DataFrame()

        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level="symbol")

        df = df.reset_index()
        df = df.rename(columns={"timestamp": "time"})
        df = df[["time", "open", "high", "low", "close", "volume"]].copy()
        df.sort_values("time", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df.tail(limit).reset_index(drop=True)

    except Exception as e:
        log.error(f"get_bars({symbol}) : {e}")
        return pd.DataFrame()

def get_latest_price(symbol: str) -> float | None:
    """Récupère le dernier prix coté via StockHistoricalDataClient."""
    try:
        req   = StockLatestTradeRequest(symbol_or_symbols=symbol, feed="iex")
        trade = data_client.get_stock_latest_trade(req)
        return float(trade[symbol].price)
    except Exception as e:
        log.error(f"get_latest_price({symbol}) : {e}")
        return None

# ─── INDICATEURS TECHNIQUES ───────────────────────────────────────────────────

def calc_sma(series: pd.Series, period: int) -> pd.Series:
    """Moyenne Mobile Simple."""
    return series.rolling(period).mean()

def calc_rsi(series: pd.Series, period: int = 14) -> float:
    """
    RSI de Wilder (EWM). Retourne la valeur RSI du dernier point.
    Retourne 50.0 si pas assez de données (neutre).
    """
    if len(series) < period + 1:
        return 50.0
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - (100 / (1 + rs))
    val   = float(rsi.iloc[-1])
    return round(val, 2) if not np.isnan(val) else 50.0

# ─── STRATÉGIE — MEAN REVERSION ───────────────────────────────────────────────

def analyze_symbol(symbol: str) -> dict:
    """
    Analyse un symbole selon la stratégie Mean Reversion :
      - Prix > SMA200  →  tendance longue haussière (filtre)
      - RSI(14) < 30   →  survente (signal d'entrée)
    """
    result = {
        "symbol": symbol,
        "signal": "HOLD",
        "reason": "",
        "rsi":    50.0,
        "sma200": None,
        "price":  None,
    }

    df = get_bars(symbol, limit=CONFIG["lookback_days"])
    if df.empty or len(df) < CONFIG["sma_period"] + 5:
        result["reason"] = f"Données insuffisantes ({len(df)} bougies)"
        return result

    close   = df["close"]
    sma200  = calc_sma(close, CONFIG["sma_period"])
    rsi     = calc_rsi(close, CONFIG["rsi_period"])
    price   = float(close.iloc[-1])
    sma_val = float(sma200.iloc[-1])

    result["rsi"]    = rsi
    result["sma200"] = round(sma_val, 2)
    result["price"]  = round(price, 2)

    above_sma = price > sma_val
    oversold  = rsi < CONFIG["rsi_oversold"]

    if above_sma and oversold:
        result["signal"] = "BUY"
        result["reason"] = f"Prix>${sma_val:.2f} (SMA200) | RSI={rsi} < {CONFIG['rsi_oversold']}"
    elif not above_sma:
        result["reason"] = f"Prix SOUS SMA200 ({sma_val:.2f}) | RSI={rsi}"
    else:
        result["reason"] = f"RSI={rsi} (seuil: {CONFIG['rsi_oversold']}) | Au-dessus SMA200"

    return result

# ─── GESTION DU RISQUE ET DIMENSIONNEMENT ─────────────────────────────────────

def calc_position_size(capital: float, buying_power: float, price: float) -> int:
    """
    Calcule le nombre d'actions à acheter selon la règle des 2%.
    Risque = 2% du capital, SL = 2% sous l'entrée.
    Retourne 0 si le buying power est insuffisant.
    """
    risk_amount = capital * CONFIG["risk_pct_per_trade"]
    sl_distance = price * CONFIG["stop_loss_pct"]
    qty_risk    = int(risk_amount / max(sl_distance, 0.01))

    cost = qty_risk * price
    if cost > buying_power * 0.95:
        qty_bp = int((buying_power * 0.95) / price)
        log.warning(
            f"Buying power limité : {qty_risk} actions → {qty_bp} "
            f"(BP disponible: ${buying_power:,.2f})"
        )
        qty_risk = qty_bp

    return max(qty_risk, 0)

# ─── ORDRES ALPACA — BRACKET ORDER (via TradingClient) ───────────────────────

def place_bracket_order(symbol: str, qty: int, entry_price: float) -> dict | None:
    """
    Soumet un Bracket Order (buy market + stop loss + take profit) via alpaca-py.
    Utilise MarketOrderRequest avec order_class=OrderClass.BRACKET,
    take_profit=TakeProfitRequest et stop_loss=StopLossRequest.
    Retourne un dict normalisé ou None en cas d'erreur.
    """
    sl_price = round(entry_price * (1 - CONFIG["stop_loss_pct"]), 2)
    tp_price = round(entry_price * (1 + CONFIG["take_profit_pct"]), 2)

    try:
        order_request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=tp_price),
            stop_loss=StopLossRequest(stop_price=sl_price),
        )

        order = trading_client.submit_order(order_request)

        log.info(
            f"✅ Bracket order soumis — {symbol} | "
            f"Qty: {qty} | SL: ${sl_price} | TP: ${tp_price} | "
            f"Order ID: {order.id}"
        )
        return {"id": str(order.id), "status": str(order.status)}

    except Exception as e:
        log.error(f"place_bracket_order({symbol}) : {e}")
        return None

# ─── GESTION DES POSITIONS OUVERTES ───────────────────────────────────────────

def sync_positions_from_alpaca(state: dict):
    """
    Synchronise les positions ouvertes entre Alpaca et notre state.json.
    Détecte les positions clôturées (SL/TP atteint) et met à jour les métriques.
    """
    alpaca_positions = get_positions_alpaca()
    alpaca_symbols   = {p["symbol"] for p in alpaca_positions}

    closed_symbols = set(state["positions"].keys()) - alpaca_symbols
    for sym in closed_symbols:
        pos       = state["positions"][sym]
        entry     = pos["entry_price"]
        qty       = pos["qty"]
        tp_price  = pos["take_profit"]
        sl_price  = pos["stop_loss"]

        exit_price = get_latest_price(sym) or entry
        pnl        = round((exit_price - entry) * qty, 2)
        trade_type = "TP" if exit_price >= (entry * (1 + CONFIG["take_profit_pct"] * 0.9)) else "SL"

        state["capital"]   = round(state["capital"] + pnl, 2)
        state["total_pnl"] = round(state["total_pnl"] + pnl, 2)
        state["day_pnl"]   = round(state["day_pnl"] + pnl, 2)
        state["total_trades"] += 1
        if pnl > 0:
            state["winning_trades"] += 1

        state["trade_history"].append({
            "type":   trade_type,
            "symbol": sym,
            "pnl":    pnl,
            "entry":  entry,
            "exit":   round(exit_price, 2),
            "qty":    qty,
            "sl":     sl_price,
            "tp":     tp_price,
            "date":   datetime.now(timezone.utc).isoformat(),
        })
        state["trade_history"] = state["trade_history"][-10:]
        del state["positions"][sym]

        icon = "✅" if pnl > 0 else "❌"
        msg  = (
            f"{icon} CLÔTURE {sym} | "
            f"Entrée: ${entry:.2f} → Sortie: ${exit_price:.2f} | "
            f"P&L: {'+' if pnl>=0 else ''}{pnl:.2f}$ | {trade_type}"
        )
        log.info(msg)
        push_log(state, msg, "sell")
        tg(
            f"{icon} <b>CLÔTURE {sym}</b>\n"
            f"Entrée: ${entry:.2f} → Sortie: ${exit_price:.2f}\n"
            f"P&L: <b>{'+' if pnl>=0 else ''}{pnl:.2f}$</b> | {trade_type}\n"
            f"Capital: ${state['capital']:,.2f}"
        )

    for pos_data in alpaca_positions:
        sym = pos_data["symbol"]
        if pos_data["current_price"]:
            state["latest_prices"][sym] = pos_data["current_price"]

    state["in_position"] = len(state["positions"]) > 0

# ─── RESET JOURNALIER ─────────────────────────────────────────────────────────

def reset_daily_if_needed(state: dict):
    """Reset le P&L journalier à minuit UTC."""
    today = str(datetime.now(timezone.utc).date())
    if state.get("last_reset_date") != today:
        state["day_pnl"]         = 0.0
        state["last_reset_date"] = today
        log.info(f"Nouveau jour UTC — reset P&L journalier ({today})")

# ─── BOUCLE PRINCIPALE ────────────────────────────────────────────────────────

def run():
    log.info("━" * 60)
    log.info("  Alpaca Equities Bot — Mean Reversion v2.0 (alpaca-py)")
    log.info(f"  Univers : {', '.join(CONFIG['symbols'])}")
    log.info("━" * 60)

    if not CONFIG["alpaca_api_key"] or not CONFIG["alpaca_secret_key"]:
        log.critical("ALPACA_API_KEY ou ALPACA_SECRET_KEY manquante dans .env — arrêt.")
        return

    # Initialisation des clients alpaca-py
    init_clients()

    # Récupération initiale des données de marché → market_data.json
    fetch_and_save_market_data()

    state = load_state()
    save_state(state)
    tg("🚀 <b>Alpaca Bot démarré</b>\nStratégie: Mean Reversion (SMA200 + RSI)\nMode: Paper Trading")

    while True:
        try:
            # ── 1. Reset journalier ────────────────────────────────────────────
            reset_daily_if_needed(state)

            # ── 2. Mise à jour quotidienne de market_data.json ─────────────────
            if should_refresh_market_data():
                fetch_and_save_market_data()

            # ── 3. Vérification Market Clock ───────────────────────────────────
            clock = get_clock()
            if clock is None:
                log.warning("Impossible de récupérer le Market Clock — retry 60s")
                time.sleep(60)
                continue

            is_open    = clock.get("is_open", False)
            next_open  = clock.get("next_open", "N/A")
            next_close = clock.get("next_close", "N/A")
            state["regime"] = "OPEN" if is_open else "CLOSED"

            if not is_open:
                log.info(f"Marché FERMÉ — prochaine ouverture : {next_open}")
                push_log(state, f"Marché fermé. Prochain open: {next_open}", "warn")
                save_state(state)
                time.sleep(CONFIG["check_interval_sec"])
                continue

            # ── 4. Récupération des infos compte ───────────────────────────────
            account = get_account()
            if account is None:
                log.warning("Impossible de récupérer le compte Alpaca — retry 60s")
                time.sleep(60)
                continue

            buying_power    = account["buying_power"]
            portfolio_value = account["portfolio_value"]

            state["capital"]      = round(portfolio_value, 2)
            state["buying_power"] = round(buying_power, 2)

            log.info(
                f"Marché OUVERT | Capital: ${portfolio_value:,.2f} | "
                f"BP disponible: ${buying_power:,.2f} | "
                f"Positions: {len(state['positions'])}"
            )

            # ── 5. Synchronisation des positions (détection clôtures SL/TP) ───
            sync_positions_from_alpaca(state)

            # ── 6. Analyse de chaque symbole de l'univers ─────────────────────
            for symbol in CONFIG["symbols"]:
                if symbol in state["positions"]:
                    price = get_latest_price(symbol)
                    if price:
                        state["latest_prices"][symbol] = price
                    pos      = state["positions"][symbol]
                    pnl_live = round((price or pos["entry_price"] - pos["entry_price"]) * pos["qty"], 2)
                    log.info(
                        f"[{symbol}] Position ouverte | "
                        f"Entrée: ${pos['entry_price']:.2f} | "
                        f"P&L live: {'+' if pnl_live>=0 else ''}{pnl_live:.2f}$"
                    )
                    continue

                analysis = analyze_symbol(symbol)
                state["latest_prices"][symbol] = analysis["price"] or 0

                log.info(
                    f"[{symbol}] Prix: ${analysis['price']} | "
                    f"SMA200: ${analysis['sma200']} | "
                    f"RSI: {analysis['rsi']} | "
                    f"Signal: {analysis['signal']}"
                )

                if analysis["signal"] != "BUY":
                    continue

                state["last_signal"] = "BUY"
                state["last_symbol"] = symbol

                price = analysis["price"]
                if not price or price <= 0:
                    log.warning(f"[{symbol}] Prix invalide, skip")
                    continue

                qty = calc_position_size(state["capital"], buying_power, price)
                if qty <= 0:
                    msg = f"[{symbol}] Sizing = 0 (BP insuffisant ou prix trop élevé)"
                    log.warning(msg)
                    push_log(state, msg, "warn")
                    continue

                cost = qty * price
                log.info(f"[{symbol}] BUY SIGNAL | Qty: {qty} | Coût: ${cost:,.2f}")

                order = place_bracket_order(symbol, qty, price)
                if order is None:
                    push_log(state, f"[{symbol}] Ordre refusé par Alpaca", "warn")
                    continue

                sl_price = round(price * (1 - CONFIG["stop_loss_pct"]), 2)
                tp_price = round(price * (1 + CONFIG["take_profit_pct"]), 2)
                state["positions"][symbol] = {
                    "entry_price": price,
                    "qty":         qty,
                    "stop_loss":   sl_price,
                    "take_profit": tp_price,
                    "entry_time":  datetime.now(timezone.utc).isoformat(),
                    "order_id":    order.get("id", ""),
                }
                state["in_position"] = True

                msg = (
                    f"🟢 ACHAT {symbol} | "
                    f"${price:.2f} × {qty} | "
                    f"SL: ${sl_price} | TP: ${tp_price} | "
                    f"Coût: ${cost:,.2f} | {analysis['reason']}"
                )
                log.info("=" * 60)
                log.info(msg)
                log.info("=" * 60)
                push_log(state, msg, "buy")
                tg(
                    f"🟢 <b>ACHAT {symbol}</b>\n"
                    f"Prix: <b>${price:.2f}</b> × {qty} actions\n"
                    f"SL: ${sl_price} | TP: ${tp_price}\n"
                    f"Coût: ${cost:,.2f}\n"
                    f"Signal: {analysis['reason']}"
                )

            # ── 7. Sauvegarde — le dashboard lit ce fichier ────────────────────
            save_state(state)
            log.info("state.json sauvegardé — cycle terminé")

        except KeyboardInterrupt:
            log.info("Arrêt manuel (KeyboardInterrupt)")
            save_state(state)
            break
        except Exception as e:
            log.error(f"Erreur boucle principale : {e}", exc_info=True)
            push_log(state, f"Erreur: {e}", "error")
            save_state(state)
            time.sleep(30)

        time.sleep(CONFIG["check_interval_sec"])


if __name__ == "__main__":
    run()