#!/usr/bin/env python3
"""
Alpaca Equities Trading Bot — Mean Reversion v4.0

Changelog v4 — Intégration IA (Ollama qwen2.5:7b) :
  ① Filtre Sentiment    : avant chaque achat, analyse les news via LLM local.
                          → Répond PANIQUE ou OK. Si PANIQUE, achat annulé.
  ② Macro-Régime        : une fois par jour, évalue le climat macro (news SPY).
                          → Répond INCERTAIN ou NORMAL. Si INCERTAIN, risk /= 2.
  ③ Reporting Narratif  : à la clôture du marché, génère un rapport CIO en prose
                          et l'envoie sur Discord à la place du message brut.

Philosophie fail-safe :
  Chaque appel Ollama est dans un try/except avec timeout strict.
  En cas d'échec (réseau, timeout, réponse inattendue), le bot trade normalement
  et log une erreur [IA] sans planter.

Architecture state.json : inchangée.
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

from alpaca.data.historical import StockHistoricalDataClient, NewsClient
from alpaca.data.requests   import StockBarsRequest, StockLatestTradeRequest, NewsRequest
from alpaca.data.timeframe  import TimeFrame

load_dotenv()

# ─── FUSEAU DE RÉFÉRENCE ──────────────────────────────────────────────────────
TZ = ZoneInfo("Europe/Paris")

def now() -> datetime:
    return datetime.now(TZ)

# ─── CONFIGURATION CENTRALE ───────────────────────────────────────────────────
CONFIG = {
    # Univers et stratégie
    # Top 100 des actions US (S&P 100 + Nasdaq Giants)
    "symbols": [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK.B", "LLY", "TSLA", "V",
        "JPM", "UNH", "XOM", "MA", "JNJ", "PG", "HD", "AVGO", "COST", "MRK",
        "ABBV", "CVX", "CRM", "AMD", "PEP", "BAC", "KO", "WMT", "TMO", "MCD",
        "CSCO", "INTC", "ABT", "INTU", "WFC", "CMCSA", "DHR", "NFLX", "ORCL", "ADBE",
        "DIS", "TXN", "VZ", "PM", "NEE", "QCOM", "PFE", "HON", "AMGN", "IBM",
        "UNP", "BA", "GE", "CAT", "GS", "MS", "SPGI", "LOW", "AXP", "RTX",
        "NOW", "BKNG", "ISRG", "BLK", "PLD", "MDT", "EL", "T", "SBUX", "SYK",
        "C", "TJX", "CB", "ZTS", "MO", "GILD", "CI", "FI", "BDX", "MMM",
        "SO", "MMC", "ADI", "CME", "D", "VRTX", "REGN", "ITW", "EOG", "NOC",
        "BSX", "HUM", "EW", "PNC", "ETN", "CSX", "KLAC", "WM", "F", "GM"
    ],
    "sma_period":         200,
    "rsi_period":         14,
    "rsi_oversold":       40,
    "lookback_days":      252,

    # Gestion du risque (modifiable dynamiquement par le module Macro-Régime)
    "risk_pct_per_trade": 0.02,
    "risk_pct_base":      0.02,   # Valeur de référence — jamais modifiée directement
    "take_profit_pct":    0.04,
    "stop_loss_pct":      0.02,

    # Exécution
    "check_interval_sec": 900,

    # Fichiers
    "state_file":         "alpaca-state.json",
    "market_data_file":   "market_data.json",
    "log_file":           "alpaca-trades.log",

    # Capital
    "capital_initial":    100.0,

    # Clés API Alpaca
    "alpaca_api_key":     os.getenv("ALPACA_API_KEY",      ""),
    "alpaca_secret_key":  os.getenv("ALPACA_SECRET_KEY",   ""),
    "discord_webhook_url":os.getenv("DISCORD_WEBHOOK_URL", ""),

    # ★ Ollama — LLM local
    "ollama_url":   "http://192.168.1.195:11434/api/generate",
    "ollama_model": "qwen2.5:7b",
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

# ─── CLIENTS ALPACA-PY ────────────────────────────────────────────────────────
trading_client = None
data_client    = None
news_client    = None   # ★ Client news pour les modules IA

def init_clients():
    global trading_client, data_client, news_client
    trading_client = TradingClient(
        api_key=CONFIG["alpaca_api_key"],
        secret_key=CONFIG["alpaca_secret_key"],
        paper=True,
    )
    data_client = StockHistoricalDataClient(
        api_key=CONFIG["alpaca_api_key"],
        secret_key=CONFIG["alpaca_secret_key"],
    )
    # NewsClient utilise les mêmes clés — pas de clé séparée nécessaire
    news_client = NewsClient(
        api_key=CONFIG["alpaca_api_key"],
        secret_key=CONFIG["alpaca_secret_key"],
    )
    log.info("Clients alpaca-py initialisés (TradingClient + DataClient + NewsClient)")

# ─── STATE JSON ───────────────────────────────────────────────────────────────
state_path       = Path(__file__).parent / CONFIG["state_file"]
market_data_path = Path(__file__).parent / CONFIG["market_data_file"]

def load_state() -> dict:
    if state_path.exists():
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                s = json.load(f)
            s.setdefault("trade_history",   [])
            s.setdefault("positions",       {})
            s.setdefault("logs",            [])
            s.setdefault("last_scan",       {})
            s.setdefault("macro_regime",    "NORMAL")    # ★ Régime macro du jour
            s.setdefault("macro_date",      None)        # ★ Date de la dernière analyse macro
            log.info(f"Reprise — capital:${s['capital']:.4f} | trades:{s['total_trades']} | régime:{s['macro_regime']}")
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
        "last_scan":        {},
        "macro_regime":     "NORMAL",   # ★ NORMAL ou INCERTAIN
        "macro_date":       None,       # ★ Date ISO de la dernière éval macro
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

# ─── MARKET DATA ──────────────────────────────────────────────────────────────
def fetch_and_save_market_data():
    log.info("Récupération market_data via alpaca-py...")
    result = {"updated_at": now().isoformat(), "symbols": {}}
    end    = now()
    start  = end - timedelta(days=365)

    for symbol in CONFIG["symbols"]:
        try:
            req  = StockBarsRequest(
                symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
                start=start, end=end, feed="iex", adjustment="split",
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
        log.info("market_data.json sauvegardé")
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

# ─── DISCORD ──────────────────────────────────────────────────────────────────
def discord_alert(msg: str, username: str = "Alpaca Bot"):
    webhook_url = CONFIG.get("discord_webhook_url", "")
    if not webhook_url:
        return
    clean = msg.replace("<b>", "**").replace("</b>", "**")
    try:
        requests.post(
            webhook_url,
            json={"content": clean, "username": username},
            timeout=5,
        )
    except Exception as e:
        log.warning(f"Discord : {e}")

# ─── API ALPACA ────────────────────────────────────────────────────────────────
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

# ─── FETCH NEWS (commun aux deux modules IA news) ─────────────────────────────
def fetch_headlines(symbol: str, limit: int = 5) -> list[str]:
    """
    Récupère les N derniers titres de presse pour un symbole via NewsClient.
    Retourne une liste de strings (headlines). Liste vide si erreur.
    """
    try:
        req  = NewsRequest(
            symbols=[symbol],
            limit=limit,
            exclude_contentless=True,
        )
        news = news_client.get_news(req)
        # news est un NewsSet ; itération directe sur les articles
        headlines = [article.headline for article in news.news if article.headline]
        return headlines[:limit]
    except Exception as e:
        log.error(f"fetch_headlines({symbol}) : {e}")
        return []

# ═══════════════════════════════════════════════════════════════════════════════
# ██  MODULE IA — COUCHE COMMUNE OLLAMA  ████████████████████████████████████████
# ═══════════════════════════════════════════════════════════════════════════════

def _call_ollama(prompt: str, timeout: int) -> str | None:
    """
    Appelle l'API Ollama (LLM local) avec le prompt fourni.
    Format de requête : {"model": "qwen2.5:7b", "prompt": "...", "stream": false}

    Paramètres :
        prompt  : texte du prompt complet
        timeout : secondes avant abandon (strict)

    Retourne :
        La réponse textuelle du modèle (stripped), ou None si échec.

    Fail-safe : toute exception est capturée → log erreur → retourne None.
    Le bot continue de trader normalement si None est retourné.
    """
    payload = {
        "model":  CONFIG["ollama_model"],
        "prompt": prompt,
        "stream": False,                 # Réponse complète en un seul JSON
    }
    try:
        resp = requests.post(
            CONFIG["ollama_url"],
            json=payload,
            timeout=timeout,             # Timeout strict passé en paramètre
        )
        resp.raise_for_status()
        data = resp.json()
        # Ollama retourne {"response": "...", "done": true, ...}
        text = data.get("response", "").strip()
        if not text:
            log.warning("[IA] Ollama a retourné une réponse vide")
            return None
        return text
    except requests.exceptions.Timeout:
        log.error(f"[IA] Ollama timeout ({timeout}s dépassé) — fail-safe : on ignore l'IA")
        return None
    except requests.exceptions.ConnectionError:
        log.error(f"[IA] Ollama injoignable ({CONFIG['ollama_url']}) — fail-safe : on ignore l'IA")
        return None
    except Exception as e:
        log.error(f"[IA] Erreur Ollama inattendue : {e} — fail-safe : on ignore l'IA")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# ██  MODULE IA ①  —  FILTRE SENTIMENT  (avant chaque achat)  ████████████████
# ═══════════════════════════════════════════════════════════════════════════════

def ia_filtre_sentiment(symbol: str, state: dict) -> bool:
    """
    Analyse le sentiment des dernières news sur `symbol` via Ollama.
    Appelé uniquement quand un signal BUY est détecté, avant de passer l'ordre.

    Retourne :
        True  → le sentiment est OK, achat autorisé
        False → PANIQUE détectée, achat bloqué

    Fail-safe : si Ollama ne répond pas ou répond de manière inattendue,
    on retourne True (on laisse trader normalement) pour ne pas bloquer le bot.
    """
    log.info(f"[IA①] Analyse sentiment pour {symbol}...")

    # 1. Récupération des headlines via Alpaca News
    headlines = fetch_headlines(symbol, limit=5)
    if not headlines:
        log.warning(f"[IA①] Aucune news disponible pour {symbol} — sentiment ignoré, achat autorisé")
        return True   # Fail-safe : pas de news → on ne bloque pas

    titres_str = "\n".join(f"- {h}" for h in headlines)
    log.info(f"[IA①] {len(headlines)} titres récupérés pour {symbol}")

    # 2. Construction du prompt
    prompt = (
        f"Tu es un analyste financier. "
        f"Voici les derniers titres sur l'action {symbol} :\n"
        f"{titres_str}\n\n"
        f"Le sentiment est-il catastrophique/panique, ou normal ? "
        f"Réponds UNIQUEMENT par le mot PANIQUE ou le mot OK. "
        f"Aucune autre phrase."
    )

    # 3. Appel Ollama — timeout court (10s) car réponse attendue en 1 mot
    reponse = _call_ollama(prompt, timeout=10)

    # 4. Interprétation avec fail-safe
    if reponse is None:
        # Ollama indisponible : fail-safe → on autorise l'achat
        push_log(state, f"[IA①] {symbol} — Ollama indisponible, sentiment ignoré", "warn")
        return True

    reponse_upper = reponse.upper()
    log.info(f"[IA①] Réponse Ollama pour {symbol} : '{reponse}'")

    if "PANIQUE" in reponse_upper:
        msg = f"[IA①] PANIQUE détectée sur {symbol} — achat annulé | News: {headlines[0][:80]}"
        log.warning(msg)
        push_log(state, msg, "warn")
        discord_alert(
            f"🚨 **[IA] PANIQUE détectée — {symbol}**\n"
            f"Achat annulé par le filtre sentiment.\n"
            f"News: _{headlines[0][:100]}_"
        )
        return False  # ← Achat bloqué

    if "OK" in reponse_upper:
        log.info(f"[IA①] {symbol} — Sentiment OK, achat autorisé")
        push_log(state, f"[IA①] {symbol} — Sentiment OK ✓", "info")
        return True   # ← Achat autorisé

    # Réponse inattendue (ni PANIQUE ni OK) : fail-safe → on autorise
    log.warning(
        f"[IA①] Réponse inattendue pour {symbol} : '{reponse}' "
        f"— fail-safe : achat autorisé"
    )
    push_log(state, f"[IA①] {symbol} — Réponse IA inattendue ('{reponse[:30]}'), achat autorisé", "warn")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# ██  MODULE IA ②  —  MACRO-RÉGIME  (une fois par jour)  ████████████████████
# ═══════════════════════════════════════════════════════════════════════════════

def ia_macro_regime(state: dict):
    """
    Analyse le climat macro-économique via les news du ticker SPY.
    Appelé une fois par jour au premier cycle du jour où le marché est ouvert.

    Effets sur CONFIG :
        - Régime INCERTAIN → risk_pct_per_trade = risk_pct_base / 2
        - Régime NORMAL    → risk_pct_per_trade = risk_pct_base (restauré)

    Le résultat est persisté dans state['macro_regime'] et state['macro_date'].

    Fail-safe : si Ollama échoue, le risque n'est PAS modifié.
    """
    today = str(now().date())

    # Ne s'exécute qu'une fois par jour
    if state.get("macro_date") == today:
        return

    log.info(f"[IA②] Analyse macro-régime du jour ({today})...")

    # 1. News macro via le ticker SPY (proxy du marché américain global)
    headlines = fetch_headlines("SPY", limit=10)
    if not headlines:
        log.warning("[IA②] Aucune news SPY disponible — régime macro inchangé")
        state["macro_date"] = today   # On marque pour ne pas retry toute la journée
        return

    titres_str = "\n".join(f"- {h}" for h in headlines)
    log.info(f"[IA②] {len(headlines)} titres macro récupérés")

    # 2. Prompt macro
    prompt = (
        f"Analyse ces titres économiques mondiaux :\n"
        f"{titres_str}\n\n"
        f"Le climat des marchés est-il très incertain "
        f"(guerre, krach, inflation galopante, panique) ou normal ? "
        f"Réponds UNIQUEMENT par INCERTAIN ou NORMAL."
    )

    # 3. Appel Ollama — timeout 15s (1 mot attendu)
    reponse = _call_ollama(prompt, timeout=15)

    # 4. Interprétation
    if reponse is None:
        log.warning("[IA②] Ollama indisponible — macro-régime inchangé")
        state["macro_date"] = today
        push_log(state, "[IA②] Ollama indisponible — risque non ajusté", "warn")
        return

    reponse_upper = reponse.upper()
    log.info(f"[IA②] Réponse Ollama macro : '{reponse}'")

    if "INCERTAIN" in reponse_upper:
        CONFIG["risk_pct_per_trade"] = round(CONFIG["risk_pct_base"] / 2, 4)
        state["macro_regime"] = "INCERTAIN"
        msg = (
            f"[IA②] Macro-régime INCERTAIN — risque réduit : "
            f"{CONFIG['risk_pct_base']*100:.1f}% → {CONFIG['risk_pct_per_trade']*100:.2f}%"
        )
        log.warning(msg)
        push_log(state, msg, "warn")
        discord_alert(
            f"⚠️ **[IA] Macro-régime INCERTAIN**\n"
            f"Risque/trade réduit de moitié : **{CONFIG['risk_pct_per_trade']*100:.2f}%**\n"
            f"News macro: _{headlines[0][:100]}_"
        )

    elif "NORMAL" in reponse_upper:
        CONFIG["risk_pct_per_trade"] = CONFIG["risk_pct_base"]
        state["macro_regime"] = "NORMAL"
        msg = f"[IA②] Macro-régime NORMAL — risque standard : {CONFIG['risk_pct_per_trade']*100:.1f}%"
        log.info(msg)
        push_log(state, msg, "info")

    else:
        # Réponse inattendue : fail-safe → on ne change rien
        log.warning(f"[IA②] Réponse inattendue : '{reponse}' — régime inchangé")
        push_log(state, f"[IA②] Réponse IA inattendue ('{reponse[:30]}') — risque inchangé", "warn")

    state["macro_date"] = today


# ═══════════════════════════════════════════════════════════════════════════════
# ██  MODULE IA ③  —  REPORTING NARRATIF  (à la clôture)  ████████████████████
# ═══════════════════════════════════════════════════════════════════════════════

def ia_rapport_narratif(state: dict) -> str | None:
    """
    Génère un rapport de fin de journée en prose via Ollama (rôle CIO).
    Appelé au moment où is_open passe de True à False.

    Retourne :
        Le texte du rapport généré, ou None si Ollama échoue.
        En cas d'échec, le rapport classique formaté est utilisé à la place.

    Timeout : 90 secondes (réponse longue ~50s mesurée sur votre infrastructure).
    """
    log.info("[IA③] Génération du rapport narratif de clôture...")

    # Construction du contexte métriques de la journée
    total        = state["total_trades"]
    winners      = state["winning_trades"]
    win_rate     = round(winners / total * 100, 1) if total > 0 else 0
    day_pnl      = state["day_pnl"]
    capital      = state["capital"]
    pos_count    = len(state["positions"])
    macro        = state.get("macro_regime", "NORMAL")
    risk_pct     = CONFIG["risk_pct_per_trade"] * 100

    # Résumé des trades du jour depuis trade_history
    trades_today = [
        t for t in state.get("trade_history", [])
        if t.get("date", "")[:10] == str(now().date())
    ]
    trades_str = ""
    if trades_today:
        trades_str = "\n".join(
            f"  - {t['symbol']} {t['type']}: P&L={'+' if t['pnl']>=0 else ''}{t['pnl']:.4f}$"
            for t in trades_today
        )
    else:
        trades_str = "  Aucun trade clôturé aujourd'hui."

    metriques = (
        f"Date: {now().strftime('%d/%m/%Y')}\n"
        f"Capital final: ${capital:.4f}\n"
        f"P&L du jour: {'+' if day_pnl>=0 else ''}{day_pnl:.4f}$\n"
        f"Trades clôturés: {len(trades_today)} (Win rate global: {win_rate}%)\n"
        f"Positions encore ouvertes: {pos_count}\n"
        f"Régime macro IA: {macro}\n"
        f"Risque/trade appliqué: {risk_pct:.2f}%\n"
        f"Détail des trades du jour:\n{trades_str}"
    )

    prompt = (
        f"Agis comme un Chief Investment Officer. "
        f"Voici les résultats bruts du bot aujourd'hui :\n\n"
        f"{metriques}\n\n"
        f"Rédige un rapport de fin de journée de 3 phrases maximum, "
        f"très professionnel et concis, en français. "
        f"Aucun formatage Markdown, aucun titre, texte pur uniquement."
    )

    # Appel Ollama — timeout long car réponse prose attendue (~50s)
    reponse = _call_ollama(prompt, timeout=90)

    if reponse is None:
        log.warning("[IA③] Ollama indisponible — rapport classique utilisé")
        return None

    log.info(f"[IA③] Rapport généré ({len(reponse)} caractères)")
    return reponse


# ═══════════════════════════════════════════════════════════════════════════════
# ██  RAPPORT DE CLÔTURE (orchestration IA③ + fallback)  ██████████████████████
# ═══════════════════════════════════════════════════════════════════════════════

def send_close_report(state: dict):
    """
    Envoi du rapport de clôture sur Discord.
    Tente d'abord le rapport narratif IA (Module ③).
    Fallback sur le message classique formaté si Ollama échoue.
    """
    day_pnl  = state["day_pnl"]
    capital  = state["capital"]
    pnl_str  = f"{'+' if day_pnl >= 0 else ''}{day_pnl:.4f}$"
    icon_pnl = "📈" if day_pnl >= 0 else "📉"
    date_str = now().strftime("%d/%m %H:%M")

    # ── Tentative rapport IA narratif ─────────────────────────────────────────
    rapport_ia = ia_rapport_narratif(state)

    if rapport_ia:
        # Le rapport IA réussit : on l'envoie tel quel avec un en-tête minimal
        msg = (
            f"🏁 **Marché fermé** | {date_str} (Paris) | {icon_pnl} {pnl_str}\n\n"
            f"_{rapport_ia}_"
        )
        push_log(state, f"Rapport IA clôture envoyé | P&L Jour: {pnl_str}", "info")
        log.info("Rapport IA de clôture envoyé sur Discord")
    else:
        # Fallback : message classique structuré
        msg = (
            f"🏁 **Marché fermé** | {date_str} (Paris)\n"
            f"Capital : **{capital:.4f}$** | P&L Jour : **{pnl_str}** {icon_pnl}\n"
            f"Positions ouvertes : {len(state['positions'])} | "
            f"Trades totaux : {state['total_trades']}"
        )
        push_log(state, f"Rapport classique clôture (IA indisponible) | P&L: {pnl_str}", "warn")
        log.info("Rapport de clôture classique envoyé (Ollama indisponible)")

    discord_alert(msg, username="Alpaca Bot — Clôture")


# ─── INDICATEURS TECHNIQUES ───────────────────────────────────────────────────
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

# ─── STRATÉGIE MEAN REVERSION ─────────────────────────────────────────────────
def analyze_symbol(symbol: str) -> dict:
    result = {
        "symbol": symbol, "signal": "HOLD",
        "reason": "", "rsi": 50.0, "sma200": None, "price": None,
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
    result.update({"rsi": rsi, "sma200": round(sma_val, 2), "price": round(price, 2)})
    if price > sma_val and rsi < CONFIG["rsi_oversold"]:
        result["signal"] = "BUY"
        result["reason"] = f"Prix>${sma_val:.2f} | RSI={rsi}<{CONFIG['rsi_oversold']}"
    elif price <= sma_val:
        result["reason"] = f"Prix SOUS SMA200 ({sma_val:.2f}) | RSI={rsi}"
    else:
        result["reason"] = f"RSI={rsi} non oversold | SMA200 OK"
    return result

# ─── SIZING FRACTIONNEL ───────────────────────────────────────────────────────
def calc_position_size(capital: float, buying_power: float, price: float) -> float:
    """
    Retourne un float (4 décimales) pour les fractions d'actions.
    Utilise CONFIG['risk_pct_per_trade'] qui peut être réduit par le Module IA ②.
    """
    risk_amount = capital * CONFIG["risk_pct_per_trade"]   # ← ajusté par IA②
    sl_distance = price   * CONFIG["stop_loss_pct"]
    qty_risk    = risk_amount / max(sl_distance, 0.0001)

    cost = qty_risk * price
    if cost > buying_power * 0.95:
        qty_bp = (buying_power * 0.95) / price
        log.warning(f"BP limité : {qty_risk:.4f} → {qty_bp:.4f} fractions (BP=${buying_power:.4f})")
        qty_risk = qty_bp

    qty_final = round(qty_risk, 4)
    return qty_final if qty_final >= 0.001 else 0.0

# ─── BRACKET ORDER (avec fallback market simple) ──────────────────────────────
def place_bracket_order(symbol: str, qty: float, entry_price: float):
    sl_price = round(entry_price * (1 - CONFIG["stop_loss_pct"]),  2)
    tp_price = round(entry_price * (1 + CONFIG["take_profit_pct"]), 2)

    # Tentative 1 : Bracket complet
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
            log.error(f"Bracket order — erreur non récupérable : {e}")
            return None

    # Tentative 2 : Market order simple (fallback)
    try:
        order = trading_client.submit_order(MarketOrderRequest(
            symbol=symbol, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
        ))
        log.warning(f"⚠️ Market simple (SANS bracket) | {symbol} | qty:{qty} | ID:{order.id}")
        return {"id": str(order.id), "status": str(order.status), "type": "market_only"}
    except Exception as e2:
        log.error(f"Fallback market refusé ({symbol}) : {e2}")
        return None

# ─── SYNC POSITIONS ───────────────────────────────────────────────────────────
def sync_positions_from_alpaca(state: dict):
    alpaca_positions = get_positions_alpaca()
    alpaca_symbols   = {p["symbol"] for p in alpaca_positions}

    for sym in set(state["positions"].keys()) - alpaca_symbols:
        pos        = state["positions"][sym]
        entry      = pos["entry_price"]
        qty        = pos["qty"]
        tp_price   = pos["take_profit"]
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
        discord_alert(
            f"{icon} **CLÔTURE {sym}**\n"
            f"${entry:.2f} → ${exit_price:.2f}\n"
            f"P&L: **{'+' if pnl>=0 else ''}{pnl:.4f}$** | {trade_type}\n"
            f"Capital: ${state['capital']:.4f}"
        )

    for p in alpaca_positions:
        if p["current_price"]:
            state["latest_prices"][p["symbol"]] = p["current_price"]

    state["in_position"] = len(state["positions"]) > 0

# ─── RESET JOURNALIER ─────────────────────────────────────────────────────────
def reset_daily_if_needed(state: dict):
    today = str(now().date())
    if state.get("last_reset_date") != today:
        state["day_pnl"]         = 0.0
        state["last_reset_date"] = today
        log.info(f"Nouveau jour Paris — reset P&L journalier ({today})")

# ─── HEARTBEAT [SCAN] ─────────────────────────────────────────────────────────
def log_scan_result(state: dict, analysis: dict):
    sym    = analysis["symbol"]
    price  = analysis["price"]  or "N/A"
    sma    = analysis["sma200"] or "N/A"
    rsi    = analysis["rsi"]
    signal = analysis["signal"]
    arrow  = "🟢 BUY!" if signal == "BUY" else "⚪"
    log.info(f"[SCAN] {sym}: ${price} | SMA200: ${sma} | RSI: {rsi} → {signal} {arrow}")
    state["last_scan"][sym] = {
        "price":      price,
        "sma200":     sma,
        "rsi":        rsi,
        "signal":     signal,
        "reason":     analysis.get("reason", ""),
        "scanned_at": now().strftime("%H:%M:%S"),
    }

# ═══════════════════════════════════════════════════════════════════════════════
# ██  BOUCLE PRINCIPALE  ███████████████████████████████████████████████████████
# ═══════════════════════════════════════════════════════════════════════════════

def run():
    log.info("━" * 60)
    log.info("  Alpaca Equities Bot — Mean Reversion v4.0 + IA Ollama")
    log.info(f"  Capital : ${CONFIG['capital_initial']:.2f} | Fuseau : Europe/Paris")
    log.info(f"  Univers : {', '.join(CONFIG['symbols'])}")
    log.info(f"  Ollama  : {CONFIG['ollama_url']} ({CONFIG['ollama_model']})")
    log.info("━" * 60)

    if not CONFIG["alpaca_api_key"] or not CONFIG["alpaca_secret_key"]:
        log.critical("ALPACA_API_KEY ou ALPACA_SECRET_KEY manquante — arrêt.")
        return

    init_clients()
    fetch_and_save_market_data()

    state = load_state()
    save_state(state)
    discord_alert(
        f"🚀 **Alpaca Bot v4.0 démarré** | {now().strftime('%d/%m %H:%M')} Paris\n"
        f"Capital: **${CONFIG['capital_initial']:.2f}** | IA Ollama activée\n"
        f"Modules : ① Sentiment | ② Macro-Régime | ③ Rapport Narratif"
    )

    was_open = False   # Détection transition OPEN → CLOSED

    while True:
        try:
            reset_daily_if_needed(state)

            if should_refresh_market_data():
                fetch_and_save_market_data()

            # ── Market Clock ───────────────────────────────────────────────────
            clock = get_clock()
            if clock is None:
                log.warning("Market Clock indisponible — retry 60s")
                time.sleep(60)
                continue

            is_open    = clock["is_open"]
            next_open  = clock["next_open"]
            next_close = clock["next_close"]
            state["regime"] = "OPEN" if is_open else "CLOSED"

            # ── Transition OPEN → CLOSED : rapport de clôture ─────────────────
            if was_open and not is_open:
                # Module IA ③ : rapport narratif CIO (avec fallback classique)
                send_close_report(state)
            was_open = is_open

            if not is_open:
                log.info(f"Marché FERMÉ — prochain open : {next_open}")
                push_log(state, f"Marché fermé. Prochain open: {next_open}", "warn")
                save_state(state)
                time.sleep(CONFIG["check_interval_sec"])
                continue

            # ── Compte Alpaca ──────────────────────────────────────────────────
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
                f"Marché OUVERT | Capital:${portfolio_value:.4f} | "
                f"BP:${buying_power:.4f} | Pos:{len(state['positions'])} | "
                f"Régime macro:{state.get('macro_regime','NORMAL')} | "
                f"Risk:{CONFIG['risk_pct_per_trade']*100:.2f}%"
            )
            push_log(
                state,
                f"Scan | BP:${buying_power:.4f} | Pos:{len(state['positions'])} | "
                f"Macro:{state.get('macro_regime','NORMAL')} | "
                f"Risk:{CONFIG['risk_pct_per_trade']*100:.2f}%",
                "info"
            )

            # ── Module IA ② : Macro-Régime (une fois par jour) ────────────────
            # Appelé ici, après confirmation que le marché est ouvert,
            # pour éviter de scraper des news la nuit sans utilité.
            ia_macro_regime(state)

            sync_positions_from_alpaca(state)

            # ── Analyse des symboles ───────────────────────────────────────────
# ── Analyse des symboles ───────────────────────────────────────────
            for symbol in CONFIG["symbols"]:

                # ── Position déjà ouverte : heartbeat + gestion SL/TP manuelle ──
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

                    # ── Clôture manuelle SL/TP pour les ordres market_only ─────
                    # Les Bracket Orders natifs sont gérés par Alpaca (sync_positions).
                    # Pour les market_only (fractions), on surveille le prix ici
                    # et on envoie nous-mêmes l'ordre de vente au déclenchement.
                    if pos.get("order_type") == "market_only" and price:
                        tp_hit = price >= pos["take_profit"]
                        sl_hit = price <= pos["stop_loss"]

                        if tp_hit or sl_hit:
                            trigger     = "TP" if tp_hit else "SL"
                            exit_price  = price

                            log.info(
                                f"[{symbol}] {trigger} déclenché manuellement | "
                                f"Prix:${exit_price:.2f} | "
                                f"SL:${pos['stop_loss']} | TP:${pos['take_profit']}"
                            )

                            # ── Ordre de vente au marché ──────────────────────
                            sell_ok = False
                            try:
                                sell_order = trading_client.submit_order(
                                    MarketOrderRequest(
                                        symbol=symbol,
                                        qty=pos["qty"],
                                        side=OrderSide.SELL,
                                        time_in_force=TimeInForce.DAY,
                                    )
                                )
                                log.info(
                                    f"✅ Ordre SELL soumis | {symbol} | "
                                    f"qty:{pos['qty']} | ID:{sell_order.id}"
                                )
                                sell_ok = True
                            except Exception as e:
                                log.error(
                                    f"❌ Échec ordre SELL {symbol} : {e} — "
                                    f"position conservée, retry au prochain cycle"
                                )

                            # ── Mise à jour du state uniquement si l'ordre est passé ──
                            if sell_ok:
                                pnl        = round((exit_price - pos["entry_price"]) * pos["qty"], 4)
                                icon       = "✅" if pnl > 0 else "❌"

                                state["capital"]      = round(state["capital"]   + pnl, 4)
                                state["total_pnl"]    = round(state["total_pnl"] + pnl, 4)
                                state["day_pnl"]      = round(state["day_pnl"]   + pnl, 4)
                                state["total_trades"] += 1
                                if pnl > 0:
                                    state["winning_trades"] += 1

                                state["trade_history"].append({
                                    "type":   trigger,
                                    "symbol": symbol,
                                    "pnl":    pnl,
                                    "entry":  pos["entry_price"],
                                    "exit":   round(exit_price, 2),
                                    "qty":    pos["qty"],
                                    "sl":     pos["stop_loss"],
                                    "tp":     pos["take_profit"],
                                    "date":   now().isoformat(),
                                })
                                state["trade_history"] = state["trade_history"][-10:]

                                del state["positions"][symbol]
                                state["in_position"] = len(state["positions"]) > 0

                                msg = (
                                    f"{icon} CLÔTURE {symbol} (market_only) | "
                                    f"${pos['entry_price']:.2f}→${exit_price:.2f} | "
                                    f"P&L:{'+' if pnl>=0 else ''}{pnl:.4f}$ | {trigger}"
                                )
                                log.info("=" * 60)
                                log.info(msg)
                                log.info("=" * 60)
                                push_log(state, msg, "sell")
                                discord_alert(
                                    f"{icon} **CLÔTURE {symbol}** _(market only)_\n"
                                    f"${pos['entry_price']:.2f} → ${exit_price:.2f} | **{trigger}**\n"
                                    f"P&L: **{'+' if pnl>=0 else ''}{pnl:.4f}$**\n"
                                    f"Capital: ${state['capital']:.4f}"
                                )

                    continue  # ← Toujours skip l'analyse technique si position ouverte

                # Analyse technique
                analysis = analyze_symbol(symbol)
                state["latest_prices"][symbol] = analysis["price"] or 0
                log_scan_result(state, analysis)

                if analysis["signal"] != "BUY":
                    continue

                # ── Signal BUY détecté ────────────────────────────────────────
                state["last_signal"] = "BUY"
                state["last_symbol"] = symbol
                price = analysis["price"]

                if not price or price <= 0:
                    log.warning(f"[{symbol}] Prix invalide, skip")
                    continue

                # ── Module IA ① : Filtre Sentiment ────────────────────────────
                if not ia_filtre_sentiment(symbol, state):
                    state["last_scan"][symbol]["signal"] = "IA_PANIQUE"
                    state["last_scan"][symbol]["reason"] = "Bloqué par filtre sentiment IA"
                    continue

                # Sizing fractionnel
                qty = calc_position_size(state["capital"], buying_power, price)
                if qty <= 0:
                    msg = f"[{symbol}] Sizing=0 (capital ${state['capital']:.4f} insuffisant)"
                    log.warning(msg)
                    push_log(state, msg, "warn")
                    continue

                cost = round(qty * price, 4)
                log.info(f"[{symbol}] BUY | qty:{qty} fractions | coût:${cost:.4f}")

                order = place_bracket_order(symbol, qty, price)
                if order is None:
                    push_log(state, f"[{symbol}] Ordre refusé par Alpaca", "warn")
                    continue

                sl_price   = round(price * (1 - CONFIG["stop_loss_pct"]),  2)
                tp_price   = round(price * (1 + CONFIG["take_profit_pct"]), 2)
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

                msg = (
                    f"🟢 ACHAT {symbol} | ${price:.2f} × {qty} fractions | "
                    f"SL:${sl_price} | TP:${tp_price} | ${cost:.4f} | {type_label}"
                )
                log.info("=" * 60)
                log.info(msg)
                log.info("=" * 60)
                push_log(state, msg, "buy")
                discord_alert(
                    f"🟢 **ACHAT {symbol}**\n"
                    f"${price:.2f} × **{qty} fractions** | {type_label}\n"
                    f"SL:${sl_price} | TP:${tp_price} | Coût:${cost:.4f}\n"
                    f"Signal: {analysis['reason']}\n"
                    f"Sentiment IA: ✅ OK | Macro: {state.get('macro_regime','NORMAL')}"
                )

            # ── Sauvegarde state.json ─────────────────────────────────────────
            save_state(state)
            log.info("state.json sauvegardé — fin de cycle")

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