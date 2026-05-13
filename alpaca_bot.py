#!/usr/bin/env python3
"""
Alpaca Equities Trading Bot — Mean Reversion
Architecture miroir du ETH/USDT Bot : backend Python → state.json → dashboard HTML
Stratégie : Mean Reversion sur actions US (SMA200 + RSI oversold)
Paper Trading uniquement via l'API Alpaca
"""

import os
import time
import json
import logging
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
from dotenv import load_dotenv

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
    "lookback_days":      250,      # Bougies historiques à récupérer (> sma_period)

    # Gestion du risque
    "risk_pct_per_trade": 0.02,     # 2% du capital risqué par trade
    "take_profit_pct":    0.04,     # TP à +4% de l'entrée
    "stop_loss_pct":      0.02,     # SL à -2% de l'entrée (ratio R:R = 2:1)

    # Exécution
    "check_interval_sec": 900,      # Vérification toutes les 15 minutes

    # Fichiers
    "state_file": "state.json",
    "log_file":   "alpaca-trades.log",

    # Capital de départ (pour calcul P&L affiché, Alpaca gère le vrai solde)
    "capital_initial": 100000.0,    # Capital paper trading Alpaca par défaut

    # Clés API (chargées depuis .env)
    "alpaca_api_key":    os.getenv("ALPACA_API_KEY", ""),
    "alpaca_secret_key": os.getenv("ALPACA_SECRET_KEY", ""),
    "telegram_token":    os.getenv("TELEGRAM_TOKEN", ""),
    "telegram_chat_id":  os.getenv("TELEGRAM_CHAT_ID", ""),
}

# URLs Alpaca Paper Trading
ALPACA_BASE_URL    = "https://paper-api.alpaca.markets"
ALPACA_DATA_URL    = "https://data.alpaca.markets"

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

# ─── HEADERS HTTP ALPACA ──────────────────────────────────────────────────────

def alpaca_headers() -> dict:
    """Construit les headers d'authentification pour l'API Alpaca."""
    return {
        "APCA-API-KEY-ID":     CONFIG["alpaca_api_key"],
        "APCA-API-SECRET-KEY": CONFIG["alpaca_secret_key"],
        "Content-Type":        "application/json",
    }

# ─── ÉTAT PERSISTANT (PONT JSON VERS LE DASHBOARD) ───────────────────────────

state_path = Path(__file__).parent / CONFIG["state_file"]

def load_state() -> dict:
    """
    Charge l'état depuis le fichier JSON.
    Si le fichier n'existe pas, initialise un état vide.
    """
    if state_path.exists():
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                s = json.load(f)
            # Garantit la présence des champs ajoutés en cours de vie
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
        "in_position":     False,       # True si au moins une position ouverte
        "positions":       {},          # {symbol: {entry, qty, sl, tp, entry_time}}
        "total_trades":    0,
        "winning_trades":  0,
        "total_pnl":       0.0,
        "day_pnl":         0.0,
        "last_reset_date": str(datetime.now(timezone.utc).date()),
        "last_signal":     "HOLD",
        "last_symbol":     None,
        "latest_prices":   {},          # {symbol: price}
        "regime":          "OPEN",      # OPEN / CLOSED / PRE_MARKET
        "started_at":      datetime.now(timezone.utc).isoformat(),
        "trade_history":   [],          # 10 derniers trades clôturés
        "logs":            [],          # 50 dernières lignes de log pour le dashboard
    }

def save_state(s: dict):
    """Sauvegarde atomique du state.json (écriture dans un fichier tmp puis rename)."""
    tmp_path = state_path.with_suffix(".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(s, f, indent=2, ensure_ascii=False)
        tmp_path.replace(state_path)
    except Exception as e:
        log.error(f"Impossible de sauvegarder state.json : {e}")

def push_log(state: dict, message: str, level: str = "info"):
    """
    Ajoute une ligne de log horodatée dans le state.json pour le dashboard.
    Conserve uniquement les 50 dernières entrées.
    """
    entry = {
        "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
        "level": level,
        "msg": message,
    }
    state["logs"].append(entry)
    state["logs"] = state["logs"][-50:]

# ─── TELEGRAM ─────────────────────────────────────────────────────────────────

def tg(msg: str):
    """Envoie une notification Telegram. Échoue silencieusement si non configuré."""
    token = CONFIG["telegram_token"]
    chat  = CONFIG["telegram_chat_id"]
    if not token or not chat:
        return
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        requests.post(
            url,
            json={"chat_id": chat, "text": msg, "parse_mode": "HTML"},
            timeout=5,
        )
    except Exception as e:
        log.warning(f"Telegram : {e}")

# ─── API ALPACA — MARCHÉ ──────────────────────────────────────────────────────

def get_clock() -> dict | None:
    """
    Interroge le Market Clock d'Alpaca.
    Retourne {'is_open': bool, 'next_open': str, 'next_close': str} ou None.
    """
    try:
        r = requests.get(
            f"{ALPACA_BASE_URL}/v2/clock",
            headers=alpaca_headers(),
            timeout=10,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.error(f"get_clock() : {e}")
        return None

def get_account() -> dict | None:
    """
    Récupère les informations du compte Alpaca (capital, buying power, etc.).
    """
    try:
        r = requests.get(
            f"{ALPACA_BASE_URL}/v2/account",
            headers=alpaca_headers(),
            timeout=10,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.error(f"get_account() : {e}")
        return None

def get_positions_alpaca() -> list:
    """
    Récupère les positions ouvertes depuis Alpaca.
    Retourne une liste de dicts (une entrée par symbole).
    """
    try:
        r = requests.get(
            f"{ALPACA_BASE_URL}/v2/positions",
            headers=alpaca_headers(),
            timeout=10,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.error(f"get_positions_alpaca() : {e}")
        return []

# ─── API ALPACA — DONNÉES HISTORIQUES ─────────────────────────────────────────

def get_bars(symbol: str, timeframe: str = "1Day", limit: int = 250) -> pd.DataFrame:
    """
    Récupère les bougies OHLCV via l'API Market Data d'Alpaca.
    timeframe : "1Day" | "1Hour" | "15Min"
    """
    try:
        r = requests.get(
            f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars",
            headers=alpaca_headers(),
            params={
                "timeframe": timeframe,
                "limit":     limit,
                "feed":      "iex",     # IEX = gratuit, SIP = données complètes (abonnement)
                "adjustment":"split",   # Ajustement splits
            },
            timeout=15,
        )
        r.raise_for_status()
        data = r.json().get("bars", [])
        if not data:
            log.warning(f"Aucune donnée reçue pour {symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df.rename(columns={"t": "time", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}, inplace=True)
        df["time"] = pd.to_datetime(df["time"])
        df = df[["time", "open", "high", "low", "close", "volume"]].copy()
        df.sort_values("time", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    except Exception as e:
        log.error(f"get_bars({symbol}) : {e}")
        return pd.DataFrame()

def get_latest_price(symbol: str) -> float | None:
    """Récupère le dernier prix coté pour un symbole."""
    try:
        r = requests.get(
            f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/trades/latest",
            headers=alpaca_headers(),
            params={"feed": "iex"},
            timeout=10,
        )
        r.raise_for_status()
        return float(r.json()["trade"]["p"])
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

    Retourne un dict avec signal, rsi, sma200, price, et un message de raison.
    """
    result = {
        "symbol":  symbol,
        "signal":  "HOLD",
        "reason":  "",
        "rsi":     50.0,
        "sma200":  None,
        "price":   None,
    }

    df = get_bars(symbol, timeframe="1Day", limit=CONFIG["lookback_days"])
    if df.empty or len(df) < CONFIG["sma_period"] + 5:
        result["reason"] = f"Données insuffisantes ({len(df)} bougies)"
        return result

    close   = df["close"]
    sma200  = calc_sma(close, CONFIG["sma_period"])
    rsi     = calc_rsi(close, CONFIG["rsi_period"])
    price   = float(close.iloc[-1])
    sma_val = float(sma200.iloc[-1])

    result["rsi"]   = rsi
    result["sma200"] = round(sma_val, 2)
    result["price"] = round(price, 2)

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
    risk_amount  = capital * CONFIG["risk_pct_per_trade"]        # ex: 2000$ sur 100k
    sl_distance  = price * CONFIG["stop_loss_pct"]               # ex: 3$ sur une action à 150$
    # Nombre d'actions pour risquer exactement risk_amount
    qty_risk     = int(risk_amount / max(sl_distance, 0.01))

    # Vérification du buying power (avec marge de sécurité de 5%)
    cost         = qty_risk * price
    if cost > buying_power * 0.95:
        qty_bp   = int((buying_power * 0.95) / price)
        log.warning(
            f"Buying power limité : {qty_risk} actions → {qty_bp} "
            f"(BP disponible: ${buying_power:,.2f})"
        )
        qty_risk = qty_bp

    return max(qty_risk, 0)

# ─── ORDRES ALPACA — BRACKET ORDER ────────────────────────────────────────────

def place_bracket_order(symbol: str, qty: int, entry_price: float) -> dict | None:
    """
    Soumet un Bracket Order (buy market + stop loss + take profit) en une seule requête.
    Le Bracket Order garantit que SL et TP sont actifs dès l'exécution du parent.
    Retourne le dict de réponse Alpaca ou None en cas d'erreur.
    """
    sl_price = round(entry_price * (1 - CONFIG["stop_loss_pct"]), 2)
    tp_price = round(entry_price * (1 + CONFIG["take_profit_pct"]), 2)

    order_payload = {
        "symbol":        symbol,
        "qty":           str(qty),
        "side":          "buy",
        "type":          "market",
        "time_in_force": "day",
        "order_class":   "bracket",
        "stop_loss": {
            "stop_price": str(sl_price),
        },
        "take_profit": {
            "limit_price": str(tp_price),
        },
    }

    try:
        r = requests.post(
            f"{ALPACA_BASE_URL}/v2/orders",
            headers=alpaca_headers(),
            json=order_payload,
            timeout=15,
        )
        r.raise_for_status()
        order = r.json()
        log.info(
            f"✅ Bracket order soumis — {symbol} | "
            f"Qty: {qty} | SL: ${sl_price} | TP: ${tp_price} | "
            f"Order ID: {order.get('id', '?')}"
        )
        return order
    except requests.exceptions.HTTPError as e:
        body = e.response.text if e.response else "N/A"
        log.error(f"Bracket order refusé ({symbol}) : {e} — Body: {body}")
        return None
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

    # Détection des clôtures (position dans notre state mais plus dans Alpaca)
    closed_symbols = set(state["positions"].keys()) - alpaca_symbols
    for sym in closed_symbols:
        pos       = state["positions"][sym]
        entry     = pos["entry_price"]
        qty       = pos["qty"]
        tp_price  = pos["take_profit"]
        sl_price  = pos["stop_loss"]

        # Récupération du prix actuel pour estimer le type de sortie
        exit_price = get_latest_price(sym) or entry
        pnl        = round((exit_price - entry) * qty, 2)
        trade_type = "TP" if exit_price >= (entry * (1 + CONFIG["take_profit_pct"] * 0.9)) else "SL"

        state["capital"]   = round(state["capital"] + pnl, 2)
        state["total_pnl"] = round(state["total_pnl"] + pnl, 2)
        state["day_pnl"]   = round(state["day_pnl"] + pnl, 2)
        state["total_trades"] += 1
        if pnl > 0:
            state["winning_trades"] += 1

        # Ajout au trade_history (lu par le graphique du dashboard)
        state["trade_history"].append({
            "type":  trade_type,
            "symbol": sym,
            "pnl":   pnl,
            "entry": entry,
            "exit":  round(exit_price, 2),
            "qty":   qty,
            "sl":    sl_price,
            "tp":    tp_price,
            "date":  datetime.now(timezone.utc).isoformat(),
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

    # Mise à jour des prix courants des positions encore ouvertes
    for pos_data in alpaca_positions:
        sym = pos_data["symbol"]
        try:
            current_price = float(pos_data.get("current_price", 0))
            state["latest_prices"][sym] = current_price
        except (ValueError, TypeError):
            pass

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
    log.info("  Alpaca Equities Bot — Mean Reversion v1.0")
    log.info(f"  Univers : {', '.join(CONFIG['symbols'])}")
    log.info(f"  Paper Trading : {ALPACA_BASE_URL}")
    log.info("━" * 60)

    # Vérification des clés API au démarrage
    if not CONFIG["alpaca_api_key"] or not CONFIG["alpaca_secret_key"]:
        log.critical("ALPACA_API_KEY ou ALPACA_SECRET_KEY manquante dans .env — arrêt.")
        return

    state = load_state()
    save_state(state)
    tg("🚀 <b>Alpaca Bot démarré</b>\nStratégie: Mean Reversion (SMA200 + RSI)\nMode: Paper Trading")

    while True:
        try:
            # ── 1. Reset journalier ────────────────────────────────────────────
            reset_daily_if_needed(state)

            # ── 2. Vérification Market Clock ───────────────────────────────────
            clock = get_clock()
            if clock is None:
                log.warning("Impossible de récupérer le Market Clock — retry 60s")
                time.sleep(60)
                continue

            is_open     = clock.get("is_open", False)
            next_open   = clock.get("next_open", "N/A")
            next_close  = clock.get("next_close", "N/A")
            state["regime"] = "OPEN" if is_open else "CLOSED"

            if not is_open:
                log.info(f"Marché FERMÉ — prochaine ouverture : {next_open}")
                push_log(state, f"Marché fermé. Prochain open: {next_open}", "warn")
                save_state(state)
                time.sleep(CONFIG["check_interval_sec"])
                continue

            # ── 3. Récupération des infos compte (buying power) ────────────────
            account = get_account()
            if account is None:
                log.warning("Impossible de récupérer le compte Alpaca — retry 60s")
                time.sleep(60)
                continue

            buying_power   = float(account.get("buying_power", 0))
            portfolio_value = float(account.get("portfolio_value", state["capital"]))

            # Mise à jour du capital depuis Alpaca (source de vérité)
            state["capital"]       = round(portfolio_value, 2)
            state["buying_power"]  = round(buying_power, 2)

            log.info(
                f"Marché OUVERT | Capital: ${portfolio_value:,.2f} | "
                f"BP disponible: ${buying_power:,.2f} | "
                f"Positions: {len(state['positions'])}"
            )

            # ── 4. Synchronisation des positions (détection clôtures SL/TP) ───
            sync_positions_from_alpaca(state)

            # ── 5. Analyse de chaque symbole de l'univers ─────────────────────
            for symbol in CONFIG["symbols"]:
                # Skip si déjà en position sur ce symbole
                if symbol in state["positions"]:
                    price = get_latest_price(symbol)
                    if price:
                        state["latest_prices"][symbol] = price
                    pos = state["positions"][symbol]
                    pnl_live = round((price or pos["entry_price"] - pos["entry_price"]) * pos["qty"], 2)
                    log.info(
                        f"[{symbol}] Position ouverte | "
                        f"Entrée: ${pos['entry_price']:.2f} | "
                        f"P&L live: {'+' if pnl_live>=0 else ''}{pnl_live:.2f}$"
                    )
                    continue

                # Analyse technique
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

                # Signal BUY détecté
                state["last_signal"] = "BUY"
                state["last_symbol"] = symbol

                price = analysis["price"]
                if not price or price <= 0:
                    log.warning(f"[{symbol}] Prix invalide, skip")
                    continue

                # Calcul du sizing et vérification buying power
                qty = calc_position_size(state["capital"], buying_power, price)
                if qty <= 0:
                    msg = f"[{symbol}] Sizing = 0 (BP insuffisant ou prix trop élevé)"
                    log.warning(msg)
                    push_log(state, msg, "warn")
                    continue

                cost = qty * price
                log.info(f"[{symbol}] BUY SIGNAL | Qty: {qty} | Coût: ${cost:,.2f}")

                # Soumission du Bracket Order
                order = place_bracket_order(symbol, qty, price)
                if order is None:
                    push_log(state, f"[{symbol}] Ordre refusé par Alpaca", "warn")
                    continue

                # Enregistrement de la position dans notre state
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

            # ── 6. Sauvegarde — le dashboard lit ce fichier ────────────────────
            save_state(state)
            log.info(f"state.json sauvegardé — cycle terminé")

        except KeyboardInterrupt:
            log.info("Arrêt manuel (KeyboardInterrupt)")
            save_state(state)
            break
        except Exception as e:
            log.error(f"Erreur boucle principale : {e}", exc_info=True)
            push_log(state, f"Erreur: {e}", "error")
            save_state(state)
            time.sleep(30)     # Pause courte avant retry en cas d'erreur réseau

        time.sleep(CONFIG["check_interval_sec"])


if __name__ == "__main__":
    run()
