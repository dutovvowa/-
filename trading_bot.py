"""Простой торговый бот для Bybit с возможностью бэктеста и live-режима.

Основная логика идентична первоначальному скрипту: используется комбинация
EMA20/EMA50, RSI и ATR для постановки ордеров и риск-менеджмента. Код
структурирован по функциям, чтобы было проще читать, тестировать и изменять,
но без усложнения архитектуры.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import ccxt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange


# --------------------------------------------------------------------------------------
# Настройки по умолчанию
# --------------------------------------------------------------------------------------


DEFAULT_SYMBOLS: List[str] = [
    "SOL/USDT:USDT",
    "DOGE/USDT:USDT",
    "TRUMP/USDT:USDT",
    "POPCAT/USDT:USDT",
    "1000PEPE/USDT:USDT",
    "WIF/USDT:USDT",
]

DEFAULT_TIMEFRAME = "1h"
DEFAULT_LOOKBACK = 600


@dataclass
class Credentials:
    """Пара ключ/секрет для Bybit."""

    api_key: str
    api_secret: str

    @classmethod
    def from_env(cls) -> "Credentials":
        return cls(os.getenv("BYBIT_KEY", ""), os.getenv("BYBIT_SECRET", ""))

    def ensure_present(self) -> None:
        if not self.api_key or not self.api_secret:
            raise SystemExit("BYBIT_KEY и BYBIT_SECRET не заданы.")


@dataclass
class TelegramConfig:
    token: str
    chat_id: str

    @classmethod
    def from_env(cls) -> "TelegramConfig":
        return cls(os.getenv("TG_TOKEN", ""), os.getenv("TG_CHAT", ""))

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)


@dataclass
class StrategyConfig:
    symbols: List[str]
    timeframe: str
    lookback: int
    use_testnet: bool
    risk_pct: float
    atr_multiplier: float
    reward_to_risk: float
    trend_filter: bool
    volume_filter: bool
    daily_max_r_loss: float
    poll_seconds: int
    stop_file: Path


@dataclass
class Trade:
    side: str
    entry: float
    stop: float
    take: float
    quantity: float
    risk_usd: float


@dataclass
class BacktestStats:
    trades: int
    equity: float
    total_pnl: float
    win_rate: float
    average_pnl: float
    max_drawdown: float


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EMA/RSI бот для Bybit")
    parser.add_argument("--mode", choices=("backtest", "live"), default="backtest")
    parser.add_argument("--symbols", nargs="*", default=DEFAULT_SYMBOLS, help="Список тикеров")
    parser.add_argument("--timeframe", default=DEFAULT_TIMEFRAME)
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    parser.add_argument("--mainnet", action="store_true", help="Использовать боевой аккаунт")
    parser.add_argument("--risk-pct", type=float, default=0.001, help="Доля риска на сделку")
    parser.add_argument("--atr-mult", type=float, default=1.5, help="ATR множитель для стопа")
    parser.add_argument("--rr", type=float, default=2.0, help="Целевое соотношение TP/SL")
    parser.add_argument("--no-trend-filter", action="store_true", help="Отключить фильтр EMA200")
    parser.add_argument("--no-volume-filter", action="store_true", help="Отключить фильтр объёма")
    parser.add_argument("--daily-max-r", type=float, default=3.0, help="Стоп-день по R")
    parser.add_argument("--poll", type=int, default=30, help="Пауза между циклами в секундах")
    parser.add_argument("--stop-file", default="stop.txt", help="Файл для ручной остановки")
    parser.add_argument("--start-equity", type=float, default=10_000.0, help="Начальный капитал в бэктесте")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def make_strategy_config(args: argparse.Namespace) -> StrategyConfig:
    return StrategyConfig(
        symbols=list(args.symbols),
        timeframe=args.timeframe,
        lookback=args.lookback,
        use_testnet=not args.mainnet,
        risk_pct=args.risk_pct,
        atr_multiplier=args.atr_mult,
        reward_to_risk=args.rr,
        trend_filter=not args.no_trend_filter,
        volume_filter=not args.no_volume_filter,
        daily_max_r_loss=args.daily_max_r,
        poll_seconds=args.poll,
        stop_file=Path(args.stop_file),
    )


# --------------------------------------------------------------------------------------
# Помощники: Telegram, сеть, индикаторы
# --------------------------------------------------------------------------------------


class TelegramNotifier:
    """Отправка уведомлений в Telegram. Ошибки не прерывают работу."""

    def __init__(self, config: TelegramConfig) -> None:
        self._config = config
        self._session = requests.Session() if config.enabled else None

    def send(self, message: str) -> None:
        if not self._session or not self._config.enabled:
            return
        try:
            self._session.post(
                f"https://api.telegram.org/bot{self._config.token}/sendMessage",
                data={"chat_id": self._config.chat_id, "text": message},
                timeout=10,
            )
        except Exception as exc:  # noqa: BLE001 - логируем и идём дальше
            logging.debug("Telegram send failed: %s", exc)


def create_exchange(credentials: Credentials, settings: StrategyConfig) -> ccxt.bybit:
    """Создание клиента Bybit с переключением тестнета и лимитом запросов."""

    exchange = ccxt.bybit(
        {
            "apiKey": credentials.api_key,
            "secret": credentials.api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        }
    )
    if settings.use_testnet:
        exchange.set_sandbox_mode(True)
    return exchange


def call_with_retries(
    fn,
    *args,
    retries: int = 5,
    base_delay: float = 3.0,
    notifier: Optional[TelegramNotifier] = None,
    **kwargs,
):
    """Повторяет вызов функции ccxt с увеличивающейся задержкой."""

    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - ccxt кидает свои исключения
            delay = base_delay * attempt
            logging.warning("[NET] %s | попытка %d/%d", exc, attempt, retries)
            time.sleep(delay)
    if notifier:
        notifier.send("♻️ Переподключение к Bybit: превышено число попыток")
    raise RuntimeError("Не удалось выполнить запрос к Bybit")


def load_ohlcv(exchange: ccxt.bybit, symbol: str, settings: StrategyConfig) -> pd.DataFrame:
    data = call_with_retries(
        exchange.fetch_ohlcv,
        symbol,
        timeframe=settings.timeframe,
        limit=settings.lookback,
    )
    if not data:
        raise RuntimeError(f"[{symbol}] Нет данных OHLCV")
    df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df["ema20"] = EMAIndicator(df["close"], 20).ema_indicator()
    df["ema50"] = EMAIndicator(df["close"], 50).ema_indicator()
    df["ema200"] = EMAIndicator(df["close"], 200).ema_indicator()
    df["rsi"] = RSIIndicator(df["close"], 14).rsi()
    atr = AverageTrueRange(df["high"], df["low"], df["close"], 14)
    df["atr"] = atr.average_true_range()
    df["vol_med"] = df["volume"].rolling(20).median()
    df["vol_ok"] = df["volume"] > df["vol_med"]
    return df


def build_signal(row: pd.Series, settings: StrategyConfig) -> str:
    long_ok = row.ema20 > row.ema50 and row.rsi > 55 and row.close > row.ema20
    short_ok = row.ema20 < row.ema50 and row.rsi < 45 and row.close < row.ema20
    if settings.trend_filter:
        if long_ok and row.close < row.ema200:
            long_ok = False
        if short_ok and row.close > row.ema200:
            short_ok = False
    if settings.volume_filter and not bool(row.vol_ok):
        return "FLAT"
    if long_ok and not short_ok:
        return "LONG"
    if short_ok and not long_ok:
        return "SHORT"
    return "FLAT"


def get_equity(exchange: ccxt.bybit) -> float:
    balance = call_with_retries(exchange.fetch_balance)
    if not balance:
        return 0.0
    total = balance.get("USDT", {}).get("total")
    if total is None:
        total = balance.get("total", {}).get("USDT", 0.0)
    return float(total or 0.0)


def calc_position_size(balance_usdt: float, risk_pct: float, entry: float, stop: float) -> Tuple[float, float]:
    risk_usd = max(0.0, balance_usdt * risk_pct)
    dist = abs(entry - stop)
    if dist <= 0:
        return 0.0, 0.0
    qty = risk_usd / dist
    return qty, risk_usd


def get_position_side(exchange: ccxt.bybit, symbol: str) -> str:
    try:
        positions = call_with_retries(exchange.fetch_positions, [symbol])
    except Exception:  # noqa: BLE001
        return "FLAT"
    if not positions:
        return "FLAT"
    for pos in positions:
        amt = float(pos.get("contracts") or 0)
        side = str(pos.get("side") or "flat").upper()
        if amt and side != "FLAT":
            return side
    return "FLAT"


def place_market_order(
    exchange: ccxt.bybit,
    notifier: TelegramNotifier,
    symbol: str,
    side: str,
    qty: float,
    stop: float,
    take: float,
) -> bool:
    try:
        order = call_with_retries(
            exchange.create_order,
            symbol,
            "market",
            side.lower(),
            qty,
            params={
                "timeInForce": "GTC",
                "reduceOnly": False,
                "stopLoss": stop,
                "takeProfit": take,
            },
        )
        if order is None:
            raise RuntimeError("create_order вернул None")
        logging.info("[LIVE] %s %s qty=%.6f sl=%.4f tp=%.4f", symbol, side, qty, stop, take)
        notifier.send(
            f"{symbol}\n{side} qty={qty:.6f}\nSL={stop:.4f}\nTP={take:.4f}"
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logging.error("[%s] Ошибка ордера: %s", symbol, exc)
        notifier.send(f"⚠️ {symbol} ошибка ордера: {exc}")
        return False


# --------------------------------------------------------------------------------------
# Бэктест
# --------------------------------------------------------------------------------------


def run_backtest(
    exchange: ccxt.bybit,
    symbol: str,
    settings: StrategyConfig,
    start_equity: float,
) -> Tuple[List[Dict[str, float]], BacktestStats]:
    df = load_ohlcv(exchange, symbol, settings)
    equity = start_equity
    trade_log: List[Dict[str, float]] = []
    position: Optional[Trade] = None

    for idx in range(200, len(df)):
        prev, current = df.iloc[idx - 1], df.iloc[idx]

        if position:
            if position.side == "LONG":
                hit_tp = current["high"] >= position.take
                hit_sl = current["low"] <= position.stop
            else:
                hit_tp = current["low"] <= position.take
                hit_sl = current["high"] >= position.stop
            exit_price: Optional[float] = None
            if hit_tp and hit_sl:
                exit_price = position.stop if abs(position.stop - position.entry) < abs(position.take - position.entry) else position.take
            elif hit_tp:
                exit_price = position.take
            elif hit_sl:
                exit_price = position.stop
            if exit_price is not None:
                if position.side == "LONG":
                    pnl = (exit_price - position.entry) * position.quantity
                else:
                    pnl = (position.entry - exit_price) * position.quantity
                equity += pnl
                trade_log.append({
                    "exit": float(exit_price),
                    "pnl": float(pnl),
                    "equity": float(equity),
                })
                position = None

        if position is None:
            signal = build_signal(prev, settings)
            if signal not in ("LONG", "SHORT"):
                continue
            entry = float(current["open"])
            atr_value = float(prev["atr"])
            if atr_value <= 0:
                continue
            if signal == "LONG":
                stop = entry - settings.atr_multiplier * atr_value
                take = entry + settings.reward_to_risk * (entry - stop)
            else:
                stop = entry + settings.atr_multiplier * atr_value
                take = entry - settings.reward_to_risk * (stop - entry)
            qty, risk_usd = calc_position_size(equity, settings.risk_pct, entry, stop)
            if qty <= 0:
                continue
            position = Trade(signal, entry, stop, take, qty, risk_usd)

    stats = summarise_backtest(trade_log, start_equity)
    return trade_log, stats


def summarise_backtest(trades: List[Dict[str, float]], start_equity: float) -> BacktestStats:
    if not trades:
        return BacktestStats(0, start_equity, 0.0, 0.0, 0.0, 0.0)
    equity_curve = start_equity + np.cumsum([t["pnl"] for t in trades])
    max_equity = np.maximum.accumulate(equity_curve)
    drawdown = equity_curve - max_equity
    wins = [t for t in trades if t["pnl"] > 0]
    return BacktestStats(
        trades=len(trades),
        equity=float(equity_curve[-1]),
        total_pnl=float(equity_curve[-1] - start_equity),
        win_rate=float(len(wins) / len(trades) * 100.0),
        average_pnl=float(np.mean([t["pnl"] for t in trades])),
        max_drawdown=float(drawdown.min()),
    )


def export_trades_csv(trades: List[Dict[str, float]], path: Path) -> None:
    if not trades:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["exit", "pnl", "equity"])
        writer.writeheader()
        writer.writerows(trades)


def plot_equity_curve(trades: List[Dict[str, float]], start_equity: float, path: Path) -> None:
    if not trades:
        return
    equity_curve = start_equity + np.cumsum([t["pnl"] for t in trades])
    peaks = np.maximum.accumulate(equity_curve)
    drawdown = equity_curve - peaks
    plt.figure(figsize=(10, 6))
    plt.plot(equity_curve, label="Equity", linewidth=1.8)
    plt.fill_between(range(len(drawdown)), equity_curve, peaks, where=(drawdown < 0), color="red", alpha=0.3)
    plt.title("Equity curve и просадки")
    plt.xlabel("Номер сделки")
    plt.ylabel("Баланс (USDT)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# --------------------------------------------------------------------------------------
# Live-цикл
# --------------------------------------------------------------------------------------


def live_loop(exchange: ccxt.bybit, notifier: TelegramNotifier, settings: StrategyConfig) -> None:
    notifier.send(
        "🤖 Бот запущен. Режим: mainnet" if not settings.use_testnet else "🤖 Бот запущен. Режим: testnet"
    )
    markets = call_with_retries(exchange.load_markets)
    available = [s for s in settings.symbols if s in markets]
    missing = [s for s in settings.symbols if s not in markets]
    for symbol in missing:
        logging.warning("[WARN] %s недоступна на Bybit — пропуск.", symbol)
    if not available:
        raise SystemExit("Нет доступных пар из списка SYMBOLS.")

    daily_r = {symbol: 0.0 for symbol in available}
    current_day = date.today()

    while True:
        try:
            if settings.stop_file.exists():
                logging.info("stop.txt найден — остановка.")
                notifier.send("🛑 Бот остановлен через stop.txt")
                settings.stop_file.unlink(missing_ok=True)
                break

            if date.today() != current_day:
                daily_r = {symbol: 0.0 for symbol in available}
                current_day = date.today()

            for symbol in available:
                if daily_r[symbol] >= settings.daily_max_r_loss:
                    logging.info("[RISK] %s: дневной лимит достигнут, пропуск.", symbol)
                    continue

                df = load_ohlcv(exchange, symbol, settings)
                prev, current = df.iloc[-2], df.iloc[-1]

                if get_position_side(exchange, symbol) != "FLAT":
                    continue

                signal = build_signal(prev, settings)
                if signal not in ("LONG", "SHORT"):
                    continue

                entry = float(current["open"])
                atr_value = float(prev["atr"])
                if atr_value <= 0:
                    continue
                if signal == "LONG":
                    stop = entry - settings.atr_multiplier * atr_value
                    take = entry + settings.reward_to_risk * (entry - stop)
                    side = "BUY"
                else:
                    stop = entry + settings.atr_multiplier * atr_value
                    take = entry - settings.reward_to_risk * (stop - entry)
                    side = "SELL"

                equity = get_equity(exchange)
                qty, risk_usd = calc_position_size(equity, settings.risk_pct, entry, stop)
                if qty <= 0:
                    continue

                if place_market_order(exchange, notifier, symbol, side, qty, stop, take):
                    daily_r[symbol] += 1.0
                    logging.info(
                        "[RISK] %s: дневной счётчик %.1fR / %.1fR",
                        symbol,
                        daily_r[symbol],
                        settings.daily_max_r_loss,
                    )

            time.sleep(settings.poll_seconds)
        except KeyboardInterrupt:
            logging.info("Остановка пользователем.")
            break
        except Exception as exc:  # noqa: BLE001
            logging.error("Ошибка главного цикла: %s", exc, exc_info=True)
            time.sleep(settings.poll_seconds)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)
    settings = make_strategy_config(args)
    credentials = Credentials.from_env()
    notifier = TelegramNotifier(TelegramConfig.from_env())

    if args.mode == "live":
        credentials.ensure_present()

    exchange = create_exchange(credentials, settings)

    markets = call_with_retries(exchange.load_markets)
    available = [s for s in settings.symbols if s in markets]
    missing = [s for s in settings.symbols if s not in markets]
    for symbol in missing:
        logging.warning("[WARN] %s недоступна на Bybit — пропуск.", symbol)
    if not available:
        raise SystemExit("Нет доступных пар из списка SYMBOLS.")

    if args.mode == "backtest":
        for symbol in available:
            logging.info("=== Бэктест %s ===", symbol)
            trades, stats = run_backtest(exchange, symbol, settings, start_equity=args.start_equity)
            logging.info("Статистика: %s", stats)
            export_trades_csv(trades, Path(f"trades_{symbol.split('/')[0]}.csv"))
            plot_equity_curve(trades, args.start_equity, Path(f"equity_{symbol.split('/')[0]}.png"))
        return 0

    live_loop(exchange, notifier, settings)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except ccxt.BaseError as exc:
        logging.error("Ошибка CCXT: %s", exc)
        raise
