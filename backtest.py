# backtest.py
import numpy as np
import pandas as pd

class Backtest:
    def __init__(
        self,
        initial_cash=100000,
        commission=0.00125,
        slippage_bps=0.0,
        min_lot=0.0001,
        allow_shorts=True,
        exit_on_opposite=True,
        max_hold_bars=None,
        use_atr_stops=False,
        atr_window=14,
        atr_sl_mult=1.0,
        atr_tp_mult=2.0,
        persistence=1,
        cooldown_bars=0,
        # NUEVO: sizing
        size_mode="fraction",         # "fraction" (antes) | "risk"
        risk_per_trade=0.005,         # 0.5% del equity si size_mode="risk"
    ):
        self.initial_cash = float(initial_cash)
        self.commission = float(commission)
        self.slippage_bps = float(slippage_bps)
        self.min_lot = float(min_lot)

        self.allow_shorts = bool(allow_shorts)
        self.exit_on_opposite = bool(exit_on_opposite)
        self.max_hold_bars = int(max_hold_bars) if max_hold_bars else None

        self.use_atr_stops = bool(use_atr_stops)
        self.atr_window = int(atr_window)
        self.atr_sl_mult = float(atr_sl_mult)
        self.atr_tp_mult = float(atr_tp_mult)

        self.persistence = int(persistence)
        self.cooldown_bars = int(cooldown_bars)

        self.size_mode = str(size_mode)
        self.risk_per_trade = float(risk_per_trade)

        self.reset()

    def reset(self):
        self.cash = float(self.initial_cash)
        self.active_long = None
        self.active_short = None
        self.portfolio_hist = []
        self.trades = []
        self.closed_trades = []
        self.pending_order = None     # {'side', 'stop_loss', 'take_profit', 'n_shares'| 'risk_pct'}
        self.pending_exit = None
        self._cooldown = 0

    def _apply_slippage(self, price, side, is_entry=True):
        price = float(price)
        if self.slippage_bps <= 0:
            return price
        slip = price * (self.slippage_bps / 10000.0)
        if side == "long":
            return price + slip if is_entry else max(0.0, price - slip)
        else:
            return price - slip if is_entry else price + slip

    def _round_lot(self, qty):
        if qty <= 0:
            return 0.0
        steps = int(qty / self.min_lot)
        return round(steps * self.min_lot, 8)

    def _has_position(self):
        return (self.active_long is not None) or (self.active_short is not None)

    def _calc_short_unrealized(self, position, current_price):
        return (position["entry_price"] - current_price) * position["n_shares"]

    def calculate_portfolio_value(self, current_price):
        total = self.cash
        if self.active_long is not None:
            total += self.active_long["n_shares"] * current_price
        if self.active_short is not None:
            total += self.active_short["margin"]
            total += self._calc_short_unrealized(self.active_short, current_price)
        return float(total)

    def _stops_pct_from_atr(self, row, fallback_sl, fallback_tp):
        if not self.use_atr_stops:
            return float(fallback_sl), float(fallback_tp)

        col = f"atr_{self.atr_window}"
        if col in row.index:
            atr = float(row[col])
            ref = float(row["open"]) if row["open"] > 0 else float(row["close"])
            if atr > 0 and ref > 0:
                sl = (self.atr_sl_mult * atr) / ref
                tp = (self.atr_tp_mult * atr) / ref
                return float(sl), float(tp)

        return float(fallback_sl), float(fallback_tp)

    # --- aperturas/cierres ------------------------------------------------
    def open_long_position(self, row, n_shares, stop_loss, take_profit):
        if self._has_position() or n_shares <= 0:
            return False

        entry = self._apply_slippage(row["open"], side="long", is_entry=True)
        gross = entry * n_shares
        cost = gross * (1 + self.commission)

        if self.cash >= cost:
            self.cash -= cost
            self.active_long = {
                "type": "long",
                "entry_price": entry,
                "n_shares": n_shares,
                "stop_loss": float(stop_loss),
                "take_profit": float(take_profit),
                "entry_date": row["date"],
                "entry_index": row.name,
                "bars": 0,
            }
            self.trades.append({
                "date": row["date"], "type": "open_long",
                "price": entry, "n_shares": n_shares, "cost": cost
            })
            return True
        return False

    def open_short_position(self, row, n_shares, stop_loss, take_profit):
        if self._has_position() or n_shares <= 0 or not self.allow_shorts:
            return False

        entry = self._apply_slippage(row["open"], side="short", is_entry=True)
        notional = entry * n_shares
        margin_required = notional * (1 + self.commission)

        if self.cash >= margin_required:
            self.cash -= margin_required
            proceeds = notional * (1 - self.commission)
            self.active_short = {
                "type": "short",
                "entry_price": entry,
                "n_shares": n_shares,
                "stop_loss": float(stop_loss),
                "take_profit": float(take_profit),
                "entry_date": row["date"],
                "entry_index": row.name,
                "margin": margin_required,
                "proceeds": proceeds,
                "bars": 0,
            }
            self.trades.append({
                "date": row["date"], "type": "open_short",
                "price": entry, "n_shares": n_shares, "margin": margin_required
            })
            return True
        return False

    def close_long_position(self, row, reason, exit_price=None):
        pos = self.active_long
        if pos is None:
            return 0.0

        px = exit_price if exit_price is not None else self._apply_slippage(row["open"], "long", False)
        n = pos["n_shares"]
        revenue = n * px * (1 - self.commission)
        entry_cost = pos["entry_price"] * n * (1 + self.commission)
        pnl = revenue - entry_cost

        self.cash += revenue
        self.closed_trades.append({
            "type": "long", "entry_date": pos["entry_date"], "exit_date": row["date"],
            "entry_price": pos["entry_price"], "exit_price": px,
            "n_shares": n, "pnl": pnl, "return": pnl / entry_cost if entry_cost > 0 else 0.0,
            "reason": reason
        })
        self.trades.append({
            "date": row["date"], "type": "close_long",
            "price": px, "n_shares": n, "pnl": pnl, "reason": reason
        })
        self.active_long = None
        self._cooldown = self.cooldown_bars
        return pnl

    def close_short_position(self, row, reason, exit_price=None):
        pos = self.active_short
        if pos is None:
            return 0.0

        px = exit_price if exit_price is not None else self._apply_slippage(row["open"], "short", False)
        n = pos["n_shares"]
        buyback_cost = px * n * (1 + self.commission)
        pnl = pos["proceeds"] - buyback_cost

        self.cash += pos["margin"]
        self.cash += pnl
        self.closed_trades.append({
            "type": "short", "entry_date": pos["entry_date"], "exit_date": row["date"],
            "entry_price": pos["entry_price"], "exit_price": px,
            "n_shares": n, "pnl": pnl, "return": pnl / pos["margin"] if pos["margin"] > 0 else 0.0,
            "reason": reason
        })
        self.trades.append({
            "date": row["date"], "type": "close_short",
            "price": px, "n_shares": n, "pnl": pnl, "reason": reason
        })
        self.active_short = None
        self._cooldown = self.cooldown_bars
        return pnl

    def _check_long_exit_intrabar(self, pos, row):
        ep = pos["entry_price"]
        stop = ep * (1 - pos["stop_loss"])
        tp = ep * (1 + pos["take_profit"])
        if row["low"] <= stop:
            return True, "stop_loss", stop
        if row["high"] >= tp:
            return True, "take_profit", tp
        return False, None, None

    def _check_short_exit_intrabar(self, pos, row):
        ep = pos["entry_price"]
        stop = ep * (1 + pos["stop_loss"])
        tp = ep * (1 - pos["take_profit"])
        if row["high"] >= stop:
            return True, "stop_loss", stop
        if row["low"] <= tp:
            return True, "take_profit", tp
        return False, None, None

    def run(self, data, n_shares=0.1, stop_loss=0.02, take_profit=0.03):
        self.reset()
        data = data.reset_index(drop=True).copy()

        # persistencia
        if "buy_signal" in data.columns and "sell_signal" in data.columns:
            if self.persistence > 1:
                data["buy_sig"] = data["buy_signal"].rolling(self.persistence).sum().ge(self.persistence).astype(int)
                data["sell_sig"] = data["sell_signal"].rolling(self.persistence).sum().ge(self.persistence).astype(int)
            else:
                data["buy_sig"] = data["buy_signal"].astype(int)
                data["sell_sig"] = data["sell_signal"].astype(int)
        else:
            data["buy_sig"] = 0
            data["sell_sig"] = 0

        for i, row in data.iterrows():
            # salidas programadas
            if self.pending_exit == "long" and self.active_long is not None:
                self.close_long_position(row, "opposite_signal")
                self.pending_exit = None
            elif self.pending_exit == "short" and self.active_short is not None:
                self.close_short_position(row, "opposite_signal")
                self.pending_exit = None

            # ejecutar orden pendiente (abrir ahora al OPEN)
            if self.pending_order is not None and not self._has_position():
                side = self.pending_order["side"]
                base_sl = self.pending_order["stop_loss"]
                base_tp = self.pending_order["take_profit"]

                # ATR → porcentajes
                sl_pct, tp_pct = self._stops_pct_from_atr(row, base_sl, base_tp)

                entry_preview = self._apply_slippage(row["open"], side=side, is_entry=True)

                # tamaño
                if self.size_mode == "risk":
                    risk_pct = float(self.pending_order["risk_pct"])
                    stop_dist = max(entry_preview * sl_pct, 1e-9)
                    raw_shares = (self.cash * risk_pct) / stop_dist
                else:
                    if n_shares < 1:
                        qty_cash = self.cash * n_shares
                        ref_price = max(row["close"], 1e-9)
                        raw_shares = qty_cash / ref_price
                    else:
                        raw_shares = float(n_shares)

                shares_to_trade = self._round_lot(raw_shares)

                if side == "long":
                    self.open_long_position(row, shares_to_trade, sl_pct, tp_pct)
                else:
                    self.open_short_position(row, shares_to_trade, sl_pct, tp_pct)

                self.pending_order = None

            # SL/TP intrabar
            if self.active_long is not None:
                close_now, reason, px = self._check_long_exit_intrabar(self.active_long, row)
                if close_now:
                    self.close_long_position(row, reason, exit_price=px)
            if self.active_short is not None:
                close_now, reason, px = self._check_short_exit_intrabar(self.active_short, row)
                if close_now:
                    self.close_short_position(row, reason, exit_price=px)

            # time-stop
            if self.active_long is not None:
                self.active_long["bars"] += 1
                if self.max_hold_bars and self.active_long["bars"] >= self.max_hold_bars:
                    self.close_long_position(row, "time_stop")
            if self.active_short is not None:
                self.active_short["bars"] += 1
                if self.max_hold_bars and self.active_short["bars"] >= self.max_hold_bars:
                    self.close_short_position(row, "time_stop")

            # programar órdenes para próxima barra
            if i < len(data) - 1:
                if self.exit_on_opposite:
                    if self.active_long is not None and data.loc[i, "sell_sig"] == 1:
                        self.pending_exit = "long"
                    if self.active_short is not None and data.loc[i, "buy_sig"] == 1:
                        self.pending_exit = "short"

                if self._cooldown == 0 and self.pending_order is None and not self._has_position():
                    buy  = data.loc[i, "buy_sig"] == 1
                    sell = data.loc[i, "sell_sig"] == 1

                    if buy or (sell and self.allow_shorts):
                        base_order = {
                            "stop_loss": float(stop_loss),
                            "take_profit": float(take_profit),
                        }
                        if self.size_mode == "risk":
                            base_order["risk_pct"] = self.risk_per_trade
                        else:
                            base_order["n_shares"] = None  # calculado arriba si fraccional

                        if buy:
                            self.pending_order = {"side": "long", **base_order}
                        elif sell and self.allow_shorts:
                            self.pending_order = {"side": "short", **base_order}

            # equity al cierre
            self.portfolio_hist.append(self.calculate_portfolio_value(row["close"]))

            if self._cooldown > 0:
                self._cooldown -= 1

        return self.portfolio_hist, self.closed_trades


def backtest(data, params=None):
    if params is None:
        params = {}

    bt = Backtest(
        initial_cash=params.get("initial_cash", 100000),
        commission=params.get("commission", 0.00125),
        slippage_bps=params.get("slippage_bps", 0.0),
        min_lot=params.get("min_lot", 0.0001),
        allow_shorts=params.get("allow_shorts", False),
        exit_on_opposite=params.get("exit_on_opposite", True),
        max_hold_bars=params.get("max_hold_bars", None),
        use_atr_stops=params.get("use_atr_stops", False),
        atr_window=params.get("atr_window", 14),
        atr_sl_mult=params.get("atr_sl_mult", 1.5),
        atr_tp_mult=params.get("atr_tp_mult", 3.0),
        persistence=params.get("persistence", 1),
        cooldown_bars=params.get("cooldown_bars", 0),
        size_mode=params.get("size_mode", "risk"),
        risk_per_trade=params.get("risk_per_trade", 0.005),
    )

    portfolio_hist, trades = bt.run(
        data,
        n_shares=params.get("n_shares", 0.1),
        stop_loss=params.get("stop_loss", 0.02),
        take_profit=params.get("take_profit", 0.03),
    )
    return portfolio_hist, trades