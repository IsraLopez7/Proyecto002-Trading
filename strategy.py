"""
Módulo de Estrategia de Trading
Implementa la lógica de trading con gestión de posiciones y riesgo
"""

import pandas as pd
import numpy as np

class TradingStrategy:
    def __init__(self, initial_capital=100000, commission=0.00125):
        self.initial_capital = initial_capital
        self.commission = commission
        self.position_size = 0.95  # Usar 95% del capital
        self.stop_loss_atr = 1.5
        self.take_profit_atr = 2.5
        
    def backtest(self, df):
        """Ejecuta el backtest"""
        # Variables iniciales
        capital = self.initial_capital
        position = 0  # 0: sin posición, 1: long, -1: short
        entry_price = 0
        entry_date = None
        trades = []
        equity = [capital]
        
        btc_held = 0
        capital_available = capital
        
        for i in range(1, len(df)):
            row = df.iloc[i]
            current_price = row['Close']
            signal = row['signal'] if 'signal' in row else 0
            date = row['Date']
            atr = row['ATR'] if 'ATR' in row else current_price * 0.02
            
            # Calcular equity actual
            if position == 1:  # Long
                current_equity = capital_available + (btc_held * current_price)
            elif position == -1:  # Short
                current_equity = capital_available + (btc_held * (2 * entry_price - current_price))
            else:
                current_equity = capital_available
            
            # Gestión de posiciones existentes
            if position != 0:
                # Calcular niveles de salida
                if position == 1:  # Long
                    stop_loss = entry_price - (atr * self.stop_loss_atr)
                    take_profit = entry_price + (atr * self.take_profit_atr)
                    should_close = (signal == -1) or (current_price <= stop_loss) or (current_price >= take_profit)
                else:  # Short
                    stop_loss = entry_price + (atr * self.stop_loss_atr)
                    take_profit = entry_price - (atr * self.take_profit_atr)
                    should_close = (signal == 1) or (current_price >= stop_loss) or (current_price <= take_profit)
                
                if should_close:
                    # Cerrar posición
                    if position == 1:
                        capital_available = btc_held * current_price * (1 - self.commission)
                        pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    else:
                        capital_available = capital_available + (btc_held * (entry_price - current_price))
                        capital_available = capital_available * (1 - self.commission)
                        pnl_pct = ((entry_price - current_price) / entry_price) * 100
                    
                    # Registrar trade
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'type': 'LONG' if position == 1 else 'SHORT',
                        'pnl_pct': pnl_pct,
                        'pnl_usd': capital_available - capital
                    })
                    
                    # Reset
                    position = 0
                    btc_held = 0
                    capital = capital_available
            
            # Abrir nueva posición
            if position == 0 and signal != 0:
                position = signal
                entry_price = current_price
                entry_date = date
                
                if signal == 1:  # Abrir long
                    btc_held = (capital_available * self.position_size) / current_price
                    capital_available = capital_available * (1 - self.position_size) * (1 - self.commission)
                else:  # Abrir short
                    btc_held = (capital_available * self.position_size) / current_price
            
            equity.append(current_equity)
        
        # Cerrar posición final si existe
        if position != 0:
            final_price = df.iloc[-1]['Close']
            if position == 1:
                capital_available = btc_held * final_price * (1 - self.commission)
            else:
                capital_available = capital_available + (btc_held * (entry_price - final_price))
                capital_available = capital_available * (1 - self.commission)
            
            equity[-1] = capital_available
        
        return equity, trades