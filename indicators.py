"""
Módulo de Indicadores Técnicos
Implementa los indicadores principales y genera señales de trading
"""

import pandas as pd
import numpy as np
import ta

class TechnicalIndicators:
    def __init__(self):
        # Parámetros optimizados
        self.params = {
            'sma_short': 10,
            'sma_long': 30,
            'ema_short': 9,
            'ema_long': 21,
            'rsi_period': 14,
            'rsi_oversold': 35,
            'rsi_overbought': 65,
            'bb_period': 20,
            'bb_std': 2,
            'stoch_period': 14,
            'atr_period': 14
        }
    
    def calculate_all(self, df):
        """Calcula todos los indicadores"""
        data = df.copy()
        
        # Moving Averages
        data['SMA_short'] = ta.trend.SMAIndicator(
            close=data['Close'], 
            window=self.params['sma_short']
        ).sma_indicator()
        
        data['SMA_long'] = ta.trend.SMAIndicator(
            close=data['Close'], 
            window=self.params['sma_long']
        ).sma_indicator()
        
        data['EMA_short'] = ta.trend.EMAIndicator(
            close=data['Close'],
            window=self.params['ema_short']
        ).ema_indicator()
        
        data['EMA_long'] = ta.trend.EMAIndicator(
            close=data['Close'],
            window=self.params['ema_long']
        ).ema_indicator()
        
        # RSI
        data['RSI'] = ta.momentum.RSIIndicator(
            close=data['Close'],
            window=self.params['rsi_period']
        ).rsi()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(
            close=data['Close'],
            window=self.params['bb_period'],
            window_dev=self.params['bb_std']
        )
        data['BB_upper'] = bb.bollinger_hband()
        data['BB_lower'] = bb.bollinger_lband()
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            window=self.params['stoch_period']
        )
        data['Stoch'] = stoch.stoch()
        
        # ATR
        data['ATR'] = ta.volatility.AverageTrueRange(
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            window=self.params['atr_period']
        ).average_true_range()
        
        # MACD
        macd = ta.trend.MACD(close=data['Close'])
        data['MACD'] = macd.macd()
        data['MACD_signal'] = macd.macd_signal()
        data['MACD_diff'] = macd.macd_diff()
        
        return data
    
    def generate_signals(self, df):
        """Genera señales de trading"""
        data = df.copy()
        
        # Señales individuales
        data['signal_ma'] = 0
        data['signal_rsi'] = 0
        data['signal_bb'] = 0
        data['signal_macd'] = 0
        
        # Moving Average
        data.loc[(data['SMA_short'] > data['SMA_long']) & 
                 (data['EMA_short'] > data['EMA_long']), 'signal_ma'] = 1
        data.loc[(data['SMA_short'] < data['SMA_long']) & 
                 (data['EMA_short'] < data['EMA_long']), 'signal_ma'] = -1
        
        # RSI
        data.loc[data['RSI'] < self.params['rsi_oversold'], 'signal_rsi'] = 1
        data.loc[data['RSI'] > self.params['rsi_overbought'], 'signal_rsi'] = -1
        
        # Bollinger Bands
        data.loc[data['Close'] < data['BB_lower'], 'signal_bb'] = 1
        data.loc[data['Close'] > data['BB_upper'], 'signal_bb'] = -1
        
        # MACD
        data.loc[data['MACD_diff'] > 0, 'signal_macd'] = 1
        data.loc[data['MACD_diff'] < 0, 'signal_macd'] = -1
        
        # Señal combinada (mínimo 2 de 4 indicadores)
        signal_cols = ['signal_ma', 'signal_rsi', 'signal_bb', 'signal_macd']
        data['signal_sum'] = data[signal_cols].sum(axis=1)
        
        data['signal'] = 0
        data.loc[data['signal_sum'] >= 2, 'signal'] = 1  # BUY
        data.loc[data['signal_sum'] <= -2, 'signal'] = -1  # SELL
        
        return data