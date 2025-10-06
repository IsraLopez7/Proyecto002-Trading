"""
indicators.py
Módulo para calcular indicadores técnicos y generar señales
"""

import pandas as pd
import numpy as np
import ta

def calculate_sma(df, window=20):
    """
    Calcula Simple Moving Average
    """
    df = df.copy()
    df[f'sma_{window}'] = ta.trend.SMAIndicator(close=df['close'], window=window).sma_indicator()
    return df

def calculate_ema(df, window=20):
    """
    Calcula Exponential Moving Average
    """
    df = df.copy()
    df[f'ema_{window}'] = ta.trend.EMAIndicator(close=df['close'], window=window).ema_indicator()
    return df

def calculate_stochastic(df, window=14, smooth_window=3):
    """
    Calcula Stochastic Oscillator
    """
    df = df.copy()
    stoch = ta.momentum.StochasticOscillator(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=window,
        smooth_window=smooth_window
    )
    df[f'stoch_k_{window}'] = stoch.stoch()
    df[f'stoch_d_{window}'] = stoch.stoch_signal()
    return df

def calculate_atr(df, window=14):
    """
    Calcula Average True Range
    """
    df = df.copy()
    df[f'atr_{window}'] = ta.volatility.AverageTrueRange(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=window
    ).average_true_range()
    return df

def calculate_rsi(df, window=14):
    """
    Calcula RSI (Relative Strength Index)
    """
    df = df.copy()
    df[f'rsi_{window}'] = ta.momentum.RSIIndicator(
        close=df['close'],
        window=window
    ).rsi()
    return df

def calculate_bollinger_bands(df, window=20, window_dev=2):
    """
    Calcula Bandas de Bollinger
    """
    df = df.copy()
    bb = ta.volatility.BollingerBands(
        close=df['close'],
        window=window,
        window_dev=window_dev
    )
    df[f'bb_upper_{window}'] = bb.bollinger_hband()
    df[f'bb_middle_{window}'] = bb.bollinger_mavg()
    df[f'bb_lower_{window}'] = bb.bollinger_lband()
    df[f'bb_width_{window}'] = bb.bollinger_wband()
    return df

def generate_sma_signals(df, short_window=10, long_window=30):
    """
    Genera señales basadas en cruce de SMAs
    """
    df = df.copy()
    
    # Calcular SMAs si no existen
    if f'sma_{short_window}' not in df.columns:
        df = calculate_sma(df, short_window)
    if f'sma_{long_window}' not in df.columns:
        df = calculate_sma(df, long_window)
    
    # Señal de compra: SMA corta cruza hacia arriba la SMA larga
    df['sma_buy_signal'] = (
        (df[f'sma_{short_window}'] > df[f'sma_{long_window}']) & 
        (df[f'sma_{short_window}'].shift(1) <= df[f'sma_{long_window}'].shift(1))
    ).astype(int)
    
    # Señal de venta: SMA corta cruza hacia abajo la SMA larga
    df['sma_sell_signal'] = (
        (df[f'sma_{short_window}'] < df[f'sma_{long_window}']) & 
        (df[f'sma_{short_window}'].shift(1) >= df[f'sma_{long_window}'].shift(1))
    ).astype(int)
    
    return df

def generate_stoch_signals(df, window=14, oversold=20, overbought=80):
    """
    Genera señales basadas en Stochastic Oscillator
    """
    df = df.copy()
    
    # Calcular Stochastic si no existe
    if f'stoch_k_{window}' not in df.columns:
        df = calculate_stochastic(df, window)
    
    # Señal de compra: Stoch K cruza hacia arriba el nivel oversold
    df['stoch_buy_signal'] = (
        (df[f'stoch_k_{window}'] > oversold) & 
        (df[f'stoch_k_{window}'].shift(1) <= oversold)
    ).astype(int)
    
    # Señal de venta: Stoch K cruza hacia abajo el nivel overbought
    df['stoch_sell_signal'] = (
        (df[f'stoch_k_{window}'] < overbought) & 
        (df[f'stoch_k_{window}'].shift(1) >= overbought)
    ).astype(int)
    
    return df

def generate_rsi_signals(df, window=14, oversold=30, overbought=70):
    """
    Genera señales basadas en RSI
    """
    df = df.copy()
    
    # Calcular RSI si no existe
    if f'rsi_{window}' not in df.columns:
        df = calculate_rsi(df, window)
    
    # Señal de compra: RSI sale de zona oversold
    df['rsi_buy_signal'] = (
        (df[f'rsi_{window}'] > oversold) & 
        (df[f'rsi_{window}'].shift(1) <= oversold)
    ).astype(int)
    
    # Señal de venta: RSI entra en zona overbought
    df['rsi_sell_signal'] = (
        (df[f'rsi_{window}'] < overbought) & 
        (df[f'rsi_{window}'].shift(1) >= overbought)
    ).astype(int)
    
    return df

def calculate_all_indicators(df, params=None):
    """
    Calcula todos los indicadores con parámetros especificados
    """
    if params is None:
        params = {
            'sma_short': 10,
            'sma_long': 30,
            'ema_period': 20,
            'stoch_window': 14,
            'rsi_window': 14,
            'atr_window': 14,
            'bb_window': 20
        }
    
    df = df.copy()
    
    # SMAs
    df = calculate_sma(df, params['sma_short'])
    df = calculate_sma(df, params['sma_long'])
    
    # EMA
    df = calculate_ema(df, params['ema_period'])
    
    # Stochastic
    df = calculate_stochastic(df, params['stoch_window'])
    
    # RSI
    df = calculate_rsi(df, params['rsi_window'])
    
    # ATR
    df = calculate_atr(df, params['atr_window'])
    
    # Bollinger Bands
    df = calculate_bollinger_bands(df, params['bb_window'])
    
    return df

def generate_combined_signals(df, params=None, min_confirmations=2):
    if params is None:
        params = {
            'sma_short': 10, 'sma_long': 30,
            'stoch_window': 14, 'stoch_oversold': 20, 'stoch_overbought': 80,
            'rsi_window': 14, 'rsi_oversold': 30, 'rsi_overbought': 70,
            'ema_trend': 200, 'trend_filter': True,
            'persistence': 1,
        }

    df = df.copy()

    # señales individuales
    df = generate_sma_signals(df, params['sma_short'], params['sma_long'])
    df = generate_stoch_signals(df, params['stoch_window'],
                                params['stoch_oversold'], params['stoch_overbought'])
    df = generate_rsi_signals(df, params['rsi_window'],
                              params['rsi_oversold'], params['rsi_overbought'])

    # confirmaciones
    df['buy_confirmations'] = (
        df['sma_buy_signal'].fillna(0) +
        df['stoch_buy_signal'].fillna(0) +
        df['rsi_buy_signal'].fillna(0)
    )
    df['sell_confirmations'] = (
        df['sma_sell_signal'].fillna(0) +
        df['stoch_sell_signal'].fillna(0) +
        df['rsi_sell_signal'].fillna(0)
    )

    ema_trend = params.get('ema_trend', 200)
    trend_filter = params.get('trend_filter', True)

    buy_raw  = df['buy_confirmations']  >= int(min_confirmations)
    sell_raw = df['sell_confirmations'] >= int(min_confirmations)

    if trend_filter and f'ema_{ema_trend}' in df.columns:
        trend_up   = df['close'] > df[f'ema_{ema_trend}']
        trend_down = df['close'] < df[f'ema_{ema_trend}']
        buy_raw  = buy_raw  & trend_up
        sell_raw = sell_raw & trend_down

    persistence = int(params.get('persistence', 1))
    if persistence > 1:
        df['buy_signal']  = buy_raw.rolling(persistence).sum().ge(persistence).astype(int)
        df['sell_signal'] = sell_raw.rolling(persistence).sum().ge(persistence).astype(int)
    else:
        df['buy_signal']  = buy_raw.astype(int)
        df['sell_signal'] = sell_raw.astype(int)

    return df


def create_signals(data, params):
    df = data.copy()
    # EMA de tendencia (necesaria si se usa trend_filter)
    df = calculate_ema(df, params.get('ema_trend', 200))

    # ← ahora min_confirmations viene del dict de params
    df = generate_combined_signals(df, params, min_confirmations=params.get('min_conf', 1))
    return df

if __name__ == "__main__":
    # Test del módulo
    import data_loader
    df = data_loader.load_data()
    df = data_loader.add_returns(df)
    df = calculate_all_indicators(df)
    df = generate_combined_signals(df)