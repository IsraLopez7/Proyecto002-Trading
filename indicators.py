"""
Módulo de Indicadores Técnicos
Implementa los indicadores principales y genera señales de trading
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import ta

class TechnicalIndicators:
    """
    Clase para calcular indicadores técnicos y generar señales
    """
    
    def __init__(self, config: Dict):
        """
        Inicializa el calculador de indicadores
        
        Parameters:
        -----------
        config : Dict
            Configuración con parámetros de indicadores
        """
        self.config = config['indicators']
        self.risk_config = config['risk_management']
        
    def calculate_sma(self, df: pd.DataFrame, short_period: int = None, 
                     long_period: int = None) -> pd.DataFrame:
        """
        Calcula Simple Moving Average (SMA)
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame con precios
        short_period : int
            Período para SMA corto
        long_period : int
            Período para SMA largo
            
        Returns:
        --------
        pd.DataFrame
            DataFrame con indicadores SMA
        """
        df = df.copy()
        
        short = short_period or self.config['sma_short']
        long = long_period or self.config['sma_long']
        
        # Calcular SMAs
        df['SMA_short'] = ta.trend.SMAIndicator(
            close=df['Close'], 
            window=short
        ).sma_indicator()
        
        df['SMA_long'] = ta.trend.SMAIndicator(
            close=df['Close'], 
            window=long
        ).sma_indicator()
        
        # Señales SMA
        df['SMA_signal'] = 0
        df.loc[df['SMA_short'] > df['SMA_long'], 'SMA_signal'] = 1  # Bullish
        df.loc[df['SMA_short'] < df['SMA_long'], 'SMA_signal'] = -1  # Bearish
        
        # Crossovers
        df['SMA_cross'] = df['SMA_signal'].diff()
        
        return df
    
    def calculate_ema(self, df: pd.DataFrame, short_period: int = None,
                     long_period: int = None) -> pd.DataFrame:
        """
        Calcula Exponential Moving Average (EMA)
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame con precios
        short_period : int
            Período para EMA corto
        long_period : int
            Período para EMA largo
            
        Returns:
        --------
        pd.DataFrame
            DataFrame con indicadores EMA
        """
        df = df.copy()
        
        short = short_period or self.config['ema_short']
        long = long_period or self.config['ema_long']
        
        # Calcular EMAs
        df['EMA_short'] = ta.trend.EMAIndicator(
            close=df['Close'],
            window=short
        ).ema_indicator()
        
        df['EMA_long'] = ta.trend.EMAIndicator(
            close=df['Close'],
            window=long
        ).ema_indicator()
        
        # Señales EMA
        df['EMA_signal'] = 0
        df.loc[df['EMA_short'] > df['EMA_long'], 'EMA_signal'] = 1  # Bullish
        df.loc[df['EMA_short'] < df['EMA_long'], 'EMA_signal'] = -1  # Bearish
        
        # MACD adicional
        df['MACD'] = df['EMA_short'] - df['EMA_long']
        df['MACD_signal_line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal_line']
        
        return df
    
    def calculate_stochastic(self, df: pd.DataFrame, window: int = None,
                           smooth_window: int = None) -> pd.DataFrame:
        """
        Calcula Stochastic Oscillator
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame con precios OHLC
        window : int
            Período para el cálculo
        smooth_window : int
            Período de suavizado
            
        Returns:
        --------
        pd.DataFrame
            DataFrame con indicador Stochastic
        """
        df = df.copy()
        
        window = window or self.config['stoch_window']
        smooth = smooth_window or self.config['stoch_smooth']
        
        # Calcular Stochastic
        stoch = ta.momentum.StochasticOscillator(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            window=window,
            smooth_window=smooth
        )
        
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        # Señales Stochastic
        df['Stoch_signal'] = 0
        # Oversold (< 20) turning up = Bullish
        df.loc[(df['Stoch_K'] > 20) & (df['Stoch_K'].shift(1) <= 20), 'Stoch_signal'] = 1
        # Overbought (> 80) turning down = Bearish
        df.loc[(df['Stoch_K'] < 80) & (df['Stoch_K'].shift(1) >= 80), 'Stoch_signal'] = -1
        
        # Divergencias K y D
        df.loc[(df['Stoch_K'] > df['Stoch_D']) & (df['Stoch_K'] < 50), 'Stoch_signal'] = 1
        df.loc[(df['Stoch_K'] < df['Stoch_D']) & (df['Stoch_K'] > 50), 'Stoch_signal'] = -1
        
        return df
    
    def calculate_atr(self, df: pd.DataFrame, window: int = None) -> pd.DataFrame:
        """
        Calcula Average True Range (ATR) para gestión de riesgo
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame con precios OHLC
        window : int
            Período para ATR
            
        Returns:
        --------
        pd.DataFrame
            DataFrame con ATR
        """
        df = df.copy()
        
        window = window or self.config['atr_window']
        
        # Calcular ATR
        atr_indicator = ta.volatility.AverageTrueRange(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            window=window
        )
        
        df['ATR'] = atr_indicator.average_true_range()
        
        # ATR como porcentaje del precio
        df['ATR_pct'] = (df['ATR'] / df['Close']) * 100
        
        # Niveles de Stop Loss y Take Profit basados en ATR
        df['SL_long'] = df['Close'] - (df['ATR'] * self.risk_config['stop_loss_atr_multiplier'])
        df['TP_long'] = df['Close'] + (df['ATR'] * self.risk_config['take_profit_atr_multiplier'])
        
        df['SL_short'] = df['Close'] + (df['ATR'] * self.risk_config['stop_loss_atr_multiplier'])
        df['TP_short'] = df['Close'] - (df['ATR'] * self.risk_config['take_profit_atr_multiplier'])
        
        return df
    
    def calculate_rsi(self, df: pd.DataFrame, window: int = None) -> pd.DataFrame:
        """
        Calcula Relative Strength Index (RSI)
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame con precios
        window : int
            Período para RSI
            
        Returns:
        --------
        pd.DataFrame
            DataFrame con RSI
        """
        df = df.copy()
        
        window = window or self.config['rsi_window']
        
        # Calcular RSI
        df['RSI'] = ta.momentum.RSIIndicator(
            close=df['Close'],
            window=window
        ).rsi()
        
        # Señales RSI
        df['RSI_signal'] = 0
        df.loc[df['RSI'] < 30, 'RSI_signal'] = 1  # Oversold - Bullish
        df.loc[df['RSI'] > 70, 'RSI_signal'] = -1  # Overbought - Bearish
        
        return df
    
    def calculate_bollinger_bands(self, df: pd.DataFrame, window: int = 20,
                                 num_std: float = 2) -> pd.DataFrame:
        """
        Calcula Bollinger Bands
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame con precios
        window : int
            Período para la media móvil
        num_std : float
            Número de desviaciones estándar
            
        Returns:
        --------
        pd.DataFrame
            DataFrame con Bollinger Bands
        """
        df = df.copy()
        
        # Calcular Bollinger Bands
        bb = ta.volatility.BollingerBands(
            close=df['Close'],
            window=window,
            window_dev=num_std
        )
        
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_middle'] = bb.bollinger_mavg()
        df['BB_lower'] = bb.bollinger_lband()
        df['BB_width'] = df['BB_upper'] - df['BB_lower']
        df['BB_pct'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
        # Señales Bollinger Bands
        df['BB_signal'] = 0
        df.loc[df['Close'] < df['BB_lower'], 'BB_signal'] = 1  # Precio bajo banda inferior - Bullish
        df.loc[df['Close'] > df['BB_upper'], 'BB_signal'] = -1  # Precio sobre banda superior - Bearish
        
        return df
    
    def calculate_all_indicators(self, df: pd.DataFrame, params: Dict = None) -> pd.DataFrame:
        """
        Calcula todos los indicadores técnicos
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame con precios OHLCV
        params : Dict
            Parámetros opcionales para los indicadores
            
        Returns:
        --------
        pd.DataFrame
            DataFrame con todos los indicadores calculados
        """
        df = df.copy()
        
        print("Calculando indicadores técnicos...")
        
        # Si se proporcionan parámetros, actualizar configuración temporal
        if params:
            original_config = self.config.copy()
            self.config.update(params)
        
        # Calcular cada indicador
        df = self.calculate_sma(df)
        df = self.calculate_ema(df)
        df = self.calculate_stochastic(df)
        df = self.calculate_atr(df)
        df = self.calculate_rsi(df)
        df = self.calculate_bollinger_bands(df)
        
        # Restaurar configuración original si se modificó
        if params:
            self.config = original_config
        
        print(f"Indicadores calculados: SMA, EMA, Stochastic, ATR, RSI, Bollinger Bands")
        
        return df
    
    def generate_combined_signal(self, df: pd.DataFrame, min_confirmations: int = 2) -> pd.DataFrame:
        """
        Genera señal combinada basada en confirmación de múltiples indicadores
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame con indicadores calculados
        min_confirmations : int
            Número mínimo de indicadores que deben coincidir (default: 2 de 3)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame con señales de trading combinadas
        """
        df = df.copy()
        
        # Lista de señales principales (3 indicadores principales)
        primary_signals = ['SMA_signal', 'EMA_signal', 'Stoch_signal']
        
        # Calcular suma de señales principales
        df['signal_sum'] = df[primary_signals].sum(axis=1)
        
        # Generar señal combinada con confirmación
        df['combined_signal'] = 0
        
        # Buy signal: al menos 2 de 3 indicadores son bullish
        df.loc[df['signal_sum'] >= min_confirmations, 'combined_signal'] = 1
        
        # Sell signal: al menos 2 de 3 indicadores son bearish
        df.loc[df['signal_sum'] <= -min_confirmations, 'combined_signal'] = -1
        
        # Agregar filtros adicionales con RSI y Bollinger Bands
        # Fortalecer señales cuando hay confirmación adicional
        df['signal_strength'] = abs(df['signal_sum'])
        
        # Si RSI confirma, aumentar fuerza
        df.loc[(df['combined_signal'] == 1) & (df['RSI_signal'] == 1), 'signal_strength'] += 1
        df.loc[(df['combined_signal'] == -1) & (df['RSI_signal'] == -1), 'signal_strength'] += 1
        
        # Si Bollinger Bands confirma, aumentar fuerza
        df.loc[(df['combined_signal'] == 1) & (df['BB_signal'] == 1), 'signal_strength'] += 1
        df.loc[(df['combined_signal'] == -1) & (df['BB_signal'] == -1), 'signal_strength'] += 1
        
        # Crear señal de entrada (cuando cambia la señal combinada)
        df['entry_signal'] = df['combined_signal'].diff()
        
        # Estadísticas de señales
        total_signals = len(df[df['entry_signal'] != 0])
        buy_signals = len(df[df['entry_signal'] > 0])
        sell_signals = len(df[df['entry_signal'] < 0])
        
        print(f"\n=== Señales Generadas ===")
        print(f"Total de señales: {total_signals}")
        print(f"Señales de compra: {buy_signals}")
        print(f"Señales de venta: {sell_signals}")
        print(f"Ratio Buy/Sell: {buy_signals/sell_signals if sell_signals > 0 else 'N/A':.2f}")
        
        return df
    
    def get_current_signals(self, df: pd.DataFrame) -> Dict:
        """
        Obtiene el estado actual de las señales
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame con indicadores y señales
            
        Returns:
        --------
        Dict
            Diccionario con el estado actual de todas las señales
        """
        last_row = df.iloc[-1]
        
        signals = {
            'timestamp': last_row['Date'] if 'Date' in df.columns else None,
            'price': last_row['Close'],
            'combined_signal': last_row['combined_signal'] if 'combined_signal' in df.columns else 0,
            'signal_strength': last_row['signal_strength'] if 'signal_strength' in df.columns else 0,
            'indicators': {
                'SMA': last_row['SMA_signal'] if 'SMA_signal' in df.columns else 0,
                'EMA': last_row['EMA_signal'] if 'EMA_signal' in df.columns else 0,
                'Stochastic': last_row['Stoch_signal'] if 'Stoch_signal' in df.columns else 0,
                'RSI': last_row['RSI_signal'] if 'RSI_signal' in df.columns else 0,
                'BB': last_row['BB_signal'] if 'BB_signal' in df.columns else 0
            },
            'values': {
                'RSI': last_row['RSI'] if 'RSI' in df.columns else None,
                'Stoch_K': last_row['Stoch_K'] if 'Stoch_K' in df.columns else None,
                'ATR': last_row['ATR'] if 'ATR' in df.columns else None,
                'ATR_pct': last_row['ATR_pct'] if 'ATR_pct' in df.columns else None
            },
            'risk_levels': {
                'stop_loss_long': last_row['SL_long'] if 'SL_long' in df.columns else None,
                'take_profit_long': last_row['TP_long'] if 'TP_long' in df.columns else None,
                'stop_loss_short': last_row['SL_short'] if 'SL_short' in df.columns else None,
                'take_profit_short': last_row['TP_short'] if 'TP_short' in df.columns else None
            }
        }
        
        return signals


# Función de prueba
def test_indicators(config, df):
    """
    Prueba el módulo de indicadores
    """
    indicators = TechnicalIndicators(config)
    
    # Calcular todos los indicadores
    df_indicators = indicators.calculate_all_indicators(df)
    
    # Generar señales combinadas
    df_signals = indicators.generate_combined_signal(df_indicators)
    
    # Obtener señales actuales
    current = indicators.get_current_signals(df_signals)
    
    print("\n=== Estado Actual de Señales ===")
    print(f"Precio actual: ${current['price']:.2f}")
    print(f"Señal combinada: {current['combined_signal']}")
    print(f"Fuerza de señal: {current['signal_strength']}")
    print(f"RSI: {current['values']['RSI']:.2f}")
    print(f"Stochastic K: {current['values']['Stoch_K']:.2f}")
    
    return df_signals


if __name__ == "__main__":
    # Código de prueba
    pass