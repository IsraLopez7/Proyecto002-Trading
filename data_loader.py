"""
Módulo de carga y preprocesamiento de datos
Maneja la lectura del CSV, limpieza, y división del dataset
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
from datetime import datetime

class DataLoader:
    """
    Clase para cargar y preprocesar datos de trading
    """
    
    def __init__(self, config: Dict):
        """
        Inicializa el cargador de datos
        
        Parameters:
        -----------
        config : Dict
            Diccionario de configuración con parámetros del proyecto
        """
        self.config = config
        self.data = None
        self.train_data = None
        self.test_data = None
        self.validation_data = None
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Carga los datos desde un archivo CSV
        
        Parameters:
        -----------
        file_path : str
            Ruta al archivo CSV
            
        Returns:
        --------
        pd.DataFrame
            DataFrame con los datos cargados y procesados
        """
        print(f"Cargando datos desde {file_path}...")
        
        # Leer CSV saltando la primera línea (comentario)
        df = pd.read_csv(file_path, skiprows=1)
        
        # Limpiar la columna Date de posibles milisegundos
        # Algunos registros tienen .000 al final
        df['Date'] = df['Date'].astype(str)
        df['Date'] = df['Date'].str.replace(r'\.000$', '', regex=True)
        
        # Convertir la columna Date a datetime
        try:
            df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d %H:%M:%S')
        except Exception as e:
            print(f"Advertencia: Usando formato mixto para fechas debido a: {e}")
            df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=False)
        
        # Ordenar por fecha (el dataset viene en orden inverso)
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Renombrar columnas para facilitar el uso
        df = df.rename(columns={
            'Volume BTC': 'Volume_BTC',
            'Volume USDT': 'Volume_USDT'
        })
        
        # Verificar y manejar valores nulos
        null_counts = df.isnull().sum()
        if null_counts.any():
            print("Valores nulos encontrados:")
            print(null_counts[null_counts > 0])
            # Rellenar valores nulos con interpolación
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume_BTC', 'Volume_USDT']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = df[col].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
        
        # Asegurar que las columnas numéricas sean del tipo correcto
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume_BTC', 'Volume_USDT', 'tradecount']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Crear columnas adicionales útiles
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Close_Open_Ratio'] = df['Close'] / df['Open']
        
        # Volatilidad rolling (para análisis posterior)
        df['Volatility_20'] = df['Returns'].rolling(window=20).std()
        
        # Verificar integridad temporal
        time_diff = df['Date'].diff()
        irregular_intervals = time_diff[time_diff != pd.Timedelta(hours=1)].dropna()
        if len(irregular_intervals) > 0:
            print(f"Advertencia: {len(irregular_intervals)} intervalos irregulares detectados")
        
        self.data = df
        print(f"Datos cargados exitosamente: {len(df)} registros")
        print(f"Período: {df['Date'].min()} a {df['Date'].max()}")
        
        return df
    
    def split_data(self, df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Divide los datos en conjuntos de entrenamiento, prueba y validación
        
        Parameters:
        -----------
        df : pd.DataFrame, optional
            DataFrame a dividir. Si no se proporciona, usa self.data
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            train_data, test_data, validation_data
        """
        if df is None:
            df = self.data
            
        if df is None:
            raise ValueError("No hay datos cargados. Ejecute load_data() primero.")
        
        n = len(df)
        train_size = int(n * self.config['data']['train_ratio'])
        test_size = int(n * self.config['data']['test_ratio'])
        
        # División temporal (no aleatoria para series de tiempo)
        self.train_data = df.iloc[:train_size].copy()
        self.test_data = df.iloc[train_size:train_size + test_size].copy()
        self.validation_data = df.iloc[train_size + test_size:].copy()
        
        # Reset índices para cada conjunto
        self.train_data.reset_index(drop=True, inplace=True)
        self.test_data.reset_index(drop=True, inplace=True)
        self.validation_data.reset_index(drop=True, inplace=True)
        
        print("\n=== División del Dataset ===")
        print(f"Train: {len(self.train_data)} registros ({self.train_data['Date'].min().date()} a {self.train_data['Date'].max().date()})")
        print(f"Test: {len(self.test_data)} registros ({self.test_data['Date'].min().date()} a {self.test_data['Date'].max().date()})")
        print(f"Validation: {len(self.validation_data)} registros ({self.validation_data['Date'].min().date()} a {self.validation_data['Date'].max().date()})")
        
        return self.train_data, self.test_data, self.validation_data
    
    def get_price_statistics(self, df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Calcula estadísticas descriptivas del dataset
        
        Parameters:
        -----------
        df : pd.DataFrame, optional
            DataFrame para calcular estadísticas
            
        Returns:
        --------
        Dict
            Diccionario con estadísticas
        """
        if df is None:
            df = self.data
            
        if df is None or len(df) == 0:
            return {}
            
        stats = {
            'count': len(df),
            'start_date': df['Date'].min(),
            'end_date': df['Date'].max(),
            'price_stats': {
                'min': df['Close'].min(),
                'max': df['Close'].max(),
                'mean': df['Close'].mean(),
                'median': df['Close'].median(),
                'std': df['Close'].std()
            },
            'volume_stats': {
                'mean_btc': df['Volume_BTC'].mean(),
                'mean_usdt': df['Volume_USDT'].mean(),
                'total_usdt': df['Volume_USDT'].sum()
            },
            'returns_stats': {
                'mean_return': df['Returns'].mean() if 'Returns' in df.columns else 0,
                'std_return': df['Returns'].std() if 'Returns' in df.columns else 0,
                'sharpe': (df['Returns'].mean() / df['Returns'].std() * np.sqrt(365 * 24)) if 'Returns' in df.columns and df['Returns'].std() != 0 else 0,
                'min_return': df['Returns'].min() if 'Returns' in df.columns else 0,
                'max_return': df['Returns'].max() if 'Returns' in df.columns else 0
            }
        }
        
        return stats
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara características adicionales para el análisis
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame con datos OHLCV
            
        Returns:
        --------
        pd.DataFrame
            DataFrame con características adicionales
        """
        df = df.copy()
        
        # Características de precio
        df['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100
        df['CO_PCT'] = (df['Close'] - df['Open']) / df['Open'] * 100
        
        # Características de volumen
        df['Volume_Ratio'] = df['Volume_USDT'] / df['Volume_USDT'].rolling(window=20).mean()
        
        # Características temporales
        df['Hour'] = df['Date'].dt.hour
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month
        
        # Patrones de velas japonesas simples
        df['Doji'] = (abs(df['Close'] - df['Open']) / (df['High'] - df['Low']) < 0.1).astype(int)
        df['Bullish'] = (df['Close'] > df['Open']).astype(int)
        df['Bearish'] = (df['Close'] < df['Open']).astype(int)
        
        return df


# Función de utilidad para prueba rápida
def test_data_loader(config):
    """
    Función de prueba para el módulo DataLoader
    """
    loader = DataLoader(config)
    
    # Cargar datos
    df = loader.load_data(config['data']['file_path'])
    
    # Dividir datos
    train, test, val = loader.split_data()
    
    # Obtener estadísticas
    print("\n=== Estadísticas del Dataset Completo ===")
    stats = loader.get_price_statistics()
    if stats:
        print(f"Precio mínimo: ${stats['price_stats']['min']:.2f}")
        print(f"Precio máximo: ${stats['price_stats']['max']:.2f}")
        print(f"Precio promedio: ${stats['price_stats']['mean']:.2f}")
        print(f"Volatilidad (std): ${stats['price_stats']['std']:.2f}")
        print(f"Sharpe Ratio anualizado: {stats['returns_stats']['sharpe']:.2f}")
    
    # Preparar características
    df_features = loader.prepare_features(train)
    print(f"\nCaracterísticas adicionales creadas: {list(df_features.columns[-10:])}")
    
    return loader


# Si se ejecuta como script principal
if __name__ == "__main__":
    # Configuración de prueba
    CONFIG = {
        'data': {
            'file_path': 'Binance_BTCUSDT_1h.csv',
            'train_ratio': 0.6,
            'test_ratio': 0.2,
            'validation_ratio': 0.2
        }
    }
    loader = test_data_loader(CONFIG)