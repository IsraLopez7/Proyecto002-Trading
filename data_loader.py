"""
Módulo simplificado de carga de datos
"""

import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, filepath='Binance_BTCUSDT_1h.csv'):
        self.filepath = filepath
        self.data = None
        
    def load_and_prepare(self):
        """Carga y prepara los datos"""
        # Cargar CSV
        df = pd.read_csv(self.filepath, skiprows=1)
        
        # Limpiar fechas
        df['Date'] = df['Date'].astype(str).str.replace(r'\.000$', '', regex=True)
        df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=False)
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Renombrar columnas
        df.rename(columns={'Volume USDT': 'Volume_USDT'}, inplace=True)
        
        # Añadir retornos
        df['Returns'] = df['Close'].pct_change()
        
        self.data = df
        return df
    
    def split_data(self, train_ratio=0.6, test_ratio=0.2):
        """Divide los datos en train, test y validation"""
        n = len(self.data)
        train_size = int(n * train_ratio)
        test_size = int(n * test_ratio)
        
        train = self.data[:train_size].copy()
        test = self.data[train_size:train_size + test_size].copy()
        val = self.data[train_size + test_size:].copy()
        
        return train, test, val