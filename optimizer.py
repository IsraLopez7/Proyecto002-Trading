"""
Módulo de Optimización de Hiperparámetros
Utiliza Optuna para optimización bayesiana maximizando el Calmar Ratio
"""

import optuna
import numpy as np
from data_loader import DataLoader
from indicators import TechnicalIndicators
from strategy import TradingStrategy
from metrics import PerformanceMetrics

# Silenciar warnings de Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

class SimpleOptimizer:
    def __init__(self, train_data):
        self.train_data = train_data
        self.best_params = None
        
    def objective(self, trial):
        """Función objetivo para Optuna"""
        
        # Sugerir parámetros
        params = {
            'sma_short': trial.suggest_int('sma_short', 5, 20),
            'sma_long': trial.suggest_int('sma_long', 20, 50),
            'rsi_period': trial.suggest_int('rsi_period', 10, 20),
            'rsi_oversold': trial.suggest_int('rsi_oversold', 20, 40),
            'rsi_overbought': trial.suggest_int('rsi_overbought', 60, 80),
        }
        
        # Asegurar que sma_long > sma_short
        if params['sma_long'] <= params['sma_short']:
            params['sma_long'] = params['sma_short'] + 10
        
        # Configurar indicadores con nuevos parámetros
        indicators = TechnicalIndicators()
        indicators.params.update(params)
        
        # Calcular indicadores y señales
        data = indicators.calculate_all(self.train_data.copy())
        data = indicators.generate_signals(data)
        
        # Ejecutar backtest
        strategy = TradingStrategy()
        equity, trades = strategy.backtest(data)
        
        # Calcular métricas
        metrics_calc = PerformanceMetrics()
        metrics = metrics_calc.calculate(equity, trades, strategy.initial_capital)
        
        # Objetivo: maximizar Calmar Ratio
        return metrics['calmar_ratio']
    
    def optimize(self, n_trials=30):
        """Ejecuta la optimización"""
        print(f"\nOptimizando con {n_trials} trials...")
        
        # Crear estudio
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)
        
        # Guardar mejores parámetros
        self.best_params = study.best_params
        
        print(f"\n✓ Optimización completada")
        print(f"Mejor Calmar Ratio: {study.best_value:.3f}")
        print(f"Mejores parámetros:")
        for param, value in self.best_params.items():
            print(f"  • {param}: {value}")
        
        return self.best_params

# Función para usar en main.py
def run_optimization():
    """Ejecuta optimización de parámetros"""
    print("\n" + "="*60)
    print("OPTIMIZACIÓN DE PARÁMETROS")
    print("="*60)
    
    # Cargar datos
    loader = DataLoader('Binance_BTCUSDT_1h.csv')
    df = loader.load_and_prepare()
    train_data, _, _ = loader.split_data()
    
    # Usar solo primeros 10000 registros para optimización rápida
    train_subset = train_data[:10000].copy()
    
    # Optimizar
    optimizer = SimpleOptimizer(train_subset)
    best_params = optimizer.optimize(n_trials=30)
    
    return best_params

if __name__ == "__main__":
    # Ejecutar optimización standalone
    best_params = run_optimization()