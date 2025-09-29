"""
Módulo de Optimización de Hiperparámetros
Utiliza Optuna para optimización bayesiana maximizando el Calmar Ratio
"""

import pandas as pd
import numpy as np
import optuna
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

from backtest_engine import BacktestEngine
from data_loader import DataLoader

class StrategyOptimizer:
    """
    Optimizador de parámetros de la estrategia usando Optuna
    """
    
    def __init__(self, config: Dict):
        """
        Inicializa el optimizador
        
        Parameters:
        -----------
        config : Dict
            Configuración del proyecto
        """
        self.config = config
        self.best_params = None
        self.study = None
        self.optimization_history = []
        
    def objective(self, trial: optuna.Trial, df: pd.DataFrame, 
                 use_cv: bool = True) -> float:
        """
        Función objetivo para Optuna
        
        Parameters:
        -----------
        trial : optuna.Trial
            Trial de Optuna
        df : pd.DataFrame
            Datos para el backtest
        use_cv : bool
            Si usar validación cruzada temporal
            
        Returns:
        --------
        float
            Calmar Ratio (objetivo a maximizar)
        """
        # Sugerir hiperparámetros
        params = {
            # Parámetros de indicadores
            'sma_short': trial.suggest_int('sma_short', 10, 30),
            'sma_long': trial.suggest_int('sma_long', 40, 80),
            'ema_short': trial.suggest_int('ema_short', 8, 20),
            'ema_long': trial.suggest_int('ema_long', 21, 40),
            'stoch_window': trial.suggest_int('stoch_window', 10, 20),
            'stoch_smooth': trial.suggest_int('stoch_smooth', 3, 7),
            'atr_window': trial.suggest_int('atr_window', 10, 20),
            'rsi_window': trial.suggest_int('rsi_window', 10, 20),
            
            # Parámetros de gestión de riesgo
            'stop_loss_atr_multiplier': trial.suggest_float('stop_loss_atr', 1.5, 3.0, step=0.1),
            'take_profit_atr_multiplier': trial.suggest_float('take_profit_atr', 2.0, 5.0, step=0.1),
            'risk_per_trade': trial.suggest_float('risk_per_trade', 0.01, 0.05, step=0.005)
        }
        
        # Asegurar que SMA long > SMA short y EMA long > EMA short
        if params['sma_long'] <= params['sma_short']:
            params['sma_long'] = params['sma_short'] + 20
        if params['ema_long'] <= params['ema_short']:
            params['ema_long'] = params['ema_short'] + 15
        
        # Actualizar configuración con los parámetros sugeridos
        test_config = self.config.copy()
        test_config['indicators'].update({
            'sma_short': params['sma_short'],
            'sma_long': params['sma_long'],
            'ema_short': params['ema_short'],
            'ema_long': params['ema_long'],
            'stoch_window': params['stoch_window'],
            'stoch_smooth': params['stoch_smooth'],
            'atr_window': params['atr_window'],
            'rsi_window': params['rsi_window']
        })
        test_config['risk_management'].update({
            'stop_loss_atr_multiplier': params['stop_loss_atr_multiplier'],
            'take_profit_atr_multiplier': params['take_profit_atr_multiplier'],
            'risk_per_trade': params['risk_per_trade']
        })
        
        if use_cv:
            # Usar validación cruzada temporal
            calmar_ratios = self._cross_validate(df, test_config, n_splits=3)
            calmar_ratio = np.mean(calmar_ratios)
        else:
            # Backtest simple
            engine = BacktestEngine(test_config)
            results = engine.run_backtest(df, params, verbose=False)
            calmar_ratio = results['performance_metrics']['calmar_ratio']
        
        # Registrar en historial
        self.optimization_history.append({
            'trial': trial.number,
            'params': params,
            'calmar_ratio': calmar_ratio
        })
        
        return calmar_ratio
    
    def _cross_validate(self, df: pd.DataFrame, config: Dict, 
                       n_splits: int = 3) -> List[float]:
        """
        Realiza validación cruzada temporal
        
        Parameters:
        -----------
        df : pd.DataFrame
            Datos para validación
        config : Dict
            Configuración para el backtest
        n_splits : int
            Número de splits para la validación cruzada
            
        Returns:
        --------
        List[float]
            Lista de Calmar Ratios de cada split
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        calmar_ratios = []
        
        for train_idx, val_idx in tscv.split(df):
            # Asegurar un tamaño mínimo para el conjunto de validación
            if len(val_idx) < 100:
                continue
                
            train_data = df.iloc[train_idx]
            val_data = df.iloc[val_idx]
            
            # Ejecutar backtest en datos de validación
            engine = BacktestEngine(config)
            results = engine.run_backtest(val_data, verbose=False)
            
            calmar_ratio = results['performance_metrics']['calmar_ratio']
            
            # Penalizar si no hay trades o si el retorno es negativo
            if results['performance_metrics']['total_trades'] == 0:
                calmar_ratio = -1000
            elif results['performance_metrics']['total_return'] < 0:
                calmar_ratio = calmar_ratio - 10  # Penalización
                
            calmar_ratios.append(calmar_ratio)
        
        return calmar_ratios
    
    def optimize(self, df: pd.DataFrame, n_trials: int = None, 
                use_cv: bool = True, seed: int = None) -> Dict:
        """
        Ejecuta la optimización de hiperparámetros
        
        Parameters:
        -----------
        df : pd.DataFrame
            Datos para optimización
        n_trials : int
            Número de trials de Optuna
        use_cv : bool
            Si usar validación cruzada
        seed : int
            Semilla para reproducibilidad
            
        Returns:
        --------
        Dict
            Mejores parámetros encontrados
        """
        n_trials = n_trials or self.config['optimization']['n_trials']
        seed = seed or self.config['optimization']['seed']
        
        print("\n=== Iniciando Optimización con Optuna ===")
        print(f"Número de trials: {n_trials}")
        print(f"Validación cruzada: {'Sí' if use_cv else 'No'}")
        print(f"Objetivo: Maximizar Calmar Ratio")
        
        # Crear estudio de Optuna
        sampler = optuna.samplers.TPESampler(seed=seed)
        self.study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            study_name='btc_trading_strategy'
        )
        
        # Ejecutar optimización
        self.study.optimize(
            lambda trial: self.objective(trial, df, use_cv),
            n_trials=n_trials,
            show_progress_bar=True
        )
        
        # Obtener mejores parámetros
        self.best_params = self.study.best_params
        
        print("\n=== Optimización Completada ===")
        print(f"Mejor Calmar Ratio: {self.study.best_value:.3f}")
        print("\nMejores Parámetros:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        
        return self.best_params
    
    def get_optimization_report(self) -> Dict:
        """
        Genera un reporte de la optimización
        
        Returns:
        --------
        Dict
            Reporte detallado de la optimización
        """
        if not self.study:
            return {}
        
        # Obtener todos los trials
        trials_df = pd.DataFrame([
            {
                'trial': t.number,
                'calmar_ratio': t.value,
                'state': t.state.name,
                **t.params
            }
            for t in self.study.trials
        ])
        
        # Top 10 mejores trials
        top_trials = trials_df.nlargest(10, 'calmar_ratio')
        
        # Estadísticas de parámetros
        param_importance = {}
        if len(trials_df) > 10:
            for param in self.best_params.keys():
                if param in trials_df.columns:
                    # Correlación con el objetivo
                    correlation = trials_df[param].corr(trials_df['calmar_ratio'])
                    param_importance[param] = abs(correlation)
        
        report = {
            'best_params': self.best_params,
            'best_value': self.study.best_value,
            'n_trials': len(self.study.trials),
            'top_10_trials': top_trials.to_dict('records'),
            'param_importance': param_importance,
            'optimization_history': trials_df.to_dict('records'),
            'convergence': {
                'best_values': [t.value for t in self.study.best_trials[:10]],
                'trial_numbers': [t.number for t in self.study.best_trials[:10]]
            }
        }
        
        return report
    
    def walk_forward_analysis(self, df: pd.DataFrame, 
                            window_size: float = 0.6,
                            step_size: float = 0.1) -> Dict:
        """
        Realiza Walk-Forward Analysis para validación robusta
        
        Parameters:
        -----------
        df : pd.DataFrame
            Datos completos
        window_size : float
            Tamaño de la ventana de entrenamiento (proporción)
        step_size : float
            Tamaño del paso (proporción)
            
        Returns:
        --------
        Dict
            Resultados del walk-forward analysis
        """
        print("\n=== Walk-Forward Analysis ===")
        
        results = []
        n_samples = len(df)
        window_samples = int(n_samples * window_size)
        step_samples = int(n_samples * step_size)
        
        current_pos = 0
        window_num = 1
        
        while current_pos + window_samples + step_samples <= n_samples:
            print(f"\nVentana {window_num}:")
            
            # Dividir datos
            train_end = current_pos + window_samples
            test_end = min(train_end + step_samples, n_samples)
            
            train_data = df.iloc[current_pos:train_end]
            test_data = df.iloc[train_end:test_end]
            
            print(f"  Train: {train_data.iloc[0]['Date'].date()} a {train_data.iloc[-1]['Date'].date()}")
            print(f"  Test: {test_data.iloc[0]['Date'].date()} a {test_data.iloc[-1]['Date'].date()}")
            
            # Optimizar en train
            temp_optimizer = StrategyOptimizer(self.config)
            best_params = temp_optimizer.optimize(
                train_data, 
                n_trials=20,  # Menos trials para walk-forward
                use_cv=False,
                seed=42 + window_num
            )
            
            # Evaluar en test
            test_config = self.config.copy()
            test_config['indicators'].update({
                k: v for k, v in best_params.items() 
                if k in test_config['indicators']
            })
            test_config['risk_management'].update({
                k.replace('_atr', '_atr_multiplier'): v 
                for k, v in best_params.items() 
                if '_atr' in k
            })
            if 'risk_per_trade' in best_params:
                test_config['risk_management']['risk_per_trade'] = best_params['risk_per_trade']
            
            engine = BacktestEngine(test_config)
            test_results = engine.run_backtest(test_data, verbose=False)
            
            results.append({
                'window': window_num,
                'train_start': train_data.iloc[0]['Date'],
                'train_end': train_data.iloc[-1]['Date'],
                'test_start': test_data.iloc[0]['Date'],
                'test_end': test_data.iloc[-1]['Date'],
                'best_params': best_params,
                'train_calmar': temp_optimizer.study.best_value,
                'test_calmar': test_results['performance_metrics']['calmar_ratio'],
                'test_return': test_results['performance_metrics']['total_return'],
                'test_sharpe': test_results['performance_metrics']['sharpe_ratio'],
                'test_trades': test_results['performance_metrics']['total_trades']
            })
            
            current_pos += step_samples
            window_num += 1
        
        # Calcular estadísticas agregadas
        results_df = pd.DataFrame(results)
        
        summary = {
            'n_windows': len(results),
            'avg_test_calmar': results_df['test_calmar'].mean(),
            'std_test_calmar': results_df['test_calmar'].std(),
            'avg_test_return': results_df['test_return'].mean(),
            'avg_test_sharpe': results_df['test_sharpe'].mean(),
            'stability_ratio': results_df['test_calmar'].mean() / (results_df['test_calmar'].std() + 1e-10),
            'windows': results
        }
        
        print("\n=== Resumen Walk-Forward ===")
        print(f"Ventanas analizadas: {summary['n_windows']}")
        print(f"Calmar promedio (test): {summary['avg_test_calmar']:.3f}")
        print(f"Desviación estándar Calmar: {summary['std_test_calmar']:.3f}")
        print(f"Ratio de estabilidad: {summary['stability_ratio']:.3f}")
        
        return summary


# Funciones de utilidad para análisis de robustez
def parameter_sensitivity_analysis(optimizer: StrategyOptimizer, 
                                  df: pd.DataFrame,
                                  param_name: str,
                                  param_range: List) -> pd.DataFrame:
    """
    Analiza la sensibilidad de un parámetro específico
    
    Parameters:
    -----------
    optimizer : StrategyOptimizer
        Optimizador
    df : pd.DataFrame
        Datos para análisis
    param_name : str
        Nombre del parámetro
    param_range : List
        Rango de valores a probar
        
    Returns:
    --------
    pd.DataFrame
        Resultados del análisis de sensibilidad
    """
    results = []
    base_config = optimizer.config.copy()
    
    for value in param_range:
        # Actualizar parámetro
        if param_name in base_config['indicators']:
            base_config['indicators'][param_name] = value
        elif param_name in base_config['risk_management']:
            base_config['risk_management'][param_name] = value
        
        # Ejecutar backtest
        engine = BacktestEngine(base_config)
        backtest_results = engine.run_backtest(df, verbose=False)
        
        results.append({
            'parameter': param_name,
            'value': value,
            'calmar_ratio': backtest_results['performance_metrics']['calmar_ratio'],
            'sharpe_ratio': backtest_results['performance_metrics']['sharpe_ratio'],
            'total_return': backtest_results['performance_metrics']['total_return'],
            'max_drawdown': backtest_results['performance_metrics']['max_drawdown'],
            'total_trades': backtest_results['performance_metrics']['total_trades']
        })
    
    return pd.DataFrame(results)


# Función de prueba
def test_optimizer(config):
    """
    Prueba el optimizador
    """
    # Cargar datos
    loader = DataLoader(config)
    df = loader.load_data(config['data']['file_path'])
    train_data, _, _ = loader.split_data()
    
    # Crear optimizador
    optimizer = StrategyOptimizer(config)
    
    # Optimizar (con menos trials para prueba)
    best_params = optimizer.optimize(
        train_data[:5000],  # Usar subset para prueba rápida
        n_trials=10,
        use_cv=True
    )
    
    # Obtener reporte
    report = optimizer.get_optimization_report()
    
    print("\n=== Reporte de Optimización ===")
    print(f"Mejor Calmar Ratio encontrado: {report['best_value']:.3f}")
    print(f"Total de trials ejecutados: {report['n_trials']}")
    
    return optimizer, report


if __name__ == "__main__":
    from config import CONFIG
    optimizer, report = test_optimizer(CONFIG)