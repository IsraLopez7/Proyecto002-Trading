# main.py
"""
Script Principal de Ejecución
Orquesta todo el flujo del proyecto de trading sistemático
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import json
import os

# Importar módulos del proyecto
from data_loader import DataLoader
from indicators import TechnicalIndicators
from strategy import TradingStrategy, StrategyValidator
from backtest_engine import BacktestEngine
from optimizer import StrategyOptimizer, parameter_sensitivity_analysis
from visualization import PerformanceVisualizer

# Configuración global
CONFIG = {
    'data': {
        'file_path': 'Binance_BTCUSDT_1h.csv',
        'train_ratio': 0.6,
        'test_ratio': 0.2,
        'validation_ratio': 0.2
    },
    'trading': {
        'commission': 0.00125,  # 0.125%
        'initial_capital': 100000,
        'position_size': 1.0,
        'allow_short': True,
        'leverage': 1
    },
    'indicators': {
        'sma_short': 20,
        'sma_long': 50,
        'ema_short': 12,
        'ema_long': 26,
        'stoch_window': 14,
        'stoch_smooth': 3,
        'atr_window': 14,
        'rsi_window': 14
    },
    'risk_management': {
        'stop_loss_atr_multiplier': 2.0,
        'take_profit_atr_multiplier': 3.0,
        'max_positions': 1,
        'risk_per_trade': 0.02
    },
    'optimization': {
        'n_trials': 100,
        'n_jobs': -1,
        'seed': 42,
        'cv_splits': 5
    }
}

def run_complete_analysis():
    """
    Ejecuta el análisis completo del proyecto
    """
    print("="*80)
    print("PROYECTO DE TRADING SISTEMÁTICO BTC/USDT")
    print("="*80)
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # ========================
    # 1. CARGA DE DATOS
    # ========================
    print("\n" + "="*50)
    print("FASE 1: CARGA Y PREPARACIÓN DE DATOS")
    print("="*50)
    
    data_loader = DataLoader(CONFIG)
    df = data_loader.load_data(CONFIG['data']['file_path'])
    
    # Dividir datos
    train_data, test_data, val_data = data_loader.split_data()
    
    # Estadísticas básicas
    stats = data_loader.get_price_statistics(df)
    print(f"\n📊 Estadísticas del Dataset Completo:")
    print(f"  - Precio mínimo: ${stats['price_stats']['min']:,.2f}")
    print(f"  - Precio máximo: ${stats['price_stats']['max']:,.2f}")
    print(f"  - Precio promedio: ${stats['price_stats']['mean']:,.2f}")
    print(f"  - Volatilidad: {stats['price_stats']['std']:,.2f}")
    
    # ========================
    # 2. OPTIMIZACIÓN
    # ========================
    print("\n" + "="*50)
    print("FASE 2: OPTIMIZACIÓN DE HIPERPARÁMETROS")
    print("="*50)
    
    optimizer = StrategyOptimizer(CONFIG)
    
    # Optimizar con subset para desarrollo rápido
    print("\n🔧 Optimizando estrategia con Optuna...")
    print("   (Usando subset de datos para desarrollo)")
    
    # Para desarrollo usar menos datos y trials
    train_subset = train_data[:10000]  # Primeros 10,000 registros
    best_params = optimizer.optimize(
        train_subset,
        n_trials=50,  # Reducido para desarrollo
        use_cv=True,
        seed=42
    )
    
    # Obtener reporte de optimización
    opt_report = optimizer.get_optimization_report()
    
    # ========================
    # 3. BACKTEST CON MEJORES PARÁMETROS
    # ========================
    print("\n" + "="*50)
    print("FASE 3: BACKTEST CON PARÁMETROS OPTIMIZADOS")
    print("="*50)
    
    # Actualizar configuración con mejores parámetros
    optimized_config = CONFIG.copy()
    for key, value in best_params.items():
        if key in ['sma_short', 'sma_long', 'ema_short', 'ema_long', 
                  'stoch_window', 'stoch_smooth', 'atr_window', 'rsi_window']:
            optimized_config['indicators'][key] = value
        elif 'atr' in key:
            optimized_config['risk_management'][key + '_multiplier'] = value
        elif key == 'risk_per_trade':
            optimized_config['risk_management'][key] = value
    
    # Crear motor de backtest con configuración optimizada
    backtest_engine = BacktestEngine(optimized_config)
    
    # Ejecutar backtest en TRAIN
    print("\n📈 Ejecutando Backtest en datos de ENTRENAMIENTO...")
    train_results = backtest_engine.run_backtest(train_data, verbose=True)
    
    # Ejecutar backtest en TEST
    print("\n📈 Ejecutando Backtest en datos de PRUEBA...")
    test_engine = BacktestEngine(optimized_config)
    test_results = test_engine.run_backtest(test_data, verbose=True)
    
    # Ejecutar backtest en VALIDATION
    print("\n📈 Ejecutando Backtest en datos de VALIDACIÓN...")
    val_engine = BacktestEngine(optimized_config)
    val_results = val_engine.run_backtest(val_data, verbose=True)
    
    # ========================
    # 4. ANÁLISIS DE ROBUSTEZ
    # ========================
    print("\n" + "="*50)
    print("FASE 4: ANÁLISIS DE ROBUSTEZ")
    print("="*50)
    
    # Walk-Forward Analysis (simplificado para desarrollo)
    print("\n🔄 Ejecutando Walk-Forward Analysis...")
    wf_results = optimizer.walk_forward_analysis(
        train_data[:20000],  # Subset para desarrollo
        window_size=0.7,
        step_size=0.15
    )
    
    # Análisis de sensibilidad de parámetros clave
    print("\n🎯 Análisis de Sensibilidad de Parámetros...")
    
    # SMA Short sensitivity
    sma_sensitivity = parameter_sensitivity_analysis(
        optimizer,
        train_subset,
        'sma_short',
        range(15, 35, 5)
    )
    
    print(f"  - Sensibilidad SMA Short:")
    print(f"    Mejor valor: {sma_sensitivity.loc[sma_sensitivity['calmar_ratio'].idxmax(), 'value']}")
    print(f"    Calmar range: [{sma_sensitivity['calmar_ratio'].min():.2f}, {sma_sensitivity['calmar_ratio'].max():.2f}]")
    
    # ========================
    # 5. VISUALIZACIÓN
    # ========================
    print("\n" + "="*50)
    print("FASE 5: GENERACIÓN DE VISUALIZACIONES Y REPORTES")
    print("="*50)
    
    # Crear directorio para reportes si no existe
    if not os.path.exists('reports'):
        os.makedirs('reports')
    
    # Visualizador para resultados de train
    print("\n📊 Generando visualizaciones...")
    
    train_visualizer = PerformanceVisualizer(train_results, optimized_config)
    test_visualizer = PerformanceVisualizer(test_results, optimized_config)
    val_visualizer = PerformanceVisualizer(val_results, optimized_config)
    
    # Dashboard principal para cada conjunto
    train_dashboard = train_visualizer.create_performance_dashboard('reports/train_dashboard.png')
    test_dashboard = test_visualizer.create_performance_dashboard('reports/test_dashboard.png')
    val_dashboard = val_visualizer.create_performance_dashboard('reports/val_dashboard.png')
    
    # Análisis de trades
    train_trade_analysis = train_visualizer.create_trade_analysis('reports/train_trades.png')
    
    # Reporte de optimización
    if optimizer.optimization_history:
        opt_viz = train_visualizer.create_optimization_report(
            optimizer.optimization_history,
            'reports/optimization_report.png'
        )
    
    # Generar tablas de rendimiento
    train_tables = train_visualizer.generate_performance_tables()
    test_tables = test_visualizer.generate_performance_tables()
    val_tables = val_visualizer.generate_performance_tables()
    
    # Generar reporte HTML
    train_visualizer.generate_html_report('reports/performance_report.html')
    
    # ========================
    # 6. RESUMEN EJECUTIVO
    # ========================
    print("\n" + "="*50)
    print("RESUMEN EJECUTIVO")
    print("="*50)
    
    print("\n📋 ESTRATEGIA IMPLEMENTADA:")
    print(f"  - Confirmación de señal: 2 de 3 indicadores")
    print(f"  - Indicadores principales: SMA, EMA, Stochastic")
    print(f"  - Gestión de riesgo: ATR-based SL/TP")
    print(f"  - Posiciones: Largas y Cortas (sin apalancamiento)")
    
    print("\n🎯 PARÁMETROS OPTIMIZADOS:")
    for param, value in best_params.items():
        print(f"  - {param}: {value}")
    
    print("\n📊 MÉTRICAS DE PERFORMANCE:")
    
    print("\n  TRAIN SET:")
    print(f"    - Retorno Total: {train_results['performance_metrics']['total_return']:.2f}%")
    print(f"    - Calmar Ratio: {train_results['performance_metrics']['calmar_ratio']:.3f}")
    print(f"    - Sharpe Ratio: {train_results['performance_metrics']['sharpe_ratio']:.3f}")
    print(f"    - Max Drawdown: {train_results['performance_metrics']['max_drawdown']:.2f}%")
    print(f"    - Win Rate: {train_results['performance_metrics']['win_rate']:.1f}%")
    
    print("\n  TEST SET:")
    print(f"    - Retorno Total: {test_results['performance_metrics']['total_return']:.2f}%")
    print(f"    - Calmar Ratio: {test_results['performance_metrics']['calmar_ratio']:.3f}")
    print(f"    - Sharpe Ratio: {test_results['performance_metrics']['sharpe_ratio']:.3f}")
    print(f"    - Max Drawdown: {test_results['performance_metrics']['max_drawdown']:.2f}%")
    print(f"    - Win Rate: {test_results['performance_metrics']['win_rate']:.1f}%")
    
    print("\n  VALIDATION SET:")
    print(f"    - Retorno Total: {val_results['performance_metrics']['total_return']:.2f}%")
    print(f"    - Calmar Ratio: {val_results['performance_metrics']['calmar_ratio']:.3f}")
    print(f"    - Sharpe Ratio: {val_results['performance_metrics']['sharpe_ratio']:.3f}")
    print(f"    - Max Drawdown: {val_results['performance_metrics']['max_drawdown']:.2f}%")
    print(f"    - Win Rate: {val_results['performance_metrics']['win_rate']:.1f}%")
    
    print("\n🔍 ANÁLISIS DE ROBUSTEZ:")
    print(f"  - Walk-Forward Calmar promedio: {wf_results['avg_test_calmar']:.3f}")
    print(f"  - Ratio de estabilidad: {wf_results['stability_ratio']:.3f}")
    
    # ========================
    # 7. GUARDAR RESULTADOS
    # ========================
    print("\n" + "="*50)
    print("GUARDANDO RESULTADOS")
    print("="*50)
    
    # Guardar configuración y mejores parámetros
    results_summary = {
        'config': CONFIG,
        'optimized_params': best_params,
        'performance': {
            'train': {
                'total_return': train_results['performance_metrics']['total_return'],
                'calmar_ratio': train_results['performance_metrics']['calmar_ratio'],
                'sharpe_ratio': train_results['performance_metrics']['sharpe_ratio'],
                'max_drawdown': train_results['performance