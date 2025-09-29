
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
    
    # Importar módulos
    from data_loader import DataLoader
    from indicators import TechnicalIndicators
    from strategy import TradingStrategy, StrategyValidator
    from backtest_engine import BacktestEngine
    from optimizer import StrategyOptimizer, parameter_sensitivity_analysis
    from visualization import PerformanceVisualizer
    
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
                'max_drawdown': train_results['performance_metrics']['max_drawdown'],
                'win_rate': train_results['performance_metrics']['win_rate'],
                'total_trades': train_results['performance_metrics']['total_trades']
            },
            'test': {
                'total_return': test_results['performance_metrics']['total_return'],
                'calmar_ratio': test_results['performance_metrics']['calmar_ratio'],
                'sharpe_ratio': test_results['performance_metrics']['sharpe_ratio'],
                'max_drawdown': test_results['performance_metrics']['max_drawdown'],
                'win_rate': test_results['performance_metrics']['win_rate'],
                'total_trades': test_results['performance_metrics']['total_trades']
            },
            'validation': {
                'total_return': val_results['performance_metrics']['total_return'],
                'calmar_ratio': val_results['performance_metrics']['calmar_ratio'],
                'sharpe_ratio': val_results['performance_metrics']['sharpe_ratio'],
                'max_drawdown': val_results['performance_metrics']['max_drawdown'],
                'win_rate': val_results['performance_metrics']['win_rate'],
                'total_trades': val_results['performance_metrics']['total_trades']
            }
        },
        'walk_forward': {
            'avg_test_calmar': wf_results['avg_test_calmar'],
            'stability_ratio': wf_results['stability_ratio'],
            'n_windows': wf_results['n_windows']
        },
        'timestamp': datetime.now().isoformat()
    }
    
    # Guardar en JSON
    with open('reports/results_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=4)
    
    print("✅ Resultados guardados en reports/results_summary.json")
    
    # Guardar trades en CSV
    if train_results['trades']:
        train_trades_df = pd.DataFrame(train_results['trades'])
        train_trades_df.to_csv('reports/train_trades.csv', index=False)
        print("✅ Trades de entrenamiento guardados en reports/train_trades.csv")
    
    if test_results['trades']:
        test_trades_df = pd.DataFrame(test_results['trades'])
        test_trades_df.to_csv('reports/test_trades.csv', index=False)
        print("✅ Trades de prueba guardados en reports/test_trades.csv")
    
    # ========================
    # 8. CONCLUSIONES
    # ========================
    print("\n" + "="*50)
    print("CONCLUSIONES Y RECOMENDACIONES")
    print("="*50)
    
    # Evaluar performance
    avg_calmar = (train_results['performance_metrics']['calmar_ratio'] + 
                  test_results['performance_metrics']['calmar_ratio'] + 
                  val_results['performance_metrics']['calmar_ratio']) / 3
    
    avg_sharpe = (train_results['performance_metrics']['sharpe_ratio'] + 
                  test_results['performance_metrics']['sharpe_ratio'] + 
                  val_results['performance_metrics']['sharpe_ratio']) / 3
    
    print("\n📈 EVALUACIÓN DE LA ESTRATEGIA:")
    
    if avg_calmar > 1.0:
        print("  ✅ Calmar Ratio promedio > 1.0: Estrategia prometedora")
    elif avg_calmar > 0.5:
        print("  ⚠️ Calmar Ratio promedio entre 0.5-1.0: Estrategia aceptable pero mejorable")
    else:
        print("  ❌ Calmar Ratio promedio < 0.5: Estrategia necesita mejoras significativas")
    
    if avg_sharpe > 1.0:
        print("  ✅ Sharpe Ratio promedio > 1.0: Buen ratio riesgo-retorno")
    elif avg_sharpe > 0.5:
        print("  ⚠️ Sharpe Ratio promedio entre 0.5-1.0: Ratio riesgo-retorno moderado")
    else:
        print("  ❌ Sharpe Ratio promedio < 0.5: Ratio riesgo-retorno insuficiente")
    
    # Verificar sobreajuste
    train_calmar = train_results['performance_metrics']['calmar_ratio']
    test_calmar = test_results['performance_metrics']['calmar_ratio']
    val_calmar = val_results['performance_metrics']['calmar_ratio']
    
    calmar_degradation = ((train_calmar - test_calmar) / train_calmar) * 100 if train_calmar > 0 else 0
    
    print("\n🔍 ANÁLISIS DE SOBREAJUSTE:")
    if calmar_degradation < 20:
        print(f"  ✅ Degradación Train->Test: {calmar_degradation:.1f}% (Bajo sobreajuste)")
    elif calmar_degradation < 40:
        print(f"  ⚠️ Degradación Train->Test: {calmar_degradation:.1f}% (Sobreajuste moderado)")
    else:
        print(f"  ❌ Degradación Train->Test: {calmar_degradation:.1f}% (Alto sobreajuste)")
    
    print("\n💡 RECOMENDACIONES:")
    
    # Recomendaciones basadas en métricas
    if train_results['performance_metrics']['total_trades'] < 100:
        print("  • Considerar reducir los umbrales de señal para generar más trades")
    
    if train_results['performance_metrics']['win_rate'] < 40:
        print("  • Mejorar los filtros de entrada para aumentar el win rate")
    
    if train_results['performance_metrics']['max_drawdown'] > 30:
        print("  • Ajustar la gestión de riesgo para reducir el drawdown máximo")
    
    if wf_results['stability_ratio'] < 1.0:
        print("  • La estrategia muestra inestabilidad en diferentes períodos")
        print("  • Considerar usar parámetros más robustos o adaptativos")
    
    if avg_calmar < 1.0:
        print("  • Explorar indicadores adicionales o diferentes combinaciones")
        print("  • Considerar técnicas de machine learning para mejorar señales")
    
    print("\n" + "="*80)
    print(f"Análisis completado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    return {
        'train_results': train_results,
        'test_results': test_results,
        'val_results': val_results,
        'optimized_config': optimized_config,
        'optimization_report': opt_report,
        'walk_forward_results': wf_results
    }


def quick_test():
    """
    Función de prueba rápida con datos reducidos
    """
    from data_loader import DataLoader
    from backtest_engine import BacktestEngine
    from visualization import PerformanceVisualizer
    
    print("\n🚀 MODO DE PRUEBA RÁPIDA")
    print("="*50)
    
    # Configuración reducida para pruebas
    test_config = CONFIG.copy()
    test_config['optimization']['n_trials'] = 10
    
    # Cargar datos
    data_loader = DataLoader(test_config)
    df = data_loader.load_data(test_config['data']['file_path'])
    
    # Usar solo primeros 5000 registros
    df_subset = df[:5000]
    
    # Backtest rápido
    engine = BacktestEngine(test_config)
    results = engine.run_backtest(df_subset, verbose=True)
    
    # Visualización rápida
    visualizer = PerformanceVisualizer(results, test_config)
    dashboard = visualizer.create_performance_dashboard()
    
    plt.show()
    
    return results


def analyze_single_parameter(param_name, param_range):
    """
    Analiza el impacto de un solo parámetro
    
    Parameters:
    -----------
    param_name : str
        Nombre del parámetro a analizar
    param_range : list
        Rango de valores a probar
    """
    from data_loader import DataLoader
    from optimizer import StrategyOptimizer, parameter_sensitivity_analysis
    
    print(f"\n📊 Analizando parámetro: {param_name}")
    print("="*50)
    
    # Cargar datos
    data_loader = DataLoader(CONFIG)
    df = data_loader.load_data(CONFIG['data']['file_path'])
    train_data, _, _ = data_loader.split_data()
    
    # Usar subset para análisis rápido
    train_subset = train_data[:10000]
    
    # Crear optimizador
    optimizer = StrategyOptimizer(CONFIG)
    
    # Análisis de sensibilidad
    sensitivity_df = parameter_sensitivity_analysis(
        optimizer,
        train_subset,
        param_name,
        param_range
    )
    
    # Graficar resultados
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Calmar Ratio
    axes[0, 0].plot(sensitivity_df['value'], sensitivity_df['calmar_ratio'], 'o-')
    axes[0, 0].set_title(f'Calmar Ratio vs {param_name}')
    axes[0, 0].set_xlabel(param_name)
    axes[0, 0].set_ylabel('Calmar Ratio')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Sharpe Ratio
    axes[0, 1].plot(sensitivity_df['value'], sensitivity_df['sharpe_ratio'], 'o-', color='orange')
    axes[0, 1].set_title(f'Sharpe Ratio vs {param_name}')
    axes[0, 1].set_xlabel(param_name)
    axes[0, 1].set_ylabel('Sharpe Ratio')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Total Return
    axes[1, 0].plot(sensitivity_df['value'], sensitivity_df['total_return'], 'o-', color='green')
    axes[1, 0].set_title(f'Total Return vs {param_name}')
    axes[1, 0].set_xlabel(param_name)
    axes[1, 0].set_ylabel('Total Return (%)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Max Drawdown
    axes[1, 1].plot(sensitivity_df['value'], sensitivity_df['max_drawdown'], 'o-', color='red')
    axes[1, 1].set_title(f'Max Drawdown vs {param_name}')
    axes[1, 1].set_xlabel(param_name)
    axes[1, 1].set_ylabel('Max Drawdown (%)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Sensitivity Analysis: {param_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Encontrar valor óptimo
    optimal_idx = sensitivity_df['calmar_ratio'].idxmax()
    optimal_value = sensitivity_df.loc[optimal_idx, 'value']
    optimal_calmar = sensitivity_df.loc[optimal_idx, 'calmar_ratio']
    
    print(f"\n✅ Valor óptimo de {param_name}: {optimal_value}")
    print(f"   Calmar Ratio: {optimal_calmar:.3f}")
    
    return sensitivity_df


def main():
    """
    Función principal con menú interactivo
    """
    print("\n" + "="*80)
    print("SISTEMA DE TRADING ALGORÍTMICO BTC/USDT")
    print("="*80)
    print("\nOpciones disponibles:")
    print("1. Ejecutar análisis completo")
    print("2. Prueba rápida")
    print("3. Analizar parámetro específico")
    print("4. Salir")
    
    choice = input("\nSeleccione opción (1-4): ")
    
    if choice == "1":
        try:
            results = run_complete_analysis()
            print("\n✅ Análisis completo finalizado")
            print("📁 Reportes guardados en carpeta 'reports/'")
        except Exception as e:
            print(f"\n❌ Error durante el análisis: {str(e)}")
            print("Asegúrese de que todos los módulos estén disponibles")
        
    elif choice == "2":
        try:
            results = quick_test()
            print("\n✅ Prueba rápida completada")
        except Exception as e:
            print(f"\n❌ Error durante la prueba: {str(e)}")
        
    elif choice == "3":
        print("\nParámetros disponibles:")
        print("- sma_short (10-30)")
        print("- sma_long (40-80)")
        print("- ema_short (8-20)")
        print("- ema_long (21-40)")
        print("- stoch_window (10-20)")
        print("- atr_window (10-20)")
        print("- stop_loss_atr_multiplier (1.5-3.0)")
        print("- take_profit_atr_multiplier (2.0-5.0)")
        
        param = input("\nIngrese nombre del parámetro: ")
        range_str = input("Ingrese rango (ej: 10,15,20,25,30): ")
        
        try:
            param_range = [float(x) for x in range_str.split(',')]
            sensitivity_df = analyze_single_parameter(param, param_range)
            print("\n✅ Análisis de sensibilidad completado")
        except Exception as e:
            print(f"\n❌ Error durante el análisis: {str(e)}")
        
    else:
        print("\n👋 Saliendo del sistema...")
    
    print("\n" + "="*80)
    print("FIN DEL PROGRAMA")
    print("="*80)


if __name__ == "__main__":
    main()