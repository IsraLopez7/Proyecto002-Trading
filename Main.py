
"""
Script Principal de Ejecución
Orquesta todo el flujo del proyecto de trading sistemático
"""

import warnings
warnings.filterwarnings('ignore')

from data_loader import DataLoader
from indicators import TechnicalIndicators
from strategy import TradingStrategy
from metrics import PerformanceMetrics
from visualization import Visualizer

def main():
    """Función principal - ejecuta el análisis completo"""
    
    print("\n" + "="*60)
    print("SISTEMA DE TRADING BTC/USDT - ANÁLISIS COMPLETO")
    print("="*60)
    
    # 1. CARGAR DATOS
    print("\n[1/5] Cargando datos...")
    loader = DataLoader('Binance_BTCUSDT_1h.csv')
    df = loader.load_and_prepare()
    print(f"✓ {len(df)} registros cargados")
    
    # 2. DIVIDIR DATOS
    print("\n[2/5] Dividiendo datos...")
    train_data, test_data, val_data = loader.split_data()
    print(f"✓ Train: {len(train_data)} | Test: {len(test_data)} | Validation: {len(val_data)}")
    
    # 3. CALCULAR INDICADORES Y SEÑALES
    print("\n[3/5] Calculando indicadores y señales...")
    indicators = TechnicalIndicators()
    
    # Procesar cada conjunto
    datasets = {
        'TRAIN': train_data,
        'TEST': test_data,
        'VALIDATION': val_data
    }
    
    for name, data in datasets.items():
        # Calcular indicadores
        data = indicators.calculate_all(data)
        # Generar señales
        data = indicators.generate_signals(data)
        datasets[name] = data
    
    print("✓ Indicadores calculados para todos los conjuntos")
    
    # 4. EJECUTAR BACKTESTS
    print("\n[4/5] Ejecutando backtests...")
    strategy = TradingStrategy()
    metrics_calc = PerformanceMetrics()
    results = {}
    
    for name, data in datasets.items():
        # Ejecutar backtest
        equity, trades = strategy.backtest(data)
        
        # Calcular métricas
        metrics = metrics_calc.calculate(equity, trades, strategy.initial_capital)
        
        # Guardar resultados
        results[name] = {
            'data': data,
            'equity': equity,
            'trades': trades,
            'metrics': metrics
        }
        
        # Imprimir resumen
        print(f"\n{name}:")
        print(f"  • Retorno: {metrics['retorno_total']:.2f}%")
        print(f"  • Sharpe: {metrics['sharpe_ratio']:.3f}")
        print(f"  • Calmar: {metrics['calmar_ratio']:.3f}")
        print(f"  • Trades: {metrics['total_trades']}")
    
    # 5. VISUALIZACIÓN
    print("\n[5/5] Generando visualizaciones...")
    visualizer = Visualizer()
    
    # Mostrar resultados del conjunto de validación
    val_results = results['VALIDATION']
    visualizer.plot_results(
        val_results['data'],
        val_results['equity'],
        val_results['trades'],
        val_results['metrics']
    )
    
    # Comparación entre conjuntos
    visualizer.plot_comparison(results)
    
    # RESUMEN FINAL
    print("\n" + "="*60)
    print("RESUMEN FINAL")
    print("="*60)
    
    # Análisis de performance
    train_metrics = results['TRAIN']['metrics']
    test_metrics = results['TEST']['metrics']
    val_metrics = results['VALIDATION']['metrics']
    
    # Evaluación
    avg_calmar = (train_metrics['calmar_ratio'] + test_metrics['calmar_ratio'] + val_metrics['calmar_ratio']) / 3
    avg_sharpe = (train_metrics['sharpe_ratio'] + test_metrics['sharpe_ratio'] + val_metrics['sharpe_ratio']) / 3
    
    print(f"\nMétricas Promedio:")
    print(f"  • Calmar Ratio: {avg_calmar:.3f}")
    print(f"  • Sharpe Ratio: {avg_sharpe:.3f}")
    
    # Análisis de sobreajuste
    calmar_degradation = ((train_metrics['calmar_ratio'] - test_metrics['calmar_ratio']) / 
                         max(abs(train_metrics['calmar_ratio']), 0.001)) * 100
    
    print(f"\nAnálisis de Sobreajuste:")
    print(f"  • Degradación Train->Test: {calmar_degradation:.1f}%")
    
    if calmar_degradation < 30:
        print("  • ✓ Bajo sobreajuste")
    elif calmar_degradation < 50:
        print("  • ⚠ Sobreajuste moderado")
    else:
        print("  • ✗ Alto sobreajuste")
    
    # Evaluación de la estrategia
    print(f"\nEvaluación de la Estrategia:")
    if avg_calmar > 1.0:
        print("  • ✓ Calmar > 1.0: Estrategia prometedora")
    elif avg_calmar > 0.5:
        print("  • ⚠ Calmar 0.5-1.0: Estrategia aceptable")
    else:
        print("  • ✗ Calmar < 0.5: Necesita mejoras")
    
    if avg_sharpe > 1.0:
        print("  • ✓ Sharpe > 1.0: Buen ratio riesgo-retorno")
    elif avg_sharpe > 0.5:
        print("  • ⚠ Sharpe 0.5-1.0: Ratio moderado")
    else:
        print("  • ✗ Sharpe < 0.5: Ratio insuficiente")
    
    # Análisis de trades por tipo
    print(f"\nAnálisis de Trades:")
    for name in ['TRAIN', 'TEST', 'VALIDATION']:
        trades = results[name]['trades']
        if trades:
            long_trades = [t for t in trades if t['type'] == 'LONG']
            short_trades = [t for t in trades if t['type'] == 'SHORT']
            
            print(f"\n{name}:")
            print(f"  • Total: {len(trades)} trades")
            print(f"  • Long: {len(long_trades)} | Short: {len(short_trades)}")
            
            if long_trades:
                long_wins = [t for t in long_trades if t['pnl_pct'] > 0]
                print(f"  • Long Win Rate: {(len(long_wins)/len(long_trades)*100):.1f}%")
            
            if short_trades:
                short_wins = [t for t in short_trades if t['pnl_pct'] > 0]
                print(f"  • Short Win Rate: {(len(short_wins)/len(short_trades)*100):.1f}%")
    
    print("\n" + "="*60)
    print("ANÁLISIS COMPLETADO")
    print("="*60)
    
    return results

if __name__ == "__main__":
    # Ejecutar análisis
    results = main()