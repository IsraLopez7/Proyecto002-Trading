"""
main.py
Programa principal del sistema de trading
Ejecuta el pipeline completo: carga datos, optimiza, backtest y genera reportes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Importar módulos del proyecto
import data_loader
import indicators
import backtest
import metrics
import optimization
import visualization as report

def print_detailed_metrics(metrics_dict, title="MÉTRICAS"):
    """
    Imprime métricas detalladas en la consola
    """
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print('='*60)
    print(f"Valor Inicial:        ${metrics_dict.get('initial_value', 0):>15,.2f}")
    print(f"Valor Final:          ${metrics_dict.get('final_value', 0):>15,.2f}")
    print(f"Retorno Total:         {metrics_dict.get('total_return', 0):>15.2f}%")
    print(f"Sharpe Ratio:          {metrics_dict.get('sharpe_ratio', 0):>15.3f}")
    print(f"Sortino Ratio:         {metrics_dict.get('sortino_ratio', 0):>15.3f}")
    print(f"Calmar Ratio:          {metrics_dict.get('calmar_ratio', 0):>15.3f}")
    print(f"Max Drawdown:          {metrics_dict.get('max_drawdown', 0):>15.2f}%")
    print(f"Volatilidad Anual:     {metrics_dict.get('volatility', 0):>15.2f}%")
    
    if 'total_trades' in metrics_dict:
        print(f"\nTotal Operaciones:     {int(metrics_dict.get('total_trades', 0)):>15}")
        print(f"Tasa de Éxito:         {metrics_dict.get('win_rate', 0):>15.2f}%")
        print(f"Factor de Beneficio:   {metrics_dict.get('profit_factor', 0):>15.2f}")
        print(f"Ganancia Promedio:     ${metrics_dict.get('avg_win', 0):>14,.2f}")
        print(f"Pérdida Promedio:      ${metrics_dict.get('avg_loss', 0):>14,.2f}")

def print_trades_summary(trades):
    """
    Imprime resumen de operaciones en la consola
    """
    if not trades:
        print("\nNo hay operaciones cerradas")
        return
    
    df_trades = pd.DataFrame(trades)
    
    print(f"\n{'='*60}")
    print(f"{'RESUMEN DE OPERACIONES':^60}")
    print('='*60)
    
    # Operaciones por tipo
    if 'type' in df_trades.columns:
        long_trades = df_trades[df_trades['type'] == 'long']
        short_trades = df_trades[df_trades['type'] == 'short']
        
        print(f"\nTotal de operaciones: {len(trades)}")
        print(f"\nOperaciones Largas: {len(long_trades)}")
        if len(long_trades) > 0:
            print(f"  - PnL Total:     ${long_trades['pnl'].sum():>12,.2f}")
            print(f"  - PnL Promedio:  ${long_trades['pnl'].mean():>12,.2f}")
            print(f"  - Ganadoras:     {len(long_trades[long_trades['pnl'] > 0]):>12}")
            print(f"  - Perdedoras:    {len(long_trades[long_trades['pnl'] < 0]):>12}")
        
        print(f"\nOperaciones Cortas: {len(short_trades)}")
        if len(short_trades) > 0:
            print(f"  - PnL Total:     ${short_trades['pnl'].sum():>12,.2f}")
            print(f"  - PnL Promedio:  ${short_trades['pnl'].mean():>12,.2f}")
            print(f"  - Ganadoras:     {len(short_trades[short_trades['pnl'] > 0]):>12}")
            print(f"  - Perdedoras:    {len(short_trades[short_trades['pnl'] < 0]):>12}")
        
        # Razón de cierre
        if 'reason' in df_trades.columns:
            print(f"\nRazón de cierre:")
            print(f"  - Stop Loss:     {len(df_trades[df_trades['reason'] == 'stop_loss']):>12}")
            print(f"  - Take Profit:   {len(df_trades[df_trades['reason'] == 'take_profit']):>12}")

def print_parameter_importance(study):
    """
    Imprime la importancia de parámetros
    """
    try:
        importance = optimization.get_parameter_importance(study)
        if importance:
            print(f"\n{'='*60}")
            print(f"{'IMPORTANCIA DE PARÁMETROS':^60}")
            print('='*60)
            
            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            for param, score in sorted_importance:
                bar = '█' * int(score * 40)
                print(f"{param:20s}: {bar} {score:.3f}")
    except:
        pass

def print_comparison_table(metrics_test, metrics_val, metrics_complete):
    """
    Imprime tabla comparativa de métricas
    """
    print(f"\n{'='*80}")
    print(f"{'COMPARACIÓN DE RENDIMIENTO POR CONJUNTO':^80}")
    print('='*80)
    print(f"{'Métrica':<20} {'Test':>15} {'Validation':>15} {'Completo':>15}")
    print('-'*80)
    
    # Retorno
    print(f"{'Retorno Total':<20} {metrics_test['total_return']:>14.2f}% "
          f"{metrics_val['total_return']:>14.2f}% "
          f"{metrics_complete['total_return']:>14.2f}%")
    
    # Sharpe
    print(f"{'Sharpe Ratio':<20} {metrics_test['sharpe_ratio']:>15.3f} "
          f"{metrics_val['sharpe_ratio']:>15.3f} "
          f"{metrics_complete['sharpe_ratio']:>15.3f}")
    
    # Calmar
    print(f"{'Calmar Ratio':<20} {metrics_test['calmar_ratio']:>15.3f} "
          f"{metrics_val['calmar_ratio']:>15.3f} "
          f"{metrics_complete['calmar_ratio']:>15.3f}")
    
    # Max DD
    print(f"{'Max Drawdown':<20} {metrics_test['max_drawdown']:>14.2f}% "
          f"{metrics_val['max_drawdown']:>14.2f}% "
          f"{metrics_complete['max_drawdown']:>14.2f}%")
    
    # Win Rate
    if 'win_rate' in metrics_test:
        print(f"{'Win Rate':<20} {metrics_test.get('win_rate', 0):>14.2f}% "
              f"{metrics_val.get('win_rate', 0):>14.2f}% "
              f"{metrics_complete.get('win_rate', 0):>14.2f}%")

def run_trading_system(config=None):
    """
    Ejecuta el sistema completo de trading
    """
    print("="*80)
    print("SISTEMA DE TRADING ALGORÍTMICO - BTCUSDT")
    print("="*80)
    
    # Configuración por defecto
    if config is None:
        config = {
            'data_file': 'Binance_BTCUSDT_1h.csv',
            'initial_cash': 100000,
            'commission': 0.00125,
            'train_ratio': 0.6,
            'test_ratio': 0.2,
            'val_ratio': 0.2,
            'optimization_trials': 100,
            'cv_splits': 5
        }
    
    # ========== PASO 1: CARGAR Y PREPARAR DATOS ==========
    print("\n[1/6] Cargando datos...")
    df = data_loader.load_data(config['data_file'])
    df = data_loader.add_returns(df)
    df = data_loader.get_price_features(df)
    print(f"   ✓ {len(df)} registros cargados")
    print(f"   ✓ Periodo: {df['date'].min()} a {df['date'].max()}")
    
    # ========== PASO 2: DIVIDIR DATOS ==========
    print("\n[2/6] Dividiendo datos...")
    train_df, test_df, val_df = data_loader.split_data(
        df, 
        config['train_ratio'], 
        config['test_ratio'], 
        config['val_ratio']
    )
    print(f"   ✓ Train: {len(train_df)} | Test: {len(test_df)} | Val: {len(val_df)}")
    
    # ========== PASO 3: OPTIMIZAR PARÁMETROS ==========
    print(f"\n[3/6] Optimizando parámetros ({config['optimization_trials']} trials)...")
    study = optimization.optimize_strategy(
        train_df,
        n_trials=config['optimization_trials'],
        n_splits=config['cv_splits']
    )
    best_params = study.best_params
    print(f"   ✓ Mejor Calmar Ratio en Train: {study.best_value:.3f}")
    
    # Imprimir mejores parámetros
    print(f"\n{'='*60}")
    print(f"{'MEJORES PARÁMETROS ENCONTRADOS':^60}")
    print('='*60)
    for param, value in best_params.items():
        if isinstance(value, float):
            print(f"{param:20s}: {value:.4f}")
        else:
            print(f"{param:20s}: {value}")
    
    # Importancia de parámetros
    print_parameter_importance(study)

    # ========== PASO 4: VALIDAR EN TEST ==========
    print("\n[4/6] Validando en conjunto de test...")
    test_df_signals = indicators.create_signals(test_df.copy(), best_params)
    
    bt_params_test = {
        'initial_cash': config['initial_cash'],
        'commission'  : config['commission'],
        'exit_on_opposite': True,

        # ——— Gestión de riesgo (clave) ———
        'size_mode'      : 'risk',   # ← fija
        'risk_per_trade' : 0.0035,   # 0.35% por trade (ajusta 0.2–0.6%)

        # Stops dinámicos
        'use_atr_stops'  : True,
        'atr_window'     : best_params.get('atr_window', 14),
        'atr_sl_mult'    : max(1.6, best_params.get('atr_sl_mult', 1.8)),
        'atr_tp_mult'    : max(2.5, best_params.get('atr_tp_mult', 3.0)),

        # Filtros de calidad de señal
        'persistence'    : best_params.get('persistence', 1),
        'cooldown_bars'  : max(2, best_params.get('cooldown_bars', 2)),
        'max_hold_bars'  : best_params.get('max_hold_bars', 48),

        # Respeta el estudio (no lo contradigas)
        'allow_shorts'   : best_params.get('allow_shorts', False),

        # Fallbacks si ATR no está (se usan poco)
        'stop_loss'      : max(0.02, best_params.get('stop_loss', 0.02)),
        'take_profit'    : max(0.03, best_params.get('take_profit', 0.04)),

        # Ignorado si size_mode='risk', pero lo dejamos por compatibilidad
        'n_shares'       : min(0.1, best_params.get('n_shares', 0.1)),
        # Opcional: añade slippage realista para evitar scalping
        'slippage_bps'   : 3,  # 0.03%
    }

    
    portfolio_hist_test, trades_test = backtest.backtest(test_df_signals, bt_params_test)
    metrics_test = metrics.calculate_all_metrics(pd.Series(portfolio_hist_test), trades_test)
    
    print_detailed_metrics(metrics_test, "MÉTRICAS EN CONJUNTO DE TEST")
    
    # ========== PASO 5: EVALUACIÓN FINAL EN VALIDATION ==========
    print("\n[5/6] Evaluando en conjunto de validación...")
    val_df_signals = indicators.create_signals(val_df.copy(), best_params)
    
    portfolio_hist_val, trades_val = backtest.backtest(val_df_signals, bt_params_test.copy())
    metrics_val = metrics.calculate_all_metrics(pd.Series(portfolio_hist_val), trades_val)
    
    print_detailed_metrics(metrics_val, "MÉTRICAS EN CONJUNTO DE VALIDACIÓN")
    
    # ========== PASO 6: BACKTEST COMPLETO ==========
    print("\n[6/6] Ejecutando backtest en dataset completo...")
    df_complete_signals = indicators.create_signals(df.copy(), best_params)
    
    portfolio_hist_complete, trades_complete = backtest.backtest(
        df_complete_signals, 
        bt_params_test
    )
    
    metrics_complete = metrics.calculate_all_metrics(
        pd.Series(portfolio_hist_complete),
        trades_complete
    )
        # ====== REPORTES ======
    try:
        # Dashboard principal
        fig_dash = report.create_comprehensive_report(
        df_complete_signals,
        portfolio_hist_complete,
        trades_complete,
        metrics_complete,
        best_params,
        save_path=None  # no guardamos
        )
        plt.show()  # <- muestra la figura

        # Distribución de retornos
        fig_dist = report.create_distribution_analysis(
            portfolio_hist_complete,
            trades_complete
        )
        plt.show()

        # Tabla de retornos mensuales (heatmap o tabla)
        fig_heat = report.create_monthly_performance_table(
            portfolio_hist_complete,
            dates=df_complete_signals['date'].values
        )
        plt.show()

        print("\nListo: se mostraron las 3 gráficas en ventanas interactivas.")
    except Exception as e:
        print(f"\n⚠️  No se pudieron generar/mostrar los reportes gráficos: {e}")


    print_detailed_metrics(metrics_complete, "MÉTRICAS EN DATASET COMPLETO")
    
    # Resumen de trades
    print_trades_summary(trades_complete)
    
    # Tabla comparativa
    print_comparison_table(metrics_test, metrics_val, metrics_complete)
    
    # ========== ANÁLISIS FINAL ==========
    print(f"\n{'='*80}")
    print(f"{'ANÁLISIS Y CONCLUSIONES':^80}")
    print('='*80)
    
    # Análisis de resultados
    print("\nEvaluación de la Estrategia:")
    if metrics_complete['calmar_ratio'] > 1:
        print("  ✅ Buen ratio riesgo/retorno (Calmar > 1)")
    else:
        print("  ⚠️  El ratio riesgo/retorno necesita mejoras (Calmar < 1)")
    
    if metrics_complete['sharpe_ratio'] > 1:
        print("  ✅ Sharpe Ratio sólido (> 1)")
    elif metrics_complete['sharpe_ratio'] > 0:
        print("  ⚠️  Sharpe Ratio positivo pero mejorable")
    else:
        print("  ❌ Sharpe Ratio negativo")
    
    if 'win_rate' in metrics_complete and metrics_complete['win_rate'] > 50:
        print(f"  ✅ Tasa de éxito superior al 50% ({metrics_complete['win_rate']:.2f}%)")
    else:
        print(f"  ⚠️  Tasa de éxito inferior al 50%")
    
    # Consistencia
    test_val_diff = abs(metrics_test['total_return'] - metrics_val['total_return'])
    if test_val_diff < 10:
        print("  ✅ Resultados consistentes entre test y validación")
    else:
        print(f"  ⚠️  Diferencia de {test_val_diff:.2f}% entre test y validación")
    
    # Recomendaciones
    print("\nRecomendaciones:")
    print("  • Monitorear el rendimiento en tiempo real antes de usar capital real")
    print("  • Considerar costos adicionales como slippage y spreads")
    print("  • Implementar gestión de riesgo adicional")
    print("  • Reoptimizar parámetros periódicamente")
    
    print("\n" + "="*80)
    
    # Retornar resultados principales
    return {
        'best_params': best_params,
        'metrics_test': metrics_test,
        'metrics_val': metrics_val,
        'metrics_complete': metrics_complete,
        'portfolio_hist': portfolio_hist_complete,
        'trades': trades_complete,
        'study': study
    }

if __name__ == "__main__":
    """
    Punto de entrada principal - Ejecuta todo el sistema automáticamente
    """
    
    print("\n" + "="*80)
    print("SISTEMA DE TRADING ALGORÍTMICO")
    print("Bitcoin (BTC/USDT) - Análisis Técnico")
    print("="*80)
    
    # Configuración
    config = {
        'data_file': 'Binance_BTCUSDT_1h.csv',
        'initial_cash': 100000,
        'commission': 0.00125,  # 0.125%
        'train_ratio': 0.6,
        'test_ratio': 0.2,
        'val_ratio': 0.2,
        'optimization_trials': 150,
        'cv_splits': 5
    }
    
    # Ejecutar sistema completo automáticamente
    results = run_trading_system(config)
    
    print("\n✅ Sistema ejecutado exitosamente!")
    print("="*80)