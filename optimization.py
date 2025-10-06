"""
optimization.py
Módulo de optimización de parámetros usando Optuna
"""

import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

import data_loader
import indicators
import backtest
import metrics

def objective(trial, data, n_splits=5):
    data = data.copy()

    params = {
        # Indicadores
        'sma_short': trial.suggest_int('sma_short', 5, 20),
        'sma_long':  trial.suggest_int('sma_long', 21, 60),
        'ema_period': trial.suggest_int('ema_period', 10, 30),
        'rsi_window': trial.suggest_int('rsi_window', 10, 20),
        'rsi_oversold': trial.suggest_int('rsi_oversold', 20, 40),
        'rsi_overbought': trial.suggest_int('rsi_overbought', 60, 80),
        'stoch_window': trial.suggest_int('stoch_window', 10, 20),
        'stoch_oversold': trial.suggest_int('stoch_oversold', 15, 30),
        'stoch_overbought': trial.suggest_int('stoch_overbought', 70, 85),
        'atr_window': trial.suggest_int('atr_window', 10, 20),
        'bb_window':  trial.suggest_int('bb_window',  15, 25),
        'ema_trend': trial.suggest_categorical('ema_trend', [100, 200]),
        'trend_filter': trial.suggest_categorical('trend_filter', [True, False]),
        'persistence': trial.suggest_int('persistence', 1, 2),
        'min_conf': trial.suggest_int('min_conf', 1, 2),

        # Trading / ejecución
        'allow_shorts': trial.suggest_categorical('allow_shorts', [False, True]),
        'use_atr_stops': trial.suggest_categorical('use_atr_stops', [True, False]),
        'atr_sl_mult': trial.suggest_float('atr_sl_mult', 0.8, 2.2),
        'atr_tp_mult': trial.suggest_float('atr_tp_mult', 1.5, 4.0),
        'cooldown_bars': trial.suggest_int('cooldown_bars', 0, 6),
        'max_hold_bars': trial.suggest_int('max_hold_bars', 8, 72),

        # sizing
        'size_mode': trial.suggest_categorical('size_mode', ['risk', 'fraction']),
        'risk_per_trade': trial.suggest_float('risk_per_trade', 0.003, 0.01),
        'n_shares': trial.suggest_float('n_shares', 0.05, 0.30),

        # fallback SL/TP si no ATR
        'stop_loss': trial.suggest_float('stop_loss', 0.01, 0.05),
        'take_profit': trial.suggest_float('take_profit', 0.01, 0.06),
    }

    if params['sma_long'] <= params['sma_short']:   # sanity
        params['sma_long'] = params['sma_short'] + 5
    if params['rsi_oversold'] >= params['rsi_overbought']:
        return -1e3
    if params['stoch_oversold'] >= params['stoch_overbought']:
        return -1e3

    warmup = max(
        params['sma_long'], params['ema_period'],
        params['rsi_window'], params['stoch_window'],
        params['atr_window'], params['bb_window']
    ) + 5

    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits)

    scores = []
    for tr_idx, va_idx in tscv.split(data):
        if len(va_idx) < 200:
            continue
        val_start, val_end = int(va_idx[0]), int(va_idx[-1])
        start_wu = max(0, val_start - warmup)
        block = data.iloc[start_wu:val_end+1].copy()

        try:
            block = indicators.create_signals(block, params)
            val_block = block.iloc[(val_start - start_wu):].copy()

            bt_params = {
                'initial_cash': 100000, 'commission': 0.00125,
                'n_shares': params['n_shares'],
                'stop_loss': params['stop_loss'], 'take_profit': params['take_profit'],
                'allow_shorts': params['allow_shorts'],
                'exit_on_opposite': True,
                'use_atr_stops': params['use_atr_stops'],
                'atr_window': params['atr_window'],
                'atr_sl_mult': params['atr_sl_mult'], 'atr_tp_mult': params['atr_tp_mult'],
                'persistence': params['persistence'], 'cooldown_bars': params['cooldown_bars'],
                'max_hold_bars': params['max_hold_bars'],
                'size_mode': params['size_mode'], 'risk_per_trade': params['risk_per_trade'],
            }

            equity, trades = backtest.backtest(val_block, bt_params)
            if len(equity) < 5:
                scores.append(-100); continue

            eq = pd.Series(equity)
            calmar = metrics.calculate_calmar_ratio(eq)
            sharpe = metrics.calculate_sharpe_ratio(eq)
            totret = (eq.iloc[-1] / eq.iloc[0] - 1) * 100.0
            pf     = metrics.calculate_profit_factor(trades)
            ntr    = len(trades)

            # Acotar para que outliers no dominen
            calmar_c = np.clip(calmar, -1.0, 3.0)
            sharpe_c = np.clip(sharpe, -1.0, 3.0)
            ret_c    = np.clip(totret / 5.0, -2.0, 2.0)  # escalar

            base_score = 0.5*calmar_c + 0.3*sharpe_c + 0.2*ret_c

            # ——— PENALIZACIONES ———
            # 1) Trades: penaliza muy pocos (<30) y demasiados (>250)
            if ntr < 30:
                trade_pen = ntr / 30.0
            elif ntr > 250:
                trade_pen = 250.0 / ntr
            else:
                trade_pen = 1.0

            # 2) Profit Factor: si < 1, recorta (nunca por debajo de 0.5)
            pf_pen = 1.0 if (pf is not None and np.isfinite(pf) and pf >= 1.0) else max(0.5, pf if np.isfinite(pf) else 0.5)

            # 3) Si habilita cortos y el retorno total es negativo, castiga un poco
            short_pen = 0.9 if (params['allow_shorts'] and totret < 0) else 1.0

            score = base_score * trade_pen * pf_pen * short_pen
            scores.append(score)

        except Exception:
            scores.append(-100)

    valid = [s for s in scores if s > -100]
    return float(np.mean(valid)) if valid else -100.0

def optimize_strategy(train_data, n_trials=100, n_splits=5, seed=42):
    """
    Optimiza los parámetros de la estrategia usando Optuna
    
    Args:
        train_data: Datos de entrenamiento
        n_trials: Número de trials para Optuna
        n_splits: Número de splits para cross-validation
        seed: Semilla para reproducibilidad
    
    Returns:
        study: Objeto study de Optuna con los mejores parámetros
    """
    # Crear estudio de Optuna
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=seed)
    )
    
    # Función objetivo con datos fijos
    def objective_with_data(trial):
        return objective(trial, train_data, n_splits)
    
    # Optimizar
    study.optimize(
        objective_with_data, 
        n_trials=n_trials,
        show_progress_bar=False  # Desactivar barra de progreso
    )
    
    return study

def get_parameter_importance(study):
    """
    Analiza la importancia de cada parámetro
    """
    try:
        importance = optuna.importance.get_param_importances(study)
        return importance
    except:
        return None

def validate_best_params(study, test_data):
    """
    Valida los mejores parámetros en datos de test
    
    Args:
        study: Study de Optuna con mejores parámetros
        test_data: Datos de test
    
    Returns:
        portfolio_hist: Historial del portafolio
        trades: Lista de trades
        test_metrics: Métricas en test
    """
    # Obtener mejores parámetros
    best_params = study.best_params
    
    # Generar señales con mejores parámetros
    test_data_with_signals = indicators.create_signals(test_data.copy(), best_params)
    
    # Ejecutar backtest
    bt_params = {
        'initial_cash': 100000,
        'commission': 0.00125,
        'n_shares': best_params['n_shares'],
        'stop_loss': best_params['stop_loss'],
        'take_profit': best_params['take_profit']
    }
    
    portfolio_hist, trades = backtest.backtest(test_data_with_signals, bt_params)
    
    # Calcular métricas
    test_metrics = metrics.calculate_all_metrics(
        pd.Series(portfolio_hist), 
        trades
    )
    
    return portfolio_hist, trades, test_metrics

def grid_search_optimization(train_data, param_grid=None):
    """
    Búsqueda en grid alternativa (más simple pero exhaustiva)
    """
    if param_grid is None:
        param_grid = {
            'sma_short': [10, 15],
            'sma_long': [30, 40],
            'rsi_window': [14],
            'rsi_oversold': [30],
            'rsi_overbought': [70],
            'stoch_window': [14],
            'stoch_oversold': [20],
            'stoch_overbought': [80],
            'stop_loss': [0.02, 0.03],
            'take_profit': [0.03, 0.04],
            'n_shares': [0.1, 0.15]
        }
    
    best_calmar = -float('inf')
    best_params = None
    
    # Generar todas las combinaciones
    from itertools import product
    
    keys = param_grid.keys()
    values = param_grid.values()
    
    total_combinations = 1
    for v in values:
        total_combinations *= len(v)
    
    print(f"Evaluando {total_combinations} combinaciones...")
    
    for combination in product(*values):
        params = dict(zip(keys, combination))
        
        # Validar parámetros
        if params['sma_long'] <= params['sma_short']:
            continue
        if params['rsi_oversold'] >= params['rsi_overbought']:
            continue
        if params['stoch_oversold'] >= params['stoch_overbought']:
            continue
        
        try:
            # Generar señales
            data_with_signals = indicators.create_signals(train_data.copy(), params)
            
            # Backtest
            bt_params = {
                'initial_cash': 100000,
                'commission': 0.00125,
                'n_shares': params['n_shares'],
                'stop_loss': params['stop_loss'],
                'take_profit': params['take_profit']
            }
            
            portfolio_hist, trades = backtest.backtest(data_with_signals, bt_params)
            
            if len(portfolio_hist) > 1 and len(trades) > 0:
                calmar = metrics.calculate_calmar_ratio(pd.Series(portfolio_hist))
                
                if calmar > best_calmar and not np.isnan(calmar) and not np.isinf(calmar):
                    best_calmar = calmar
                    best_params = params.copy()
        except:
            continue
    
    if best_params:
        return best_params, best_calmar
    
    return None, -float('inf')

if __name__ == "__main__":
    # Test del módulo
    df = data_loader.load_data()
    df = data_loader.add_returns(df)
    train_df, test_df, val_df = data_loader.split_data(df)
    train_df_sample = train_df.tail(2000).reset_index(drop=True)
    study = optimize_strategy(train_df_sample, n_trials=20, n_splits=3)
    portfolio_hist, trades, test_metrics = validate_best_params(study, test_df.head(500))
