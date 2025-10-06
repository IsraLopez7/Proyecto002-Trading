"""
metrics.py
Módulo para calcular métricas de rendimiento
"""

import pandas as pd
import numpy as np
from datetime import datetime

def calculate_returns(portfolio_values):
    """
    Calcula los retornos del portafolio
    """
    if isinstance(portfolio_values, list):
        portfolio_values = pd.Series(portfolio_values)
    
    returns = portfolio_values.pct_change().dropna()
    return returns

def calculate_sharpe_ratio(portfolio_values, periods_per_year=8760, risk_free_rate=0.02):
    returns = calculate_returns(portfolio_values)
    if len(returns) == 0:
        return 0.0
    mean_return = returns.mean() * periods_per_year
    std_return = returns.std() * np.sqrt(periods_per_year)
    if std_return < 1e-6:   # ← piso numérico para evitar ratios absurdos
        return 0.0
    return (mean_return - risk_free_rate) / std_return


def calculate_sortino_ratio(portfolio_values, periods_per_year=8760, risk_free_rate=0.02):
    returns = calculate_returns(portfolio_values)
    if len(returns) == 0:
        return 0.0
    mean_return = returns.mean() * periods_per_year
    negative_returns = returns[returns < 0]
    if len(negative_returns) == 0:
        return 0.0  # sin pérdidas → ratio no informativo
    downside_std = negative_returns.std() * np.sqrt(periods_per_year)
    if downside_std < 1e-6:
        return 0.0
    return (mean_return - risk_free_rate) / downside_std


def calculate_calmar_ratio(portfolio_values, periods_per_year=8760):
    if isinstance(portfolio_values, list):
        portfolio_values = pd.Series(portfolio_values)
    if len(portfolio_values) < 2: return 0.0
    vi, vf = float(portfolio_values.iloc[0]), float(portfolio_values.iloc[-1])
    if vi <= 0 or vf <= 0: return 0.0
    n = len(portfolio_values)
    annualized_return = (vf/vi)**(periods_per_year/n) - 1.0
    max_dd = calculate_max_drawdown(portfolio_values)
    eps = 1e-12
    if not np.isfinite(max_dd) or max_dd < eps:
        return 0.0
    return float(annualized_return / max_dd)

def calculate_max_drawdown(portfolio_values):
    """
    Calcula el Maximum Drawdown (pérdida máxima desde un pico)
    """
    if isinstance(portfolio_values, list):
        portfolio_values = pd.Series(portfolio_values)
    
    # Calcular el máximo acumulado
    cummax = portfolio_values.expanding().max()
    
    # Calcular drawdown
    drawdown = (portfolio_values - cummax) / cummax
    
    max_drawdown = drawdown.min()
    return abs(max_drawdown)

def calculate_win_rate(trades):
    """
    Calcula la tasa de éxito (% de operaciones ganadoras)
    trades: lista de diccionarios con información de trades
    """
    if not trades:
        return 0
    
    winning_trades = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
    total_trades = len(trades)
    
    if total_trades == 0:
        return 0
    
    return winning_trades / total_trades

def calculate_profit_factor(trades):
    """
    Calcula el factor de beneficio (ganancias totales / pérdidas totales)
    """
    if not trades:
        return 0
    
    gross_profit = sum(trade['pnl'] for trade in trades if trade.get('pnl', 0) > 0)
    gross_loss = abs(sum(trade['pnl'] for trade in trades if trade.get('pnl', 0) < 0))
    
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0
    
    return gross_profit / gross_loss

def calculate_recovery_factor(portfolio_values):
    """
    Calcula el factor de recuperación (retorno total / max drawdown)
    """
    if len(portfolio_values) < 2:
        return 0
    
    total_return = portfolio_values.iloc[-1] - portfolio_values.iloc[0]
    max_dd = calculate_max_drawdown(portfolio_values) * portfolio_values.iloc[0]
    
    if max_dd == 0:
        return float('inf') if total_return > 0 else 0
    
    return total_return / max_dd

def calculate_cagr(portfolio_values, periods_per_year=8760):
    """
    CAGR basado en número de periodos y frecuencia anual efectiva.
    """
    if isinstance(portfolio_values, list):
        portfolio_values = pd.Series(portfolio_values)
    if len(portfolio_values) < 2:
        return 0.0
    vi = float(portfolio_values.iloc[0])
    vf = float(portfolio_values.iloc[-1])
    n = len(portfolio_values)
    if vi <= 0 or vf <= 0:
        return 0.0
    return (vf / vi) ** (periods_per_year / n) - 1.0

def calculate_all_metrics(portfolio_hist, trades=None, periods_per_year=8760):
    """
    Calcula todas las métricas de rendimiento (incluye CAGR).
    """
    if isinstance(portfolio_hist, list):
        portfolio_hist = pd.Series(portfolio_hist)

    metrics = {}

    initial_value = float(portfolio_hist.iloc[0])
    final_value = float(portfolio_hist.iloc[-1])
    total_return = (final_value / initial_value - 1) * 100

    metrics['initial_value'] = initial_value
    metrics['final_value'] = final_value
    metrics['total_return'] = total_return

    # Ratios
    metrics['sharpe_ratio']  = calculate_sharpe_ratio(portfolio_hist, periods_per_year)
    metrics['sortino_ratio'] = calculate_sortino_ratio(portfolio_hist, periods_per_year)
    metrics['calmar_ratio']  = calculate_calmar_ratio(portfolio_hist, periods_per_year)
    metrics['cagr']          = calculate_cagr(portfolio_hist, periods_per_year) * 100

    # Riesgo
    metrics['max_drawdown'] = calculate_max_drawdown(portfolio_hist) * 100
    metrics['volatility']   = calculate_returns(portfolio_hist).std() * np.sqrt(periods_per_year) * 100

    # Trading
    if trades:
        metrics['win_rate'] = calculate_win_rate(trades) * 100
        metrics['profit_factor'] = calculate_profit_factor(trades)
        metrics['total_trades'] = len(trades)

        winning_trades = [t['pnl'] for t in trades if t.get('pnl', 0) > 0]
        losing_trades  = [t['pnl'] for t in trades if t.get('pnl', 0) < 0]

        metrics['avg_win']  = np.mean(winning_trades) if winning_trades else 0
        metrics['avg_loss'] = np.mean(losing_trades) if losing_trades else 0

    return metrics

def print_metrics(metrics):
    """
    Imprime las métricas de forma formateada
    """
    print("\n" + "="*60)
    print("MÉTRICAS DE RENDIMIENTO")
    print("="*60)
    
    print("\n--- Rendimiento General ---")
    print(f"Valor Inicial: ${metrics['initial_value']:,.2f}")
    print(f"Valor Final: ${metrics['final_value']:,.2f}")
    print(f"Retorno Total: {metrics['total_return']:.2f}%")
    
    print("\n--- Ratios de Riesgo-Retorno ---")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"Sortino Ratio: {metrics['sortino_ratio']:.3f}")
    print(f"Calmar Ratio: {metrics['calmar_ratio']:.3f}")
    
    print("\n--- Métricas de Riesgo ---")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"Volatilidad Anual: {metrics['volatility']:.2f}%")
    
    if 'total_trades' in metrics:
        print("\n--- Estadísticas de Trading ---")
        print(f"Total de Operaciones: {metrics['total_trades']}")
        print(f"Tasa de Éxito: {metrics['win_rate']:.2f}%")
        print(f"Factor de Beneficio: {metrics['profit_factor']:.2f}")
        print(f"Ganancia Promedio: ${metrics['avg_win']:.2f}")
        print(f"Pérdida Promedio: ${metrics['avg_loss']:.2f}")

def get_monthly_returns(portfolio_values, dates):
    """
    Calcula retornos mensuales
    """
    df = pd.DataFrame({
        'date': dates,
        'value': portfolio_values
    })
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # Resample a mensual
    monthly = df.resample('M').last()
    monthly_returns = monthly.pct_change() * 100
    
    return monthly_returns

def get_quarterly_returns(portfolio_values, dates):
    """
    Calcula retornos trimestrales
    """
    df = pd.DataFrame({
        'date': dates,
        'value': portfolio_values
    })
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # Resample a trimestral
    quarterly = df.resample('Q').last()
    quarterly_returns = quarterly.pct_change() * 100
    
    return quarterly_returns

def get_annual_returns(portfolio_values, dates):
    """
    Calcula retornos anuales
    """
    df = pd.DataFrame({
        'date': dates,
        'value': portfolio_values
    })
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # Resample a anual
    annual = df.resample('Y').last()
    annual_returns = annual.pct_change() * 100
    
    return annual_returns

if __name__ == "__main__":
    # Test del módulo
    import numpy as np
    np.random.seed(42)
    
    # Simular valores de portafolio
    initial_value = 100000
    returns = np.random.randn(1000) * 0.01
    portfolio_values = [initial_value]
    
    for r in returns:
        portfolio_values.append(portfolio_values[-1] * (1 + r))
    
    portfolio_values = pd.Series(portfolio_values)
    
    # Simular algunos trades
    trades = [
        {'pnl': 100},
        {'pnl': -50},
        {'pnl': 200},
        {'pnl': -30},
        {'pnl': 150}
    ]
    
    # Calcular métricas
    metrics = calculate_all_metrics(portfolio_values, trades)