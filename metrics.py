"""
Módulo simplificado de cálculo de métricas
"""

import numpy as np

class PerformanceMetrics:
    def __init__(self):
        pass
    
    def calculate(self, equity, trades, initial_capital):
        """Calcula todas las métricas de performance"""
        
        equity_array = np.array(equity)
        final_capital = equity_array[-1]
        
        # Retorno total
        total_return = ((final_capital / initial_capital) - 1) * 100
        
        # Calcular retornos
        returns = np.diff(equity_array) / equity_array[:-1]
        returns = returns[~np.isnan(returns)]
        
        # Sharpe Ratio
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(365 * 24)
        else:
            sharpe = 0
        
        # Maximum Drawdown
        cummax = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - cummax) / cummax
        max_drawdown = np.min(drawdown) * 100
        
        # Calmar Ratio
        years = len(equity_array) / (365 * 24)
        annual_return = total_return / years if years > 0 else 0
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Métricas de trades
        if len(trades) > 0:
            winning_trades = [t for t in trades if t['pnl_pct'] > 0]
            win_rate = (len(winning_trades) / len(trades)) * 100
            
            # Promedio de ganancias y pérdidas
            wins = [t['pnl_pct'] for t in winning_trades]
            losses = [t['pnl_pct'] for t in trades if t['pnl_pct'] <= 0]
            
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            
            # Profit Factor
            total_wins = sum([t['pnl_usd'] for t in trades if t['pnl_usd'] > 0])
            total_losses = abs(sum([t['pnl_usd'] for t in trades if t['pnl_usd'] < 0]))
            profit_factor = total_wins / total_losses if total_losses > 0 else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        return {
            'capital_final': final_capital,
            'retorno_total': total_return,
            'retorno_anual': annual_return,
            'sharpe_ratio': sharpe,
            'calmar_ratio': calmar,
            'max_drawdown': max_drawdown,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }
    
    def print_summary(self, metrics, label=""):
        """Imprime resumen de métricas"""
        print(f"\n{'='*40}")
        print(f"MÉTRICAS {label}")
        print(f"{'='*40}")
        print(f"Capital Final: ${metrics['capital_final']:,.2f}")
        print(f"Retorno Total: {metrics['retorno_total']:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"Calmar Ratio: {metrics['calmar_ratio']:.3f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.1f}%")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")